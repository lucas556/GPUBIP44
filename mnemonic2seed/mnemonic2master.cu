// mnemonic2master.cu   (原 pybind_mnemonic2master.cu 改造版)
//
// 功能：
//   从一个“每行一个助记词”的文本文件中，按批次在 GPU 上计算：
//     BIP39 PBKDF2-HMAC-SHA512(mnemonic, "mnemonic"+passphrase, 2048, 64B)
//     -> HMAC-SHA512("Bitcoin seed", seed) -> master I = IL||IR (64B)
//   并将所有 master I 顺序写入输出二进制文件：
//     每条记录恰好 64 字节：32B IL + 32B IR
//
// 使用方式：
//   nvcc -O3 -std=c++17 -Xcompiler "-march=native" -o mnemonic2master \
//        mnemonic2master.cu -L/usr/local/cuda/lib64 -lcudart
//
//   ./mnemonic2master <input_txt> <output_bin> \
//       [--batch-size N] [--threads-per-block T] [--passphrase P]
//
//   例如：
//   ./mnemonic2master mnemo_256_100000000.txt master_i.bin \
//       --batch-size 500000 \
//       --threads-per-block 128 \
//       --passphrase ""
//
// 说明：
//   - input_txt：每行一个 BIP39 助记词（可以有空格），空行会被忽略。
//   - output_bin：不带任何头部，严格为 (记录数 * 64) 字节。
//   - --passphrase 默认 ""（空字符串）。
//   - 会打印总条数、每批耗时、当前/整体速度和 ETA。

#include <cuda_runtime.h>

#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <mutex>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <algorithm>

#include "GPUSHA512.cuh"
#include "GPUPBKDF2.cuh"

using Byte = unsigned char;

// ---------------------- CUDA 工具宏 ----------------------
#define CUDA_CHECK(expr)                                                         \
    do {                                                                         \
        cudaError_t _err = (expr);                                               \
        if (_err != cudaSuccess) {                                               \
            throw std::runtime_error(std::string("CUDA error: ") +               \
                                     cudaGetErrorString(_err));                  \
        }                                                                        \
    } while (0)

static inline int clamp_tpb(int t) {
    if (t < 64)  t = 128;
    if (t > 256) t = 256;
    if (t % 32 != 0) t = ((t + 31) / 32) * 32; // warp 对齐
    return t;
}

static inline void CUDA_THROW_LAST(const char* where) {
    cudaError_t e = cudaPeekAtLastError();
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(e));
    }
}

static inline void CUDA_THROW_SYNC(const char* where) {
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(e));
    }
}

// ---------------------- 简单 RAII：GpuBuffer<T> ----------------------
template<typename T>
class GpuBuffer {
public:
    GpuBuffer() = default;
    explicit GpuBuffer(size_t n) { allocate(n); }

    ~GpuBuffer() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }

    GpuBuffer(const GpuBuffer&)            = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;

    GpuBuffer(GpuBuffer&& other) noexcept {
        ptr_  = other.ptr_;
        size_ = other.size_;
        other.ptr_  = nullptr;
        other.size_ = 0;
    }

    GpuBuffer& operator=(GpuBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_  = other.ptr_;
            size_ = other.size_;
            other.ptr_  = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    void allocate(size_t n) {
        if (n == 0) n = 1;  // 避免 cudaMalloc(0)
        if (ptr_) cudaFree(ptr_);
        CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
        size_ = n;
    }

    T* data()             { return ptr_; }
    const T* data() const { return ptr_; }
    size_t size()   const { return size_; }

private:
    T*     ptr_  = nullptr;
    size_t size_ = 0;
};

// ======================================================
//  公共：一次性初始化 SHA512 常量
// ======================================================
static std::once_flag g_sha512_once;

static inline void ensure_sha512_inited() {
    std::call_once(g_sha512_once, [](){
        initSHA512Constants();
    });
}

// ======================================================
//  扁平化助记词 + same_salt 视图构建（主机端）
// ======================================================
struct MnemonicDeviceView {
    GpuBuffer<Byte>     d_blob;
    GpuBuffer<uint32_t> d_off;
    GpuBuffer<uint32_t> d_len;
    GpuBuffer<Byte>     d_salt;
    uint32_t            salt_len = 0;
    int                 count    = 0;
};

static MnemonicDeviceView build_mnemonic_device_view(
    const std::vector<std::string>& mnemonics,
    const std::string& passphrase
) {
    MnemonicDeviceView view;
    const int n = static_cast<int>(mnemonics.size());
    if (n <= 0) {
        view.count = 0;
        return view;
    }
    view.count = n;
    const size_t n64 = static_cast<size_t>(n);

    // same_salt = "mnemonic" + passphrase
    std::string salt_text = std::string("mnemonic") + passphrase;
    uint32_t salt_len = static_cast<uint32_t>(salt_text.size());
    if (salt_len == 0) {
        throw std::runtime_error("Salt length should not be zero");
    }
    view.salt_len = salt_len;

    view.d_salt.allocate(salt_len);
    CUDA_CHECK(cudaMemcpy(
        view.d_salt.data(),
        salt_text.data(),
        salt_len,
        cudaMemcpyHostToDevice
    ));

    // 扁平化 mnemonics
    size_t m_total_len = 0;
    std::vector<uint32_t> h_off(n64);
    std::vector<uint32_t> h_len(n64);

    for (size_t i = 0; i < n64; ++i) {
        const auto& s = mnemonics[i];
        size_t len = s.size();
        if (len > 0xFFFFFFFFu) {
            throw std::runtime_error("mnemonic too long (>4GB)");
        }
        h_off[i] = static_cast<uint32_t>(m_total_len);
        h_len[i] = static_cast<uint32_t>(len);
        m_total_len += len;
        if (m_total_len > 0xFFFFFFFFu) {
            throw std::runtime_error("mnemonics blob too large (>4GB)");
        }
    }

    std::vector<Byte> h_blob(m_total_len ? m_total_len : 1);
    for (size_t i = 0; i < n64; ++i) {
        uint32_t off = h_off[i];
        uint32_t len = h_len[i];
        if (len) {
            std::memcpy(h_blob.data() + off,
                        mnemonics[i].data(),
                        len);
        }
    }

    view.d_blob.allocate(m_total_len ? m_total_len : 1);
    view.d_off.allocate(n64);
    view.d_len.allocate(n64);

    if (m_total_len) {
        CUDA_CHECK(cudaMemcpy(
            view.d_blob.data(),
            h_blob.data(),
            m_total_len,
            cudaMemcpyHostToDevice
        ));
    }
    CUDA_CHECK(cudaMemcpy(
        view.d_off.data(),
        h_off.data(),
        n64 * sizeof(uint32_t),
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        view.d_len.data(),
        h_len.data(),
        n64 * sizeof(uint32_t),
        cudaMemcpyHostToDevice
    ));

    return view;
}

// ======================================================
//  BIP39 + BIP32 master fused （内部用；只算到 master I）
//   d_out_I: n * 64 字节，master IL||IR
// ======================================================
static void run_fused_master_I_kernel(
    const MnemonicDeviceView& view,
    int threads_per_block,
    Byte* d_out_I
) {
    const int n   = view.count;
    if (n <= 0) return;
    const int TPB = clamp_tpb(threads_per_block);

    int blocks = (n + TPB - 1) / TPB;
    pbkdf2_hmac_bitcoinseed_kernel_flat_same_salt<<<blocks, TPB>>>(
        view.d_blob.data(),
        view.d_off.data(),
        view.d_len.data(),
        view.d_salt.data(),
        view.salt_len,
        PBKDF2_HMAC_SHA512_ITERATIONS,
        d_out_I,
        n
    );
    CUDA_THROW_LAST("pbkdf2_hmac_bitcoinseed_kernel_flat_same_salt launch");
}

// ======================================================
//  小工具：去掉行首尾空白
// ======================================================
static inline void trim(std::string& s) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
}

// ======================================================
//  CLI 参数解析
// ======================================================
struct CliOptions {
    std::string input_path;
    std::string output_path;
    std::string passphrase = "";   // 默认空字符串
    size_t      batch_size = 500000;
    int         tpb        = 128;
};

static CliOptions parse_args(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "用法:\n"
                  << "  " << argv[0] << " <input_txt> <output_bin> "
                  << "[--batch-size N] [--threads-per-block T] [--passphrase P]\n\n"
                  << "示例:\n"
                  << "  " << argv[0]
                  << " mnemo_256_100000000.txt master_i.bin "
                  << "--batch-size 500000 --threads-per-block 128 --passphrase \"\"\n";
        std::exit(1);
    }

    CliOptions opt;
    opt.input_path  = argv[1];
    opt.output_path = argv[2];

    int i = 3;
    while (i < argc) {
        std::string arg = argv[i];
        auto next = [&](int& idx) -> const char* {
            if (idx + 1 >= argc) {
                std::cerr << "缺少参数值: " << arg << "\n";
                std::exit(1);
            }
            return argv[++idx];
        };

        if (arg == "--batch-size") {
            opt.batch_size = std::stoull(next(i));
        } else if (arg.rfind("--batch-size=", 0) == 0) {
            opt.batch_size = std::stoull(arg.substr(std::strlen("--batch-size=")));
        } else if (arg == "--threads-per-block") {
            opt.tpb = std::stoi(next(i));
        } else if (arg.rfind("--threads-per-block=", 0) == 0) {
            opt.tpb = std::stoi(arg.substr(std::strlen("--threads-per-block=")));
        } else if (arg == "--passphrase") {
            opt.passphrase = next(i);
        } else if (arg.rfind("--passphrase=", 0) == 0) {
            opt.passphrase = arg.substr(std::strlen("--passphrase="));
        } else {
            std::cerr << "未知参数: " << arg << "\n";
            std::exit(1);
        }
        ++i;
    }

    opt.tpb = clamp_tpb(opt.tpb);
    return opt;
}

// ======================================================
//  main：按批次读取助记词 -> GPU 计算 master I -> 写文件
// ======================================================
int main(int argc, char** argv) {
    try {
        CliOptions opt = parse_args(argc, argv);

        // 第一次遍历：统计总行数
        std::ifstream fin_count(opt.input_path);
        if (!fin_count) {
            std::cerr << "无法打开输入文件: " << opt.input_path << "\n";
            return 1;
        }
        size_t total = 0;
        std::string line;
        while (std::getline(fin_count, line)) {
            trim(line);
            if (line.empty()) continue;
            ++total;
        }
        fin_count.close();

        if (total == 0) {
            std::cerr << "输入文件中没有助记词行\n";
            return 1;
        }

        std::cout << "总助记词数量: " << total << "\n";
        std::cout << "输出文件: " << opt.output_path << "\n";
        std::cout << "批大小: " << opt.batch_size
                  << ", threads_per_block: " << opt.tpb << "\n";
        std::cout << "BIP39 passphrase: \"" << opt.passphrase << "\"\n";

        // 打开输入/输出
        std::ifstream fin(opt.input_path);
        if (!fin) {
            std::cerr << "无法重新打开输入文件: " << opt.input_path << "\n";
            return 1;
        }
        std::ofstream fout(opt.output_path, std::ios::binary);
        if (!fout) {
            std::cerr << "无法打开输出文件: " << opt.output_path << "\n";
            return 1;
        }

        ensure_sha512_inited();

        size_t processed = 0;
        size_t batch_id  = 0;

        auto t_start_all = std::chrono::steady_clock::now();

        while (processed < total) {
            // 读一批助记词
            std::vector<std::string> mnemos;
            mnemos.reserve(opt.batch_size);

            while (mnemos.size() < opt.batch_size && std::getline(fin, line)) {
                trim(line);
                if (line.empty()) continue;
                mnemos.push_back(line);
            }
            if (mnemos.empty()) break;

            size_t this_batch = mnemos.size();
            batch_id++;

            auto t_batch_start = std::chrono::steady_clock::now();

            // 构建 device 视图
            MnemonicDeviceView view = build_mnemonic_device_view(mnemos, opt.passphrase);

            // 在 GPU 上计算 master I
            GpuBuffer<Byte> d_I(view.count * SHA512_DIGEST_SIZE);
            run_fused_master_I_kernel(view, opt.tpb, d_I.data());
            CUDA_THROW_SYNC("run_fused_master_I_kernel");

            // 拷回 host
            std::vector<Byte> h_I(view.count * SHA512_DIGEST_SIZE);
            CUDA_CHECK(cudaMemcpy(
                h_I.data(),
                d_I.data(),
                h_I.size(),
                cudaMemcpyDeviceToHost
            ));

            // 写文件（每条 64B）
            fout.write(reinterpret_cast<const char*>(h_I.data()), h_I.size());
            if (!fout) {
                std::cerr << "写输出文件失败\n";
                return 1;
            }

            processed += this_batch;

            auto t_batch_end = std::chrono::steady_clock::now();
            double batch_sec = std::chrono::duration<double>(t_batch_end - t_batch_start).count();
            auto   t_now     = t_batch_end;
            double all_sec   = std::chrono::duration<double>(t_now - t_start_all).count();

            double batch_speed   = this_batch / batch_sec;
            double overall_speed = processed / all_sec;
            double remaining     = (total > processed)
                                   ? ( (double)(total - processed) / overall_speed )
                                   : 0.0;

            int rem_min = (int)(remaining / 60.0);
            int rem_sec = (int)(remaining - rem_min * 60);

            std::cout << "[batch " << std::setw(4) << batch_id << "] "
                      << "processed "
                      << std::setw(10) << processed << "/"
                      << std::setw(10) << total
                      << "  batch_time=" << std::fixed << std::setprecision(2) << batch_sec << "s"
                      << "  batch_speed="   << std::setw(10) << (uint64_t)batch_speed   << " rec/s"
                      << "  overall_speed=" << std::setw(10) << (uint64_t)overall_speed << " rec/s"
                      << "  ETA=" << rem_min << " min " << rem_sec << " s"
                      << std::endl;
        }

        fout.close();
        fin.close();

        std::cout << "完成。输出文件应该是 " << (processed * 64)
                  << " 字节（" << processed << " 条 master I 记录，每条 64B）\n";

        // 可选：强制把 CUDA 错误吐出来
        CUDA_THROW_SYNC("final sync");

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "异常: " << ex.what() << "\n";
        return 1;
    }
}
