// pybind_mnemonic2master.cu
//
// 功能：GPU 整条链路到 BIP32 master I：
//   BIP39 PBKDF2-HMAC-SHA512(mnemonic, "mnemonic"+passphrase, 2048, 64B)
//   → HMAC-SHA512("Bitcoin seed", seed) → master I = IL||IR
//
// 对外暴露：
//   derive_master_I_raw(mnemonics, passphrase="", threads_per_block=128)
//     - mnemonics: [string, ...]
//     - passphrase: BIP39 口令，可为空字符串
//     - 返回 bytes，长度 = n * 64，按 (IL||IR) 顺序拼接
//
// 依赖：GPUSHA512.cuh, GPUPBKDF2.cuh
//   - initSHA512Constants()
//   - pbkdf2_hmac_bitcoinseed_kernel_flat_same_salt(...)
//   - PBKDF2_HMAC_SHA512_ITERATIONS

#include <cuda_runtime.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <mutex>

#include "GPUSHA512.cuh"
#include "GPUPBKDF2.cuh"

namespace py = pybind11;
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
//  对外接口：mnemonics (+passphrase) → master I(IL||IR)
// ======================================================

static py::bytes derive_master_I_raw_impl(
    const std::vector<std::string>& mnemonics,
    const std::string&              passphrase,
    int                             threads_per_block
) {
    const int n = static_cast<int>(mnemonics.size());
    if (n <= 0) {
        return py::bytes();
    }

    ensure_sha512_inited();

    // 1) 构建助记词 device 视图
    MnemonicDeviceView view = build_mnemonic_device_view(mnemonics, passphrase);

    const int    count  = view.count;
    const size_t n64    = static_cast<size_t>(count);
    const int    TPB    = clamp_tpb(threads_per_block);

    // 2) master I buffer on device
    GpuBuffer<Byte> d_I_master(n64 * SHA512_DIGEST_SIZE);
    run_fused_master_I_kernel(view, TPB, d_I_master.data());

    CUDA_THROW_SYNC("derive_master_I_raw sync");

    // 3) 拷回 host
    std::vector<Byte> h_I(n64 * SHA512_DIGEST_SIZE);
    CUDA_CHECK(cudaMemcpy(
        h_I.data(),
        d_I_master.data(),
        h_I.size(),
        cudaMemcpyDeviceToHost
    ));

    return py::bytes(reinterpret_cast<const char*>(h_I.data()), h_I.size());
}

// ---------------------- PyBind11 模块 ----------------------

PYBIND11_MODULE(pybind_mnemonic2master, m) {
    m.doc() =
        "GPU BIP39 PBKDF2 + HMAC_SHA512('Bitcoin seed') "
        "-> BIP32 master I(IL,IR) raw bytes";

    m.def("derive_master_I_raw",
          &derive_master_I_raw_impl,
          py::arg("mnemonics"),
          py::arg("passphrase") = "",
          py::arg("threads_per_block") = 128,
          py::call_guard<py::gil_scoped_release>());
}
