// pybind_xroot2xkey.cpp
//
// xroot (= IL||IR = 32-byte priv + 32-byte chain) 批量 → BIP32 xprv Base58Check
//
// Python 接口：
//   import pybind_xroot2xkey as x2k
//   xprvs = x2k.xroots_to_xprv(buf, version_xprv=0x0488ADE4, depth=0, child_num=0)
//     - buf: 一维 bytes / numpy.uint8 buffer，长度必须是 64 * N
//     - version_xprv: 4 字节 BIP32 版本号（uint32，大端写入，例如 0x0488ADE4）
//     - depth:        BIP32 depth（root 通常为 0）
//     - child_num:    BIP32 child number（root 通常为 0）
//
// 返回：vector<string>，每条记录一个 xprv（不再包含 xpub）

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>

#include <omp.h>

#include <libbase58.h>
#include <openssl/evp.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// ---- Helpers ----

static inline void u32_be(uint8_t out[4], uint32_t v) {
    out[0] = static_cast<uint8_t>((v >> 24) & 0xFF);
    out[1] = static_cast<uint8_t>((v >> 16) & 0xFF);
    out[2] = static_cast<uint8_t>((v >> 8)  & 0xFF);
    out[3] = static_cast<uint8_t>(v & 0xFF);
}

// 使用调用方提供的 EVP_MD_CTX（线程本地），避免每条记录 new/free
static inline void sha256_once_ctx(EVP_MD_CTX* ctx,
                                   const uint8_t* data,
                                   size_t len,
                                   uint8_t out32[32]) {
    unsigned int out_len = 0;

    if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1 ||
        EVP_DigestUpdate(ctx, data, len) != 1 ||
        EVP_DigestFinal_ex(ctx, out32, &out_len) != 1) {
        throw std::runtime_error("EVP_sha256 digest failed");
    }

    if (out_len != 32) {
        throw std::runtime_error("EVP_sha256 produced unexpected length");
    }
}

static inline void sha256d_first4_ctx(EVP_MD_CTX* ctx1,
                                      EVP_MD_CTX* ctx2,
                                      const uint8_t* data,
                                      size_t len,
                                      uint8_t out4[4]) {
    uint8_t d1[32];
    uint8_t d2[32];

    sha256_once_ctx(ctx1, data, len, d1);
    sha256_once_ctx(ctx2, d1, sizeof(d1), d2);

    out4[0] = d2[0];
    out4[1] = d2[1];
    out4[2] = d2[2];
    out4[3] = d2[3];
}

// data(78) + checksum(4) → Base58Check string (no newline)
// 使用线程本地的 EVP_MD_CTX
static inline std::string b58check_encode_78_ctx(const uint8_t data[78],
                                                 EVP_MD_CTX* ctx1,
                                                 EVP_MD_CTX* ctx2) {
    uint8_t buf[78 + 4];
    std::memcpy(buf, data, 78);

    // 双 SHA256 checksum（前 4 字节）
    sha256d_first4_ctx(ctx1, ctx2, data, 78, buf + 78);

    char b58[128];
    size_t b58sz = sizeof(b58);
    if (!b58enc(b58, &b58sz, buf, sizeof(buf))) {
        throw std::runtime_error("b58enc failed");
    }

    // libbase58: b58sz = 实际长度（包含末尾 '\0'）
    if (b58sz == 0) {
        throw std::runtime_error("b58enc returned empty");
    }
    return std::string(b58, b58sz - 1);
}

// 构造 78 字节 BIP32 payload 并 Base58Check（只用于 xprv）
//
// payload:
//   [0..3]  version (big-endian)
//   [4]     depth
//   [5..8]  parent fingerprint (root: 0x00000000)
//   [9..12] child number (big-endian)
//   [13..44] chain code (32 bytes)
//   [45..77] key data (33 bytes): 0x00 || priv32
//
static inline std::string encode_xprv_ctx(const uint8_t version_be[4],
                                          uint8_t depth,
                                          uint32_t child_num,
                                          const uint8_t chain_code[32],
                                          const uint8_t priv32[32],
                                          EVP_MD_CTX* ctx1,
                                          EVP_MD_CTX* ctx2) {
    uint8_t payload[78];

    // version
    payload[0] = version_be[0];
    payload[1] = version_be[1];
    payload[2] = version_be[2];
    payload[3] = version_be[3];

    // depth
    payload[4] = depth;

    // parent fingerprint (root: 0)
    payload[5] = 0;
    payload[6] = 0;
    payload[7] = 0;
    payload[8] = 0;

    // child number
    u32_be(payload + 9, child_num);

    // chain code
    std::memcpy(payload + 13, chain_code, 32);

    // key data: 0x00 || priv32
    payload[45] = 0x00;
    std::memcpy(payload + 46, priv32, 32);

    return b58check_encode_78_ctx(payload, ctx1, ctx2);
}

// ---- Core function: xroots buffer -> vector<string>(xprv only) ----

static std::vector<std::string> xroots_to_xprv(py::buffer xroots_buf,
                                               uint32_t version_xprv,
                                               uint8_t depth,
                                               uint32_t child_num) {
    const size_t RECORD_SIZE = 64;  // 32 priv (IL) + 32 chain (IR)

    py::buffer_info info = xroots_buf.request();
    if (info.ndim != 1) {
        throw std::runtime_error("xroots buffer must be 1-D contiguous");
    }
    if (info.itemsize != 1) {
        throw std::runtime_error("xroots buffer must have itemsize == 1 (uint8/bytes)");
    }

    const size_t total_bytes = static_cast<size_t>(info.size);
    if (total_bytes % RECORD_SIZE != 0) {
        throw std::runtime_error("xroots buffer length must be multiple of 64 bytes");
    }

    const size_t num_records = total_bytes / RECORD_SIZE;
    auto* base_ptr = static_cast<uint8_t*>(info.ptr);

    uint8_t ver_prv_be[4];
    u32_be(ver_prv_be, version_xprv);

    std::vector<std::string> xprvs(num_records);

    // OpenMP: 记录级并行；每个线程有自己的 EVP_MD_CTX
    #pragma omp parallel
    {
        EVP_MD_CTX* md1 = EVP_MD_CTX_new();
        EVP_MD_CTX* md2 = EVP_MD_CTX_new();

        if (!md1 || !md2) {
            if (md1) EVP_MD_CTX_free(md1);
            if (md2) EVP_MD_CTX_free(md2);
            throw std::runtime_error("Failed to create EVP_MD_CTX");
        }

        #pragma omp for schedule(static)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(num_records); ++i) {
            const uint8_t* rec    = base_ptr + i * RECORD_SIZE;
            const uint8_t* priv32 = rec;        // IL
            const uint8_t* chain32= rec + 32;   // IR

            std::string xprv_str = encode_xprv_ctx(
                ver_prv_be,
                depth,
                child_num,
                chain32,
                priv32,
                md1,
                md2
            );

            xprvs[static_cast<size_t>(i)] = std::move(xprv_str);
        }

        EVP_MD_CTX_free(md1);
        EVP_MD_CTX_free(md2);
    }

    return xprvs;
}

// ---- pybind11 module ----

PYBIND11_MODULE(pybind_xroot2xkey, m) {
    m.doc() = "xroot(IL||IR, 32+32 bytes) -> BIP32 xprv Base58Check (OpenMP + libbase58 + OpenSSL)";

    m.def(
        "xroots_to_xprv",
        &xroots_to_xprv,
        py::arg("xroots"),
        py::arg("version_xprv") = 0x0488ADE4,  // mainnet xprv
        py::arg("depth")        = 0,
        py::arg("child_num")    = 0
        // 不使用 gil_scoped_release，避免 GIL assert + OpenMP 崩溃
    );
}