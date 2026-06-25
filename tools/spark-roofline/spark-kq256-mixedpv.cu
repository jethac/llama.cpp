#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define GGML_COMMON_DECL_CUDA
#include "ggml-common.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CUDA_CHECK(err) spark_cuda_check((err), __FILE__, __LINE__)

__device__ __constant__ int8_t k_e2m1_values_x2[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};

static void spark_cuda_check(cudaError_t err, const char * file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s:%d: CUDA error: %s\n", file, line, cudaGetErrorString(err));
        std::exit(1);
    }
}

struct bench_config {
    int      device   = 0;
    int      blocks   = 0;
    int      threads  = 128;
    int      mtp_rows = 4;
    uint64_t iters    = 2000;
};

static void print_usage(const char * exe) {
    std::printf(
        "usage: %s [options]\n"
        "\n"
        "D=256 native-FP4 KQ plus mixed-PV probe at MTP tile height.\n"
        "\n"
        "options:\n"
        "  --device N    CUDA device id (default: 0)\n"
        "  --blocks N    CUDA blocks (default: 4 * SM count)\n"
        "  --threads N   CUDA threads per block (default: 128)\n"
        "  --mtp-rows N  useful MTP verification rows in an m16 tile (default: 4)\n"
        "  --iters N     loop iterations per warp (default: 2000)\n"
        "  --help        print this help\n",
        exe);
}

static uint64_t parse_u64(const char * s, const char * name) {
    char * end = nullptr;
    const unsigned long long v = std::strtoull(s, &end, 10);
    if (end == s || *end != '\0') {
        std::fprintf(stderr, "invalid value for %s: %s\n", name, s);
        std::exit(1);
    }
    return (uint64_t) v;
}

static bench_config parse_args(int argc, char ** argv) {
    bench_config cfg;

    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];
        const auto require_value = [&](const char * name) -> const char * {
            if (++i >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(1);
            }
            return argv[i];
        };

        if (std::strcmp(arg, "--device") == 0) {
            cfg.device = (int) parse_u64(require_value(arg), arg);
        } else if (std::strcmp(arg, "--blocks") == 0) {
            cfg.blocks = (int) parse_u64(require_value(arg), arg);
        } else if (std::strcmp(arg, "--threads") == 0) {
            cfg.threads = (int) parse_u64(require_value(arg), arg);
        } else if (std::strcmp(arg, "--mtp-rows") == 0) {
            cfg.mtp_rows = (int) parse_u64(require_value(arg), arg);
        } else if (std::strcmp(arg, "--iters") == 0) {
            cfg.iters = parse_u64(require_value(arg), arg);
        } else if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg);
            print_usage(argv[0]);
            std::exit(1);
        }
    }

    if (cfg.threads <= 0 || cfg.threads % 32 != 0 || cfg.mtp_rows <= 0 || cfg.mtp_rows > 16 || cfg.iters == 0) {
        std::fprintf(stderr, "invalid benchmark configuration\n");
        std::exit(1);
    }

    return cfg;
}

static __device__ __forceinline__ uint8_t fp32_to_ue4m3(float x) {
    if (!(x > 0.0f)) {
        return 0;
    }
    if (x > 448.0f) {
        x = 448.0f;
    }

    const uint32_t bits = __float_as_uint(x);
    const int fp32_exp = (int) ((bits >> 23) & 0xffu) - 127;
    const int fp32_man = (int) ((bits >> 20) & 0x7u);
    int ue4m3_exp = fp32_exp + 7;

    if (ue4m3_exp <= 0) {
        int man = (int) (x * 512.0f + 0.5f);
        man = max(0, min(7, man));
        return (uint8_t) man;
    }
    if (ue4m3_exp >= 15) {
        return 0x7e;
    }

    const int round_bit = (int) ((bits >> 19) & 1u);
    int ue4m3_man = fp32_man + round_bit;
    if (ue4m3_man > 7) {
        ue4m3_man = 0;
        ++ue4m3_exp;
        if (ue4m3_exp >= 15) {
            return 0x7e;
        }
    }
    return (uint8_t) ((ue4m3_exp << 3) | ue4m3_man);
}

static __device__ __forceinline__ float ue4m3_to_fp32(uint8_t x) {
    if (x == 0 || x == 0x7f) {
        return 0.0f;
    }
    const int exp = (int) ((x >> 3) & 0xfu);
    const int man = (int) (x & 0x7u);
    return exp == 0 ? ldexpf((float) man, -9) : ldexpf(1.0f + (float) man / 8.0f, exp - 7);
}

static __device__ __forceinline__ float ue4m3_to_fp32_halfscale(uint8_t x) {
    return ue4m3_to_fp32(x) * 0.5f;
}

static __device__ __forceinline__ float fp4_e2m1_to_fp32(uint8_t x) {
    const uint8_t sign = x & 0x08u;
    const uint8_t exp  = x & 0x06u;
    const uint8_t mant = x & 0x01u;

    float v = 0.0f;
    if (exp == 0) {
        v = mant ? 0.5f : 0.0f;
    } else if (exp == 2) {
        v = mant ? 1.5f : 1.0f;
    } else if (exp == 4) {
        v = mant ? 3.0f : 2.0f;
    } else {
        v = mant ? 6.0f : 4.0f;
    }
    return sign ? -v : v;
}

static __device__ __forceinline__ uint32_t half2_bits(__half2 v) {
    union {
        __half2  h;
        uint32_t u;
    } cvt;
    cvt.h = v;
    return cvt.u;
}

static __device__ __forceinline__ __half2 dequant_nvfp4_byte_to_half2(uint8_t q, float d) {
    const float lo = d * fp4_e2m1_to_fp32(q & 0x0f);
    const float hi = d * fp4_e2m1_to_fp32(q >> 4);
    return __floats2half2_rn(lo, hi);
}

static __device__ __forceinline__ uint8_t nearest_mxfp4_code(float x, float d) {
    uint8_t best = 0;
    float best_err = fabsf((float) k_e2m1_values_x2[0] * d - x);
#pragma unroll
    for (int i = 1; i < 16; ++i) {
        const float err = fabsf((float) k_e2m1_values_x2[i] * d - x);
        if (err < best_err) {
            best_err = err;
            best = (uint8_t) i;
        }
    }
    return best;
}

__global__ void fill_f32_kernel(float * data, size_t n) {
    const size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t) gridDim.x * blockDim.x;

    for (size_t i = tid; i < n; i += stride) {
        const uint32_t x = (uint32_t) i * 1664525u + 1013904223u;
        data[i] = ((float) (int) (x & 0x3ffu) - 511.5f) * 0.00390625f;
    }
}

__global__ void fill_nvfp4_kernel(block_nvfp4 * data, size_t n) {
    const size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t) gridDim.x * blockDim.x;

    for (size_t i = tid; i < n; i += stride) {
#pragma unroll
        for (int s = 0; s < QK_NVFP4 / QK_NVFP4_SUB; ++s) {
            data[i].d[s] = (uint8_t) (0x38 + ((i + s) & 0x7));
        }
#pragma unroll
        for (int q = 0; q < QK_NVFP4 / 2; ++q) {
            data[i].qs[q] = (uint8_t) ((q + i) * 17u);
        }
    }
}

__global__ void quantize_q_tile_kernel(const float * src, block_nvfp4 * dst, size_t n_blocks, uint64_t repeats) {
    const size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t) gridDim.x * blockDim.x;

    for (uint64_t r = 0; r < repeats; ++r) {
        for (size_t b = tid; b < n_blocks; b += stride) {
            const float * x = src + b * QK_NVFP4;
            block_nvfp4 out;

#pragma unroll
            for (int s = 0; s < QK_NVFP4 / QK_NVFP4_SUB; ++s) {
                float amax = 0.0f;
#pragma unroll
                for (int j = 0; j < QK_NVFP4_SUB; ++j) {
                    amax = fmaxf(amax, fabsf(x[s * QK_NVFP4_SUB + j]));
                }

                const uint8_t ue = fp32_to_ue4m3(amax / 6.0f);
                const float d = ue4m3_to_fp32_halfscale(ue);
                out.d[s] = ue;

#pragma unroll
                for (int j = 0; j < QK_NVFP4_SUB / 2; ++j) {
                    const uint8_t lo = nearest_mxfp4_code(x[s * QK_NVFP4_SUB + j], d);
                    const uint8_t hi = nearest_mxfp4_code(x[s * QK_NVFP4_SUB + j + QK_NVFP4_SUB / 2], d);
                    out.qs[s * (QK_NVFP4_SUB / 2) + j] = (uint8_t) (lo | (hi << 4));
                }
            }

            dst[b] = out;
        }
    }
}

__device__ __forceinline__ void kq_mma_accumulate(
        const int      ax0,
        const int      ax1,
        const int      ax2,
        const int      ax3,
        const int      bx0,
        const int      bx1,
        const uint32_t q_scale,
        const uint32_t k_scale,
        float &        d0,
        float &        d1,
        float &        d2,
        float &        d3) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
        "%10, {0, 0}, %11, {0, 0};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(ax0), "r"(ax1), "r"(ax2), "r"(ax3), "r"(bx0), "r"(bx1), "r"(q_scale), "r"(k_scale));
#else
    (void) ax0; (void) ax1; (void) ax2; (void) ax3; (void) bx0; (void) bx1; (void) q_scale; (void) k_scale;
    (void) d0; (void) d1; (void) d2; (void) d3;
#endif
}

__device__ __forceinline__ void pv_mma_accumulate(
        const uint32_t ax0,
        const uint32_t ax1,
        const uint32_t ax2,
        const uint32_t ax3,
        const uint32_t bx0,
        const uint32_t bx1,
        float &        d0,
        float &        d1,
        float &        d2,
        float &        d3) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(ax0), "r"(ax1), "r"(ax2), "r"(ax3), "r"(bx0), "r"(bx1));
#else
    (void) ax0; (void) ax1; (void) ax2; (void) ax3; (void) bx0; (void) bx1;
    (void) d0; (void) d1; (void) d2; (void) d3;
#endif
}

__global__ void kq256_only_kernel(
        const block_nvfp4 * q,
        const block_nvfp4 * k,
        size_t              block_mask,
        float *             out,
        uint64_t            iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    constexpr int nfrags = 256 / QK_NVFP4;
    constexpr int k_tiles = 64;

    const int lane = threadIdx.x & 31;
    const int warp = ((int) blockIdx.x * blockDim.x + threadIdx.x) >> 5;

    int q_a[nfrags][4];
    uint32_t q_scale[nfrags];

#pragma unroll
    for (int frag = 0; frag < nfrags; ++frag) {
        const size_t q_idx = (((size_t) warp * (size_t) nfrags + (size_t) frag) * 32u + (size_t) lane) & block_mask;
        const block_nvfp4 q_blk = q[q_idx];
        const uint32_t * q_qs = reinterpret_cast<const uint32_t *>(q_blk.qs);
        q_a[frag][0] = (int) q_qs[(lane + 0) & 7];
        q_a[frag][1] = (int) q_qs[(lane + 1) & 7];
        q_a[frag][2] = (int) q_qs[(lane + 2) & 7];
        q_a[frag][3] = (int) q_qs[(lane + 3) & 7];
        q_scale[frag] = *reinterpret_cast<const uint32_t *>(q_blk.d);
    }

    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;

    for (uint64_t i = 0; i < iters; ++i) {
#pragma unroll
        for (int tile = 0; tile < k_tiles; ++tile) {
#pragma unroll
            for (int frag = 0; frag < nfrags; ++frag) {
                const size_t k_idx =
                    (((size_t) warp + (size_t) i * (size_t) gridDim.x) * (size_t) k_tiles * (size_t) nfrags * 32u +
                     (size_t) tile * (size_t) nfrags * 32u + (size_t) frag * 32u + (size_t) lane) & block_mask;

                const block_nvfp4 k_blk = k[k_idx];
                const uint32_t * k_qs = reinterpret_cast<const uint32_t *>(k_blk.qs);
                const int bx0 = (int) k_qs[(lane + 0) & 7];
                const int bx1 = (int) k_qs[(lane + 1) & 7];
                const uint32_t k_scale = *reinterpret_cast<const uint32_t *>(k_blk.d);
                kq_mma_accumulate(q_a[frag][0], q_a[frag][1], q_a[frag][2], q_a[frag][3], bx0, bx1, q_scale[frag], k_scale, d0, d1, d2, d3);
            }
        }
    }

    if (lane == 0) {
        out[warp] = d0 + d1 + d2 + d3;
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = 0.0f;
    }
    (void) q; (void) k; (void) block_mask; (void) out; (void) iters;
#endif
}

template <bool with_kq>
__global__ void mixedpv_kernel(
        const block_nvfp4 * q,
        const block_nvfp4 * k,
        const block_nvfp4 * v,
        size_t              block_mask,
        float *             out,
        uint64_t            iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    constexpr int nfrags = 256 / QK_NVFP4;
    constexpr int k_tiles = 64;
    constexpr int groups_per_frag = QK_NVFP4 / 8;

    const int lane = threadIdx.x & 31;
    const int warp = ((int) blockIdx.x * blockDim.x + threadIdx.x) >> 5;

    int q_a[nfrags][4];
    uint32_t q_scale[nfrags];

#pragma unroll
    for (int frag = 0; frag < nfrags; ++frag) {
        const size_t q_idx = (((size_t) warp * (size_t) nfrags + (size_t) frag) * 32u + (size_t) lane) & block_mask;
        const block_nvfp4 q_blk = q[q_idx];
        const uint32_t * q_qs = reinterpret_cast<const uint32_t *>(q_blk.qs);
        q_a[frag][0] = (int) q_qs[(lane + 0) & 7];
        q_a[frag][1] = (int) q_qs[(lane + 1) & 7];
        q_a[frag][2] = (int) q_qs[(lane + 2) & 7];
        q_a[frag][3] = (int) q_qs[(lane + 3) & 7];
        q_scale[frag] = *reinterpret_cast<const uint32_t *>(q_blk.d);
    }

    float kq0 = 0.0f;
    float kq1 = 0.0f;
    float kq2 = 0.0f;
    float kq3 = 0.0f;
    float pv0 = 0.0f;
    float pv1 = 0.0f;
    float pv2 = 0.0f;
    float pv3 = 0.0f;
    float row_m0 = -64.0f;
    float row_l0 = 0.0f;
    float sink = 0.0f;

    for (uint64_t i = 0; i < iters; ++i) {
#pragma unroll
        for (int tile = 0; tile < k_tiles; ++tile) {
            if (with_kq) {
#pragma unroll
                for (int frag = 0; frag < nfrags; ++frag) {
                    const size_t k_idx =
                        (((size_t) warp + (size_t) i * (size_t) gridDim.x) * (size_t) k_tiles * (size_t) nfrags * 32u +
                         (size_t) tile * (size_t) nfrags * 32u + (size_t) frag * 32u + (size_t) lane) & block_mask;

                    const block_nvfp4 k_blk = k[k_idx];
                    const uint32_t * k_qs = reinterpret_cast<const uint32_t *>(k_blk.qs);
                    const int bx0 = (int) k_qs[(lane + 0) & 7];
                    const int bx1 = (int) k_qs[(lane + 1) & 7];
                    const uint32_t k_scale = *reinterpret_cast<const uint32_t *>(k_blk.d);
                    kq_mma_accumulate(q_a[frag][0], q_a[frag][1], q_a[frag][2], q_a[frag][3], bx0, bx1, q_scale[frag], k_scale, kq0, kq1, kq2, kq3);
                }

                const int q_row = lane & 15;
                const int base_pos = (int) (i & 1023u);
                const bool causal = tile <= base_pos + q_row;
                const bool swa = tile + 1024 >= base_pos + q_row;
                const float score = (causal && swa) ? (kq0 + kq1 + kq2 + kq3) * 0.000244140625f : -64.0f;
                const float next_m0 = fmaxf(row_m0, score);
                const float alpha0 = row_l0 == 0.0f ? 0.0f : exp2f(row_m0 - next_m0);
                row_l0 = row_l0 * alpha0 + exp2f(score - next_m0);
                row_m0 = next_m0;
            }

            const float p_base = with_kq ? 1.0f / (row_l0 + 1.0f) : 0.015625f;
            const uint32_t pax0 = half2_bits(__floats2half2_rn(p_base, p_base * 0.9375f));
            const uint32_t pax1 = half2_bits(__floats2half2_rn(p_base * 0.875f, p_base * 0.8125f));
            const uint32_t pax2 = half2_bits(__floats2half2_rn(p_base * 0.7500f, p_base * 0.6875f));
            const uint32_t pax3 = half2_bits(__floats2half2_rn(p_base * 0.6250f, p_base * 0.5625f));

            const size_t row_base = (((size_t) warp + (size_t) i * (size_t) gridDim.x) * (size_t) k_tiles * 64u +
                                     (size_t) tile * 64u) & block_mask;

#pragma unroll
            for (int frag = 0; frag < nfrags; ++frag) {
                const block_nvfp4 blk0 = v[(((row_base + (size_t) (2 * lane + 0)) * (size_t) nfrags) + (size_t) frag) & block_mask];
                const block_nvfp4 blk1 = v[(((row_base + (size_t) (2 * lane + 1)) * (size_t) nfrags) + (size_t) frag) & block_mask];
                const uint32_t * v0 = reinterpret_cast<const uint32_t *>(blk0.qs);
                const uint32_t * v1 = reinterpret_cast<const uint32_t *>(blk1.qs);

#pragma unroll
                for (int group = 0; group < groups_per_frag; ++group) {
                    const uint32_t word0 = v0[group];
                    const uint32_t word1 = v1[group];
                    const float d0 = ue4m3_to_fp32(blk0.d[group / 2]);
                    const float d1 = ue4m3_to_fp32(blk1.d[group / 2]);

#pragma unroll
                    for (int kk = 0; kk < 4; ++kk) {
                        const uint8_t q0b = (uint8_t) (word0 >> (8 * kk));
                        const uint8_t q1b = (uint8_t) (word1 >> (8 * kk));
                        const uint32_t bx0 = half2_bits(dequant_nvfp4_byte_to_half2(q0b, d0));
                        const uint32_t bx1 = half2_bits(dequant_nvfp4_byte_to_half2(q1b, d1));
                        pv_mma_accumulate(pax0, pax1, pax2, pax3, bx0, bx1, pv0, pv1, pv2, pv3);
                    }

                    sink += (pv0 + pv1 + pv2 + pv3) * (float) (frag * groups_per_frag + group + 1) * 0.0000001f;
                }
            }
        }
    }

    if (lane == 0) {
        out[warp] = kq0 + kq1 + kq2 + kq3 + pv0 + pv1 + pv2 + pv3 + row_m0 + row_l0 + sink;
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = 0.0f;
    }
    (void) q; (void) k; (void) v; (void) block_mask; (void) out; (void) iters;
#endif
}

template <int pv_groups>
__global__ void mixedpv_stripmine_kernel(
        const block_nvfp4 * q,
        const block_nvfp4 * k,
        const block_nvfp4 * v,
        size_t              block_mask,
        float *             out,
        uint64_t            iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    static_assert(pv_groups == 1 || pv_groups == 2 || pv_groups == 4, "unsupported strip-mined PV group count");
    constexpr int nfrags = 256 / QK_NVFP4;
    constexpr int k_tiles = 64;

    const int lane = threadIdx.x & 31;
    const int warp = ((int) blockIdx.x * blockDim.x + threadIdx.x) >> 5;

    int q_a[nfrags][4];
    uint32_t q_scale[nfrags];

#pragma unroll
    for (int frag = 0; frag < nfrags; ++frag) {
        const size_t q_idx = (((size_t) warp * (size_t) nfrags + (size_t) frag) * 32u + (size_t) lane) & block_mask;
        const block_nvfp4 q_blk = q[q_idx];
        const uint32_t * q_qs = reinterpret_cast<const uint32_t *>(q_blk.qs);
        q_a[frag][0] = (int) q_qs[(lane + 0) & 7];
        q_a[frag][1] = (int) q_qs[(lane + 1) & 7];
        q_a[frag][2] = (int) q_qs[(lane + 2) & 7];
        q_a[frag][3] = (int) q_qs[(lane + 3) & 7];
        q_scale[frag] = *reinterpret_cast<const uint32_t *>(q_blk.d);
    }

    float kq0 = 0.0f;
    float kq1 = 0.0f;
    float kq2 = 0.0f;
    float kq3 = 0.0f;
    float pv0 = 0.0f;
    float pv1 = 0.0f;
    float pv2 = 0.0f;
    float pv3 = 0.0f;
    float row_m0 = -64.0f;
    float row_l0 = 0.0f;
    float sink = 0.0f;

    for (uint64_t i = 0; i < iters; ++i) {
#pragma unroll
        for (int tile = 0; tile < k_tiles; ++tile) {
#pragma unroll
            for (int frag = 0; frag < nfrags; ++frag) {
                const size_t k_idx =
                    (((size_t) warp + (size_t) i * (size_t) gridDim.x) * (size_t) k_tiles * (size_t) nfrags * 32u +
                     (size_t) tile * (size_t) nfrags * 32u + (size_t) frag * 32u + (size_t) lane) & block_mask;

                const block_nvfp4 k_blk = k[k_idx];
                const uint32_t * k_qs = reinterpret_cast<const uint32_t *>(k_blk.qs);
                const int bx0 = (int) k_qs[(lane + 0) & 7];
                const int bx1 = (int) k_qs[(lane + 1) & 7];
                const uint32_t k_scale = *reinterpret_cast<const uint32_t *>(k_blk.d);
                kq_mma_accumulate(q_a[frag][0], q_a[frag][1], q_a[frag][2], q_a[frag][3], bx0, bx1, q_scale[frag], k_scale, kq0, kq1, kq2, kq3);
            }

            const int q_row = lane & 15;
            const int base_pos = (int) (i & 1023u);
            const bool causal = tile <= base_pos + q_row;
            const bool swa = tile + 1024 >= base_pos + q_row;
            const float score = (causal && swa) ? (kq0 + kq1 + kq2 + kq3) * 0.000244140625f : -64.0f;
            const float next_m0 = fmaxf(row_m0, score);
            const float alpha0 = row_l0 == 0.0f ? 0.0f : exp2f(row_m0 - next_m0);
            row_l0 = row_l0 * alpha0 + exp2f(score - next_m0);
            row_m0 = next_m0;

            const float p_base = 1.0f / (row_l0 + 1.0f);
            const uint32_t pax0 = half2_bits(__floats2half2_rn(p_base, p_base * 0.9375f));
            const uint32_t pax1 = half2_bits(__floats2half2_rn(p_base * 0.875f, p_base * 0.8125f));
            const uint32_t pax2 = half2_bits(__floats2half2_rn(p_base * 0.7500f, p_base * 0.6875f));
            const uint32_t pax3 = half2_bits(__floats2half2_rn(p_base * 0.6250f, p_base * 0.5625f));

            const size_t row_base = (((size_t) warp + (size_t) i * (size_t) gridDim.x) * (size_t) k_tiles * 64u +
                                     (size_t) tile * 64u) & block_mask;

#pragma unroll
            for (int frag = 0; frag < nfrags; ++frag) {
                const block_nvfp4 blk0 = v[(((row_base + (size_t) (2 * lane + 0)) * (size_t) nfrags) + (size_t) frag) & block_mask];
                const block_nvfp4 blk1 = v[(((row_base + (size_t) (2 * lane + 1)) * (size_t) nfrags) + (size_t) frag) & block_mask];
                const uint32_t * v0 = reinterpret_cast<const uint32_t *>(blk0.qs);
                const uint32_t * v1 = reinterpret_cast<const uint32_t *>(blk1.qs);

#pragma unroll
                for (int group = 0; group < pv_groups; ++group) {
                    const uint32_t word0 = v0[group];
                    const uint32_t word1 = v1[group];
                    const float d0 = ue4m3_to_fp32(blk0.d[group / 2]);
                    const float d1 = ue4m3_to_fp32(blk1.d[group / 2]);

#pragma unroll
                    for (int kk = 0; kk < 4; ++kk) {
                        const uint8_t q0b = (uint8_t) (word0 >> (8 * kk));
                        const uint8_t q1b = (uint8_t) (word1 >> (8 * kk));
                        const uint32_t bx0 = half2_bits(dequant_nvfp4_byte_to_half2(q0b, d0));
                        const uint32_t bx1 = half2_bits(dequant_nvfp4_byte_to_half2(q1b, d1));
                        pv_mma_accumulate(pax0, pax1, pax2, pax3, bx0, bx1, pv0, pv1, pv2, pv3);
                    }

                    sink += (pv0 + pv1 + pv2 + pv3) * (float) (frag * pv_groups + group + 1) * 0.0000001f;
                }
            }
        }
    }

    if (lane == 0) {
        out[warp] = kq0 + kq1 + kq2 + kq3 + pv0 + pv1 + pv2 + pv3 + row_m0 + row_l0 + sink;
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = 0.0f;
    }
    (void) q; (void) k; (void) v; (void) block_mask; (void) out; (void) iters;
#endif
}

__global__ void mixedpv_stagehalf_kernel(
        const block_nvfp4 * q,
        const block_nvfp4 * k,
        const block_nvfp4 * v,
        size_t              block_mask,
        float *             out,
        uint64_t            iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    constexpr int nfrags = 256 / QK_NVFP4;
    constexpr int k_tiles = 64;
    constexpr int groups_per_frag = QK_NVFP4 / 8;
    constexpr int stage_entries_per_warp = nfrags * 4 * 32 * 2;

    extern __shared__ uint32_t smem[];

    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warp = ((int) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    uint32_t * warp_stage = smem + (size_t) warp_in_block * stage_entries_per_warp;

    int q_a[nfrags][4];
    uint32_t q_scale[nfrags];

#pragma unroll
    for (int frag = 0; frag < nfrags; ++frag) {
        const size_t q_idx = (((size_t) warp * (size_t) nfrags + (size_t) frag) * 32u + (size_t) lane) & block_mask;
        const block_nvfp4 q_blk = q[q_idx];
        const uint32_t * q_qs = reinterpret_cast<const uint32_t *>(q_blk.qs);
        q_a[frag][0] = (int) q_qs[(lane + 0) & 7];
        q_a[frag][1] = (int) q_qs[(lane + 1) & 7];
        q_a[frag][2] = (int) q_qs[(lane + 2) & 7];
        q_a[frag][3] = (int) q_qs[(lane + 3) & 7];
        q_scale[frag] = *reinterpret_cast<const uint32_t *>(q_blk.d);
    }

    float kq0 = 0.0f;
    float kq1 = 0.0f;
    float kq2 = 0.0f;
    float kq3 = 0.0f;
    float pv0 = 0.0f;
    float pv1 = 0.0f;
    float pv2 = 0.0f;
    float pv3 = 0.0f;
    float row_m0 = -64.0f;
    float row_l0 = 0.0f;
    float sink = 0.0f;

    for (uint64_t i = 0; i < iters; ++i) {
#pragma unroll
        for (int tile = 0; tile < k_tiles; ++tile) {
#pragma unroll
            for (int frag = 0; frag < nfrags; ++frag) {
                const size_t k_idx =
                    (((size_t) warp + (size_t) i * (size_t) gridDim.x) * (size_t) k_tiles * (size_t) nfrags * 32u +
                     (size_t) tile * (size_t) nfrags * 32u + (size_t) frag * 32u + (size_t) lane) & block_mask;

                const block_nvfp4 k_blk = k[k_idx];
                const uint32_t * k_qs = reinterpret_cast<const uint32_t *>(k_blk.qs);
                const int bx0 = (int) k_qs[(lane + 0) & 7];
                const int bx1 = (int) k_qs[(lane + 1) & 7];
                const uint32_t k_scale = *reinterpret_cast<const uint32_t *>(k_blk.d);
                kq_mma_accumulate(q_a[frag][0], q_a[frag][1], q_a[frag][2], q_a[frag][3], bx0, bx1, q_scale[frag], k_scale, kq0, kq1, kq2, kq3);
            }

            const int q_row = lane & 15;
            const int base_pos = (int) (i & 1023u);
            const bool causal = tile <= base_pos + q_row;
            const bool swa = tile + 1024 >= base_pos + q_row;
            const float score = (causal && swa) ? (kq0 + kq1 + kq2 + kq3) * 0.000244140625f : -64.0f;
            const float next_m0 = fmaxf(row_m0, score);
            const float alpha0 = row_l0 == 0.0f ? 0.0f : exp2f(row_m0 - next_m0);
            row_l0 = row_l0 * alpha0 + exp2f(score - next_m0);
            row_m0 = next_m0;

            const float p_base = 1.0f / (row_l0 + 1.0f);
            const uint32_t pax0 = half2_bits(__floats2half2_rn(p_base, p_base * 0.9375f));
            const uint32_t pax1 = half2_bits(__floats2half2_rn(p_base * 0.875f, p_base * 0.8125f));
            const uint32_t pax2 = half2_bits(__floats2half2_rn(p_base * 0.7500f, p_base * 0.6875f));
            const uint32_t pax3 = half2_bits(__floats2half2_rn(p_base * 0.6250f, p_base * 0.5625f));

            const size_t row_base = (((size_t) warp + (size_t) i * (size_t) gridDim.x) * (size_t) k_tiles * 64u +
                                     (size_t) tile * 64u) & block_mask;

#pragma unroll
            for (int group = 0; group < groups_per_frag; ++group) {
#pragma unroll
                for (int frag = 0; frag < nfrags; ++frag) {
                    const block_nvfp4 blk0 = v[(((row_base + (size_t) (2 * lane + 0)) * (size_t) nfrags) + (size_t) frag) & block_mask];
                    const block_nvfp4 blk1 = v[(((row_base + (size_t) (2 * lane + 1)) * (size_t) nfrags) + (size_t) frag) & block_mask];
                    const uint32_t * v0 = reinterpret_cast<const uint32_t *>(blk0.qs);
                    const uint32_t * v1 = reinterpret_cast<const uint32_t *>(blk1.qs);
                    const uint32_t word0 = v0[group];
                    const uint32_t word1 = v1[group];
                    const float d0 = ue4m3_to_fp32(blk0.d[group / 2]);
                    const float d1 = ue4m3_to_fp32(blk1.d[group / 2]);

#pragma unroll
                    for (int kk = 0; kk < 4; ++kk) {
                        const uint8_t q0b = (uint8_t) (word0 >> (8 * kk));
                        const uint8_t q1b = (uint8_t) (word1 >> (8 * kk));
                        const size_t stage_idx = (((size_t) frag * 4u + (size_t) kk) * 32u + (size_t) lane) * 2u;
                        warp_stage[stage_idx + 0] = half2_bits(dequant_nvfp4_byte_to_half2(q0b, d0));
                        warp_stage[stage_idx + 1] = half2_bits(dequant_nvfp4_byte_to_half2(q1b, d1));
                    }
                }

                __syncwarp();

#pragma unroll
                for (int frag = 0; frag < nfrags; ++frag) {
#pragma unroll
                    for (int kk = 0; kk < 4; ++kk) {
                        const size_t stage_idx = (((size_t) frag * 4u + (size_t) kk) * 32u + (size_t) lane) * 2u;
                        const uint32_t bx0 = warp_stage[stage_idx + 0];
                        const uint32_t bx1 = warp_stage[stage_idx + 1];
                        pv_mma_accumulate(pax0, pax1, pax2, pax3, bx0, bx1, pv0, pv1, pv2, pv3);
                    }

                    sink += (pv0 + pv1 + pv2 + pv3) * (float) (frag * groups_per_frag + group + 1) * 0.0000001f;
                }

                __syncwarp();
            }
        }
    }

    if (lane == 0) {
        out[warp] = kq0 + kq1 + kq2 + kq3 + pv0 + pv1 + pv2 + pv3 + row_m0 + row_l0 + sink;
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = 0.0f;
    }
    (void) q; (void) k; (void) v; (void) block_mask; (void) out; (void) iters;
#endif
}

__global__ void mixedpv_localacc_kernel(
        const block_nvfp4 * q,
        const block_nvfp4 * k,
        const block_nvfp4 * v,
        size_t              block_mask,
        float *             out,
        uint64_t            iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    constexpr int nfrags = 256 / QK_NVFP4;
    constexpr int k_tiles = 64;
    constexpr int groups_per_frag = QK_NVFP4 / 8;

    const int lane = threadIdx.x & 31;
    const int warp = ((int) blockIdx.x * blockDim.x + threadIdx.x) >> 5;

    int q_a[nfrags][4];
    uint32_t q_scale[nfrags];

#pragma unroll
    for (int frag = 0; frag < nfrags; ++frag) {
        const size_t q_idx = (((size_t) warp * (size_t) nfrags + (size_t) frag) * 32u + (size_t) lane) & block_mask;
        const block_nvfp4 q_blk = q[q_idx];
        const uint32_t * q_qs = reinterpret_cast<const uint32_t *>(q_blk.qs);
        q_a[frag][0] = (int) q_qs[(lane + 0) & 7];
        q_a[frag][1] = (int) q_qs[(lane + 1) & 7];
        q_a[frag][2] = (int) q_qs[(lane + 2) & 7];
        q_a[frag][3] = (int) q_qs[(lane + 3) & 7];
        q_scale[frag] = *reinterpret_cast<const uint32_t *>(q_blk.d);
    }

    float kq0 = 0.0f;
    float kq1 = 0.0f;
    float kq2 = 0.0f;
    float kq3 = 0.0f;
    float row_m0 = -64.0f;
    float row_l0 = 0.0f;
    float sink = 0.0f;

    for (uint64_t i = 0; i < iters; ++i) {
#pragma unroll
        for (int tile = 0; tile < k_tiles; ++tile) {
#pragma unroll
            for (int frag = 0; frag < nfrags; ++frag) {
                const size_t k_idx =
                    (((size_t) warp + (size_t) i * (size_t) gridDim.x) * (size_t) k_tiles * (size_t) nfrags * 32u +
                     (size_t) tile * (size_t) nfrags * 32u + (size_t) frag * 32u + (size_t) lane) & block_mask;

                const block_nvfp4 k_blk = k[k_idx];
                const uint32_t * k_qs = reinterpret_cast<const uint32_t *>(k_blk.qs);
                const int bx0 = (int) k_qs[(lane + 0) & 7];
                const int bx1 = (int) k_qs[(lane + 1) & 7];
                const uint32_t k_scale = *reinterpret_cast<const uint32_t *>(k_blk.d);
                kq_mma_accumulate(q_a[frag][0], q_a[frag][1], q_a[frag][2], q_a[frag][3], bx0, bx1, q_scale[frag], k_scale, kq0, kq1, kq2, kq3);
            }

            const int q_row = lane & 15;
            const int base_pos = (int) (i & 1023u);
            const bool causal = tile <= base_pos + q_row;
            const bool swa = tile + 1024 >= base_pos + q_row;
            const float score = (causal && swa) ? (kq0 + kq1 + kq2 + kq3) * 0.000244140625f : -64.0f;
            const float next_m0 = fmaxf(row_m0, score);
            const float alpha0 = row_l0 == 0.0f ? 0.0f : exp2f(row_m0 - next_m0);
            row_l0 = row_l0 * alpha0 + exp2f(score - next_m0);
            row_m0 = next_m0;

            const float p_base = 1.0f / (row_l0 + 1.0f);
            const uint32_t pax0 = half2_bits(__floats2half2_rn(p_base, p_base * 0.9375f));
            const uint32_t pax1 = half2_bits(__floats2half2_rn(p_base * 0.875f, p_base * 0.8125f));
            const uint32_t pax2 = half2_bits(__floats2half2_rn(p_base * 0.7500f, p_base * 0.6875f));
            const uint32_t pax3 = half2_bits(__floats2half2_rn(p_base * 0.6250f, p_base * 0.5625f));

            const size_t row_base = (((size_t) warp + (size_t) i * (size_t) gridDim.x) * (size_t) k_tiles * 64u +
                                     (size_t) tile * 64u) & block_mask;

#pragma unroll
            for (int frag = 0; frag < nfrags; ++frag) {
                const block_nvfp4 blk0 = v[(((row_base + (size_t) (2 * lane + 0)) * (size_t) nfrags) + (size_t) frag) & block_mask];
                const block_nvfp4 blk1 = v[(((row_base + (size_t) (2 * lane + 1)) * (size_t) nfrags) + (size_t) frag) & block_mask];
                const uint32_t * v0 = reinterpret_cast<const uint32_t *>(blk0.qs);
                const uint32_t * v1 = reinterpret_cast<const uint32_t *>(blk1.qs);

#pragma unroll
                for (int group = 0; group < groups_per_frag; ++group) {
                    const uint32_t word0 = v0[group];
                    const uint32_t word1 = v1[group];
                    const float d0 = ue4m3_to_fp32(blk0.d[group / 2]);
                    const float d1 = ue4m3_to_fp32(blk1.d[group / 2]);
                    float pv0 = 0.0f;
                    float pv1 = 0.0f;
                    float pv2 = 0.0f;
                    float pv3 = 0.0f;

#pragma unroll
                    for (int kk = 0; kk < 4; ++kk) {
                        const uint8_t q0b = (uint8_t) (word0 >> (8 * kk));
                        const uint8_t q1b = (uint8_t) (word1 >> (8 * kk));
                        const uint32_t bx0 = half2_bits(dequant_nvfp4_byte_to_half2(q0b, d0));
                        const uint32_t bx1 = half2_bits(dequant_nvfp4_byte_to_half2(q1b, d1));
                        pv_mma_accumulate(pax0, pax1, pax2, pax3, bx0, bx1, pv0, pv1, pv2, pv3);
                    }

                    sink += (pv0 + pv1 + pv2 + pv3) * (float) (frag * groups_per_frag + group + 1) * 0.0000001f;
                }
            }
        }
    }

    if (lane == 0) {
        out[warp] = kq0 + kq1 + kq2 + kq3 + row_m0 + row_l0 + sink;
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = 0.0f;
    }
    (void) q; (void) k; (void) v; (void) block_mask; (void) out; (void) iters;
#endif
}

__global__ void mixedpv_bypassvdequant_kernel(
        const block_nvfp4 * q,
        const block_nvfp4 * k,
        const block_nvfp4 * v,
        size_t              block_mask,
        float *             out,
        uint64_t            iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    constexpr int nfrags = 256 / QK_NVFP4;
    constexpr int k_tiles = 64;
    constexpr int groups_per_frag = QK_NVFP4 / 8;

    const int lane = threadIdx.x & 31;
    const int warp = ((int) blockIdx.x * blockDim.x + threadIdx.x) >> 5;

    int q_a[nfrags][4];
    uint32_t q_scale[nfrags];

#pragma unroll
    for (int frag = 0; frag < nfrags; ++frag) {
        const size_t q_idx = (((size_t) warp * (size_t) nfrags + (size_t) frag) * 32u + (size_t) lane) & block_mask;
        const block_nvfp4 q_blk = q[q_idx];
        const uint32_t * q_qs = reinterpret_cast<const uint32_t *>(q_blk.qs);
        q_a[frag][0] = (int) q_qs[(lane + 0) & 7];
        q_a[frag][1] = (int) q_qs[(lane + 1) & 7];
        q_a[frag][2] = (int) q_qs[(lane + 2) & 7];
        q_a[frag][3] = (int) q_qs[(lane + 3) & 7];
        q_scale[frag] = *reinterpret_cast<const uint32_t *>(q_blk.d);
    }

    float kq0 = 0.0f;
    float kq1 = 0.0f;
    float kq2 = 0.0f;
    float kq3 = 0.0f;
    float pv0 = 0.0f;
    float pv1 = 0.0f;
    float pv2 = 0.0f;
    float pv3 = 0.0f;
    float row_m0 = -64.0f;
    float row_l0 = 0.0f;
    float sink = 0.0f;

    for (uint64_t i = 0; i < iters; ++i) {
#pragma unroll
        for (int tile = 0; tile < k_tiles; ++tile) {
#pragma unroll
            for (int frag = 0; frag < nfrags; ++frag) {
                const size_t k_idx =
                    (((size_t) warp + (size_t) i * (size_t) gridDim.x) * (size_t) k_tiles * (size_t) nfrags * 32u +
                     (size_t) tile * (size_t) nfrags * 32u + (size_t) frag * 32u + (size_t) lane) & block_mask;

                const block_nvfp4 k_blk = k[k_idx];
                const uint32_t * k_qs = reinterpret_cast<const uint32_t *>(k_blk.qs);
                const int bx0 = (int) k_qs[(lane + 0) & 7];
                const int bx1 = (int) k_qs[(lane + 1) & 7];
                const uint32_t k_scale = *reinterpret_cast<const uint32_t *>(k_blk.d);
                kq_mma_accumulate(q_a[frag][0], q_a[frag][1], q_a[frag][2], q_a[frag][3], bx0, bx1, q_scale[frag], k_scale, kq0, kq1, kq2, kq3);
            }

            const int q_row = lane & 15;
            const int base_pos = (int) (i & 1023u);
            const bool causal = tile <= base_pos + q_row;
            const bool swa = tile + 1024 >= base_pos + q_row;
            const float score = (causal && swa) ? (kq0 + kq1 + kq2 + kq3) * 0.000244140625f : -64.0f;
            const float next_m0 = fmaxf(row_m0, score);
            const float alpha0 = row_l0 == 0.0f ? 0.0f : exp2f(row_m0 - next_m0);
            row_l0 = row_l0 * alpha0 + exp2f(score - next_m0);
            row_m0 = next_m0;

            const float p_base = 1.0f / (row_l0 + 1.0f);
            const uint32_t pax0 = half2_bits(__floats2half2_rn(p_base, p_base * 0.9375f));
            const uint32_t pax1 = half2_bits(__floats2half2_rn(p_base * 0.875f, p_base * 0.8125f));
            const uint32_t pax2 = half2_bits(__floats2half2_rn(p_base * 0.7500f, p_base * 0.6875f));
            const uint32_t pax3 = half2_bits(__floats2half2_rn(p_base * 0.6250f, p_base * 0.5625f));

            const size_t row_base = (((size_t) warp + (size_t) i * (size_t) gridDim.x) * (size_t) k_tiles * 64u +
                                     (size_t) tile * 64u) & block_mask;

#pragma unroll
            for (int frag = 0; frag < nfrags; ++frag) {
                const block_nvfp4 blk0 = v[(((row_base + (size_t) (2 * lane + 0)) * (size_t) nfrags) + (size_t) frag) & block_mask];
                const block_nvfp4 blk1 = v[(((row_base + (size_t) (2 * lane + 1)) * (size_t) nfrags) + (size_t) frag) & block_mask];
                const uint32_t * v0 = reinterpret_cast<const uint32_t *>(blk0.qs);
                const uint32_t * v1 = reinterpret_cast<const uint32_t *>(blk1.qs);

#pragma unroll
                for (int group = 0; group < groups_per_frag; ++group) {
                    const uint32_t bx0 = v0[group];
                    const uint32_t bx1 = v1[group];

#pragma unroll
                    for (int kk = 0; kk < 4; ++kk) {
                        pv_mma_accumulate(pax0, pax1, pax2, pax3, bx0, bx1, pv0, pv1, pv2, pv3);
                    }

                    sink += (pv0 + pv1 + pv2 + pv3) * (float) (frag * groups_per_frag + group + 1) * 0.0000001f;
                }
            }
        }
    }

    if (lane == 0) {
        out[warp] = kq0 + kq1 + kq2 + kq3 + pv0 + pv1 + pv2 + pv3 + row_m0 + row_l0 + sink;
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = 0.0f;
    }
    (void) q; (void) k; (void) v; (void) block_mask; (void) out; (void) iters;
#endif
}

static float time_events(cudaEvent_t start, cudaEvent_t stop) {
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

static double useful_mtp_fraction(const bench_config & cfg) {
    return (double) cfg.mtp_rows / 16.0;
}

static int print_occupancy(
        const bench_config & cfg,
        const cudaDeviceProp & prop,
        const char * label,
        const void * kernel,
        size_t dynamic_shared_bytes = 0) {
    if (dynamic_shared_bytes > (size_t) prop.sharedMemPerBlock) {
        const size_t optin_limit = prop.sharedMemPerBlockOptin > 0 ? (size_t) prop.sharedMemPerBlockOptin : (size_t) prop.sharedMemPerBlock;
        if (dynamic_shared_bytes > optin_limit) {
            std::printf("%s occupancy: active_blocks_per_sm=0 active_warps_per_sm=0 occupancy=0.0%% shared=%.3f KiB\n",
                        label, (double) dynamic_shared_bytes / 1024.0);
            std::printf("%s: skipped reason=dynamic shared memory %.3f KiB exceeds device opt-in limit %.3f KiB\n",
                        label, (double) dynamic_shared_bytes / 1024.0, (double) optin_limit / 1024.0);
            return 0;
        }

        const cudaError_t attr_err = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) dynamic_shared_bytes);
        if (attr_err != cudaSuccess) {
            std::printf("%s occupancy: active_blocks_per_sm=0 active_warps_per_sm=0 occupancy=0.0%% shared=%.3f KiB\n",
                        label, (double) dynamic_shared_bytes / 1024.0);
            std::printf("%s: skipped reason=cudaFuncSetAttribute(MaxDynamicSharedMemorySize=%.3f KiB) failed: %s\n",
                        label, (double) dynamic_shared_bytes / 1024.0, cudaGetErrorString(attr_err));
            return 0;
        }
    }

    int active_blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks_per_sm, kernel, cfg.threads, dynamic_shared_bytes));
    const int active_warps_per_sm = active_blocks_per_sm * (cfg.threads / 32);
    const double occupancy = prop.maxThreadsPerMultiProcessor > 0 ?
        (double) active_blocks_per_sm * (double) cfg.threads / (double) prop.maxThreadsPerMultiProcessor : 0.0;
    std::printf("%s occupancy: active_blocks_per_sm=%d active_warps_per_sm=%d occupancy=%.1f%% shared=%.3f KiB\n",
                label, active_blocks_per_sm, active_warps_per_sm, 100.0 * occupancy, (double) dynamic_shared_bytes / 1024.0);
    return active_blocks_per_sm;
}

static size_t stagehalf_shared_bytes(const bench_config & cfg) {
    constexpr int nfrags = 256 / QK_NVFP4;
    constexpr int stage_entries_per_warp = nfrags * 4 * 32 * 2;
    return (size_t) (cfg.threads / 32) * (size_t) stage_entries_per_warp * sizeof(uint32_t);
}

template <int pv_groups>
static int run_stripmine_benchmark(
        const bench_config & cfg,
        const cudaDeviceProp & prop,
        const block_nvfp4 * q,
        const block_nvfp4 * k,
        const block_nvfp4 * v,
        size_t n_blocks,
        size_t n_warps,
        float * out);

static int run_stagehalf_benchmark(
        const bench_config & cfg,
        const cudaDeviceProp & prop,
        const block_nvfp4 * q,
        const block_nvfp4 * k,
        const block_nvfp4 * v,
        size_t n_blocks,
        size_t n_warps,
        float * out);

static int run_localacc_benchmark(
        const bench_config & cfg,
        const cudaDeviceProp & prop,
        const block_nvfp4 * q,
        const block_nvfp4 * k,
        const block_nvfp4 * v,
        size_t n_blocks,
        size_t n_warps,
        float * out);

static int run_bypassvdequant_benchmark(
        const bench_config & cfg,
        const cudaDeviceProp & prop,
        const block_nvfp4 * q,
        const block_nvfp4 * k,
        const block_nvfp4 * v,
        size_t n_blocks,
        size_t n_warps,
        float * out);

static int run_benchmark(const bench_config & cfg, const cudaDeviceProp & prop, int cc) {
    std::printf("config:               blocks=%d threads=%d mtp_rows=%d useful_m16=%.1f%% iters=%" PRIu64 "\n",
                cfg.blocks, cfg.threads, cfg.mtp_rows, 100.0 * useful_mtp_fraction(cfg), cfg.iters);

    if (cc < 1200 || cc >= 1300) {
        std::printf("kq256_mixedpv: skipped reason=requires Blackwell sm_120/sm_121 device\n");
        return 0;
    }

    constexpr int nfrags = 256 / QK_NVFP4;
    constexpr int k_tiles = 64;
    constexpr int groups_per_frag = QK_NVFP4 / 8;

    const int kq_active = print_occupancy(cfg, prop, "kq256_only", (const void *) kq256_only_kernel);
    const int pv_active = print_occupancy(cfg, prop, "pv256_mixed", (const void *) mixedpv_kernel<false>);
    const int combined_active = print_occupancy(cfg, prop, "combined256_mixed", (const void *) mixedpv_kernel<true>);
    if (kq_active == 0 || pv_active == 0 || combined_active == 0) {
        std::printf("kq256_mixedpv: skipped reason=at least one required kernel has zero active blocks for threads=%d\n", cfg.threads);
        return 2;
    }

    const size_t n_threads = (size_t) cfg.blocks * (size_t) cfg.threads;
    const size_t n_warps = n_threads / 32;
    const size_t n_blocks = 1ull << 22;
    const size_t q_float_count = n_blocks * (size_t) QK_NVFP4;

    float * q_f32 = nullptr;
    block_nvfp4 * q = nullptr;
    block_nvfp4 * k = nullptr;
    block_nvfp4 * v = nullptr;
    float * out = nullptr;
    CUDA_CHECK(cudaMalloc(&q_f32, q_float_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&q, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&k, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&v, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&out, n_warps * sizeof(float)));

    fill_f32_kernel<<<cfg.blocks, cfg.threads>>>(q_f32, q_float_count);
    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(q, n_blocks);
    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(k, n_blocks);
    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(v, n_blocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    quantize_q_tile_kernel<<<cfg.blocks, cfg.threads>>>(q_f32, q, n_blocks, 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    constexpr uint64_t quant_repeats = 4;
    CUDA_CHECK(cudaEventRecord(start));
    quantize_q_tile_kernel<<<cfg.blocks, cfg.threads>>>(q_f32, q, n_blocks, quant_repeats);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float quant_ms = time_events(start, stop);
    const double quant_seconds = quant_ms / 1000.0;
    const double quant_in_gb = (double) q_float_count * sizeof(float) * (double) quant_repeats / 1.0e9;
    const double quant_out_gb = (double) n_blocks * sizeof(block_nvfp4) * (double) quant_repeats / 1.0e9;
    std::printf("q_quant: %.3f GB/s-input  %.3f GB/s-output  %.3f GB/s-total  blocks=%zu repeats=%" PRIu64 " time=%.3f ms\n",
                quant_in_gb / quant_seconds,
                quant_out_gb / quant_seconds,
                (quant_in_gb + quant_out_gb) / quant_seconds,
                n_blocks,
                quant_repeats,
                quant_ms);

    kq256_only_kernel<<<cfg.blocks, cfg.threads>>>(q, k, n_blocks - 1, out, 10);
    mixedpv_kernel<false><<<cfg.blocks, cfg.threads>>>(q, k, v, n_blocks - 1, out, 2);
    mixedpv_kernel<true><<<cfg.blocks, cfg.threads>>>(q, k, v, n_blocks - 1, out, 2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    kq256_only_kernel<<<cfg.blocks, cfg.threads>>>(q, k, n_blocks - 1, out, cfg.iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float kq_ms = time_events(start, stop);

    CUDA_CHECK(cudaEventRecord(start));
    mixedpv_kernel<false><<<cfg.blocks, cfg.threads>>>(q, k, v, n_blocks - 1, out, cfg.iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float pv_ms = time_events(start, stop);

    CUDA_CHECK(cudaEventRecord(start));
    mixedpv_kernel<true><<<cfg.blocks, cfg.threads>>>(q, k, v, n_blocks - 1, out, cfg.iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float combined_ms = time_events(start, stop);

    const double kq_ops = (double) n_warps * (double) cfg.iters * (double) k_tiles * (double) nfrags * 2.0 * 16.0 * 8.0 * 64.0;
    const double pv_ops = (double) n_warps * (double) cfg.iters * (double) k_tiles * (double) nfrags *
                          (double) groups_per_frag * 4.0 * 2.0 * 16.0 * 8.0 * 16.0;
    const double k_compact_gb = (double) n_warps * (double) cfg.iters * (double) k_tiles * (double) nfrags * 32.0 * (double) sizeof(block_nvfp4) / 1.0e9;
    const double pv_compact_gb = (double) n_warps * (double) cfg.iters * (double) k_tiles * 64.0 * (double) nfrags * (double) sizeof(block_nvfp4) / 1.0e9;
    const double q_compact_gb = (double) n_warps * (double) nfrags * 32.0 * (double) sizeof(block_nvfp4) / 1.0e9;
    const double kq_seconds = kq_ms / 1000.0;
    const double pv_seconds = pv_ms / 1000.0;
    const double combined_seconds = combined_ms / 1000.0;

    std::printf("kq256: %.3f KQ-TOPS  %.3f useful-mtp-KQ-TOPS  %.3f GB/s-K-compact-read  %.6f GB-Q-compact-once  blocks=%zu warps=%zu iters=%" PRIu64 " time=%.3f ms\n",
                kq_ops / kq_seconds / 1.0e12,
                kq_ops / kq_seconds / 1.0e12 * useful_mtp_fraction(cfg),
                k_compact_gb / kq_seconds,
                q_compact_gb,
                n_blocks,
                n_warps,
                cfg.iters,
                kq_ms);

    std::printf("pv256_mixed: %.3f mixedPV-TOPS  %.3f useful-mtp-mixedPV-TOPS  %.3f GB/s-V-compact-read  blocks=%zu warps=%zu iters=%" PRIu64 " time=%.3f ms\n",
                pv_ops / pv_seconds / 1.0e12,
                pv_ops / pv_seconds / 1.0e12 * useful_mtp_fraction(cfg),
                pv_compact_gb / pv_seconds,
                n_blocks,
                n_warps,
                cfg.iters,
                pv_ms);

    std::printf("combined256_mixed: %.3f modeled-total-TOPS  %.3f useful-mtp-modeled-total-TOPS  %.3f KQ-issue-TOPS  %.3f mixedPV-issue-TOPS  %.3f GB/s-K-compact-read  %.3f GB/s-V-compact-read  blocks=%zu warps=%zu iters=%" PRIu64 " time=%.3f ms\n",
                (kq_ops + pv_ops) / combined_seconds / 1.0e12,
                (kq_ops + pv_ops) / combined_seconds / 1.0e12 * useful_mtp_fraction(cfg),
                kq_ops / combined_seconds / 1.0e12,
                pv_ops / combined_seconds / 1.0e12,
                k_compact_gb / combined_seconds,
                pv_compact_gb / combined_seconds,
                n_blocks,
                n_warps,
                cfg.iters,
                combined_ms);

    run_stripmine_benchmark<1>(cfg, prop, q, k, v, n_blocks, n_warps, out);
    run_stripmine_benchmark<2>(cfg, prop, q, k, v, n_blocks, n_warps, out);
    run_stripmine_benchmark<4>(cfg, prop, q, k, v, n_blocks, n_warps, out);
    run_stagehalf_benchmark(cfg, prop, q, k, v, n_blocks, n_warps, out);
    run_localacc_benchmark(cfg, prop, q, k, v, n_blocks, n_warps, out);
    run_bypassvdequant_benchmark(cfg, prop, q, k, v, n_blocks, n_warps, out);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(v));
    CUDA_CHECK(cudaFree(k));
    CUDA_CHECK(cudaFree(q));
    CUDA_CHECK(cudaFree(q_f32));
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}

template <int pv_groups>
static int run_stripmine_benchmark(
        const bench_config & cfg,
        const cudaDeviceProp & prop,
        const block_nvfp4 * q,
        const block_nvfp4 * k,
        const block_nvfp4 * v,
        const size_t n_blocks,
        const size_t n_warps,
        float * out) {
    constexpr int nfrags = 256 / QK_NVFP4;
    constexpr int k_tiles = 64;
    constexpr int full_groups_per_frag = QK_NVFP4 / 8;

    char label[64];
    std::snprintf(label, sizeof(label), "combined256_stripmine_g%d", pv_groups);
    const int active = print_occupancy(cfg, prop, label, (const void *) mixedpv_stripmine_kernel<pv_groups>);
    if (active == 0) {
        std::printf("%s: skipped reason=kernel has zero active blocks for threads=%d\n", label, cfg.threads);
        return 0;
    }

    mixedpv_stripmine_kernel<pv_groups><<<cfg.blocks, cfg.threads>>>(q, k, v, n_blocks - 1, out, 2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    mixedpv_stripmine_kernel<pv_groups><<<cfg.blocks, cfg.threads>>>(q, k, v, n_blocks - 1, out, cfg.iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float ms = time_events(start, stop);

    const double seconds = ms / 1000.0;
    const double kq_ops = (double) n_warps * (double) cfg.iters * (double) k_tiles * (double) nfrags * 2.0 * 16.0 * 8.0 * 64.0;
    const double pv_measured_ops = (double) n_warps * (double) cfg.iters * (double) k_tiles * (double) nfrags *
                                   (double) pv_groups * 4.0 * 2.0 * 16.0 * 8.0 * 16.0;
    const double pv_projected_full_ops = pv_measured_ops * (double) full_groups_per_frag / (double) pv_groups;
    const double k_compact_gb = (double) n_warps * (double) cfg.iters * (double) k_tiles * (double) nfrags * 32.0 * (double) sizeof(block_nvfp4) / 1.0e9;
    const double v_compact_gb = (double) n_warps * (double) cfg.iters * (double) k_tiles * 64.0 *
                               (double) nfrags * (double) sizeof(block_nvfp4) * (double) pv_groups / (double) full_groups_per_frag / 1.0e9;

    std::printf("%s: %.3f measured-total-TOPS  %.3f useful-mtp-measured-total-TOPS  %.3f KQ-issue-TOPS  %.3f measured-mixedPV-TOPS  %.3f projected-fullPV-TOPS-at-same-time  %.3f GB/s-K-compact-read  %.3f GB/s-V-compact-read  pv_group_fraction=%.3f pv_groups=%d blocks=%zu warps=%zu iters=%" PRIu64 " time=%.3f ms\n",
                label,
                (kq_ops + pv_measured_ops) / seconds / 1.0e12,
                (kq_ops + pv_measured_ops) / seconds / 1.0e12 * useful_mtp_fraction(cfg),
                kq_ops / seconds / 1.0e12,
                pv_measured_ops / seconds / 1.0e12,
                pv_projected_full_ops / seconds / 1.0e12,
                k_compact_gb / seconds,
                v_compact_gb / seconds,
                (double) pv_groups / (double) full_groups_per_frag,
                pv_groups,
                n_blocks,
                n_warps,
                cfg.iters,
                ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return 0;
}

static int run_bypassvdequant_benchmark(
        const bench_config & cfg,
        const cudaDeviceProp & prop,
        const block_nvfp4 * q,
        const block_nvfp4 * k,
        const block_nvfp4 * v,
        const size_t n_blocks,
        const size_t n_warps,
        float * out) {
    constexpr int nfrags = 256 / QK_NVFP4;
    constexpr int k_tiles = 64;
    constexpr int groups_per_frag = QK_NVFP4 / 8;

    const char * label = "combined256_bypassvdequant";
    const int active = print_occupancy(cfg, prop, label, (const void *) mixedpv_bypassvdequant_kernel);
    if (active == 0) {
        std::printf("%s: skipped reason=kernel has zero active blocks for threads=%d\n", label, cfg.threads);
        return 0;
    }

    mixedpv_bypassvdequant_kernel<<<cfg.blocks, cfg.threads>>>(q, k, v, n_blocks - 1, out, 2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    mixedpv_bypassvdequant_kernel<<<cfg.blocks, cfg.threads>>>(q, k, v, n_blocks - 1, out, cfg.iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float ms = time_events(start, stop);

    const double seconds = ms / 1000.0;
    const double kq_ops = (double) n_warps * (double) cfg.iters * (double) k_tiles * (double) nfrags * 2.0 * 16.0 * 8.0 * 64.0;
    const double pv_ops = (double) n_warps * (double) cfg.iters * (double) k_tiles * (double) nfrags *
                          (double) groups_per_frag * 4.0 * 2.0 * 16.0 * 8.0 * 16.0;
    const double k_compact_gb = (double) n_warps * (double) cfg.iters * (double) k_tiles * (double) nfrags * 32.0 * (double) sizeof(block_nvfp4) / 1.0e9;
    const double v_payload_gb = (double) n_warps * (double) cfg.iters * (double) k_tiles * 64.0 *
                               (double) nfrags * (double) groups_per_frag * 2.0 * (double) sizeof(uint32_t) / 1.0e9;

    std::printf("%s: %.3f modeled-total-TOPS  %.3f useful-mtp-modeled-total-TOPS  %.3f KQ-issue-TOPS  %.3f mixedPV-issue-TOPS  %.3f GB/s-K-compact-read  %.3f GB/s-V-payload-read  blocks=%zu warps=%zu iters=%" PRIu64 " time=%.3f ms\n",
                label,
                (kq_ops + pv_ops) / seconds / 1.0e12,
                (kq_ops + pv_ops) / seconds / 1.0e12 * useful_mtp_fraction(cfg),
                kq_ops / seconds / 1.0e12,
                pv_ops / seconds / 1.0e12,
                k_compact_gb / seconds,
                v_payload_gb / seconds,
                n_blocks,
                n_warps,
                cfg.iters,
                ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return 0;
}

static int run_localacc_benchmark(
        const bench_config & cfg,
        const cudaDeviceProp & prop,
        const block_nvfp4 * q,
        const block_nvfp4 * k,
        const block_nvfp4 * v,
        const size_t n_blocks,
        const size_t n_warps,
        float * out) {
    constexpr int nfrags = 256 / QK_NVFP4;
    constexpr int k_tiles = 64;
    constexpr int groups_per_frag = QK_NVFP4 / 8;

    const char * label = "combined256_localacc";
    const int active = print_occupancy(cfg, prop, label, (const void *) mixedpv_localacc_kernel);
    if (active == 0) {
        std::printf("%s: skipped reason=kernel has zero active blocks for threads=%d\n", label, cfg.threads);
        return 0;
    }

    mixedpv_localacc_kernel<<<cfg.blocks, cfg.threads>>>(q, k, v, n_blocks - 1, out, 2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    mixedpv_localacc_kernel<<<cfg.blocks, cfg.threads>>>(q, k, v, n_blocks - 1, out, cfg.iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float ms = time_events(start, stop);

    const double seconds = ms / 1000.0;
    const double kq_ops = (double) n_warps * (double) cfg.iters * (double) k_tiles * (double) nfrags * 2.0 * 16.0 * 8.0 * 64.0;
    const double pv_ops = (double) n_warps * (double) cfg.iters * (double) k_tiles * (double) nfrags *
                          (double) groups_per_frag * 4.0 * 2.0 * 16.0 * 8.0 * 16.0;
    const double k_compact_gb = (double) n_warps * (double) cfg.iters * (double) k_tiles * (double) nfrags * 32.0 * (double) sizeof(block_nvfp4) / 1.0e9;
    const double v_compact_gb = (double) n_warps * (double) cfg.iters * (double) k_tiles * 64.0 *
                               (double) nfrags * (double) sizeof(block_nvfp4) / 1.0e9;

    std::printf("%s: %.3f modeled-total-TOPS  %.3f useful-mtp-modeled-total-TOPS  %.3f KQ-issue-TOPS  %.3f mixedPV-issue-TOPS  %.3f GB/s-K-compact-read  %.3f GB/s-V-compact-read  blocks=%zu warps=%zu iters=%" PRIu64 " time=%.3f ms\n",
                label,
                (kq_ops + pv_ops) / seconds / 1.0e12,
                (kq_ops + pv_ops) / seconds / 1.0e12 * useful_mtp_fraction(cfg),
                kq_ops / seconds / 1.0e12,
                pv_ops / seconds / 1.0e12,
                k_compact_gb / seconds,
                v_compact_gb / seconds,
                n_blocks,
                n_warps,
                cfg.iters,
                ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return 0;
}

static int run_stagehalf_benchmark(
        const bench_config & cfg,
        const cudaDeviceProp & prop,
        const block_nvfp4 * q,
        const block_nvfp4 * k,
        const block_nvfp4 * v,
        const size_t n_blocks,
        const size_t n_warps,
        float * out) {
    constexpr int nfrags = 256 / QK_NVFP4;
    constexpr int k_tiles = 64;
    constexpr int groups_per_frag = QK_NVFP4 / 8;
    constexpr int stage_entries_per_warp = nfrags * 4 * 32 * 2;

    const char * label = "combined256_stagehalf";
    const size_t shared_bytes = stagehalf_shared_bytes(cfg);
    const int active = print_occupancy(cfg, prop, label, (const void *) mixedpv_stagehalf_kernel, shared_bytes);
    if (active == 0) {
        std::printf("%s: skipped reason=kernel has zero active blocks for threads=%d\n", label, cfg.threads);
        return 0;
    }

    mixedpv_stagehalf_kernel<<<cfg.blocks, cfg.threads, shared_bytes>>>(q, k, v, n_blocks - 1, out, 2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    mixedpv_stagehalf_kernel<<<cfg.blocks, cfg.threads, shared_bytes>>>(q, k, v, n_blocks - 1, out, cfg.iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float ms = time_events(start, stop);

    const double seconds = ms / 1000.0;
    const double kq_ops = (double) n_warps * (double) cfg.iters * (double) k_tiles * (double) nfrags * 2.0 * 16.0 * 8.0 * 64.0;
    const double pv_ops = (double) n_warps * (double) cfg.iters * (double) k_tiles * (double) nfrags *
                          (double) groups_per_frag * 4.0 * 2.0 * 16.0 * 8.0 * 16.0;
    const double k_compact_gb = (double) n_warps * (double) cfg.iters * (double) k_tiles * (double) nfrags * 32.0 * (double) sizeof(block_nvfp4) / 1.0e9;
    const double v_compact_gb = (double) n_warps * (double) cfg.iters * (double) k_tiles * 64.0 *
                               (double) nfrags * (double) sizeof(block_nvfp4) / 1.0e9;
    const double stage_rw_gb = (double) n_warps * (double) cfg.iters * (double) k_tiles *
                               (double) groups_per_frag * (double) stage_entries_per_warp *
                               (double) sizeof(uint32_t) * 2.0 / 1.0e9;

    std::printf("%s: %.3f modeled-total-TOPS  %.3f useful-mtp-modeled-total-TOPS  %.3f KQ-issue-TOPS  %.3f mixedPV-issue-TOPS  %.3f GB/s-K-compact-read  %.3f GB/s-V-compact-read  %.3f GB/s-shared-stage-rw  shared=%.3f KiB blocks=%zu warps=%zu iters=%" PRIu64 " time=%.3f ms\n",
                label,
                (kq_ops + pv_ops) / seconds / 1.0e12,
                (kq_ops + pv_ops) / seconds / 1.0e12 * useful_mtp_fraction(cfg),
                kq_ops / seconds / 1.0e12,
                pv_ops / seconds / 1.0e12,
                k_compact_gb / seconds,
                v_compact_gb / seconds,
                stage_rw_gb / seconds,
                (double) shared_bytes / 1024.0,
                n_blocks,
                n_warps,
                cfg.iters,
                ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return 0;
}

int main(int argc, char ** argv) {
    bench_config cfg = parse_args(argc, argv);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (cfg.device < 0 || cfg.device >= device_count) {
        std::fprintf(stderr, "invalid CUDA device %d, device_count=%d\n", cfg.device, device_count);
        return 1;
    }

    CUDA_CHECK(cudaSetDevice(cfg.device));

    cudaDeviceProp prop = {};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.device));

    const int cc = prop.major * 100 + prop.minor * 10;
    if (cfg.blocks == 0) {
        cfg.blocks = prop.multiProcessorCount * 4;
    }

    std::printf("device:               %d %s\n", cfg.device, prop.name);
    std::printf("compute_capability:   sm_%d%d cc=%d\n", prop.major, prop.minor, cc);
    std::printf("spark_target:         %s\n", cc == 1210 ? "yes" : "no");
    std::printf("blackwell_fp4_target: %s\n", cc >= 1200 && cc < 1300 ? "yes" : "no");
    std::printf("sms:                  %d\n", prop.multiProcessorCount);

    return run_benchmark(cfg, prop, cc);
}
