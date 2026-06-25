#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define GGML_COMMON_DECL_CUDA
#include "ggml-common.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CUDA_CHECK(err) spark_cuda_check((err), __FILE__, __LINE__)

static void spark_cuda_check(cudaError_t err, const char * file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s:%d: CUDA error: %s\n", file, line, cudaGetErrorString(err));
        std::exit(1);
    }
}

struct bench_config {
    int      device    = 0;
    int      seconds   = 1;
    int      blocks    = 0;
    int      threads   = 256;
    int      mtp_rows  = 4;
    int      max_threads_per_sm = 0;
    bool     combined_only = false;
    bool     kreuse_subtile = false;
    bool     kreuse_smem = false;
    bool     kreuse_smem_onegroup = false;
    bool     mixed_pv = false;
    bool     mixed_pv_prepacked = false;
    bool     mixed_pv_transform = false;
    int      kreuse_smem_groups = 0;
    int      mixed_pv_smem_groups = 0;
    int      mixed_pv_compact_smem_groups = 0;
    int      mixed_pv_rolling_groups = 0;
    int      mixed_pv_transform_groups = 0;
    int      mixed_pv_stripmine_groups = 0;
    int      mixed_pv_partial_groups = 0;
    size_t   bytes     = 256ull * 1024ull * 1024ull;
    uint64_t mma_iters = 20000;
};

struct kq_mma_lane_frag {
    int      a[4];
    int      b[2];
    uint32_t a_scale;
    uint32_t b_scale;
};

static_assert(sizeof(kq_mma_lane_frag) == 32, "unexpected staged KQ fragment size");

struct kq_mma_compact_frag {
    uint32_t q[8];
    uint32_t k[8];
    uint32_t q_scale;
    uint32_t k_scale;
};

static_assert(sizeof(kq_mma_compact_frag) == 72, "unexpected compact KQ fragment size");

struct pv_mma_b_lane_frag {
    int      b0;
    int      b1;
    uint32_t b_scale;
    uint32_t pad;
};

static_assert(sizeof(pv_mma_b_lane_frag) == 16, "unexpected packed PV B fragment size");

struct mixed_pv_b_lane_frag {
    uint32_t b0;
    uint32_t b1;
};

static_assert(sizeof(mixed_pv_b_lane_frag) == 8, "unexpected mixed PV B fragment size");

static constexpr int KQ_MMA_A_ROWS      = 16;
static constexpr int KQ_MMA_B_ROWS      = 8;
static constexpr int KQ_MMA_WORDS       = 8;
static constexpr int KQ_MMA_A_STRIDE    = 76;
static constexpr int KQ_MMA_B_STRIDE    = 8;
static constexpr int KQ_MMA_SHARED_INTS = KQ_MMA_A_ROWS * KQ_MMA_A_STRIDE + KQ_MMA_B_ROWS * KQ_MMA_B_STRIDE + KQ_MMA_A_ROWS + KQ_MMA_B_ROWS;
static constexpr int KQ_MMA_Q_SHARED_INTS = KQ_MMA_A_ROWS * KQ_MMA_A_STRIDE + KQ_MMA_A_ROWS;
static constexpr int KREUSE_STAGE_WORD_INTS = KQ_MMA_B_ROWS * 8 * KQ_MMA_WORDS;
static constexpr int KREUSE_STAGE_SCALE_INTS = KQ_MMA_B_ROWS * 8;
static constexpr int KREUSE_STAGE_INTS = KREUSE_STAGE_WORD_INTS + KREUSE_STAGE_SCALE_INTS;
static constexpr int GEMMA4_GLOBAL_KV_HEADS = 4;
static constexpr int GEMMA4_GLOBAL_HEAD_DIM = 512;
static constexpr int PV_TILE_ROWS = 8;

static void print_usage(const char * exe) {
    std::printf(
        "usage: %s [options]\n"
        "\n"
        "Spark roofline probes for native NVFP4 KV work.\n"
        "\n"
        "options:\n"
        "  --device N       CUDA device id (default: 0)\n"
        "  --seconds N      approximate seconds per bandwidth probe (default: 1)\n"
        "  --blocks N       CUDA blocks for probes (default: 4 * SM count)\n"
        "  --threads N      CUDA threads per block (default: 256)\n"
        "  --mtp-rows N     useful MTP verification rows in an m16 tile (default: 4)\n"
        "  --bytes N        bytes in read buffers (default: 268435456)\n"
        "  --mma-iters N    FP4 MMA loop iterations per warp (default: 20000)\n"
        "  --combined-only  run only the combined D=512 KQ/PV probes\n"
        "  --kreuse-subtile run the experimental one-group K-reuse PV probe\n"
        "  --kreuse-smem    run the experimental short-lifetime shared K-reuse PV probe\n"
        "  --kreuse-smem-onegroup  run the one-group shared K-reuse PV probe\n"
        "  --kreuse-smem-groups N  run the grouped shared K-reuse PV probe (N=1,2,4)\n"
        "  --mixed-pv       run the experimental mixed FP16-probability PV probe\n"
        "  --mixed-pv-prepacked  run the prepacked mixed-PV consume-ceiling probe\n"
        "  --mixed-pv-transform  run the compact NVFP4 to packed mixed-PV transform probe\n"
        "  --mixed-pv-transform-groups N  run the partial compact-to-packed mixed-PV transform probe (N=1,2,4)\n"
        "  --mixed-pv-partial-groups N  run the partial compact-to-mixed-PV consume probe (N=1,2,4)\n"
        "  --mixed-pv-smem-groups N  run the grouped mixed-PV half shared-stage probe (N=1,2,4)\n"
        "  --mixed-pv-compact-smem-groups N  run the grouped mixed-PV compact shared-stage probe (N=1,2,4,8)\n"
        "  --mixed-pv-rolling-groups N  run the rolling mixed-PV compact-stage plus inline-remainder probe (N=1,2,4)\n"
        "  --mixed-pv-stripmine-groups N  run the direct strip-mined mixed-PV probe (N=1,2,4,8)\n"
        "  --help           print this help\n",
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
        } else if (std::strcmp(arg, "--seconds") == 0) {
            cfg.seconds = (int) parse_u64(require_value(arg), arg);
        } else if (std::strcmp(arg, "--blocks") == 0) {
            cfg.blocks = (int) parse_u64(require_value(arg), arg);
        } else if (std::strcmp(arg, "--threads") == 0) {
            cfg.threads = (int) parse_u64(require_value(arg), arg);
        } else if (std::strcmp(arg, "--mtp-rows") == 0) {
            cfg.mtp_rows = (int) parse_u64(require_value(arg), arg);
        } else if (std::strcmp(arg, "--bytes") == 0) {
            cfg.bytes = (size_t) parse_u64(require_value(arg), arg);
        } else if (std::strcmp(arg, "--mma-iters") == 0) {
            cfg.mma_iters = parse_u64(require_value(arg), arg);
        } else if (std::strcmp(arg, "--combined-only") == 0) {
            cfg.combined_only = true;
        } else if (std::strcmp(arg, "--kreuse-subtile") == 0) {
            cfg.kreuse_subtile = true;
        } else if (std::strcmp(arg, "--kreuse-smem") == 0) {
            cfg.kreuse_smem = true;
        } else if (std::strcmp(arg, "--kreuse-smem-onegroup") == 0) {
            cfg.kreuse_smem_onegroup = true;
            cfg.kreuse_smem_groups = 1;
        } else if (std::strcmp(arg, "--kreuse-smem-groups") == 0) {
            cfg.kreuse_smem_groups = (int) parse_u64(require_value(arg), arg);
        } else if (std::strcmp(arg, "--mixed-pv") == 0) {
            cfg.mixed_pv = true;
        } else if (std::strcmp(arg, "--mixed-pv-prepacked") == 0) {
            cfg.mixed_pv_prepacked = true;
        } else if (std::strcmp(arg, "--mixed-pv-transform") == 0) {
            cfg.mixed_pv_transform = true;
        } else if (std::strcmp(arg, "--mixed-pv-transform-groups") == 0) {
            cfg.mixed_pv_transform_groups = (int) parse_u64(require_value(arg), arg);
        } else if (std::strcmp(arg, "--mixed-pv-partial-groups") == 0) {
            cfg.mixed_pv_partial_groups = (int) parse_u64(require_value(arg), arg);
        } else if (std::strcmp(arg, "--mixed-pv-smem-groups") == 0) {
            cfg.mixed_pv_smem_groups = (int) parse_u64(require_value(arg), arg);
        } else if (std::strcmp(arg, "--mixed-pv-compact-smem-groups") == 0) {
            cfg.mixed_pv_compact_smem_groups = (int) parse_u64(require_value(arg), arg);
        } else if (std::strcmp(arg, "--mixed-pv-rolling-groups") == 0) {
            cfg.mixed_pv_rolling_groups = (int) parse_u64(require_value(arg), arg);
        } else if (std::strcmp(arg, "--mixed-pv-stripmine-groups") == 0) {
            cfg.mixed_pv_stripmine_groups = (int) parse_u64(require_value(arg), arg);
        } else if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg);
            print_usage(argv[0]);
            std::exit(1);
        }
    }

    if (cfg.seconds <= 0 || cfg.threads <= 0 || cfg.mtp_rows <= 0 || cfg.mtp_rows > KQ_MMA_A_ROWS || cfg.bytes < 4096 || cfg.mma_iters == 0) {
        std::fprintf(stderr, "invalid benchmark configuration\n");
        std::exit(1);
    }

    if (cfg.kreuse_smem_groups != 0 && cfg.kreuse_smem_groups != 1 && cfg.kreuse_smem_groups != 2 && cfg.kreuse_smem_groups != 4) {
        std::fprintf(stderr, "invalid --kreuse-smem-groups value: %d (expected 1, 2, or 4)\n", cfg.kreuse_smem_groups);
        std::exit(1);
    }
    if (cfg.mixed_pv_smem_groups != 0 && cfg.mixed_pv_smem_groups != 1 && cfg.mixed_pv_smem_groups != 2 && cfg.mixed_pv_smem_groups != 4) {
        std::fprintf(stderr, "invalid --mixed-pv-smem-groups value: %d (expected 1, 2, or 4)\n", cfg.mixed_pv_smem_groups);
        std::exit(1);
    }
    if (cfg.mixed_pv_compact_smem_groups != 0 && cfg.mixed_pv_compact_smem_groups != 1 && cfg.mixed_pv_compact_smem_groups != 2 && cfg.mixed_pv_compact_smem_groups != 4 && cfg.mixed_pv_compact_smem_groups != 8) {
        std::fprintf(stderr, "invalid --mixed-pv-compact-smem-groups value: %d (expected 1, 2, 4, or 8)\n", cfg.mixed_pv_compact_smem_groups);
        std::exit(1);
    }
    if (cfg.mixed_pv_rolling_groups != 0 && cfg.mixed_pv_rolling_groups != 1 && cfg.mixed_pv_rolling_groups != 2 && cfg.mixed_pv_rolling_groups != 4) {
        std::fprintf(stderr, "invalid --mixed-pv-rolling-groups value: %d (expected 1, 2, or 4)\n", cfg.mixed_pv_rolling_groups);
        std::exit(1);
    }
    if (cfg.mixed_pv_transform_groups != 0 && cfg.mixed_pv_transform_groups != 1 && cfg.mixed_pv_transform_groups != 2 && cfg.mixed_pv_transform_groups != 4) {
        std::fprintf(stderr, "invalid --mixed-pv-transform-groups value: %d (expected 1, 2, or 4)\n", cfg.mixed_pv_transform_groups);
        std::exit(1);
    }
    if (cfg.mixed_pv_stripmine_groups != 0 && cfg.mixed_pv_stripmine_groups != 1 && cfg.mixed_pv_stripmine_groups != 2 && cfg.mixed_pv_stripmine_groups != 4 && cfg.mixed_pv_stripmine_groups != 8) {
        std::fprintf(stderr, "invalid --mixed-pv-stripmine-groups value: %d (expected 1, 2, 4, or 8)\n", cfg.mixed_pv_stripmine_groups);
        std::exit(1);
    }
    if (cfg.mixed_pv_partial_groups != 0 && cfg.mixed_pv_partial_groups != 1 && cfg.mixed_pv_partial_groups != 2 && cfg.mixed_pv_partial_groups != 4) {
        std::fprintf(stderr, "invalid --mixed-pv-partial-groups value: %d (expected 1, 2, or 4)\n", cfg.mixed_pv_partial_groups);
        std::exit(1);
    }

    return cfg;
}

__global__ void fill_u32_kernel(uint32_t * data, size_t n) {
    const size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t) gridDim.x * blockDim.x;

    for (size_t i = tid; i < n; i += stride) {
        data[i] = (uint32_t) (0x9e3779b9u * (uint32_t) (i + 1));
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

template <int nfrags>
__global__ void fill_kq_mma_stage_kernel(
        const block_nvfp4 * q,
        const block_nvfp4 * k,
        kq_mma_lane_frag *  stage,
        size_t              n_tiles,
        size_t              block_mask) {
    const size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    const size_t n = n_tiles * (size_t) nfrags * 32u;

    for (size_t i = tid; i < n; i += stride) {
        const int lane = (int) (i & 31u);
        const int frag = (int) ((i >> 5) % (size_t) nfrags);
        const size_t tile = i / ((size_t) nfrags * 32u);
        const size_t idx = (tile * (size_t) nfrags + (size_t) frag) & block_mask;

        const block_nvfp4 q_blk = q[idx];
        const block_nvfp4 k_blk = k[idx];

        const uint32_t * q_qs = reinterpret_cast<const uint32_t *>(q_blk.qs);
        const uint32_t * k_qs = reinterpret_cast<const uint32_t *>(k_blk.qs);

        kq_mma_lane_frag f;
        f.a[0] = (int) q_qs[(lane + 0) & 7];
        f.a[1] = (int) q_qs[(lane + 1) & 7];
        f.a[2] = (int) q_qs[(lane + 2) & 7];
        f.a[3] = (int) q_qs[(lane + 3) & 7];
        f.b[0] = (int) k_qs[(lane + 0) & 7];
        f.b[1] = (int) k_qs[(lane + 1) & 7];
        f.a_scale = ((uint32_t) q_blk.d[0]) | ((uint32_t) q_blk.d[1] << 8) | ((uint32_t) q_blk.d[2] << 16) | ((uint32_t) q_blk.d[3] << 24);
        f.b_scale = ((uint32_t) k_blk.d[0]) | ((uint32_t) k_blk.d[1] << 8) | ((uint32_t) k_blk.d[2] << 16) | ((uint32_t) k_blk.d[3] << 24);
        stage[i] = f;
    }
}

template <int nfrags>
__global__ void fill_kq_mma_compact_kernel(
        const block_nvfp4 *   q,
        const block_nvfp4 *   k,
        kq_mma_compact_frag * stage,
        size_t                n_tiles,
        size_t                block_mask) {
    const size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    const size_t n = n_tiles * (size_t) nfrags;

    for (size_t i = tid; i < n; i += stride) {
        const int frag = (int) (i % (size_t) nfrags);
        const size_t tile = i / (size_t) nfrags;
        const size_t idx = (tile * (size_t) nfrags + (size_t) frag) & block_mask;

        const block_nvfp4 q_blk = q[idx];
        const block_nvfp4 k_blk = k[idx];

        const uint32_t * q_qs = reinterpret_cast<const uint32_t *>(q_blk.qs);
        const uint32_t * k_qs = reinterpret_cast<const uint32_t *>(k_blk.qs);

        kq_mma_compact_frag f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            f.q[j] = q_qs[j];
            f.k[j] = k_qs[j];
        }
        f.q_scale = ((uint32_t) q_blk.d[0]) | ((uint32_t) q_blk.d[1] << 8) | ((uint32_t) q_blk.d[2] << 16) | ((uint32_t) q_blk.d[3] << 24);
        f.k_scale = ((uint32_t) k_blk.d[0]) | ((uint32_t) k_blk.d[1] << 8) | ((uint32_t) k_blk.d[2] << 16) | ((uint32_t) k_blk.d[3] << 24);
        stage[i] = f;
    }
}

__global__ void read_u4_kernel(const uint4 * data, size_t n, uint64_t iters, uint64_t * out) {
    const size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t) gridDim.x * blockDim.x;

    uint64_t acc = 0;
    for (uint64_t r = 0; r < iters; ++r) {
        for (size_t i = tid; i < n; i += stride) {
            const uint4 v = data[i];
            acc += v.x;
            acc += v.y;
            acc += v.z;
            acc += v.w;
        }
    }

    out[tid] = acc;
}

__global__ void read_nvfp4_kernel(const block_nvfp4 * data, size_t n, uint64_t iters, uint64_t * out) {
    const size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t) gridDim.x * blockDim.x;

    uint64_t acc = 0;
    for (uint64_t r = 0; r < iters; ++r) {
        for (size_t i = tid; i < n; i += stride) {
            const block_nvfp4 v = data[i];
#pragma unroll
            for (int s = 0; s < QK_NVFP4 / QK_NVFP4_SUB; ++s) {
                acc += v.d[s];
            }
#pragma unroll
            for (int q = 0; q < QK_NVFP4 / 2; ++q) {
                acc += v.qs[q];
            }
        }
    }

    out[tid] = acc;
}

__global__ void fp4_mma_kernel(const int * a, const int * b, const uint32_t * scales, float * out, uint64_t iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    const int lane = threadIdx.x & 31;
    const int warp = ((int) blockIdx.x * blockDim.x + threadIdx.x) >> 5;

    int ax0 = a[(4 * lane + 0) & 1023];
    int ax1 = a[(4 * lane + 1) & 1023];
    int ax2 = a[(4 * lane + 2) & 1023];
    int ax3 = a[(4 * lane + 3) & 1023];
    int bx0 = b[(2 * lane + 0) & 1023];
    int bx1 = b[(2 * lane + 1) & 1023];

    const uint32_t a_scale = scales[lane & 255];
    const uint32_t b_scale = scales[(lane + 32) & 255];

    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;

    for (uint64_t i = 0; i < iters; ++i) {
        asm volatile(
            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
            "%10, {0, 0}, %11, {0, 0};"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(ax0), "r"(ax1), "r"(ax2), "r"(ax3), "r"(bx0), "r"(bx1), "r"(a_scale), "r"(b_scale));
    }

    if (lane == 0) {
        out[warp] = d0 + d1 + d2 + d3;
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = 0.0f;
    }
    (void) a;
    (void) b;
    (void) scales;
    (void) iters;
#endif
}

template <int dkq>
__global__ void pv_fp4_mma_native_kernel(
        const int *      a,
        const int *      b,
        const uint32_t * scales,
        float *          out,
        uint64_t         iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    static_assert(dkq == 256 || dkq == 512, "unsupported native FP4 PV width");
    constexpr int n_out_frags = dkq / 8;

    const int lane = threadIdx.x & 31;
    const int warp = ((int) blockIdx.x * blockDim.x + threadIdx.x) >> 5;

    const int ax0 = a[(4 * lane + 0) & 1023];
    const int ax1 = a[(4 * lane + 1) & 1023];
    const int ax2 = a[(4 * lane + 2) & 1023];
    const int ax3 = a[(4 * lane + 3) & 1023];
    const uint32_t a_scale = scales[lane & 255];

    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;

    for (uint64_t i = 0; i < iters; ++i) {
#pragma unroll
        for (int frag = 0; frag < n_out_frags; ++frag) {
            const int bx0 = b[(frag * 64 + 2 * lane + 0) & 4095];
            const int bx1 = b[(frag * 64 + 2 * lane + 1) & 4095];
            const uint32_t b_scale = scales[(frag * 32 + lane + 32) & 255];

            asm volatile(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                "%10, {0, 0}, %11, {0, 0};"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(ax0), "r"(ax1), "r"(ax2), "r"(ax3), "r"(bx0), "r"(bx1), "r"(a_scale), "r"(b_scale));
        }
    }

    if (lane == 0) {
        out[warp] = d0 + d1 + d2 + d3;
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = 0.0f;
    }
    (void) a;
    (void) b;
    (void) scales;
    (void) out;
    (void) iters;
#endif
}

static __device__ __forceinline__ int spark_nvfp4_dim_group_word(const block_nvfp4 & blk, int group) {
    const int sub = group >> 1;
    const bool high = (group & 1) != 0;

    int word = 0;
#pragma unroll
    for (int j = 0; j < QK_NVFP4_SUB / 2; ++j) {
        const uint8_t q = blk.qs[sub * (QK_NVFP4_SUB / 2) + j];
        const int code = high ? (q >> 4) : (q & 0x0f);
        word |= code << (4 * j);
    }
    return word;
}

template <int dkq>
__global__ void pv_fp4_mma_compact_b_kernel(
        const int *        a,
        const block_nvfp4 * v,
        const uint32_t *   scales,
        size_t             block_mask,
        float *            out,
        uint64_t           iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    static_assert(dkq == 256 || dkq == 512, "unsupported compact FP4 PV width");
    constexpr int nfrags = dkq / QK_NVFP4;
    constexpr int groups_per_frag = QK_NVFP4 / 8;

    const int lane = threadIdx.x & 31;
    const int warp = ((int) blockIdx.x * blockDim.x + threadIdx.x) >> 5;

    const int ax0 = a[(4 * lane + 0) & 1023];
    const int ax1 = a[(4 * lane + 1) & 1023];
    const int ax2 = a[(4 * lane + 2) & 1023];
    const int ax3 = a[(4 * lane + 3) & 1023];
    const uint32_t a_scale = scales[lane & 255];

    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;

    for (uint64_t i = 0; i < iters; ++i) {
        const size_t base = ((size_t) warp * (size_t) nfrags * 64u) +
                            ((size_t) i * (size_t) gridDim.x * (size_t) nfrags * 64u);
#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const int row0 = 2 * lane + 0;
            const int row1 = 2 * lane + 1;
            const block_nvfp4 blk0 = v[(base + (size_t) row0 * (size_t) nfrags + (size_t) frag) & block_mask];
            const block_nvfp4 blk1 = v[(base + (size_t) row1 * (size_t) nfrags + (size_t) frag) & block_mask];
            const uint32_t b_scale = ((uint32_t) blk0.d[0]) | ((uint32_t) blk0.d[1] << 8) | ((uint32_t) blk0.d[2] << 16) | ((uint32_t) blk0.d[3] << 24);

#pragma unroll
            for (int group = 0; group < groups_per_frag; ++group) {
                const int bx0 = spark_nvfp4_dim_group_word(blk0, group);
                const int bx1 = spark_nvfp4_dim_group_word(blk1, group);

                asm volatile(
                    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                    "%10, {0, 0}, %11, {0, 0};"
                    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                    : "r"(ax0), "r"(ax1), "r"(ax2), "r"(ax3), "r"(bx0), "r"(bx1), "r"(a_scale), "r"(b_scale));
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
    (void) a;
    (void) v;
    (void) scales;
    (void) block_mask;
    (void) out;
    (void) iters;
#endif
}

template <int dkq>
__global__ void fill_pv_mma_b_stage_kernel(
        const block_nvfp4 *  v,
        pv_mma_b_lane_frag * stage,
        size_t               n_tiles,
        size_t               block_mask) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    static_assert(dkq == 256 || dkq == 512, "unsupported packed FP4 PV width");
    constexpr int nfrags = dkq / QK_NVFP4;
    constexpr int groups_per_frag = QK_NVFP4 / 8;
    constexpr int n_out_frags = dkq / 8;

    const size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    const size_t n = n_tiles * (size_t) n_out_frags * 32u;

    for (size_t i = tid; i < n; i += stride) {
        const int lane = (int) (i & 31u);
        const int out_frag = (int) ((i >> 5) % (size_t) n_out_frags);
        const int frag = out_frag / groups_per_frag;
        const int group = out_frag - frag * groups_per_frag;
        const size_t tile = i / ((size_t) n_out_frags * 32u);
        const size_t base = tile * (size_t) nfrags * 64u;

        const int row0 = 2 * lane + 0;
        const int row1 = 2 * lane + 1;
        const block_nvfp4 blk0 = v[(base + (size_t) row0 * (size_t) nfrags + (size_t) frag) & block_mask];
        const block_nvfp4 blk1 = v[(base + (size_t) row1 * (size_t) nfrags + (size_t) frag) & block_mask];

        pv_mma_b_lane_frag f;
        f.b0 = spark_nvfp4_dim_group_word(blk0, group);
        f.b1 = spark_nvfp4_dim_group_word(blk1, group);
        f.b_scale = ((uint32_t) blk0.d[0]) | ((uint32_t) blk0.d[1] << 8) | ((uint32_t) blk0.d[2] << 16) | ((uint32_t) blk0.d[3] << 24);
        f.pad = 0;
        stage[i] = f;
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        stage[0].b0 = 0;
    }
    (void) v;
    (void) n_tiles;
    (void) block_mask;
#endif
}

template <int dkq>
__global__ void fill_pv_mma_b_stage_reuse_kernel(
        const block_nvfp4 *  v,
        pv_mma_b_lane_frag * stage,
        size_t               n_tiles,
        size_t               block_mask) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    static_assert(dkq == 256 || dkq == 512, "unsupported packed FP4 PV width");
    constexpr int nfrags = dkq / QK_NVFP4;
    constexpr int groups_per_frag = QK_NVFP4 / 8;
    constexpr int n_out_frags = dkq / 8;

    const size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    const size_t n = n_tiles * (size_t) nfrags * 32u;

    for (size_t i = tid; i < n; i += stride) {
        const int lane = (int) (i & 31u);
        const int frag = (int) ((i >> 5) % (size_t) nfrags);
        const size_t tile = i / ((size_t) nfrags * 32u);
        const size_t src_base = tile * (size_t) nfrags * 64u;
        const size_t dst_base = tile * (size_t) n_out_frags * 32u;

        const int row0 = 2 * lane + 0;
        const int row1 = 2 * lane + 1;
        const block_nvfp4 blk0 = v[(src_base + (size_t) row0 * (size_t) nfrags + (size_t) frag) & block_mask];
        const block_nvfp4 blk1 = v[(src_base + (size_t) row1 * (size_t) nfrags + (size_t) frag) & block_mask];
        const uint32_t b_scale = ((uint32_t) blk0.d[0]) | ((uint32_t) blk0.d[1] << 8) | ((uint32_t) blk0.d[2] << 16) | ((uint32_t) blk0.d[3] << 24);

#pragma unroll
        for (int group = 0; group < groups_per_frag; ++group) {
            pv_mma_b_lane_frag f;
            f.b0 = spark_nvfp4_dim_group_word(blk0, group);
            f.b1 = spark_nvfp4_dim_group_word(blk1, group);
            f.b_scale = b_scale;
            f.pad = 0;
            stage[dst_base + (size_t) (frag * groups_per_frag + group) * 32u + (size_t) lane] = f;
        }
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        stage[0].b0 = 0;
    }
    (void) v;
    (void) n_tiles;
    (void) block_mask;
#endif
}

__global__ void fill_mixed_pv_b_stage_kernel(
        mixed_pv_b_lane_frag * stage,
        size_t                 n_entries) {
    const size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t) gridDim.x * blockDim.x;

    for (size_t i = tid; i < n_entries; i += stride) {
        stage[i].b0 = 0x3c003800u + (uint32_t) (i & 0x1fu);
        stage[i].b1 = 0x34003000u + (uint32_t) ((i >> 5) & 0x1fu);
    }
}

template <int dkq>
__global__ void pv_fp4_mma_packed_b_kernel(
        const int *                 a,
        const pv_mma_b_lane_frag *  stage,
        const uint32_t *            scales,
        size_t                      tile_mask,
        float *                     out,
        uint64_t                    iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    static_assert(dkq == 256 || dkq == 512, "unsupported packed FP4 PV width");
    constexpr int n_out_frags = dkq / 8;

    const int lane = threadIdx.x & 31;
    const int warp = ((int) blockIdx.x * blockDim.x + threadIdx.x) >> 5;

    const int ax0 = a[(4 * lane + 0) & 1023];
    const int ax1 = a[(4 * lane + 1) & 1023];
    const int ax2 = a[(4 * lane + 2) & 1023];
    const int ax3 = a[(4 * lane + 3) & 1023];
    const uint32_t a_scale = scales[lane & 255];

    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;

    for (uint64_t i = 0; i < iters; ++i) {
        const size_t tile = ((size_t) warp + (size_t) i * (size_t) gridDim.x) & tile_mask;
        const size_t base = tile * (size_t) n_out_frags * 32u;
#pragma unroll
        for (int out_frag = 0; out_frag < n_out_frags; ++out_frag) {
            const pv_mma_b_lane_frag f = stage[base + (size_t) out_frag * 32u + (size_t) lane];

            asm volatile(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                "%10, {0, 0}, %11, {0, 0};"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(ax0), "r"(ax1), "r"(ax2), "r"(ax3), "r"(f.b0), "r"(f.b1), "r"(a_scale), "r"(f.b_scale));
        }
    }

    if (lane == 0) {
        out[warp] = d0 + d1 + d2 + d3;
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = 0.0f;
    }
    (void) a;
    (void) stage;
    (void) scales;
    (void) tile_mask;
    (void) out;
    (void) iters;
#endif
}

template <int dkq>
__global__ void pv_mixed_half_mma_packed_b_kernel(
        const mixed_pv_b_lane_frag * stage,
        size_t                       tile_mask,
        float *                      out,
        uint64_t                     iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    static_assert(dkq == 256 || dkq == 512, "unsupported mixed packed PV width");
    constexpr int n_out_frags = dkq / 8;

    const int lane = threadIdx.x & 31;
    const int warp = ((int) blockIdx.x * blockDim.x + threadIdx.x) >> 5;

    const uint32_t ax0 = 0x3c003800u + (uint32_t) lane;
    const uint32_t ax1 = 0x34003000u + (uint32_t) lane;
    const uint32_t ax2 = 0x2c002800u + (uint32_t) lane;
    const uint32_t ax3 = 0x24002000u + (uint32_t) lane;

    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;

    for (uint64_t i = 0; i < iters; ++i) {
        const size_t tile = ((size_t) warp + (size_t) i * (size_t) gridDim.x) & tile_mask;
        const size_t base = tile * (size_t) n_out_frags * 4u * 32u;
#pragma unroll
        for (int out_frag = 0; out_frag < n_out_frags; ++out_frag) {
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                const mixed_pv_b_lane_frag f = stage[base + ((size_t) out_frag * 4u + (size_t) kk) * 32u + (size_t) lane];
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                    : "r"(ax0), "r"(ax1), "r"(ax2), "r"(ax3), "r"(f.b0), "r"(f.b1));
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
    (void) stage;
    (void) tile_mask;
    (void) out;
    (void) iters;
#endif
}

template <int dkq, int staged_groups>
__global__ void pv_mixed_half_mma_packed_b_groups_kernel(
        const mixed_pv_b_lane_frag * stage,
        size_t                       tile_mask,
        float *                      out,
        uint64_t                     iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    static_assert(dkq == 256 || dkq == 512, "unsupported partial mixed packed PV width");
    static_assert(staged_groups == 1 || staged_groups == 2 || staged_groups == 4, "unsupported staged group count");
    constexpr int nfrags = dkq / QK_NVFP4;
    constexpr int measured_out_frags = nfrags * staged_groups;

    const int lane = threadIdx.x & 31;
    const int warp = ((int) blockIdx.x * blockDim.x + threadIdx.x) >> 5;

    const uint32_t ax0 = 0x3c003800u + (uint32_t) lane;
    const uint32_t ax1 = 0x34003000u + (uint32_t) lane;
    const uint32_t ax2 = 0x2c002800u + (uint32_t) lane;
    const uint32_t ax3 = 0x24002000u + (uint32_t) lane;

    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;

    for (uint64_t i = 0; i < iters; ++i) {
        const size_t tile = ((size_t) warp + (size_t) i * (size_t) gridDim.x) & tile_mask;
        const size_t base = tile * (size_t) measured_out_frags * 4u * 32u;
#pragma unroll
        for (int out_frag = 0; out_frag < measured_out_frags; ++out_frag) {
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                const mixed_pv_b_lane_frag f = stage[base + ((size_t) out_frag * 4u + (size_t) kk) * 32u + (size_t) lane];
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                    : "r"(ax0), "r"(ax1), "r"(ax2), "r"(ax3), "r"(f.b0), "r"(f.b1));
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
    (void) stage;
    (void) tile_mask;
    (void) out;
    (void) iters;
#endif
}

static __device__ __forceinline__ float spark_ue4m3_to_fp32(uint8_t x) {
    if (x == 0) {
        return 0.0f;
    }

    const int exp = (int) (x >> 3);
    const int man = (int) (x & 7u);
    return ldexpf(1.0f + (float) man * 0.125f, exp - 7);
}

static __device__ __forceinline__ float spark_fp4_e2m1_to_fp32(uint8_t x) {
    const int mag_code = (int) (x & 7u);
    const int mag2 =
        mag_code == 0 ? 0 :
        mag_code == 1 ? 1 :
        mag_code == 2 ? 2 :
        mag_code == 3 ? 3 :
        mag_code == 4 ? 4 :
        mag_code == 5 ? 6 :
        mag_code == 6 ? 8 : 12;
    const float v = 0.5f * (float) mag2;
    return (x & 8u) ? -v : v;
}

static __device__ __forceinline__ float spark_dequant_sum_nvfp4_block(const block_nvfp4 & blk) {
    float sum = 0.0f;
#pragma unroll
    for (int sub = 0; sub < QK_NVFP4 / QK_NVFP4_SUB; ++sub) {
        const float d = spark_ue4m3_to_fp32(blk.d[sub]);
#pragma unroll
        for (int j = 0; j < QK_NVFP4_SUB / 2; ++j) {
            const uint8_t q = blk.qs[sub * (QK_NVFP4_SUB / 2) + j];
            sum += d * spark_fp4_e2m1_to_fp32(q & 0x0f);
            sum += d * spark_fp4_e2m1_to_fp32(q >> 4);
        }
    }
    return sum;
}

static __device__ __forceinline__ __half2 spark_dequant_nvfp4_byte_to_half2(uint8_t q, float d) {
    const float lo = d * spark_fp4_e2m1_to_fp32(q & 0x0f);
    const float hi = d * spark_fp4_e2m1_to_fp32(q >> 4);
    return __floats2half2_rn(lo, hi);
}

static __device__ __forceinline__ uint32_t spark_half2_bits(__half2 v) {
    union {
        __half2  h;
        uint32_t u;
    } cvt;
    cvt.h = v;
    return cvt.u;
}

template <int dkq>
__global__ void fill_mixed_pv_b_stage_from_nvfp4_reuse_kernel(
        const block_nvfp4 *    v,
        mixed_pv_b_lane_frag * stage,
        size_t                 n_tiles,
        size_t                 block_mask) {
    static_assert(dkq == 256 || dkq == 512, "unsupported packed mixed PV width");
    constexpr int nfrags = dkq / QK_NVFP4;
    constexpr int groups_per_frag = QK_NVFP4 / 8;
    constexpr int n_out_frags = dkq / 8;

    const size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    const size_t n = n_tiles * (size_t) nfrags * 32u;

    for (size_t i = tid; i < n; i += stride) {
        const int lane = (int) (i & 31u);
        const int frag = (int) ((i >> 5) % (size_t) nfrags);
        const size_t tile = i / ((size_t) nfrags * 32u);
        const size_t src_base = tile * (size_t) nfrags * 64u;
        const size_t dst_base = tile * (size_t) n_out_frags * 4u * 32u;

        const int row0 = 2 * lane + 0;
        const int row1 = 2 * lane + 1;
        const block_nvfp4 blk0 = v[(src_base + (size_t) row0 * (size_t) nfrags + (size_t) frag) & block_mask];
        const block_nvfp4 blk1 = v[(src_base + (size_t) row1 * (size_t) nfrags + (size_t) frag) & block_mask];

#pragma unroll
        for (int group = 0; group < groups_per_frag; ++group) {
            const int out_frag = frag * groups_per_frag + group;
            const int word0 = spark_nvfp4_dim_group_word(blk0, group);
            const int word1 = spark_nvfp4_dim_group_word(blk1, group);
            const float d0 = spark_ue4m3_to_fp32(blk0.d[group / 2]);
            const float d1 = spark_ue4m3_to_fp32(blk1.d[group / 2]);

#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                const uint8_t q0 = (uint8_t) ((uint32_t) word0 >> (8 * kk));
                const uint8_t q1 = (uint8_t) ((uint32_t) word1 >> (8 * kk));
                mixed_pv_b_lane_frag f;
                f.b0 = spark_half2_bits(spark_dequant_nvfp4_byte_to_half2(q0, d0));
                f.b1 = spark_half2_bits(spark_dequant_nvfp4_byte_to_half2(q1, d1));
                stage[dst_base + ((size_t) out_frag * 4u + (size_t) kk) * 32u + (size_t) lane] = f;
            }
        }
    }
}

template <int dkq, int staged_groups>
__global__ void fill_mixed_pv_b_stage_from_nvfp4_groups_kernel(
        const block_nvfp4 *    v,
        mixed_pv_b_lane_frag * stage,
        size_t                 n_tiles,
        size_t                 block_mask) {
    static_assert(dkq == 256 || dkq == 512, "unsupported partial packed mixed PV width");
    static_assert(staged_groups == 1 || staged_groups == 2 || staged_groups == 4, "unsupported staged group count");
    constexpr int nfrags = dkq / QK_NVFP4;
    constexpr int measured_out_frags = nfrags * staged_groups;

    const size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    const size_t n = n_tiles * (size_t) nfrags * 32u;

    for (size_t i = tid; i < n; i += stride) {
        const int lane = (int) (i & 31u);
        const int frag = (int) ((i >> 5) % (size_t) nfrags);
        const size_t tile = i / ((size_t) nfrags * 32u);
        const size_t src_base = tile * (size_t) nfrags * 64u;
        const size_t dst_base = tile * (size_t) measured_out_frags * 4u * 32u;

        const int row0 = 2 * lane + 0;
        const int row1 = 2 * lane + 1;
        const block_nvfp4 blk0 = v[(src_base + (size_t) row0 * (size_t) nfrags + (size_t) frag) & block_mask];
        const block_nvfp4 blk1 = v[(src_base + (size_t) row1 * (size_t) nfrags + (size_t) frag) & block_mask];

#pragma unroll
        for (int group = 0; group < staged_groups; ++group) {
            const int out_frag = frag * staged_groups + group;
            const int word0 = spark_nvfp4_dim_group_word(blk0, group);
            const int word1 = spark_nvfp4_dim_group_word(blk1, group);
            const float d0 = spark_ue4m3_to_fp32(blk0.d[group / 2]);
            const float d1 = spark_ue4m3_to_fp32(blk1.d[group / 2]);

#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                const uint8_t q0 = (uint8_t) ((uint32_t) word0 >> (8 * kk));
                const uint8_t q1 = (uint8_t) ((uint32_t) word1 >> (8 * kk));
                mixed_pv_b_lane_frag f;
                f.b0 = spark_half2_bits(spark_dequant_nvfp4_byte_to_half2(q0, d0));
                f.b1 = spark_half2_bits(spark_dequant_nvfp4_byte_to_half2(q1, d1));
                stage[dst_base + ((size_t) out_frag * 4u + (size_t) kk) * 32u + (size_t) lane] = f;
            }
        }
    }
}

template <int dkq, int staged_groups>
__global__ void pv_mixed_half_partial_nvfp4_kernel(
        const block_nvfp4 * v,
        size_t              block_mask,
        float *             out,
        uint64_t            iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    static_assert(dkq == 256 || dkq == 512, "unsupported partial mixed PV width");
    static_assert(staged_groups == 1 || staged_groups == 2 || staged_groups == 4, "unsupported staged group count");
    constexpr int nfrags = dkq / QK_NVFP4;

    const int lane = threadIdx.x & 31;
    const int warp = ((int) blockIdx.x * blockDim.x + threadIdx.x) >> 5;

    const uint32_t ax0 = 0x3c003800u + (uint32_t) lane;
    const uint32_t ax1 = 0x34003000u + (uint32_t) lane;
    const uint32_t ax2 = 0x2c002800u + (uint32_t) lane;
    const uint32_t ax3 = 0x24002000u + (uint32_t) lane;

    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;

    for (uint64_t i = 0; i < iters; ++i) {
        const size_t base = ((size_t) warp * (size_t) nfrags * 64u) +
                            ((size_t) i * (size_t) gridDim.x * (size_t) nfrags * 64u);
#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const int row0 = 2 * lane + 0;
            const int row1 = 2 * lane + 1;
            const block_nvfp4 blk0 = v[(base + (size_t) row0 * (size_t) nfrags + (size_t) frag) & block_mask];
            const block_nvfp4 blk1 = v[(base + (size_t) row1 * (size_t) nfrags + (size_t) frag) & block_mask];

#pragma unroll
            for (int group = 0; group < staged_groups; ++group) {
                const int word0 = spark_nvfp4_dim_group_word(blk0, group);
                const int word1 = spark_nvfp4_dim_group_word(blk1, group);
                const float scale0 = spark_ue4m3_to_fp32(blk0.d[group / 2]);
                const float scale1 = spark_ue4m3_to_fp32(blk1.d[group / 2]);

#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const uint8_t q0 = (uint8_t) ((uint32_t) word0 >> (8 * kk));
                    const uint8_t q1 = (uint8_t) ((uint32_t) word1 >> (8 * kk));
                    const uint32_t bx0 = spark_half2_bits(spark_dequant_nvfp4_byte_to_half2(q0, scale0));
                    const uint32_t bx1 = spark_half2_bits(spark_dequant_nvfp4_byte_to_half2(q1, scale1));
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                        : "r"(ax0), "r"(ax1), "r"(ax2), "r"(ax3), "r"(bx0), "r"(bx1));
                }
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
    (void) v;
    (void) block_mask;
    (void) out;
    (void) iters;
#endif
}

template <int dkq, int v_rows>
__global__ void pv_dequant_v_eq_k_nvfp4_kernel(
        const block_nvfp4 * v,
        size_t              block_mask,
        float *             out,
        uint64_t            iters) {
    static_assert(dkq == 256 || dkq == 512, "unsupported PV width");
    static_assert(v_rows == 2 || v_rows == 4 || v_rows == PV_TILE_ROWS, "unsupported PV tile rows");
    constexpr int nfrags = dkq / QK_NVFP4;
    constexpr int blocks_per_tile = v_rows * nfrags;

    const int lane = threadIdx.x & 31;
    const int warp = ((int) blockIdx.x * blockDim.x + threadIdx.x) >> 5;

    float acc = 0.0f;
    for (uint64_t i = 0; i < iters; ++i) {
        const size_t base = ((size_t) warp * (size_t) blocks_per_tile) +
                            ((size_t) i * (size_t) gridDim.x * (size_t) blocks_per_tile);
#pragma unroll
        for (int b = lane; b < blocks_per_tile; b += 32) {
            const block_nvfp4 blk = v[(base + (size_t) b) & block_mask];
            acc += spark_dequant_sum_nvfp4_block(blk);
        }
    }

    if (lane == 0) {
        out[warp] = acc;
    }
}

template <int dkq, int v_rows>
__global__ void pv_stage_half2_v_eq_k_nvfp4_kernel(
        const block_nvfp4 * v,
        size_t              block_mask,
        float *             out,
        uint64_t            iters) {
    static_assert(dkq == 256 || dkq == 512, "unsupported PV width");
    static_assert(v_rows == 2 || v_rows == 4 || v_rows == PV_TILE_ROWS, "unsupported PV tile rows");
    constexpr int nfrags = dkq / QK_NVFP4;
    constexpr int blocks_per_tile = v_rows * nfrags;
    constexpr int half2_per_block = QK_NVFP4 / 2;
    constexpr int half2_per_tile = blocks_per_tile * half2_per_block;

    extern __shared__ __half2 smem_h2[];

    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warp = ((int) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    __half2 * tile = smem_h2 + (size_t) warp_in_block * (size_t) half2_per_tile;

    float acc = 0.0f;
    for (uint64_t i = 0; i < iters; ++i) {
        const size_t base = ((size_t) warp * (size_t) blocks_per_tile) +
                            ((size_t) i * (size_t) gridDim.x * (size_t) blocks_per_tile);
#pragma unroll
        for (int b = lane; b < blocks_per_tile; b += 32) {
            const block_nvfp4 blk = v[(base + (size_t) b) & block_mask];
#pragma unroll
            for (int sub = 0; sub < QK_NVFP4 / QK_NVFP4_SUB; ++sub) {
                const float d = spark_ue4m3_to_fp32(blk.d[sub]);
#pragma unroll
                for (int j = 0; j < QK_NVFP4_SUB / 2; ++j) {
                    const int q = sub * (QK_NVFP4_SUB / 2) + j;
                    tile[b * half2_per_block + q] = spark_dequant_nvfp4_byte_to_half2(blk.qs[q], d);
                }
            }
        }

        __syncwarp();

#pragma unroll
        for (int h = lane; h < half2_per_tile; h += 32) {
            const __half2 hv = tile[h];
            acc += __half2float(__low2half(hv));
            acc += __half2float(__high2half(hv));
        }
    }

    if (lane == 0) {
        out[warp] = acc;
    }
}

__global__ void kq256_mma_nvfp4_kernel(
        const block_nvfp4 * q,
        const block_nvfp4 * k,
        size_t              block_mask,
        float *             out,
        uint64_t            iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    const int lane = threadIdx.x & 31;
    const int warp = ((int) blockIdx.x * blockDim.x + threadIdx.x) >> 5;

    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;

    for (uint64_t i = 0; i < iters; ++i) {
#pragma unroll
        for (int frag = 0; frag < 4; ++frag) {
            const size_t idx = (((size_t) warp * 4u) + (size_t) frag + (size_t) i * (size_t) gridDim.x) & block_mask;

            const block_nvfp4 q_blk = q[idx];
            const block_nvfp4 k_blk = k[idx];

            const uint32_t * q_qs = reinterpret_cast<const uint32_t *>(q_blk.qs);
            const uint32_t * k_qs = reinterpret_cast<const uint32_t *>(k_blk.qs);

            const int ax0 = (int) q_qs[(lane + 0) & 7];
            const int ax1 = (int) q_qs[(lane + 1) & 7];
            const int ax2 = (int) q_qs[(lane + 2) & 7];
            const int ax3 = (int) q_qs[(lane + 3) & 7];
            const int bx0 = (int) k_qs[(lane + 0) & 7];
            const int bx1 = (int) k_qs[(lane + 1) & 7];

            const uint32_t q_scale = *reinterpret_cast<const uint32_t *>(q_blk.d);
            const uint32_t k_scale = *reinterpret_cast<const uint32_t *>(k_blk.d);

            asm volatile(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                "%10, {0, 0}, %11, {0, 0};"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(ax0), "r"(ax1), "r"(ax2), "r"(ax3), "r"(bx0), "r"(bx1), "r"(q_scale), "r"(k_scale));
        }
    }

    if (lane == 0) {
        out[warp] = d0 + d1 + d2 + d3;
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = 0.0f;
    }
    (void) q;
    (void) k;
    (void) block_mask;
    (void) iters;
#endif
}

template <int nfrags>
__global__ void kq_mma_staged_nvfp4_kernel(
        const kq_mma_lane_frag * stage,
        size_t                   tile_mask,
        float *                  out,
        uint64_t                 iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    const int lane = threadIdx.x & 31;
    const int warp = ((int) blockIdx.x * blockDim.x + threadIdx.x) >> 5;

    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;

    for (uint64_t i = 0; i < iters; ++i) {
#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const size_t tile = ((size_t) warp + (size_t) i * (size_t) gridDim.x) & tile_mask;
            const size_t idx = ((tile * (size_t) nfrags + (size_t) frag) * 32u) + (size_t) lane;
            const kq_mma_lane_frag f = stage[idx];

            asm volatile(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                "%10, {0, 0}, %11, {0, 0};"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(f.a[0]), "r"(f.a[1]), "r"(f.a[2]), "r"(f.a[3]), "r"(f.b[0]), "r"(f.b[1]), "r"(f.a_scale), "r"(f.b_scale));
        }
    }

    if (lane == 0) {
        out[warp] = d0 + d1 + d2 + d3;
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = 0.0f;
    }
    (void) stage;
    (void) tile_mask;
    (void) iters;
#endif
}

template <int nfrags>
__global__ void kq_mma_compact_nvfp4_kernel(
        const kq_mma_compact_frag * stage,
        size_t                       tile_mask,
        float *                      out,
        uint64_t                     iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    const int lane = threadIdx.x & 31;
    const int warp = ((int) blockIdx.x * blockDim.x + threadIdx.x) >> 5;

    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;

    for (uint64_t i = 0; i < iters; ++i) {
#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const size_t tile = ((size_t) warp + (size_t) i * (size_t) gridDim.x) & tile_mask;
            const kq_mma_compact_frag f = stage[tile * (size_t) nfrags + (size_t) frag];

            const int ax0 = (int) f.q[(lane + 0) & 7];
            const int ax1 = (int) f.q[(lane + 1) & 7];
            const int ax2 = (int) f.q[(lane + 2) & 7];
            const int ax3 = (int) f.q[(lane + 3) & 7];
            const int bx0 = (int) f.k[(lane + 0) & 7];
            const int bx1 = (int) f.k[(lane + 1) & 7];

            asm volatile(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                "%10, {0, 0}, %11, {0, 0};"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(ax0), "r"(ax1), "r"(ax2), "r"(ax3), "r"(bx0), "r"(bx1), "r"(f.q_scale), "r"(f.k_scale));
        }
    }

    if (lane == 0) {
        out[warp] = d0 + d1 + d2 + d3;
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = 0.0f;
    }
    (void) stage;
    (void) tile_mask;
    (void) iters;
#endif
}

template <int nfrags>
__global__ void kq_mma_shared_nvfp4_kernel(
        const block_nvfp4 * q,
        const block_nvfp4 * k,
        size_t              block_mask,
        float *             out,
        uint64_t            iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    extern __shared__ unsigned char smem[];

    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int warp = ((int) blockIdx.x * blockDim.x + threadIdx.x) >> 5;

    block_nvfp4 * q_s = reinterpret_cast<block_nvfp4 *>(smem);
    block_nvfp4 * k_s = q_s + warps_per_block * nfrags;

    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;

    for (uint64_t i = 0; i < iters; ++i) {
        if (lane < nfrags) {
            const size_t idx = (((size_t) warp * (size_t) nfrags) + (size_t) lane + (size_t) i * (size_t) gridDim.x) & block_mask;
            q_s[warp_in_block * nfrags + lane] = q[idx];
            k_s[warp_in_block * nfrags + lane] = k[idx];
        }

        __syncthreads();

#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const block_nvfp4 q_blk = q_s[warp_in_block * nfrags + frag];
            const block_nvfp4 k_blk = k_s[warp_in_block * nfrags + frag];

            const uint32_t * q_qs = reinterpret_cast<const uint32_t *>(q_blk.qs);
            const uint32_t * k_qs = reinterpret_cast<const uint32_t *>(k_blk.qs);

            const int ax0 = (int) q_qs[(lane + 0) & 7];
            const int ax1 = (int) q_qs[(lane + 1) & 7];
            const int ax2 = (int) q_qs[(lane + 2) & 7];
            const int ax3 = (int) q_qs[(lane + 3) & 7];
            const int bx0 = (int) k_qs[(lane + 0) & 7];
            const int bx1 = (int) k_qs[(lane + 1) & 7];

            const uint32_t q_scale = ((uint32_t) q_blk.d[0]) | ((uint32_t) q_blk.d[1] << 8) | ((uint32_t) q_blk.d[2] << 16) | ((uint32_t) q_blk.d[3] << 24);
            const uint32_t k_scale = ((uint32_t) k_blk.d[0]) | ((uint32_t) k_blk.d[1] << 8) | ((uint32_t) k_blk.d[2] << 16) | ((uint32_t) k_blk.d[3] << 24);

            asm volatile(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                "%10, {0, 0}, %11, {0, 0};"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(ax0), "r"(ax1), "r"(ax2), "r"(ax3), "r"(bx0), "r"(bx1), "r"(q_scale), "r"(k_scale));
        }

        __syncthreads();
    }

    if (lane == 0) {
        out[warp] = d0 + d1 + d2 + d3;
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = 0.0f;
    }
    (void) q;
    (void) k;
    (void) block_mask;
    (void) iters;
#endif
}

template <int nfrags>
__global__ void kq_mma_warp_nvfp4_kernel(
        const block_nvfp4 * q,
        const block_nvfp4 * k,
        size_t              block_mask,
        float *             out,
        uint64_t            iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    const int lane = threadIdx.x & 31;
    const int warp = ((int) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const unsigned mask = 0xffffffffu;

    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;

    for (uint64_t i = 0; i < iters; ++i) {
#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            uint32_t qv0 = 0;
            uint32_t qv1 = 0;
            uint32_t qv2 = 0;
            uint32_t qv3 = 0;
            uint32_t qv4 = 0;
            uint32_t qv5 = 0;
            uint32_t qv6 = 0;
            uint32_t qv7 = 0;
            uint32_t kv0 = 0;
            uint32_t kv1 = 0;
            uint32_t kv2 = 0;
            uint32_t kv3 = 0;
            uint32_t kv4 = 0;
            uint32_t kv5 = 0;
            uint32_t kv6 = 0;
            uint32_t kv7 = 0;
            uint32_t q_scale = 0;
            uint32_t k_scale = 0;

            if (lane == frag) {
                const size_t idx = (((size_t) warp * (size_t) nfrags) + (size_t) frag + (size_t) i * (size_t) gridDim.x) & block_mask;
                const block_nvfp4 q_blk = q[idx];
                const block_nvfp4 k_blk = k[idx];
                const uint32_t * q_qs = reinterpret_cast<const uint32_t *>(q_blk.qs);
                const uint32_t * k_qs = reinterpret_cast<const uint32_t *>(k_blk.qs);

                qv0 = q_qs[0];
                qv1 = q_qs[1];
                qv2 = q_qs[2];
                qv3 = q_qs[3];
                qv4 = q_qs[4];
                qv5 = q_qs[5];
                qv6 = q_qs[6];
                qv7 = q_qs[7];
                kv0 = k_qs[0];
                kv1 = k_qs[1];
                kv2 = k_qs[2];
                kv3 = k_qs[3];
                kv4 = k_qs[4];
                kv5 = k_qs[5];
                kv6 = k_qs[6];
                kv7 = k_qs[7];
                q_scale = ((uint32_t) q_blk.d[0]) | ((uint32_t) q_blk.d[1] << 8) | ((uint32_t) q_blk.d[2] << 16) | ((uint32_t) q_blk.d[3] << 24);
                k_scale = ((uint32_t) k_blk.d[0]) | ((uint32_t) k_blk.d[1] << 8) | ((uint32_t) k_blk.d[2] << 16) | ((uint32_t) k_blk.d[3] << 24);
            }

            qv0 = __shfl_sync(mask, qv0, frag);
            qv1 = __shfl_sync(mask, qv1, frag);
            qv2 = __shfl_sync(mask, qv2, frag);
            qv3 = __shfl_sync(mask, qv3, frag);
            qv4 = __shfl_sync(mask, qv4, frag);
            qv5 = __shfl_sync(mask, qv5, frag);
            qv6 = __shfl_sync(mask, qv6, frag);
            qv7 = __shfl_sync(mask, qv7, frag);
            kv0 = __shfl_sync(mask, kv0, frag);
            kv1 = __shfl_sync(mask, kv1, frag);
            kv2 = __shfl_sync(mask, kv2, frag);
            kv3 = __shfl_sync(mask, kv3, frag);
            kv4 = __shfl_sync(mask, kv4, frag);
            kv5 = __shfl_sync(mask, kv5, frag);
            kv6 = __shfl_sync(mask, kv6, frag);
            kv7 = __shfl_sync(mask, kv7, frag);
            q_scale = __shfl_sync(mask, q_scale, frag);
            k_scale = __shfl_sync(mask, k_scale, frag);

            const uint32_t qv[8] = { qv0, qv1, qv2, qv3, qv4, qv5, qv6, qv7 };
            const uint32_t kv[8] = { kv0, kv1, kv2, kv3, kv4, kv5, kv6, kv7 };
            const int ax0 = (int) qv[(lane + 0) & 7];
            const int ax1 = (int) qv[(lane + 1) & 7];
            const int ax2 = (int) qv[(lane + 2) & 7];
            const int ax3 = (int) qv[(lane + 3) & 7];
            const int bx0 = (int) kv[(lane + 0) & 7];
            const int bx1 = (int) kv[(lane + 1) & 7];

            asm volatile(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                "%10, {0, 0}, %11, {0, 0};"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(ax0), "r"(ax1), "r"(ax2), "r"(ax3), "r"(bx0), "r"(bx1), "r"(q_scale), "r"(k_scale));
        }
    }

    if (lane == 0) {
        out[warp] = d0 + d1 + d2 + d3;
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = 0.0f;
    }
    (void) q;
    (void) k;
    (void) block_mask;
    (void) iters;
#endif
}

template <int nfrags>
__global__ void kq_mma_ldmatrix_nvfp4_kernel(
        const block_nvfp4 * q,
        const block_nvfp4 * k,
        size_t              block_mask,
        float *             out,
        uint64_t            iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    extern __shared__ int smem_i[];

    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warp = ((int) blockIdx.x * blockDim.x + threadIdx.x) >> 5;

    int * warp_s = smem_i + warp_in_block * KQ_MMA_SHARED_INTS;
    int * q_s = warp_s;
    int * k_s = q_s + KQ_MMA_A_ROWS * KQ_MMA_A_STRIDE;
    uint32_t * q_sc = reinterpret_cast<uint32_t *>(k_s + KQ_MMA_B_ROWS * KQ_MMA_B_STRIDE);
    uint32_t * k_sc = q_sc + KQ_MMA_A_ROWS;

    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;

    for (uint64_t i = 0; i < iters; ++i) {
#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const size_t base = ((size_t) warp * (size_t) nfrags * (size_t) (KQ_MMA_A_ROWS + KQ_MMA_B_ROWS)) +
                                ((size_t) frag * (size_t) (KQ_MMA_A_ROWS + KQ_MMA_B_ROWS)) +
                                ((size_t) i * (size_t) gridDim.x * (size_t) nfrags * (size_t) (KQ_MMA_A_ROWS + KQ_MMA_B_ROWS));

            if (lane < KQ_MMA_A_ROWS) {
                const block_nvfp4 q_blk = q[(base + (size_t) lane) & block_mask];
                const uint32_t * q_qs = reinterpret_cast<const uint32_t *>(q_blk.qs);
#pragma unroll
                for (int c = 0; c < KQ_MMA_WORDS; ++c) {
                    q_s[lane * KQ_MMA_A_STRIDE + c] = (int) q_qs[c];
                }
                q_sc[lane] = ((uint32_t) q_blk.d[0]) | ((uint32_t) q_blk.d[1] << 8) | ((uint32_t) q_blk.d[2] << 16) | ((uint32_t) q_blk.d[3] << 24);
            } else if (lane < KQ_MMA_A_ROWS + KQ_MMA_B_ROWS) {
                const int row = lane - KQ_MMA_A_ROWS;
                const block_nvfp4 k_blk = k[(base + (size_t) KQ_MMA_A_ROWS + (size_t) row) & block_mask];
                const uint32_t * k_qs = reinterpret_cast<const uint32_t *>(k_blk.qs);
#pragma unroll
                for (int c = 0; c < KQ_MMA_WORDS; ++c) {
                    k_s[row * KQ_MMA_B_STRIDE + c] = (int) k_qs[c];
                }
                k_sc[row] = ((uint32_t) k_blk.d[0]) | ((uint32_t) k_blk.d[1] << 8) | ((uint32_t) k_blk.d[2] << 16) | ((uint32_t) k_blk.d[3] << 24);
            }

            __syncthreads();

            int ax0;
            int ax1;
            int ax2;
            int ax3;
            const int * a_ptr = q_s + (lane % KQ_MMA_A_ROWS) * KQ_MMA_A_STRIDE + (lane / KQ_MMA_A_ROWS) * 4;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                : "=r"(ax0), "=r"(ax1), "=r"(ax2), "=r"(ax3)
                : "l"(a_ptr));

            const int b_i = lane / 4;
            const int b_j0 = lane % 4;
            const int bx0 = k_s[b_i * KQ_MMA_B_STRIDE + b_j0];
            const int bx1 = k_s[b_i * KQ_MMA_B_STRIDE + 4 + b_j0];

            const int tidx_a = lane / 4 + (lane % 2) * 8;
            const int tidx_b = lane / 4;
            const uint32_t q_scale = q_sc[tidx_a];
            const uint32_t k_scale = k_sc[tidx_b];

            asm volatile(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                "%10, {0, 0}, %11, {0, 0};"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(ax0), "r"(ax1), "r"(ax2), "r"(ax3), "r"(bx0), "r"(bx1), "r"(q_scale), "r"(k_scale));

            __syncthreads();
        }
    }

    if (lane == 0) {
        out[warp] = d0 + d1 + d2 + d3;
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = 0.0f;
    }
    (void) q;
    (void) k;
    (void) block_mask;
    (void) iters;
#endif
}

template <int nfrags, int k_reuse>
__global__ void kq_mma_ldmatrix_reuse_nvfp4_kernel(
        const block_nvfp4 * q,
        const block_nvfp4 * k,
        size_t              block_mask,
        float *             out,
        uint64_t            iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    extern __shared__ int smem_i[];

    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warp = ((int) blockIdx.x * blockDim.x + threadIdx.x) >> 5;

    int * warp_s = smem_i + warp_in_block * KQ_MMA_SHARED_INTS;
    int * q_s = warp_s;
    int * k_s = q_s + KQ_MMA_A_ROWS * KQ_MMA_A_STRIDE;
    uint32_t * q_sc = reinterpret_cast<uint32_t *>(k_s + KQ_MMA_B_ROWS * KQ_MMA_B_STRIDE);
    uint32_t * k_sc = q_sc + KQ_MMA_A_ROWS;

    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;

    for (uint64_t i = 0; i < iters; ++i) {
#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const size_t q_base = ((size_t) warp * (size_t) nfrags * (size_t) KQ_MMA_A_ROWS) +
                                  ((size_t) frag * (size_t) KQ_MMA_A_ROWS) +
                                  ((size_t) i * (size_t) gridDim.x * (size_t) nfrags * (size_t) KQ_MMA_A_ROWS);

            if (lane < KQ_MMA_A_ROWS) {
                const block_nvfp4 q_blk = q[(q_base + (size_t) lane) & block_mask];
                const uint32_t * q_qs = reinterpret_cast<const uint32_t *>(q_blk.qs);
#pragma unroll
                for (int c = 0; c < KQ_MMA_WORDS; ++c) {
                    q_s[lane * KQ_MMA_A_STRIDE + c] = (int) q_qs[c];
                }
                q_sc[lane] = ((uint32_t) q_blk.d[0]) | ((uint32_t) q_blk.d[1] << 8) | ((uint32_t) q_blk.d[2] << 16) | ((uint32_t) q_blk.d[3] << 24);
            }

            __syncthreads();

#pragma unroll
            for (int kr = 0; kr < k_reuse; ++kr) {
                const size_t k_base = ((size_t) warp * (size_t) nfrags * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) frag * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) kr * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) i * (size_t) gridDim.x * (size_t) nfrags * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS);

                if (lane < KQ_MMA_B_ROWS) {
                    const block_nvfp4 k_blk = k[(k_base + (size_t) lane) & block_mask];
                    const uint32_t * k_qs = reinterpret_cast<const uint32_t *>(k_blk.qs);
#pragma unroll
                    for (int c = 0; c < KQ_MMA_WORDS; ++c) {
                        k_s[lane * KQ_MMA_B_STRIDE + c] = (int) k_qs[c];
                    }
                    k_sc[lane] = ((uint32_t) k_blk.d[0]) | ((uint32_t) k_blk.d[1] << 8) | ((uint32_t) k_blk.d[2] << 16) | ((uint32_t) k_blk.d[3] << 24);
                }

                __syncthreads();

                int ax0;
                int ax1;
                int ax2;
                int ax3;
                const int * a_ptr = q_s + (lane % KQ_MMA_A_ROWS) * KQ_MMA_A_STRIDE + (lane / KQ_MMA_A_ROWS) * 4;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                    : "=r"(ax0), "=r"(ax1), "=r"(ax2), "=r"(ax3)
                    : "l"(a_ptr));

                const int b_i = lane / 4;
                const int b_j0 = lane % 4;
                const int bx0 = k_s[b_i * KQ_MMA_B_STRIDE + b_j0];
                const int bx1 = k_s[b_i * KQ_MMA_B_STRIDE + 4 + b_j0];

                const int tidx_a = lane / 4 + (lane % 2) * 8;
                const int tidx_b = lane / 4;
                const uint32_t q_scale = q_sc[tidx_a];
                const uint32_t k_scale = k_sc[tidx_b];

                asm volatile(
                    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                    "%10, {0, 0}, %11, {0, 0};"
                    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                    : "r"(ax0), "r"(ax1), "r"(ax2), "r"(ax3), "r"(bx0), "r"(bx1), "r"(q_scale), "r"(k_scale));

                __syncthreads();
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
    (void) q;
    (void) k;
    (void) block_mask;
    (void) iters;
#endif
}

template <int nfrags, int k_reuse>
__global__ void kq_mma_ldmatrix_qreg_nvfp4_kernel(
        const block_nvfp4 * q,
        const block_nvfp4 * k,
        size_t              block_mask,
        float *             out,
        uint64_t            iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    extern __shared__ int smem_i[];

    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warp = ((int) blockIdx.x * blockDim.x + threadIdx.x) >> 5;

    int * warp_s = smem_i + warp_in_block * KQ_MMA_SHARED_INTS;
    int * q_s = warp_s;
    int * k_s = q_s + KQ_MMA_A_ROWS * KQ_MMA_A_STRIDE;
    uint32_t * q_sc = reinterpret_cast<uint32_t *>(k_s + KQ_MMA_B_ROWS * KQ_MMA_B_STRIDE);
    uint32_t * k_sc = q_sc + KQ_MMA_A_ROWS;

    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;

    for (uint64_t i = 0; i < iters; ++i) {
#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const size_t q_base = ((size_t) warp * (size_t) nfrags * (size_t) KQ_MMA_A_ROWS) +
                                  ((size_t) frag * (size_t) KQ_MMA_A_ROWS) +
                                  ((size_t) i * (size_t) gridDim.x * (size_t) nfrags * (size_t) KQ_MMA_A_ROWS);

            if (lane < KQ_MMA_A_ROWS) {
                const block_nvfp4 q_blk = q[(q_base + (size_t) lane) & block_mask];
                const uint32_t * q_qs = reinterpret_cast<const uint32_t *>(q_blk.qs);
#pragma unroll
                for (int c = 0; c < KQ_MMA_WORDS; ++c) {
                    q_s[lane * KQ_MMA_A_STRIDE + c] = (int) q_qs[c];
                }
                q_sc[lane] = ((uint32_t) q_blk.d[0]) | ((uint32_t) q_blk.d[1] << 8) | ((uint32_t) q_blk.d[2] << 16) | ((uint32_t) q_blk.d[3] << 24);
            }

            __syncthreads();

            int ax0;
            int ax1;
            int ax2;
            int ax3;
            const int * a_ptr = q_s + (lane % KQ_MMA_A_ROWS) * KQ_MMA_A_STRIDE + (lane / KQ_MMA_A_ROWS) * 4;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                : "=r"(ax0), "=r"(ax1), "=r"(ax2), "=r"(ax3)
                : "l"(a_ptr));

            const int tidx_a = lane / 4 + (lane % 2) * 8;
            const uint32_t q_scale = q_sc[tidx_a];

#pragma unroll
            for (int kr = 0; kr < k_reuse; ++kr) {
                const size_t k_base = ((size_t) warp * (size_t) nfrags * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) frag * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) kr * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) i * (size_t) gridDim.x * (size_t) nfrags * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS);

                if (lane < KQ_MMA_B_ROWS) {
                    const block_nvfp4 k_blk = k[(k_base + (size_t) lane) & block_mask];
                    const uint32_t * k_qs = reinterpret_cast<const uint32_t *>(k_blk.qs);
#pragma unroll
                    for (int c = 0; c < KQ_MMA_WORDS; ++c) {
                        k_s[lane * KQ_MMA_B_STRIDE + c] = (int) k_qs[c];
                    }
                    k_sc[lane] = ((uint32_t) k_blk.d[0]) | ((uint32_t) k_blk.d[1] << 8) | ((uint32_t) k_blk.d[2] << 16) | ((uint32_t) k_blk.d[3] << 24);
                }

                __syncthreads();

                const int b_i = lane / 4;
                const int b_j0 = lane % 4;
                const int bx0 = k_s[b_i * KQ_MMA_B_STRIDE + b_j0];
                const int bx1 = k_s[b_i * KQ_MMA_B_STRIDE + 4 + b_j0];

                const int tidx_b = lane / 4;
                const uint32_t k_scale = k_sc[tidx_b];

                asm volatile(
                    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                    "%10, {0, 0}, %11, {0, 0};"
                    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                    : "r"(ax0), "r"(ax1), "r"(ax2), "r"(ax3), "r"(bx0), "r"(bx1), "r"(q_scale), "r"(k_scale));

                __syncthreads();
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
    (void) q;
    (void) k;
    (void) block_mask;
    (void) iters;
#endif
}

template <int nfrags, int k_reuse>
__global__ void kq_mma_ldmatrix_qreg_warpk_nvfp4_kernel(
        const block_nvfp4 * q,
        const block_nvfp4 * k,
        size_t              block_mask,
        float *             out,
        uint64_t            iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    extern __shared__ int smem_i[];

    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warp = ((int) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const unsigned mask = 0xffffffffu;

    int * warp_s = smem_i + warp_in_block * KQ_MMA_Q_SHARED_INTS;
    int * q_s = warp_s;
    uint32_t * q_sc = reinterpret_cast<uint32_t *>(q_s + KQ_MMA_A_ROWS * KQ_MMA_A_STRIDE);

    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;

    for (uint64_t i = 0; i < iters; ++i) {
#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const size_t q_base = ((size_t) warp * (size_t) nfrags * (size_t) KQ_MMA_A_ROWS) +
                                  ((size_t) frag * (size_t) KQ_MMA_A_ROWS) +
                                  ((size_t) i * (size_t) gridDim.x * (size_t) nfrags * (size_t) KQ_MMA_A_ROWS);

            if (lane < KQ_MMA_A_ROWS) {
                const block_nvfp4 q_blk = q[(q_base + (size_t) lane) & block_mask];
                const uint32_t * q_qs = reinterpret_cast<const uint32_t *>(q_blk.qs);
#pragma unroll
                for (int c = 0; c < KQ_MMA_WORDS; ++c) {
                    q_s[lane * KQ_MMA_A_STRIDE + c] = (int) q_qs[c];
                }
                q_sc[lane] = ((uint32_t) q_blk.d[0]) | ((uint32_t) q_blk.d[1] << 8) | ((uint32_t) q_blk.d[2] << 16) | ((uint32_t) q_blk.d[3] << 24);
            }

            __syncwarp(mask);

            int ax0;
            int ax1;
            int ax2;
            int ax3;
            const int * a_ptr = q_s + (lane % KQ_MMA_A_ROWS) * KQ_MMA_A_STRIDE + (lane / KQ_MMA_A_ROWS) * 4;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                : "=r"(ax0), "=r"(ax1), "=r"(ax2), "=r"(ax3)
                : "l"(a_ptr));

            const int tidx_a = lane / 4 + (lane % 2) * 8;
            const uint32_t q_scale = q_sc[tidx_a];

#pragma unroll
            for (int kr = 0; kr < k_reuse; ++kr) {
                const size_t k_base = ((size_t) warp * (size_t) nfrags * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) frag * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) kr * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) i * (size_t) gridDim.x * (size_t) nfrags * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS);

                uint32_t kv0 = 0;
                uint32_t kv1 = 0;
                uint32_t kv2 = 0;
                uint32_t kv3 = 0;
                uint32_t kv4 = 0;
                uint32_t kv5 = 0;
                uint32_t kv6 = 0;
                uint32_t kv7 = 0;
                uint32_t k_scale_row = 0;

                if (lane < KQ_MMA_B_ROWS) {
                    const block_nvfp4 k_blk = k[(k_base + (size_t) lane) & block_mask];
                    const uint32_t * k_qs = reinterpret_cast<const uint32_t *>(k_blk.qs);
                    kv0 = k_qs[0];
                    kv1 = k_qs[1];
                    kv2 = k_qs[2];
                    kv3 = k_qs[3];
                    kv4 = k_qs[4];
                    kv5 = k_qs[5];
                    kv6 = k_qs[6];
                    kv7 = k_qs[7];
                    k_scale_row = ((uint32_t) k_blk.d[0]) | ((uint32_t) k_blk.d[1] << 8) | ((uint32_t) k_blk.d[2] << 16) | ((uint32_t) k_blk.d[3] << 24);
                }

                const int b_i = lane / 4;
                const int b_j0 = lane % 4;
                const int row_v0 = (int) __shfl_sync(mask, kv0, b_i);
                const int row_v1 = (int) __shfl_sync(mask, kv1, b_i);
                const int row_v2 = (int) __shfl_sync(mask, kv2, b_i);
                const int row_v3 = (int) __shfl_sync(mask, kv3, b_i);
                const int row_v4 = (int) __shfl_sync(mask, kv4, b_i);
                const int row_v5 = (int) __shfl_sync(mask, kv5, b_i);
                const int row_v6 = (int) __shfl_sync(mask, kv6, b_i);
                const int row_v7 = (int) __shfl_sync(mask, kv7, b_i);

                const int bx0 = b_j0 == 0 ? row_v0 : (b_j0 == 1 ? row_v1 : (b_j0 == 2 ? row_v2 : row_v3));
                const int bx1 = b_j0 == 0 ? row_v4 : (b_j0 == 1 ? row_v5 : (b_j0 == 2 ? row_v6 : row_v7));
                const uint32_t k_scale = __shfl_sync(mask, k_scale_row, b_i);

                asm volatile(
                    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                    "%10, {0, 0}, %11, {0, 0};"
                    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                    : "r"(ax0), "r"(ax1), "r"(ax2), "r"(ax3), "r"(bx0), "r"(bx1), "r"(q_scale), "r"(k_scale));
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
    (void) q;
    (void) k;
    (void) block_mask;
    (void) iters;
#endif
}

template <int nfrags, int k_reuse, bool attentionish, bool inline_pv>
__global__ void combined_kq_packedpv_budget_nvfp4_kernel(
        const block_nvfp4 *         q,
        const block_nvfp4 *         k,
        const pv_mma_b_lane_frag *  pv_stage,
        const uint32_t *            pv_scales,
        size_t                      block_mask,
        size_t                      pv_tile_mask,
        float *                     out,
        uint64_t                    iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    static_assert(nfrags == 4 || nfrags == 8, "unsupported combined KQ/PV width");
    static_assert(k_reuse == 8, "combined probe currently models Gemma-style reuse 8");
    extern __shared__ int smem_i[];

    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warp = ((int) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const unsigned mask = 0xffffffffu;

    int * warp_s = smem_i + warp_in_block * KQ_MMA_Q_SHARED_INTS;
    int * q_s = warp_s;
    uint32_t * q_sc = reinterpret_cast<uint32_t *>(q_s + KQ_MMA_A_ROWS * KQ_MMA_A_STRIDE);

    float kq0 = 0.0f;
    float kq1 = 0.0f;
    float kq2 = 0.0f;
    float kq3 = 0.0f;
    float pv0 = 0.0f;
    float pv1 = 0.0f;
    float pv2 = 0.0f;
    float pv3 = 0.0f;
    float row_m0 = -64.0f;
    float row_m1 = -64.0f;
    float row_l0 = 0.0f;
    float row_l1 = 0.0f;
    float out0 = 0.0f;
    float out1 = 0.0f;
    float out2 = 0.0f;
    float out3 = 0.0f;
    float out4 = 0.0f;
    float out5 = 0.0f;
    float out6 = 0.0f;
    float out7 = 0.0f;

    for (uint64_t i = 0; i < iters; ++i) {
        const int pax0 = (int) (0x11111111u + (uint32_t) lane);
        const int pax1 = (int) (0x22222222u + (uint32_t) lane);
        const int pax2 = (int) (0x33333333u + (uint32_t) lane);
        const int pax3 = (int) (0x44444444u + (uint32_t) lane);
        const uint32_t p_scale = pv_scales[lane & 255];

        const auto accumulate_pv = [&](int out_frag, int bx0, int bx1, uint32_t b_scale) {
            asm volatile(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                "%10, {0, 0}, %11, {0, 0};"
                : "+f"(pv0), "+f"(pv1), "+f"(pv2), "+f"(pv3)
                : "r"(pax0), "r"(pax1), "r"(pax2), "r"(pax3), "r"(bx0), "r"(bx1), "r"(p_scale), "r"(b_scale));

            float mix = (float) (out_frag + 1) * 0.000001f;
            if constexpr (attentionish) {
                const float norm = 1.0f / (row_l0 + row_l1 + 0.000001f);
                mix *= norm;
            }
            out0 += pv0 * mix;
            out1 += pv1 * mix;
            out2 += pv2 * mix;
            out3 += pv3 * mix;
            out4 += (pv0 + row_m0) * mix;
            out5 += (pv1 + row_l0) * mix;
            out6 += (pv2 + row_m1) * mix;
            out7 += (pv3 + row_l1) * mix;
        };

#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const size_t q_base = ((size_t) warp * (size_t) nfrags * (size_t) KQ_MMA_A_ROWS) +
                                  ((size_t) frag * (size_t) KQ_MMA_A_ROWS) +
                                  ((size_t) i * (size_t) gridDim.x * (size_t) nfrags * (size_t) KQ_MMA_A_ROWS);

            if (lane < KQ_MMA_A_ROWS) {
                const block_nvfp4 q_blk = q[(q_base + (size_t) lane) & block_mask];
                const uint32_t * q_qs = reinterpret_cast<const uint32_t *>(q_blk.qs);
#pragma unroll
                for (int c = 0; c < KQ_MMA_WORDS; ++c) {
                    q_s[lane * KQ_MMA_A_STRIDE + c] = (int) q_qs[c];
                }
                q_sc[lane] = ((uint32_t) q_blk.d[0]) | ((uint32_t) q_blk.d[1] << 8) | ((uint32_t) q_blk.d[2] << 16) | ((uint32_t) q_blk.d[3] << 24);
            }

            __syncwarp(mask);

            int ax0;
            int ax1;
            int ax2;
            int ax3;
            const int * a_ptr = q_s + (lane % KQ_MMA_A_ROWS) * KQ_MMA_A_STRIDE + (lane / KQ_MMA_A_ROWS) * 4;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                : "=r"(ax0), "=r"(ax1), "=r"(ax2), "=r"(ax3)
                : "l"(a_ptr));

            const int tidx_a = lane / 4 + (lane % 2) * 8;
            const uint32_t q_scale = q_sc[tidx_a];

#pragma unroll
            for (int kr = 0; kr < k_reuse; ++kr) {
                const size_t k_base = ((size_t) warp * (size_t) nfrags * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) frag * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) kr * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) i * (size_t) gridDim.x * (size_t) nfrags * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS);

                uint32_t kv0 = 0;
                uint32_t kv1 = 0;
                uint32_t kv2 = 0;
                uint32_t kv3 = 0;
                uint32_t kv4 = 0;
                uint32_t kv5 = 0;
                uint32_t kv6 = 0;
                uint32_t kv7 = 0;
                uint32_t k_scale_row = 0;

                if (lane < KQ_MMA_B_ROWS) {
                    const block_nvfp4 k_blk = k[(k_base + (size_t) lane) & block_mask];
                    const uint32_t * k_qs = reinterpret_cast<const uint32_t *>(k_blk.qs);
                    kv0 = k_qs[0];
                    kv1 = k_qs[1];
                    kv2 = k_qs[2];
                    kv3 = k_qs[3];
                    kv4 = k_qs[4];
                    kv5 = k_qs[5];
                    kv6 = k_qs[6];
                    kv7 = k_qs[7];
                    k_scale_row = ((uint32_t) k_blk.d[0]) | ((uint32_t) k_blk.d[1] << 8) | ((uint32_t) k_blk.d[2] << 16) | ((uint32_t) k_blk.d[3] << 24);
                }

                const int b_i = lane / 4;
                const int b_j0 = lane % 4;
                const int row_v0 = (int) __shfl_sync(mask, kv0, b_i);
                const int row_v1 = (int) __shfl_sync(mask, kv1, b_i);
                const int row_v2 = (int) __shfl_sync(mask, kv2, b_i);
                const int row_v3 = (int) __shfl_sync(mask, kv3, b_i);
                const int row_v4 = (int) __shfl_sync(mask, kv4, b_i);
                const int row_v5 = (int) __shfl_sync(mask, kv5, b_i);
                const int row_v6 = (int) __shfl_sync(mask, kv6, b_i);
                const int row_v7 = (int) __shfl_sync(mask, kv7, b_i);

                const int bx0 = b_j0 == 0 ? row_v0 : (b_j0 == 1 ? row_v1 : (b_j0 == 2 ? row_v2 : row_v3));
                const int bx1 = b_j0 == 0 ? row_v4 : (b_j0 == 1 ? row_v5 : (b_j0 == 2 ? row_v6 : row_v7));
                const uint32_t k_scale = __shfl_sync(mask, k_scale_row, b_i);

                asm volatile(
                    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                    "%10, {0, 0}, %11, {0, 0};"
                    : "+f"(kq0), "+f"(kq1), "+f"(kq2), "+f"(kq3)
                    : "r"(ax0), "r"(ax1), "r"(ax2), "r"(ax3), "r"(bx0), "r"(bx1), "r"(q_scale), "r"(k_scale));

                if constexpr (attentionish) {
                    const int q_row = lane & (KQ_MMA_A_ROWS - 1);
                    const int key_pos = frag * k_reuse + kr;
                    const int base_pos = (int) (i & 1023u);
                    const bool causal = key_pos <= base_pos + q_row;
                    const bool swa = key_pos + 1024 >= base_pos + q_row;

                    const float score0 = (causal && swa) ? kq0 * 0.000244140625f : -64.0f;
                    const float score1 = (causal && swa) ? kq2 * 0.000244140625f : -64.0f;
                    const float next_m0 = fmaxf(row_m0, score0);
                    const float next_m1 = fmaxf(row_m1, score1);
                    const float alpha0 = row_l0 == 0.0f ? 0.0f : exp2f(row_m0 - next_m0);
                    const float alpha1 = row_l1 == 0.0f ? 0.0f : exp2f(row_m1 - next_m1);
                    const float beta0 = exp2f(score0 - next_m0);
                    const float beta1 = exp2f(score1 - next_m1);
                    row_l0 = row_l0 * alpha0 + beta0;
                    row_l1 = row_l1 * alpha1 + beta1;
                    row_m0 = next_m0;
                    row_m1 = next_m1;
                } else {
                    row_m0 = fmaxf(row_m0, kq0);
                    row_m1 = fmaxf(row_m1, kq2);
                    row_l0 += fabsf(kq1) * 0.000001f + 1.0f;
                    row_l1 += fabsf(kq3) * 0.000001f + 1.0f;
                }
            }
        }

        if constexpr (inline_pv) {
            constexpr int groups_per_frag = QK_NVFP4 / 8;
            const size_t pv_base = ((size_t) warp * (size_t) nfrags * 64u) +
                                   ((size_t) i * (size_t) gridDim.x * (size_t) nfrags * 64u);
#pragma unroll
            for (int frag = 0; frag < nfrags; ++frag) {
                const int row0 = 2 * lane + 0;
                const int row1 = 2 * lane + 1;
                const block_nvfp4 blk0 = k[(pv_base + (size_t) row0 * (size_t) nfrags + (size_t) frag) & block_mask];
                const block_nvfp4 blk1 = k[(pv_base + (size_t) row1 * (size_t) nfrags + (size_t) frag) & block_mask];
                const uint32_t b_scale = ((uint32_t) blk0.d[0]) | ((uint32_t) blk0.d[1] << 8) | ((uint32_t) blk0.d[2] << 16) | ((uint32_t) blk0.d[3] << 24);

#pragma unroll
                for (int group = 0; group < groups_per_frag; ++group) {
                    const int out_frag = frag * groups_per_frag + group;
                    const int bx0 = spark_nvfp4_dim_group_word(blk0, group);
                    const int bx1 = spark_nvfp4_dim_group_word(blk1, group);
                    accumulate_pv(out_frag, bx0, bx1, b_scale);
                }
            }
        } else {
            constexpr int n_out_frags = (nfrags * QK_NVFP4) / 8;
            const size_t pv_tile = ((size_t) warp + (size_t) i * (size_t) gridDim.x) & pv_tile_mask;
            const size_t pv_base = pv_tile * (size_t) n_out_frags * 32u;
#pragma unroll
            for (int out_frag = 0; out_frag < n_out_frags; ++out_frag) {
                const pv_mma_b_lane_frag f = pv_stage[pv_base + (size_t) out_frag * 32u + (size_t) lane];
                accumulate_pv(out_frag, f.b0, f.b1, f.b_scale);
            }
        }
    }

    if (lane == 0) {
        out[warp] = kq0 + kq1 + kq2 + kq3 + pv0 + pv1 + pv2 + pv3 +
                    row_m0 + row_m1 + row_l0 + row_l1 +
                    out0 + out1 + out2 + out3 + out4 + out5 + out6 + out7;
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = 0.0f;
    }
    (void) q;
    (void) k;
    (void) pv_stage;
    (void) pv_scales;
    (void) block_mask;
    (void) pv_tile_mask;
    (void) out;
    (void) iters;
#endif
}

template <int nfrags, int k_reuse, bool attentionish>
__global__ void combined_kq_mixedpv_half_nvfp4_kernel(
        const block_nvfp4 * q,
        const block_nvfp4 * k,
        size_t              block_mask,
        float *             out,
        uint64_t            iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    static_assert(nfrags == 8, "mixed-PV probe currently models only D=512");
    static_assert(k_reuse == 8, "mixed-PV probe currently models Gemma-style reuse 8");
    extern __shared__ int smem_i[];

    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warp = ((int) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const unsigned mask = 0xffffffffu;

    int * warp_s = smem_i + warp_in_block * KQ_MMA_Q_SHARED_INTS;
    int * q_s = warp_s;
    uint32_t * q_sc = reinterpret_cast<uint32_t *>(q_s + KQ_MMA_A_ROWS * KQ_MMA_A_STRIDE);

    float kq0 = 0.0f;
    float kq1 = 0.0f;
    float kq2 = 0.0f;
    float kq3 = 0.0f;
    float pv0 = 0.0f;
    float pv1 = 0.0f;
    float pv2 = 0.0f;
    float pv3 = 0.0f;
    float row_m0 = -64.0f;
    float row_m1 = -64.0f;
    float row_l0 = 0.0f;
    float row_l1 = 0.0f;
    float out0 = 0.0f;
    float out1 = 0.0f;
    float out2 = 0.0f;
    float out3 = 0.0f;
    float out4 = 0.0f;
    float out5 = 0.0f;
    float out6 = 0.0f;
    float out7 = 0.0f;

    for (uint64_t i = 0; i < iters; ++i) {
        const float p_base = attentionish ? 1.0f / (row_l0 + row_l1 + 1.0f) : 0.015625f;
        const uint32_t pax0 = spark_half2_bits(__floats2half2_rn(p_base, p_base * 0.9375f));
        const uint32_t pax1 = spark_half2_bits(__floats2half2_rn(p_base * 0.875f, p_base * 0.8125f));
        const uint32_t pax2 = spark_half2_bits(__floats2half2_rn(p_base * 0.75f, p_base * 0.6875f));
        const uint32_t pax3 = spark_half2_bits(__floats2half2_rn(p_base * 0.625f, p_base * 0.5625f));

        const auto accumulate_mixed_pv = [&](int out_frag, uint32_t bx0, uint32_t bx1) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+f"(pv0), "+f"(pv1), "+f"(pv2), "+f"(pv3)
                : "r"(pax0), "r"(pax1), "r"(pax2), "r"(pax3), "r"(bx0), "r"(bx1));

            float mix = (float) (out_frag + 1) * 0.000001f;
            if constexpr (attentionish) {
                const float norm = 1.0f / (row_l0 + row_l1 + 0.000001f);
                mix *= norm;
            }
            out0 += pv0 * mix;
            out1 += pv1 * mix;
            out2 += pv2 * mix;
            out3 += pv3 * mix;
            out4 += (pv0 + row_m0) * mix;
            out5 += (pv1 + row_l0) * mix;
            out6 += (pv2 + row_m1) * mix;
            out7 += (pv3 + row_l1) * mix;
        };

#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const size_t q_base = ((size_t) warp * (size_t) nfrags * (size_t) KQ_MMA_A_ROWS) +
                                  ((size_t) frag * (size_t) KQ_MMA_A_ROWS) +
                                  ((size_t) i * (size_t) gridDim.x * (size_t) nfrags * (size_t) KQ_MMA_A_ROWS);

            if (lane < KQ_MMA_A_ROWS) {
                const block_nvfp4 q_blk = q[(q_base + (size_t) lane) & block_mask];
                const uint32_t * q_qs = reinterpret_cast<const uint32_t *>(q_blk.qs);
#pragma unroll
                for (int c = 0; c < KQ_MMA_WORDS; ++c) {
                    q_s[lane * KQ_MMA_A_STRIDE + c] = (int) q_qs[c];
                }
                q_sc[lane] = ((uint32_t) q_blk.d[0]) | ((uint32_t) q_blk.d[1] << 8) | ((uint32_t) q_blk.d[2] << 16) | ((uint32_t) q_blk.d[3] << 24);
            }

            __syncwarp(mask);

            int ax0;
            int ax1;
            int ax2;
            int ax3;
            const int * a_ptr = q_s + (lane % KQ_MMA_A_ROWS) * KQ_MMA_A_STRIDE + (lane / KQ_MMA_A_ROWS) * 4;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                : "=r"(ax0), "=r"(ax1), "=r"(ax2), "=r"(ax3)
                : "l"(a_ptr));

            const int tidx_a = lane / 4 + (lane % 2) * 8;
            const uint32_t q_scale = q_sc[tidx_a];

#pragma unroll
            for (int kr = 0; kr < k_reuse; ++kr) {
                const size_t k_base = ((size_t) warp * (size_t) nfrags * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) frag * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) kr * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) i * (size_t) gridDim.x * (size_t) nfrags * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS);

                uint32_t kv0 = 0;
                uint32_t kv1 = 0;
                uint32_t kv2 = 0;
                uint32_t kv3 = 0;
                uint32_t kv4 = 0;
                uint32_t kv5 = 0;
                uint32_t kv6 = 0;
                uint32_t kv7 = 0;
                uint32_t k_scale_row = 0;

                if (lane < KQ_MMA_B_ROWS) {
                    const block_nvfp4 k_blk = k[(k_base + (size_t) lane) & block_mask];
                    const uint32_t * k_qs = reinterpret_cast<const uint32_t *>(k_blk.qs);
                    kv0 = k_qs[0];
                    kv1 = k_qs[1];
                    kv2 = k_qs[2];
                    kv3 = k_qs[3];
                    kv4 = k_qs[4];
                    kv5 = k_qs[5];
                    kv6 = k_qs[6];
                    kv7 = k_qs[7];
                    k_scale_row = ((uint32_t) k_blk.d[0]) | ((uint32_t) k_blk.d[1] << 8) | ((uint32_t) k_blk.d[2] << 16) | ((uint32_t) k_blk.d[3] << 24);
                }

                const int b_i = lane / 4;
                const int b_j0 = lane % 4;
                const int row_v0 = (int) __shfl_sync(mask, kv0, b_i);
                const int row_v1 = (int) __shfl_sync(mask, kv1, b_i);
                const int row_v2 = (int) __shfl_sync(mask, kv2, b_i);
                const int row_v3 = (int) __shfl_sync(mask, kv3, b_i);
                const int row_v4 = (int) __shfl_sync(mask, kv4, b_i);
                const int row_v5 = (int) __shfl_sync(mask, kv5, b_i);
                const int row_v6 = (int) __shfl_sync(mask, kv6, b_i);
                const int row_v7 = (int) __shfl_sync(mask, kv7, b_i);

                const int bx0 = b_j0 == 0 ? row_v0 : (b_j0 == 1 ? row_v1 : (b_j0 == 2 ? row_v2 : row_v3));
                const int bx1 = b_j0 == 0 ? row_v4 : (b_j0 == 1 ? row_v5 : (b_j0 == 2 ? row_v6 : row_v7));
                const uint32_t k_scale = __shfl_sync(mask, k_scale_row, b_i);

                asm volatile(
                    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                    "%10, {0, 0}, %11, {0, 0};"
                    : "+f"(kq0), "+f"(kq1), "+f"(kq2), "+f"(kq3)
                    : "r"(ax0), "r"(ax1), "r"(ax2), "r"(ax3), "r"(bx0), "r"(bx1), "r"(q_scale), "r"(k_scale));

                if constexpr (attentionish) {
                    const int q_row = lane & (KQ_MMA_A_ROWS - 1);
                    const int key_pos = frag * k_reuse + kr;
                    const int base_pos = (int) (i & 1023u);
                    const bool causal = key_pos <= base_pos + q_row;
                    const bool swa = key_pos + 1024 >= base_pos + q_row;

                    const float score0 = (causal && swa) ? kq0 * 0.000244140625f : -64.0f;
                    const float score1 = (causal && swa) ? kq2 * 0.000244140625f : -64.0f;
                    const float next_m0 = fmaxf(row_m0, score0);
                    const float next_m1 = fmaxf(row_m1, score1);
                    const float alpha0 = row_l0 == 0.0f ? 0.0f : exp2f(row_m0 - next_m0);
                    const float alpha1 = row_l1 == 0.0f ? 0.0f : exp2f(row_m1 - next_m1);
                    const float beta0 = exp2f(score0 - next_m0);
                    const float beta1 = exp2f(score1 - next_m1);
                    row_l0 = row_l0 * alpha0 + beta0;
                    row_l1 = row_l1 * alpha1 + beta1;
                    row_m0 = next_m0;
                    row_m1 = next_m1;
                } else {
                    row_m0 = fmaxf(row_m0, kq0);
                    row_m1 = fmaxf(row_m1, kq2);
                    row_l0 += fabsf(kq1) * 0.000001f + 1.0f;
                    row_l1 += fabsf(kq3) * 0.000001f + 1.0f;
                }
            }
        }

        constexpr int groups_per_frag = QK_NVFP4 / 8;
        const size_t pv_base = ((size_t) warp * (size_t) nfrags * 64u) +
                               ((size_t) i * (size_t) gridDim.x * (size_t) nfrags * 64u);
#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const int row0 = 2 * lane + 0;
            const int row1 = 2 * lane + 1;
            const block_nvfp4 blk0 = k[(pv_base + (size_t) row0 * (size_t) nfrags + (size_t) frag) & block_mask];
            const block_nvfp4 blk1 = k[(pv_base + (size_t) row1 * (size_t) nfrags + (size_t) frag) & block_mask];

#pragma unroll
            for (int group = 0; group < groups_per_frag; ++group) {
                const int out_frag = frag * groups_per_frag + group;
                const int word0 = spark_nvfp4_dim_group_word(blk0, group);
                const int word1 = spark_nvfp4_dim_group_word(blk1, group);
                const float d0 = spark_ue4m3_to_fp32(blk0.d[group / 2]);
                const float d1 = spark_ue4m3_to_fp32(blk1.d[group / 2]);

#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const uint8_t q0 = (uint8_t) ((uint32_t) word0 >> (8 * kk));
                    const uint8_t q1 = (uint8_t) ((uint32_t) word1 >> (8 * kk));
                    const uint32_t bx0 = spark_half2_bits(spark_dequant_nvfp4_byte_to_half2(q0, d0));
                    const uint32_t bx1 = spark_half2_bits(spark_dequant_nvfp4_byte_to_half2(q1, d1));
                    accumulate_mixed_pv(out_frag, bx0, bx1);
                }
            }
        }
    }

    if (lane == 0) {
        out[warp] = kq0 + kq1 + kq2 + kq3 + pv0 + pv1 + pv2 + pv3 +
                    row_m0 + row_m1 + row_l0 + row_l1 +
                    out0 + out1 + out2 + out3 + out4 + out5 + out6 + out7;
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = 0.0f;
    }
    (void) q;
    (void) k;
    (void) block_mask;
    (void) out;
    (void) iters;
#endif
}

template <int nfrags, int k_reuse, int staged_groups, bool attentionish>
__global__ void combined_kq_mixedpv_stripmine_nvfp4_kernel(
        const block_nvfp4 * q,
        const block_nvfp4 * k,
        size_t              block_mask,
        float *             out,
        uint64_t            iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    static_assert(nfrags == 8, "mixed-PV stripmine probe currently models only D=512");
    static_assert(k_reuse == 8, "mixed-PV stripmine probe currently models Gemma-style reuse 8");
    static_assert(staged_groups == 1 || staged_groups == 2 || staged_groups == 4 || staged_groups == 8, "unsupported staged group count");
    extern __shared__ int smem_i[];

    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warp = ((int) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const unsigned mask = 0xffffffffu;

    int * warp_s = smem_i + warp_in_block * KQ_MMA_Q_SHARED_INTS;
    int * q_s = warp_s;
    uint32_t * q_sc = reinterpret_cast<uint32_t *>(q_s + KQ_MMA_A_ROWS * KQ_MMA_A_STRIDE);

    float kq0 = 0.0f;
    float kq1 = 0.0f;
    float kq2 = 0.0f;
    float kq3 = 0.0f;
    float pv0 = 0.0f;
    float pv1 = 0.0f;
    float pv2 = 0.0f;
    float pv3 = 0.0f;
    float row_m0 = -64.0f;
    float row_m1 = -64.0f;
    float row_l0 = 0.0f;
    float row_l1 = 0.0f;
    float out0 = 0.0f;
    float out1 = 0.0f;
    float out2 = 0.0f;
    float out3 = 0.0f;
    float out4 = 0.0f;
    float out5 = 0.0f;
    float out6 = 0.0f;
    float out7 = 0.0f;

    for (uint64_t i = 0; i < iters; ++i) {
#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const size_t q_base = ((size_t) warp * (size_t) nfrags * (size_t) KQ_MMA_A_ROWS) +
                                  ((size_t) frag * (size_t) KQ_MMA_A_ROWS) +
                                  ((size_t) i * (size_t) gridDim.x * (size_t) nfrags * (size_t) KQ_MMA_A_ROWS);

            if (lane < KQ_MMA_A_ROWS) {
                const block_nvfp4 q_blk = q[(q_base + (size_t) lane) & block_mask];
                const uint32_t * q_qs = reinterpret_cast<const uint32_t *>(q_blk.qs);
#pragma unroll
                for (int c = 0; c < KQ_MMA_WORDS; ++c) {
                    q_s[lane * KQ_MMA_A_STRIDE + c] = (int) q_qs[c];
                }
                q_sc[lane] = ((uint32_t) q_blk.d[0]) | ((uint32_t) q_blk.d[1] << 8) | ((uint32_t) q_blk.d[2] << 16) | ((uint32_t) q_blk.d[3] << 24);
            }

            __syncwarp(mask);

            int ax0;
            int ax1;
            int ax2;
            int ax3;
            const int * a_ptr = q_s + (lane % KQ_MMA_A_ROWS) * KQ_MMA_A_STRIDE + (lane / KQ_MMA_A_ROWS) * 4;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                : "=r"(ax0), "=r"(ax1), "=r"(ax2), "=r"(ax3)
                : "l"(a_ptr));

            const int tidx_a = lane / 4 + (lane % 2) * 8;
            const uint32_t q_scale = q_sc[tidx_a];

#pragma unroll
            for (int kr = 0; kr < k_reuse; ++kr) {
                const size_t k_base = ((size_t) warp * (size_t) nfrags * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) frag * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) kr * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) i * (size_t) gridDim.x * (size_t) nfrags * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS);

                uint32_t kv0 = 0;
                uint32_t kv1 = 0;
                uint32_t kv2 = 0;
                uint32_t kv3 = 0;
                uint32_t kv4 = 0;
                uint32_t kv5 = 0;
                uint32_t kv6 = 0;
                uint32_t kv7 = 0;
                uint32_t k_scale_row = 0;

                if (lane < KQ_MMA_B_ROWS) {
                    const block_nvfp4 k_blk = k[(k_base + (size_t) lane) & block_mask];
                    const uint32_t * k_qs = reinterpret_cast<const uint32_t *>(k_blk.qs);
                    kv0 = k_qs[0];
                    kv1 = k_qs[1];
                    kv2 = k_qs[2];
                    kv3 = k_qs[3];
                    kv4 = k_qs[4];
                    kv5 = k_qs[5];
                    kv6 = k_qs[6];
                    kv7 = k_qs[7];
                    k_scale_row = ((uint32_t) k_blk.d[0]) | ((uint32_t) k_blk.d[1] << 8) | ((uint32_t) k_blk.d[2] << 16) | ((uint32_t) k_blk.d[3] << 24);
                }

                const int b_i = lane / 4;
                const int b_j0 = lane % 4;
                const int row_v0 = (int) __shfl_sync(mask, kv0, b_i);
                const int row_v1 = (int) __shfl_sync(mask, kv1, b_i);
                const int row_v2 = (int) __shfl_sync(mask, kv2, b_i);
                const int row_v3 = (int) __shfl_sync(mask, kv3, b_i);
                const int row_v4 = (int) __shfl_sync(mask, kv4, b_i);
                const int row_v5 = (int) __shfl_sync(mask, kv5, b_i);
                const int row_v6 = (int) __shfl_sync(mask, kv6, b_i);
                const int row_v7 = (int) __shfl_sync(mask, kv7, b_i);

                const int bx0_kq = b_j0 == 0 ? row_v0 : (b_j0 == 1 ? row_v1 : (b_j0 == 2 ? row_v2 : row_v3));
                const int bx1_kq = b_j0 == 0 ? row_v4 : (b_j0 == 1 ? row_v5 : (b_j0 == 2 ? row_v6 : row_v7));
                const uint32_t k_scale = __shfl_sync(mask, k_scale_row, b_i);

                asm volatile(
                    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                    "%10, {0, 0}, %11, {0, 0};"
                    : "+f"(kq0), "+f"(kq1), "+f"(kq2), "+f"(kq3)
                    : "r"(ax0), "r"(ax1), "r"(ax2), "r"(ax3), "r"(bx0_kq), "r"(bx1_kq), "r"(q_scale), "r"(k_scale));

                float beta0 = 1.0f;
                float beta1 = 1.0f;
                if constexpr (attentionish) {
                    const int q_row = lane & (KQ_MMA_A_ROWS - 1);
                    const int key_pos = frag * k_reuse + kr;
                    const int base_pos = (int) (i & 1023u);
                    const bool causal = key_pos <= base_pos + q_row;
                    const bool swa = key_pos + 1024 >= base_pos + q_row;

                    const float score0 = (causal && swa) ? kq0 * 0.000244140625f : -64.0f;
                    const float score1 = (causal && swa) ? kq2 * 0.000244140625f : -64.0f;
                    const float next_m0 = fmaxf(row_m0, score0);
                    const float next_m1 = fmaxf(row_m1, score1);
                    const float alpha0 = row_l0 == 0.0f ? 0.0f : exp2f(row_m0 - next_m0);
                    const float alpha1 = row_l1 == 0.0f ? 0.0f : exp2f(row_m1 - next_m1);
                    beta0 = exp2f(score0 - next_m0);
                    beta1 = exp2f(score1 - next_m1);
                    row_l0 = row_l0 * alpha0 + beta0;
                    row_l1 = row_l1 * alpha1 + beta1;
                    row_m0 = next_m0;
                    row_m1 = next_m1;
                } else {
                    row_m0 = fmaxf(row_m0, kq0);
                    row_m1 = fmaxf(row_m1, kq2);
                    row_l0 += fabsf(kq1) * 0.000001f + 1.0f;
                    row_l1 += fabsf(kq3) * 0.000001f + 1.0f;
                }

                const float p_base = attentionish ? 1.0f / (row_l0 + row_l1 + 1.0f) : 0.015625f;
                const uint32_t pax0 = spark_half2_bits(__floats2half2_rn(p_base * beta0, p_base * beta1));
                const uint32_t pax1 = spark_half2_bits(__floats2half2_rn(p_base * 0.875f, p_base * 0.8125f));
                const uint32_t pax2 = spark_half2_bits(__floats2half2_rn(p_base * 0.75f, p_base * 0.6875f));
                const uint32_t pax3 = spark_half2_bits(__floats2half2_rn(p_base * 0.625f, p_base * 0.5625f));

                const int b_i1 = (b_i + 4) & 7;
                const uint32_t scale0 = __shfl_sync(mask, k_scale_row, b_i);
                const uint32_t scale1 = __shfl_sync(mask, k_scale_row, b_i1);

#pragma unroll
                for (int group = 0; group < staged_groups; ++group) {
                    const uint32_t word0 =
                        group == 0 ? (uint32_t) row_v0 :
                        group == 1 ? (uint32_t) row_v1 :
                        group == 2 ? (uint32_t) row_v2 :
                        group == 3 ? (uint32_t) row_v3 :
                        group == 4 ? (uint32_t) row_v4 :
                        group == 5 ? (uint32_t) row_v5 :
                        group == 6 ? (uint32_t) row_v6 : (uint32_t) row_v7;
                    const uint32_t word1 = __shfl_sync(mask,
                        group == 0 ? kv0 :
                        group == 1 ? kv1 :
                        group == 2 ? kv2 :
                        group == 3 ? kv3 :
                        group == 4 ? kv4 :
                        group == 5 ? kv5 :
                        group == 6 ? kv6 : kv7,
                        b_i1);
                    const float d0 = spark_ue4m3_to_fp32((uint8_t) (scale0 >> (8 * (group / 2))));
                    const float d1 = spark_ue4m3_to_fp32((uint8_t) (scale1 >> (8 * (group / 2))));

#pragma unroll
                    for (int kk = 0; kk < 4; ++kk) {
                        const uint8_t q0 = (uint8_t) (word0 >> (8 * kk));
                        const uint8_t q1 = (uint8_t) (word1 >> (8 * kk));
                        const uint32_t bx0 = spark_half2_bits(spark_dequant_nvfp4_byte_to_half2(q0, d0));
                        const uint32_t bx1 = spark_half2_bits(spark_dequant_nvfp4_byte_to_half2(q1, d1));
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                            : "+f"(pv0), "+f"(pv1), "+f"(pv2), "+f"(pv3)
                            : "r"(pax0), "r"(pax1), "r"(pax2), "r"(pax3), "r"(bx0), "r"(bx1));
                    }

                    const int out_frag = (frag * k_reuse + kr) * staged_groups + group;
                    float mix = (float) (out_frag + 1) * 0.000001f;
                    if constexpr (attentionish) {
                        const float norm = 1.0f / (row_l0 + row_l1 + 0.000001f);
                        mix *= norm;
                    }
                    out0 += pv0 * mix;
                    out1 += pv1 * mix;
                    out2 += pv2 * mix;
                    out3 += pv3 * mix;
                    out4 += (pv0 + row_m0) * mix;
                    out5 += (pv1 + row_l0) * mix;
                    out6 += (pv2 + row_m1) * mix;
                    out7 += (pv3 + row_l1) * mix;
                }
            }
        }
    }

    if (lane == 0) {
        out[warp] = kq0 + kq1 + kq2 + kq3 + pv0 + pv1 + pv2 + pv3 +
                    row_m0 + row_m1 + row_l0 + row_l1 +
                    out0 + out1 + out2 + out3 + out4 + out5 + out6 + out7;
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = 0.0f;
    }
    (void) q;
    (void) k;
    (void) block_mask;
    (void) out;
    (void) iters;
#endif
}

template <int nfrags, int k_reuse, int staged_groups, bool attentionish>
__global__ void combined_kq_mixedpv_smem_half_nvfp4_kernel(
        const block_nvfp4 * q,
        const block_nvfp4 * k,
        size_t              block_mask,
        float *             out,
        uint64_t            iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    static_assert(nfrags == 8, "mixed-PV shared-stage probe currently models only D=512");
    static_assert(k_reuse == 8, "mixed-PV shared-stage probe currently models Gemma-style reuse 8");
    static_assert(staged_groups == 1 || staged_groups == 2 || staged_groups == 4, "unsupported staged group count");
    extern __shared__ int smem_i[];

    constexpr int mixed_stage_half2_per_row = staged_groups * 4;
    constexpr int mixed_stage_ints = KQ_MMA_B_ROWS * k_reuse * mixed_stage_half2_per_row;

    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warp = ((int) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const unsigned mask = 0xffffffffu;

    int * warp_s = smem_i + warp_in_block * (KQ_MMA_Q_SHARED_INTS + mixed_stage_ints);
    int * q_s = warp_s;
    uint32_t * q_sc = reinterpret_cast<uint32_t *>(q_s + KQ_MMA_A_ROWS * KQ_MMA_A_STRIDE);
    uint32_t * h_stage = reinterpret_cast<uint32_t *>(warp_s + KQ_MMA_Q_SHARED_INTS);

    float kq0 = 0.0f;
    float kq1 = 0.0f;
    float kq2 = 0.0f;
    float kq3 = 0.0f;
    float pv0 = 0.0f;
    float pv1 = 0.0f;
    float pv2 = 0.0f;
    float pv3 = 0.0f;
    float row_m0 = -64.0f;
    float row_m1 = -64.0f;
    float row_l0 = 0.0f;
    float row_l1 = 0.0f;
    float out0 = 0.0f;
    float out1 = 0.0f;
    float out2 = 0.0f;
    float out3 = 0.0f;
    float out4 = 0.0f;
    float out5 = 0.0f;
    float out6 = 0.0f;
    float out7 = 0.0f;

    for (uint64_t i = 0; i < iters; ++i) {
        const float p_base = attentionish ? 1.0f / (row_l0 + row_l1 + 1.0f) : 0.015625f;
        const uint32_t pax0 = spark_half2_bits(__floats2half2_rn(p_base, p_base * 0.9375f));
        const uint32_t pax1 = spark_half2_bits(__floats2half2_rn(p_base * 0.875f, p_base * 0.8125f));
        const uint32_t pax2 = spark_half2_bits(__floats2half2_rn(p_base * 0.75f, p_base * 0.6875f));
        const uint32_t pax3 = spark_half2_bits(__floats2half2_rn(p_base * 0.625f, p_base * 0.5625f));

        const auto accumulate_mixed_pv = [&](int out_frag, uint32_t bx0, uint32_t bx1) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+f"(pv0), "+f"(pv1), "+f"(pv2), "+f"(pv3)
                : "r"(pax0), "r"(pax1), "r"(pax2), "r"(pax3), "r"(bx0), "r"(bx1));

            float mix = (float) (out_frag + 1) * 0.000001f;
            if constexpr (attentionish) {
                const float norm = 1.0f / (row_l0 + row_l1 + 0.000001f);
                mix *= norm;
            }
            out0 += pv0 * mix;
            out1 += pv1 * mix;
            out2 += pv2 * mix;
            out3 += pv3 * mix;
            out4 += (pv0 + row_m0) * mix;
            out5 += (pv1 + row_l0) * mix;
            out6 += (pv2 + row_m1) * mix;
            out7 += (pv3 + row_l1) * mix;
        };

#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const size_t q_base = ((size_t) warp * (size_t) nfrags * (size_t) KQ_MMA_A_ROWS) +
                                  ((size_t) frag * (size_t) KQ_MMA_A_ROWS) +
                                  ((size_t) i * (size_t) gridDim.x * (size_t) nfrags * (size_t) KQ_MMA_A_ROWS);

            if (lane < KQ_MMA_A_ROWS) {
                const block_nvfp4 q_blk = q[(q_base + (size_t) lane) & block_mask];
                const uint32_t * q_qs = reinterpret_cast<const uint32_t *>(q_blk.qs);
#pragma unroll
                for (int c = 0; c < KQ_MMA_WORDS; ++c) {
                    q_s[lane * KQ_MMA_A_STRIDE + c] = (int) q_qs[c];
                }
                q_sc[lane] = ((uint32_t) q_blk.d[0]) | ((uint32_t) q_blk.d[1] << 8) | ((uint32_t) q_blk.d[2] << 16) | ((uint32_t) q_blk.d[3] << 24);
            }

            __syncwarp(mask);

            int ax0;
            int ax1;
            int ax2;
            int ax3;
            const int * a_ptr = q_s + (lane % KQ_MMA_A_ROWS) * KQ_MMA_A_STRIDE + (lane / KQ_MMA_A_ROWS) * 4;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                : "=r"(ax0), "=r"(ax1), "=r"(ax2), "=r"(ax3)
                : "l"(a_ptr));

            const int tidx_a = lane / 4 + (lane % 2) * 8;
            const uint32_t q_scale = q_sc[tidx_a];

#pragma unroll
            for (int kr = 0; kr < k_reuse; ++kr) {
                const size_t k_base = ((size_t) warp * (size_t) nfrags * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) frag * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) kr * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) i * (size_t) gridDim.x * (size_t) nfrags * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS);

                uint32_t kv0 = 0;
                uint32_t kv1 = 0;
                uint32_t kv2 = 0;
                uint32_t kv3 = 0;
                uint32_t kv4 = 0;
                uint32_t kv5 = 0;
                uint32_t kv6 = 0;
                uint32_t kv7 = 0;
                uint32_t k_scale_row = 0;

                if (lane < KQ_MMA_B_ROWS) {
                    const block_nvfp4 k_blk = k[(k_base + (size_t) lane) & block_mask];
                    const uint32_t * k_qs = reinterpret_cast<const uint32_t *>(k_blk.qs);
                    kv0 = k_qs[0];
                    kv1 = k_qs[1];
                    kv2 = k_qs[2];
                    kv3 = k_qs[3];
                    kv4 = k_qs[4];
                    kv5 = k_qs[5];
                    kv6 = k_qs[6];
                    kv7 = k_qs[7];
                    k_scale_row = ((uint32_t) k_blk.d[0]) | ((uint32_t) k_blk.d[1] << 8) | ((uint32_t) k_blk.d[2] << 16) | ((uint32_t) k_blk.d[3] << 24);

                    const uint32_t words[8] = { kv0, kv1, kv2, kv3, kv4, kv5, kv6, kv7 };
                    const int stage_row = kr * KQ_MMA_B_ROWS + lane;
#pragma unroll
                    for (int group = 0; group < staged_groups; ++group) {
                        const float d = spark_ue4m3_to_fp32((uint8_t) (k_scale_row >> (8 * (group / 2))));
                        const uint32_t word = words[group];
#pragma unroll
                        for (int kk = 0; kk < 4; ++kk) {
                            const uint8_t qv = (uint8_t) (word >> (8 * kk));
                            h_stage[stage_row * mixed_stage_half2_per_row + group * 4 + kk] =
                                spark_half2_bits(spark_dequant_nvfp4_byte_to_half2(qv, d));
                        }
                    }
                }

                const int b_i = lane / 4;
                const int b_j0 = lane % 4;
                const int row_v0 = (int) __shfl_sync(mask, kv0, b_i);
                const int row_v1 = (int) __shfl_sync(mask, kv1, b_i);
                const int row_v2 = (int) __shfl_sync(mask, kv2, b_i);
                const int row_v3 = (int) __shfl_sync(mask, kv3, b_i);
                const int row_v4 = (int) __shfl_sync(mask, kv4, b_i);
                const int row_v5 = (int) __shfl_sync(mask, kv5, b_i);
                const int row_v6 = (int) __shfl_sync(mask, kv6, b_i);
                const int row_v7 = (int) __shfl_sync(mask, kv7, b_i);

                const int bx0 = b_j0 == 0 ? row_v0 : (b_j0 == 1 ? row_v1 : (b_j0 == 2 ? row_v2 : row_v3));
                const int bx1 = b_j0 == 0 ? row_v4 : (b_j0 == 1 ? row_v5 : (b_j0 == 2 ? row_v6 : row_v7));
                const uint32_t k_scale = __shfl_sync(mask, k_scale_row, b_i);

                asm volatile(
                    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                    "%10, {0, 0}, %11, {0, 0};"
                    : "+f"(kq0), "+f"(kq1), "+f"(kq2), "+f"(kq3)
                    : "r"(ax0), "r"(ax1), "r"(ax2), "r"(ax3), "r"(bx0), "r"(bx1), "r"(q_scale), "r"(k_scale));

                if constexpr (attentionish) {
                    const int q_row = lane & (KQ_MMA_A_ROWS - 1);
                    const int key_pos = frag * k_reuse + kr;
                    const int base_pos = (int) (i & 1023u);
                    const bool causal = key_pos <= base_pos + q_row;
                    const bool swa = key_pos + 1024 >= base_pos + q_row;

                    const float score0 = (causal && swa) ? kq0 * 0.000244140625f : -64.0f;
                    const float score1 = (causal && swa) ? kq2 * 0.000244140625f : -64.0f;
                    const float next_m0 = fmaxf(row_m0, score0);
                    const float next_m1 = fmaxf(row_m1, score1);
                    const float alpha0 = row_l0 == 0.0f ? 0.0f : exp2f(row_m0 - next_m0);
                    const float alpha1 = row_l1 == 0.0f ? 0.0f : exp2f(row_m1 - next_m1);
                    const float beta0 = exp2f(score0 - next_m0);
                    const float beta1 = exp2f(score1 - next_m1);
                    row_l0 = row_l0 * alpha0 + beta0;
                    row_l1 = row_l1 * alpha1 + beta1;
                    row_m0 = next_m0;
                    row_m1 = next_m1;
                } else {
                    row_m0 = fmaxf(row_m0, kq0);
                    row_m1 = fmaxf(row_m1, kq2);
                    row_l0 += fabsf(kq1) * 0.000001f + 1.0f;
                    row_l1 += fabsf(kq3) * 0.000001f + 1.0f;
                }
            }

            __syncwarp(mask);

#pragma unroll
            for (int group = 0; group < staged_groups; ++group) {
                const int row0 = 2 * lane + 0;
                const int row1 = 2 * lane + 1;
                const int out_frag = frag * (QK_NVFP4 / 8) + group;
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const uint32_t bx0 = h_stage[row0 * mixed_stage_half2_per_row + group * 4 + kk];
                    const uint32_t bx1 = h_stage[row1 * mixed_stage_half2_per_row + group * 4 + kk];
                    accumulate_mixed_pv(out_frag, bx0, bx1);
                }
            }
        }
    }

    if (lane == 0) {
        out[warp] = kq0 + kq1 + kq2 + kq3 + pv0 + pv1 + pv2 + pv3 +
                    row_m0 + row_m1 + row_l0 + row_l1 +
                    out0 + out1 + out2 + out3 + out4 + out5 + out6 + out7;
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = 0.0f;
    }
    (void) q;
    (void) k;
    (void) block_mask;
    (void) out;
    (void) iters;
#endif
}

template <int nfrags, int k_reuse, int staged_groups, bool attentionish, bool consume_remaining_inline = false>
__global__ void combined_kq_mixedpv_smem_compact_nvfp4_kernel(
        const block_nvfp4 * q,
        const block_nvfp4 * k,
        size_t              block_mask,
        float *             out,
        uint64_t            iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    static_assert(nfrags == 8, "mixed-PV compact-stage probe currently models only D=512");
    static_assert(k_reuse == 8, "mixed-PV compact-stage probe currently models Gemma-style reuse 8");
    static_assert(staged_groups == 1 || staged_groups == 2 || staged_groups == 4 || staged_groups == 8, "unsupported staged group count");
    extern __shared__ int smem_i[];

    constexpr int stage_word_ints = KQ_MMA_B_ROWS * k_reuse * staged_groups;
    constexpr int stage_scale_ints = KQ_MMA_B_ROWS * k_reuse;
    constexpr int stage_ints = stage_word_ints + stage_scale_ints;

    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warp = ((int) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const unsigned mask = 0xffffffffu;

    int * warp_s = smem_i + warp_in_block * (KQ_MMA_Q_SHARED_INTS + stage_ints);
    int * q_s = warp_s;
    uint32_t * q_sc = reinterpret_cast<uint32_t *>(q_s + KQ_MMA_A_ROWS * KQ_MMA_A_STRIDE);
    int * k_stage = warp_s + KQ_MMA_Q_SHARED_INTS;
    uint32_t * k_stage_sc = reinterpret_cast<uint32_t *>(k_stage + stage_word_ints);

    float kq0 = 0.0f;
    float kq1 = 0.0f;
    float kq2 = 0.0f;
    float kq3 = 0.0f;
    float pv0 = 0.0f;
    float pv1 = 0.0f;
    float pv2 = 0.0f;
    float pv3 = 0.0f;
    float row_m0 = -64.0f;
    float row_m1 = -64.0f;
    float row_l0 = 0.0f;
    float row_l1 = 0.0f;
    float out0 = 0.0f;
    float out1 = 0.0f;
    float out2 = 0.0f;
    float out3 = 0.0f;
    float out4 = 0.0f;
    float out5 = 0.0f;
    float out6 = 0.0f;
    float out7 = 0.0f;

    for (uint64_t i = 0; i < iters; ++i) {
        const float p_base = attentionish ? 1.0f / (row_l0 + row_l1 + 1.0f) : 0.015625f;
        const uint32_t pax0 = spark_half2_bits(__floats2half2_rn(p_base, p_base * 0.9375f));
        const uint32_t pax1 = spark_half2_bits(__floats2half2_rn(p_base * 0.875f, p_base * 0.8125f));
        const uint32_t pax2 = spark_half2_bits(__floats2half2_rn(p_base * 0.75f, p_base * 0.6875f));
        const uint32_t pax3 = spark_half2_bits(__floats2half2_rn(p_base * 0.625f, p_base * 0.5625f));

        const auto accumulate_mixed_pv = [&](int out_frag, int word0, int word1, uint32_t scale0, uint32_t scale1) {
            const float d0 = spark_ue4m3_to_fp32((uint8_t) (scale0 >> (8 * ((out_frag & 7) / 2))));
            const float d1 = spark_ue4m3_to_fp32((uint8_t) (scale1 >> (8 * ((out_frag & 7) / 2))));
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                const uint8_t q0 = (uint8_t) ((uint32_t) word0 >> (8 * kk));
                const uint8_t q1 = (uint8_t) ((uint32_t) word1 >> (8 * kk));
                const uint32_t bx0 = spark_half2_bits(spark_dequant_nvfp4_byte_to_half2(q0, d0));
                const uint32_t bx1 = spark_half2_bits(spark_dequant_nvfp4_byte_to_half2(q1, d1));
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                    : "+f"(pv0), "+f"(pv1), "+f"(pv2), "+f"(pv3)
                    : "r"(pax0), "r"(pax1), "r"(pax2), "r"(pax3), "r"(bx0), "r"(bx1));
            }

            float mix = (float) (out_frag + 1) * 0.000001f;
            if constexpr (attentionish) {
                const float norm = 1.0f / (row_l0 + row_l1 + 0.000001f);
                mix *= norm;
            }
            out0 += pv0 * mix;
            out1 += pv1 * mix;
            out2 += pv2 * mix;
            out3 += pv3 * mix;
            out4 += (pv0 + row_m0) * mix;
            out5 += (pv1 + row_l0) * mix;
            out6 += (pv2 + row_m1) * mix;
            out7 += (pv3 + row_l1) * mix;
        };

#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const size_t q_base = ((size_t) warp * (size_t) nfrags * (size_t) KQ_MMA_A_ROWS) +
                                  ((size_t) frag * (size_t) KQ_MMA_A_ROWS) +
                                  ((size_t) i * (size_t) gridDim.x * (size_t) nfrags * (size_t) KQ_MMA_A_ROWS);

            if (lane < KQ_MMA_A_ROWS) {
                const block_nvfp4 q_blk = q[(q_base + (size_t) lane) & block_mask];
                const uint32_t * q_qs = reinterpret_cast<const uint32_t *>(q_blk.qs);
#pragma unroll
                for (int c = 0; c < KQ_MMA_WORDS; ++c) {
                    q_s[lane * KQ_MMA_A_STRIDE + c] = (int) q_qs[c];
                }
                q_sc[lane] = ((uint32_t) q_blk.d[0]) | ((uint32_t) q_blk.d[1] << 8) | ((uint32_t) q_blk.d[2] << 16) | ((uint32_t) q_blk.d[3] << 24);
            }

            __syncwarp(mask);

            int ax0;
            int ax1;
            int ax2;
            int ax3;
            const int * a_ptr = q_s + (lane % KQ_MMA_A_ROWS) * KQ_MMA_A_STRIDE + (lane / KQ_MMA_A_ROWS) * 4;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                : "=r"(ax0), "=r"(ax1), "=r"(ax2), "=r"(ax3)
                : "l"(a_ptr));

            const int tidx_a = lane / 4 + (lane % 2) * 8;
            const uint32_t q_scale = q_sc[tidx_a];

#pragma unroll
            for (int kr = 0; kr < k_reuse; ++kr) {
                const size_t k_base = ((size_t) warp * (size_t) nfrags * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) frag * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) kr * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) i * (size_t) gridDim.x * (size_t) nfrags * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS);

                uint32_t kv0 = 0;
                uint32_t kv1 = 0;
                uint32_t kv2 = 0;
                uint32_t kv3 = 0;
                uint32_t kv4 = 0;
                uint32_t kv5 = 0;
                uint32_t kv6 = 0;
                uint32_t kv7 = 0;
                uint32_t k_scale_row = 0;

                if (lane < KQ_MMA_B_ROWS) {
                    const block_nvfp4 k_blk = k[(k_base + (size_t) lane) & block_mask];
                    const uint32_t * k_qs = reinterpret_cast<const uint32_t *>(k_blk.qs);
                    kv0 = k_qs[0];
                    kv1 = k_qs[1];
                    kv2 = k_qs[2];
                    kv3 = k_qs[3];
                    kv4 = k_qs[4];
                    kv5 = k_qs[5];
                    kv6 = k_qs[6];
                    kv7 = k_qs[7];
                    k_scale_row = ((uint32_t) k_blk.d[0]) | ((uint32_t) k_blk.d[1] << 8) | ((uint32_t) k_blk.d[2] << 16) | ((uint32_t) k_blk.d[3] << 24);

                    const int stage_row = kr * KQ_MMA_B_ROWS + lane;
#pragma unroll
                    for (int group = 0; group < staged_groups; ++group) {
                        const int word =
                            group == 0 ? (int) kv0 :
                            group == 1 ? (int) kv1 :
                            group == 2 ? (int) kv2 :
                            group == 3 ? (int) kv3 :
                            group == 4 ? (int) kv4 :
                            group == 5 ? (int) kv5 :
                            group == 6 ? (int) kv6 : (int) kv7;
                        k_stage[stage_row * staged_groups + group] = word;
                    }
                    k_stage_sc[stage_row] = k_scale_row;
                }

                const int b_i = lane / 4;
                const int b_j0 = lane % 4;
                const int row_v0 = (int) __shfl_sync(mask, kv0, b_i);
                const int row_v1 = (int) __shfl_sync(mask, kv1, b_i);
                const int row_v2 = (int) __shfl_sync(mask, kv2, b_i);
                const int row_v3 = (int) __shfl_sync(mask, kv3, b_i);
                const int row_v4 = (int) __shfl_sync(mask, kv4, b_i);
                const int row_v5 = (int) __shfl_sync(mask, kv5, b_i);
                const int row_v6 = (int) __shfl_sync(mask, kv6, b_i);
                const int row_v7 = (int) __shfl_sync(mask, kv7, b_i);

                const int bx0 = b_j0 == 0 ? row_v0 : (b_j0 == 1 ? row_v1 : (b_j0 == 2 ? row_v2 : row_v3));
                const int bx1 = b_j0 == 0 ? row_v4 : (b_j0 == 1 ? row_v5 : (b_j0 == 2 ? row_v6 : row_v7));
                const uint32_t k_scale = __shfl_sync(mask, k_scale_row, b_i);

                asm volatile(
                    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                    "%10, {0, 0}, %11, {0, 0};"
                    : "+f"(kq0), "+f"(kq1), "+f"(kq2), "+f"(kq3)
                    : "r"(ax0), "r"(ax1), "r"(ax2), "r"(ax3), "r"(bx0), "r"(bx1), "r"(q_scale), "r"(k_scale));

                if constexpr (attentionish) {
                    const int q_row = lane & (KQ_MMA_A_ROWS - 1);
                    const int key_pos = frag * k_reuse + kr;
                    const int base_pos = (int) (i & 1023u);
                    const bool causal = key_pos <= base_pos + q_row;
                    const bool swa = key_pos + 1024 >= base_pos + q_row;

                    const float score0 = (causal && swa) ? kq0 * 0.000244140625f : -64.0f;
                    const float score1 = (causal && swa) ? kq2 * 0.000244140625f : -64.0f;
                    const float next_m0 = fmaxf(row_m0, score0);
                    const float next_m1 = fmaxf(row_m1, score1);
                    const float alpha0 = row_l0 == 0.0f ? 0.0f : exp2f(row_m0 - next_m0);
                    const float alpha1 = row_l1 == 0.0f ? 0.0f : exp2f(row_m1 - next_m1);
                    const float beta0 = exp2f(score0 - next_m0);
                    const float beta1 = exp2f(score1 - next_m1);
                    row_l0 = row_l0 * alpha0 + beta0;
                    row_l1 = row_l1 * alpha1 + beta1;
                    row_m0 = next_m0;
                    row_m1 = next_m1;
                } else {
                    row_m0 = fmaxf(row_m0, kq0);
                    row_m1 = fmaxf(row_m1, kq2);
                    row_l0 += fabsf(kq1) * 0.000001f + 1.0f;
                    row_l1 += fabsf(kq3) * 0.000001f + 1.0f;
                }
            }

            __syncwarp(mask);

#pragma unroll
            for (int group = 0; group < staged_groups; ++group) {
                const int row0 = 2 * lane + 0;
                const int row1 = 2 * lane + 1;
                const int word0 = k_stage[row0 * staged_groups + group];
                const int word1 = k_stage[row1 * staged_groups + group];
                const uint32_t scale0 = k_stage_sc[row0];
                const uint32_t scale1 = k_stage_sc[row1];
                accumulate_mixed_pv(frag * (QK_NVFP4 / 8) + group, word0, word1, scale0, scale1);
            }

            if constexpr (consume_remaining_inline) {
                constexpr int groups_per_frag = QK_NVFP4 / 8;
                const int row0 = 2 * lane + 0;
                const int row1 = 2 * lane + 1;
                const size_t pv_base = ((size_t) warp * (size_t) nfrags * 64u) +
                                       ((size_t) i * (size_t) gridDim.x * (size_t) nfrags * 64u);
                const block_nvfp4 blk0 = k[(pv_base + (size_t) row0 * (size_t) nfrags + (size_t) frag) & block_mask];
                const block_nvfp4 blk1 = k[(pv_base + (size_t) row1 * (size_t) nfrags + (size_t) frag) & block_mask];
#pragma unroll
                for (int group = staged_groups; group < groups_per_frag; ++group) {
                    const int out_frag = frag * groups_per_frag + group;
                    const int word0 = spark_nvfp4_dim_group_word(blk0, group);
                    const int word1 = spark_nvfp4_dim_group_word(blk1, group);
                    const uint32_t scale0 = ((uint32_t) blk0.d[0]) | ((uint32_t) blk0.d[1] << 8) | ((uint32_t) blk0.d[2] << 16) | ((uint32_t) blk0.d[3] << 24);
                    const uint32_t scale1 = ((uint32_t) blk1.d[0]) | ((uint32_t) blk1.d[1] << 8) | ((uint32_t) blk1.d[2] << 16) | ((uint32_t) blk1.d[3] << 24);
                    accumulate_mixed_pv(out_frag, word0, word1, scale0, scale1);
                }
            }
        }
    }

    if (lane == 0) {
        out[warp] = kq0 + kq1 + kq2 + kq3 + pv0 + pv1 + pv2 + pv3 +
                    row_m0 + row_m1 + row_l0 + row_l1 +
                    out0 + out1 + out2 + out3 + out4 + out5 + out6 + out7;
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = 0.0f;
    }
    (void) q;
    (void) k;
    (void) block_mask;
    (void) out;
    (void) iters;
#endif
}

// Experimental only: this register-carry K-reuse PV shape compiles, but the low-iteration
// smoke is pathological on the local sm_120 proxy, so it is not wired into main().
template <int nfrags, int k_reuse, bool attentionish>
__global__ void combined_kq_kreusepv_budget_nvfp4_kernel(
        const block_nvfp4 * q,
        const block_nvfp4 * k,
        const uint32_t *    pv_scales,
        size_t              block_mask,
        float *             out,
        uint64_t            iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    static_assert(nfrags == 8, "K-reuse PV diagnostic currently models only D=512");
    static_assert(k_reuse == 8, "K-reuse PV diagnostic currently models reuse 8");
    extern __shared__ int smem_i[];

    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warp = ((int) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const unsigned mask = 0xffffffffu;

    int * warp_s = smem_i + warp_in_block * KQ_MMA_Q_SHARED_INTS;
    int * q_s = warp_s;
    uint32_t * q_sc = reinterpret_cast<uint32_t *>(q_s + KQ_MMA_A_ROWS * KQ_MMA_A_STRIDE);

    float kq0 = 0.0f;
    float kq1 = 0.0f;
    float kq2 = 0.0f;
    float kq3 = 0.0f;
    float pv0 = 0.0f;
    float pv1 = 0.0f;
    float pv2 = 0.0f;
    float pv3 = 0.0f;
    float row_m0 = -64.0f;
    float row_m1 = -64.0f;
    float row_l0 = 0.0f;
    float row_l1 = 0.0f;
    float out0 = 0.0f;
    float out1 = 0.0f;
    float out2 = 0.0f;
    float out3 = 0.0f;
    float out4 = 0.0f;
    float out5 = 0.0f;
    float out6 = 0.0f;
    float out7 = 0.0f;

    for (uint64_t i = 0; i < iters; ++i) {
        const int pax0 = (int) (0x11111111u + (uint32_t) lane);
        const int pax1 = (int) (0x22222222u + (uint32_t) lane);
        const int pax2 = (int) (0x33333333u + (uint32_t) lane);
        const int pax3 = (int) (0x44444444u + (uint32_t) lane);
        const uint32_t p_scale = pv_scales[lane & 255];

        const auto accumulate_pv = [&](int out_frag, int bx0, int bx1, uint32_t b_scale) {
            asm volatile(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                "%10, {0, 0}, %11, {0, 0};"
                : "+f"(pv0), "+f"(pv1), "+f"(pv2), "+f"(pv3)
                : "r"(pax0), "r"(pax1), "r"(pax2), "r"(pax3), "r"(bx0), "r"(bx1), "r"(p_scale), "r"(b_scale));

            float mix = (float) (out_frag + 1) * 0.000001f;
            if constexpr (attentionish) {
                const float norm = 1.0f / (row_l0 + row_l1 + 0.000001f);
                mix *= norm;
            }
            out0 += pv0 * mix;
            out1 += pv1 * mix;
            out2 += pv2 * mix;
            out3 += pv3 * mix;
            out4 += (pv0 + row_m0) * mix;
            out5 += (pv1 + row_l0) * mix;
            out6 += (pv2 + row_m1) * mix;
            out7 += (pv3 + row_l1) * mix;
        };

#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const size_t q_base = ((size_t) warp * (size_t) nfrags * (size_t) KQ_MMA_A_ROWS) +
                                  ((size_t) frag * (size_t) KQ_MMA_A_ROWS) +
                                  ((size_t) i * (size_t) gridDim.x * (size_t) nfrags * (size_t) KQ_MMA_A_ROWS);

            if (lane < KQ_MMA_A_ROWS) {
                const block_nvfp4 q_blk = q[(q_base + (size_t) lane) & block_mask];
                const uint32_t * q_qs = reinterpret_cast<const uint32_t *>(q_blk.qs);
#pragma unroll
                for (int c = 0; c < KQ_MMA_WORDS; ++c) {
                    q_s[lane * KQ_MMA_A_STRIDE + c] = (int) q_qs[c];
                }
                q_sc[lane] = ((uint32_t) q_blk.d[0]) | ((uint32_t) q_blk.d[1] << 8) | ((uint32_t) q_blk.d[2] << 16) | ((uint32_t) q_blk.d[3] << 24);
            }

            __syncwarp(mask);

            int ax0;
            int ax1;
            int ax2;
            int ax3;
            const int * a_ptr = q_s + (lane % KQ_MMA_A_ROWS) * KQ_MMA_A_STRIDE + (lane / KQ_MMA_A_ROWS) * 4;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                : "=r"(ax0), "=r"(ax1), "=r"(ax2), "=r"(ax3)
                : "l"(a_ptr));

            const int tidx_a = lane / 4 + (lane % 2) * 8;
            const uint32_t q_scale = q_sc[tidx_a];

            uint32_t pv00 = 0;
            uint32_t pv01 = 0;
            uint32_t pv02 = 0;
            uint32_t pv03 = 0;
            uint32_t pv04 = 0;
            uint32_t pv05 = 0;
            uint32_t pv06 = 0;
            uint32_t pv07 = 0;
            uint32_t pv10 = 0;
            uint32_t pv11 = 0;
            uint32_t pv12 = 0;
            uint32_t pv13 = 0;
            uint32_t pv14 = 0;
            uint32_t pv15 = 0;
            uint32_t pv16 = 0;
            uint32_t pv17 = 0;
            uint32_t pv_scale = 0;

#pragma unroll
            for (int kr = 0; kr < k_reuse; ++kr) {
                const size_t k_base = ((size_t) warp * (size_t) nfrags * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) frag * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) kr * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) i * (size_t) gridDim.x * (size_t) nfrags * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS);

                uint32_t kv0 = 0;
                uint32_t kv1 = 0;
                uint32_t kv2 = 0;
                uint32_t kv3 = 0;
                uint32_t kv4 = 0;
                uint32_t kv5 = 0;
                uint32_t kv6 = 0;
                uint32_t kv7 = 0;
                uint32_t k_scale_row = 0;

                if (lane < KQ_MMA_B_ROWS) {
                    const block_nvfp4 k_blk = k[(k_base + (size_t) lane) & block_mask];
                    const uint32_t * k_qs = reinterpret_cast<const uint32_t *>(k_blk.qs);
                    kv0 = k_qs[0];
                    kv1 = k_qs[1];
                    kv2 = k_qs[2];
                    kv3 = k_qs[3];
                    kv4 = k_qs[4];
                    kv5 = k_qs[5];
                    kv6 = k_qs[6];
                    kv7 = k_qs[7];
                    k_scale_row = ((uint32_t) k_blk.d[0]) | ((uint32_t) k_blk.d[1] << 8) | ((uint32_t) k_blk.d[2] << 16) | ((uint32_t) k_blk.d[3] << 24);
                }

                const int pv_row0 = 2 * lane + 0;
                const int pv_row1 = 2 * lane + 1;
                const int pv_kr0 = pv_row0 >> 3;
                const int pv_kr1 = pv_row1 >> 3;
                const int pv_src0 = pv_row0 & 7;
                const int pv_src1 = pv_row1 & 7;

                if (kr == pv_kr0) {
                    pv00 = __shfl_sync(mask, kv0, pv_src0);
                    pv01 = __shfl_sync(mask, kv1, pv_src0);
                    pv02 = __shfl_sync(mask, kv2, pv_src0);
                    pv03 = __shfl_sync(mask, kv3, pv_src0);
                    pv04 = __shfl_sync(mask, kv4, pv_src0);
                    pv05 = __shfl_sync(mask, kv5, pv_src0);
                    pv06 = __shfl_sync(mask, kv6, pv_src0);
                    pv07 = __shfl_sync(mask, kv7, pv_src0);
                    pv_scale = __shfl_sync(mask, k_scale_row, pv_src0);
                }

                if (kr == pv_kr1) {
                    pv10 = __shfl_sync(mask, kv0, pv_src1);
                    pv11 = __shfl_sync(mask, kv1, pv_src1);
                    pv12 = __shfl_sync(mask, kv2, pv_src1);
                    pv13 = __shfl_sync(mask, kv3, pv_src1);
                    pv14 = __shfl_sync(mask, kv4, pv_src1);
                    pv15 = __shfl_sync(mask, kv5, pv_src1);
                    pv16 = __shfl_sync(mask, kv6, pv_src1);
                    pv17 = __shfl_sync(mask, kv7, pv_src1);
                }

                const int b_i = lane / 4;
                const int b_j0 = lane % 4;
                const int row_v0 = (int) __shfl_sync(mask, kv0, b_i);
                const int row_v1 = (int) __shfl_sync(mask, kv1, b_i);
                const int row_v2 = (int) __shfl_sync(mask, kv2, b_i);
                const int row_v3 = (int) __shfl_sync(mask, kv3, b_i);
                const int row_v4 = (int) __shfl_sync(mask, kv4, b_i);
                const int row_v5 = (int) __shfl_sync(mask, kv5, b_i);
                const int row_v6 = (int) __shfl_sync(mask, kv6, b_i);
                const int row_v7 = (int) __shfl_sync(mask, kv7, b_i);

                const int bx0 = b_j0 == 0 ? row_v0 : (b_j0 == 1 ? row_v1 : (b_j0 == 2 ? row_v2 : row_v3));
                const int bx1 = b_j0 == 0 ? row_v4 : (b_j0 == 1 ? row_v5 : (b_j0 == 2 ? row_v6 : row_v7));
                const uint32_t k_scale = __shfl_sync(mask, k_scale_row, b_i);

                asm volatile(
                    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                    "%10, {0, 0}, %11, {0, 0};"
                    : "+f"(kq0), "+f"(kq1), "+f"(kq2), "+f"(kq3)
                    : "r"(ax0), "r"(ax1), "r"(ax2), "r"(ax3), "r"(bx0), "r"(bx1), "r"(q_scale), "r"(k_scale));

                if constexpr (attentionish) {
                    const int q_row = lane & (KQ_MMA_A_ROWS - 1);
                    const int key_pos = frag * k_reuse + kr;
                    const int base_pos = (int) (i & 1023u);
                    const bool causal = key_pos <= base_pos + q_row;
                    const bool swa = key_pos + 1024 >= base_pos + q_row;

                    const float score0 = (causal && swa) ? kq0 * 0.000244140625f : -64.0f;
                    const float score1 = (causal && swa) ? kq2 * 0.000244140625f : -64.0f;
                    const float next_m0 = fmaxf(row_m0, score0);
                    const float next_m1 = fmaxf(row_m1, score1);
                    const float alpha0 = row_l0 == 0.0f ? 0.0f : exp2f(row_m0 - next_m0);
                    const float alpha1 = row_l1 == 0.0f ? 0.0f : exp2f(row_m1 - next_m1);
                    const float beta0 = exp2f(score0 - next_m0);
                    const float beta1 = exp2f(score1 - next_m1);
                    row_l0 = row_l0 * alpha0 + beta0;
                    row_l1 = row_l1 * alpha1 + beta1;
                    row_m0 = next_m0;
                    row_m1 = next_m1;
                } else {
                    row_m0 = fmaxf(row_m0, kq0);
                    row_m1 = fmaxf(row_m1, kq2);
                    row_l0 += fabsf(kq1) * 0.000001f + 1.0f;
                    row_l1 += fabsf(kq3) * 0.000001f + 1.0f;
                }
            }

            accumulate_pv(frag * 8 + 0, (int) pv00, (int) pv10, pv_scale);
            accumulate_pv(frag * 8 + 1, (int) pv01, (int) pv11, pv_scale);
            accumulate_pv(frag * 8 + 2, (int) pv02, (int) pv12, pv_scale);
            accumulate_pv(frag * 8 + 3, (int) pv03, (int) pv13, pv_scale);
            accumulate_pv(frag * 8 + 4, (int) pv04, (int) pv14, pv_scale);
            accumulate_pv(frag * 8 + 5, (int) pv05, (int) pv15, pv_scale);
            accumulate_pv(frag * 8 + 6, (int) pv06, (int) pv16, pv_scale);
            accumulate_pv(frag * 8 + 7, (int) pv07, (int) pv17, pv_scale);
        }
    }

    if (lane == 0) {
        out[warp] = kq0 + kq1 + kq2 + kq3 + pv0 + pv1 + pv2 + pv3 +
                    row_m0 + row_m1 + row_l0 + row_l1 +
                    out0 + out1 + out2 + out3 + out4 + out5 + out6 + out7;
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = 0.0f;
    }
    (void) q;
    (void) k;
    (void) pv_scales;
    (void) block_mask;
    (void) out;
    (void) iters;
#endif
}

template <int nfrags, int k_reuse>
__global__ void fused_kq_pv_v_eq_k_qreg_warpk_nvfp4_kernel(
        const block_nvfp4 * q,
        const block_nvfp4 * k,
        size_t              block_mask,
        float *             out,
        uint64_t            iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    extern __shared__ int smem_i[];

    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warp = ((int) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const unsigned mask = 0xffffffffu;

    int * warp_s = smem_i + warp_in_block * KQ_MMA_Q_SHARED_INTS;
    int * q_s = warp_s;
    uint32_t * q_sc = reinterpret_cast<uint32_t *>(q_s + KQ_MMA_A_ROWS * KQ_MMA_A_STRIDE);

    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
    float pv_acc = 0.0f;

    for (uint64_t i = 0; i < iters; ++i) {
#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const size_t q_base = ((size_t) warp * (size_t) nfrags * (size_t) KQ_MMA_A_ROWS) +
                                  ((size_t) frag * (size_t) KQ_MMA_A_ROWS) +
                                  ((size_t) i * (size_t) gridDim.x * (size_t) nfrags * (size_t) KQ_MMA_A_ROWS);

            if (lane < KQ_MMA_A_ROWS) {
                const block_nvfp4 q_blk = q[(q_base + (size_t) lane) & block_mask];
                const uint32_t * q_qs = reinterpret_cast<const uint32_t *>(q_blk.qs);
#pragma unroll
                for (int c = 0; c < KQ_MMA_WORDS; ++c) {
                    q_s[lane * KQ_MMA_A_STRIDE + c] = (int) q_qs[c];
                }
                q_sc[lane] = ((uint32_t) q_blk.d[0]) | ((uint32_t) q_blk.d[1] << 8) | ((uint32_t) q_blk.d[2] << 16) | ((uint32_t) q_blk.d[3] << 24);
            }

            __syncwarp(mask);

            int ax0;
            int ax1;
            int ax2;
            int ax3;
            const int * a_ptr = q_s + (lane % KQ_MMA_A_ROWS) * KQ_MMA_A_STRIDE + (lane / KQ_MMA_A_ROWS) * 4;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                : "=r"(ax0), "=r"(ax1), "=r"(ax2), "=r"(ax3)
                : "l"(a_ptr));

            const int tidx_a = lane / 4 + (lane % 2) * 8;
            const uint32_t q_scale = q_sc[tidx_a];

#pragma unroll
            for (int kr = 0; kr < k_reuse; ++kr) {
                const size_t k_base = ((size_t) warp * (size_t) nfrags * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) frag * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) kr * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) i * (size_t) gridDim.x * (size_t) nfrags * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS);

                uint32_t kv0 = 0;
                uint32_t kv1 = 0;
                uint32_t kv2 = 0;
                uint32_t kv3 = 0;
                uint32_t kv4 = 0;
                uint32_t kv5 = 0;
                uint32_t kv6 = 0;
                uint32_t kv7 = 0;
                uint32_t k_scale_row = 0;

                if (lane < KQ_MMA_B_ROWS) {
                    const block_nvfp4 k_blk = k[(k_base + (size_t) lane) & block_mask];
                    const uint32_t * k_qs = reinterpret_cast<const uint32_t *>(k_blk.qs);
                    kv0 = k_qs[0];
                    kv1 = k_qs[1];
                    kv2 = k_qs[2];
                    kv3 = k_qs[3];
                    kv4 = k_qs[4];
                    kv5 = k_qs[5];
                    kv6 = k_qs[6];
                    kv7 = k_qs[7];
                    k_scale_row = ((uint32_t) k_blk.d[0]) | ((uint32_t) k_blk.d[1] << 8) | ((uint32_t) k_blk.d[2] << 16) | ((uint32_t) k_blk.d[3] << 24);
                    pv_acc += spark_dequant_sum_nvfp4_block(k_blk);
                }

                const int b_i = lane / 4;
                const int b_j0 = lane % 4;
                const int row_v0 = (int) __shfl_sync(mask, kv0, b_i);
                const int row_v1 = (int) __shfl_sync(mask, kv1, b_i);
                const int row_v2 = (int) __shfl_sync(mask, kv2, b_i);
                const int row_v3 = (int) __shfl_sync(mask, kv3, b_i);
                const int row_v4 = (int) __shfl_sync(mask, kv4, b_i);
                const int row_v5 = (int) __shfl_sync(mask, kv5, b_i);
                const int row_v6 = (int) __shfl_sync(mask, kv6, b_i);
                const int row_v7 = (int) __shfl_sync(mask, kv7, b_i);

                const int bx0 = b_j0 == 0 ? row_v0 : (b_j0 == 1 ? row_v1 : (b_j0 == 2 ? row_v2 : row_v3));
                const int bx1 = b_j0 == 0 ? row_v4 : (b_j0 == 1 ? row_v5 : (b_j0 == 2 ? row_v6 : row_v7));
                const uint32_t k_scale = __shfl_sync(mask, k_scale_row, b_i);

                asm volatile(
                    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                    "%10, {0, 0}, %11, {0, 0};"
                    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                    : "r"(ax0), "r"(ax1), "r"(ax2), "r"(ax3), "r"(bx0), "r"(bx1), "r"(q_scale), "r"(k_scale));
            }
        }
    }

    float pv_sum = pv_acc;
    pv_sum += __shfl_down_sync(mask, pv_sum, 4);
    pv_sum += __shfl_down_sync(mask, pv_sum, 2);
    pv_sum += __shfl_down_sync(mask, pv_sum, 1);

    if (lane == 0) {
        out[warp] = d0 + d1 + d2 + d3 + pv_sum;
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = 0.0f;
    }
    (void) q;
    (void) k;
    (void) block_mask;
    (void) iters;
#endif
}

template <int nfrags, int k_reuse>
__global__ void kq_mma_ldmatrix_qreg_kpair_nvfp4_kernel(
        const block_nvfp4 * q,
        const block_nvfp4 * k,
        size_t              block_mask,
        float *             out,
        uint64_t            iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    extern __shared__ int smem_i[];

    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warp = ((int) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const unsigned mask = 0xffffffffu;

    int * warp_s = smem_i + warp_in_block * KQ_MMA_Q_SHARED_INTS;
    int * q_s = warp_s;
    uint32_t * q_sc = reinterpret_cast<uint32_t *>(q_s + KQ_MMA_A_ROWS * KQ_MMA_A_STRIDE);

    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;

    for (uint64_t i = 0; i < iters; ++i) {
#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const size_t q_base = ((size_t) warp * (size_t) nfrags * (size_t) KQ_MMA_A_ROWS) +
                                  ((size_t) frag * (size_t) KQ_MMA_A_ROWS) +
                                  ((size_t) i * (size_t) gridDim.x * (size_t) nfrags * (size_t) KQ_MMA_A_ROWS);

            if (lane < KQ_MMA_A_ROWS) {
                const block_nvfp4 q_blk = q[(q_base + (size_t) lane) & block_mask];
                const uint32_t * q_qs = reinterpret_cast<const uint32_t *>(q_blk.qs);
#pragma unroll
                for (int c = 0; c < KQ_MMA_WORDS; ++c) {
                    q_s[lane * KQ_MMA_A_STRIDE + c] = (int) q_qs[c];
                }
                q_sc[lane] = ((uint32_t) q_blk.d[0]) | ((uint32_t) q_blk.d[1] << 8) | ((uint32_t) q_blk.d[2] << 16) | ((uint32_t) q_blk.d[3] << 24);
            }

            __syncwarp(mask);

            int ax0;
            int ax1;
            int ax2;
            int ax3;
            const int * a_ptr = q_s + (lane % KQ_MMA_A_ROWS) * KQ_MMA_A_STRIDE + (lane / KQ_MMA_A_ROWS) * 4;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                : "=r"(ax0), "=r"(ax1), "=r"(ax2), "=r"(ax3)
                : "l"(a_ptr));

            const int tidx_a = lane / 4 + (lane % 2) * 8;
            const uint32_t q_scale = q_sc[tidx_a];

#pragma unroll
            for (int kr = 0; kr < k_reuse; ++kr) {
                const size_t k_base = ((size_t) warp * (size_t) nfrags * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) frag * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) kr * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) i * (size_t) gridDim.x * (size_t) nfrags * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS);

                const int b_i = lane / 4;
                const int b_j0 = lane % 4;
                const block_nvfp4 k_blk = k[(k_base + (size_t) b_i) & block_mask];
                const uint32_t * k_qs = reinterpret_cast<const uint32_t *>(k_blk.qs);
                const int bx0 = (int) k_qs[b_j0];
                const int bx1 = (int) k_qs[4 + b_j0];
                const uint32_t k_scale = ((uint32_t) k_blk.d[0]) | ((uint32_t) k_blk.d[1] << 8) | ((uint32_t) k_blk.d[2] << 16) | ((uint32_t) k_blk.d[3] << 24);

                asm volatile(
                    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                    "%10, {0, 0}, %11, {0, 0};"
                    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                    : "r"(ax0), "r"(ax1), "r"(ax2), "r"(ax3), "r"(bx0), "r"(bx1), "r"(q_scale), "r"(k_scale));
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
    (void) q;
    (void) k;
    (void) block_mask;
    (void) iters;
#endif
}

static float time_events(cudaEvent_t start, cudaEvent_t stop) {
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

static uint64_t calibrate_iters(float one_iter_ms, int seconds) {
    if (one_iter_ms <= 0.0f) {
        return 1;
    }

    const double target_ms = 1000.0 * (double) seconds;
    double iters = target_ms / (double) one_iter_ms;
    if (iters < 1.0) {
        return 1;
    }
    if (iters > 10000000.0) {
        return 10000000;
    }

    return (uint64_t) iters;
}

static uint64_t adjust_iters(uint64_t iters, float actual_ms, int seconds) {
    const double target_ms = 1000.0 * (double) seconds;
    if (actual_ms <= 0.0f || actual_ms >= 0.8 * target_ms || iters >= 10000000) {
        return iters;
    }

    const double scale = target_ms / (double) actual_ms;
    uint64_t next = (uint64_t) ((double) iters * scale);
    if (next <= iters) {
        next = iters + 1;
    }
    if (next > 10000000) {
        next = 10000000;
    }
    return next;
}

static size_t floor_power_of_two(size_t v) {
    size_t out = 1;
    while ((out << 1) != 0 && (out << 1) <= v) {
        out <<= 1;
    }
    return out;
}

static size_t ceil_power_of_two(size_t v) {
    size_t out = 1;
    while (out < v && (out << 1) != 0) {
        out <<= 1;
    }
    return out;
}

static double useful_mtp_fraction(const bench_config & cfg) {
    return (double) cfg.mtp_rows / (double) KQ_MMA_A_ROWS;
}

template <typename Kernel>
static void print_occupancy_line(const bench_config & cfg, const char * label, Kernel kernel, size_t shared_bytes) {
    int active_blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks_per_sm, kernel, cfg.threads, shared_bytes));

    const int active_warps_per_sm = active_blocks_per_sm * (cfg.threads / 32);
    const int max_warps_per_sm = cfg.max_threads_per_sm > 0 ? cfg.max_threads_per_sm / 32 : 0;
    const double occupancy = max_warps_per_sm > 0 ? 100.0 * (double) active_warps_per_sm / (double) max_warps_per_sm : 0.0;

    std::printf("%s_occupancy: active_blocks_per_sm=%d active_warps_per_sm=%d max_warps_per_sm=%d occupancy=%.1f%% shared=%.3f KiB mtp_rows=%d useful_m16=%.1f%%\n",
                label,
                active_blocks_per_sm,
                active_warps_per_sm,
                max_warps_per_sm,
                occupancy,
                (double) shared_bytes / 1024.0,
                cfg.mtp_rows,
                100.0 * useful_mtp_fraction(cfg));
}

static void print_gemma4_global_memory_model() {
    constexpr int nvfp4_blocks_per_head = GEMMA4_GLOBAL_HEAD_DIM / QK_NVFP4;
    constexpr size_t nvfp4_single_kv_bytes = (size_t) GEMMA4_GLOBAL_KV_HEADS * (size_t) nvfp4_blocks_per_head * sizeof(block_nvfp4);
    constexpr size_t f16_single_kv_bytes = (size_t) GEMMA4_GLOBAL_KV_HEADS * (size_t) GEMMA4_GLOBAL_HEAD_DIM * sizeof(uint16_t);
    constexpr size_t f16_separate_kv_bytes = 2u * f16_single_kv_bytes;

    std::printf("gemma4_global_d512_v_eq_k: nvfp4_single_tensor=%zu B/token/layer f16_single_tensor=%zu B/token/layer f16_separate_kv=%zu B/token/layer nvfp4_vs_f16_single=%.3fx nvfp4_vs_f16_separate=%.3fx\n",
                nvfp4_single_kv_bytes,
                f16_single_kv_bytes,
                f16_separate_kv_bytes,
                (double) f16_single_kv_bytes / (double) nvfp4_single_kv_bytes,
                (double) f16_separate_kv_bytes / (double) nvfp4_single_kv_bytes);
}

static void run_u4_read_bench(const bench_config & cfg) {
    const size_t n_u4 = cfg.bytes / sizeof(uint4);
    const size_t bytes = n_u4 * sizeof(uint4);
    const size_t n_threads = (size_t) cfg.blocks * cfg.threads;

    uint4 * data = nullptr;
    uint64_t * out = nullptr;
    CUDA_CHECK(cudaMalloc(&data, bytes));
    CUDA_CHECK(cudaMalloc(&out, n_threads * sizeof(uint64_t)));

    fill_u32_kernel<<<cfg.blocks, cfg.threads>>>((uint32_t *) data, bytes / sizeof(uint32_t));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    read_u4_kernel<<<cfg.blocks, cfg.threads>>>(data, n_u4, 1, out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    read_u4_kernel<<<cfg.blocks, cfg.threads>>>(data, n_u4, 1, out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float one_iter_ms = time_events(start, stop);
    uint64_t iters = calibrate_iters(one_iter_ms, cfg.seconds);

    float ms = 0.0f;
    for (int attempt = 0; attempt < 3; ++attempt) {
        CUDA_CHECK(cudaEventRecord(start));
        read_u4_kernel<<<cfg.blocks, cfg.threads>>>(data, n_u4, iters, out);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        ms = time_events(start, stop);

        const uint64_t next_iters = adjust_iters(iters, ms, cfg.seconds);
        if (next_iters == iters) {
            break;
        }
        iters = next_iters;
    }

    const double gb = (double) bytes * (double) iters / 1.0e9;
    std::printf("read_u4:              %.3f GB/s  bytes=%zu iters=%" PRIu64 " time=%.3f ms calib=%.3f ms\n",
                gb / (ms / 1000.0), bytes, iters, ms, one_iter_ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(data));
}

static void run_nvfp4_read_bench(const bench_config & cfg) {
    const size_t n_blocks = cfg.bytes / sizeof(block_nvfp4);
    const size_t bytes = n_blocks * sizeof(block_nvfp4);
    const size_t n_threads = (size_t) cfg.blocks * cfg.threads;

    block_nvfp4 * data = nullptr;
    uint64_t * out = nullptr;
    CUDA_CHECK(cudaMalloc(&data, bytes));
    CUDA_CHECK(cudaMalloc(&out, n_threads * sizeof(uint64_t)));

    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(data, n_blocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    read_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(data, n_blocks, 1, out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    read_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(data, n_blocks, 1, out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float one_iter_ms = time_events(start, stop);
    uint64_t iters = calibrate_iters(one_iter_ms, cfg.seconds);

    float ms = 0.0f;
    for (int attempt = 0; attempt < 3; ++attempt) {
        CUDA_CHECK(cudaEventRecord(start));
        read_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(data, n_blocks, iters, out);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        ms = time_events(start, stop);

        const uint64_t next_iters = adjust_iters(iters, ms, cfg.seconds);
        if (next_iters == iters) {
            break;
        }
        iters = next_iters;
    }

    const double gb = (double) bytes * (double) iters / 1.0e9;
    std::printf("read_block_nvfp4:     %.3f GB/s  bytes=%zu blocks=%zu iters=%" PRIu64 " time=%.3f ms calib=%.3f ms\n",
                gb / (ms / 1000.0), bytes, n_blocks, iters, ms, one_iter_ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(data));
}

static void run_fp4_mma_bench(const bench_config & cfg, int cc) {
    if (cc < 1200 || cc >= 1300) {
        std::printf("fp4_mma_mxf4nvf4:    skipped  reason=requires Blackwell sm_120/sm_121 device\n");
        return;
    }

    const size_t n_threads = (size_t) cfg.blocks * cfg.threads;
    const size_t n_warps = n_threads / 32;
    int * a = nullptr;
    int * b = nullptr;
    uint32_t * scales = nullptr;
    float * out = nullptr;
    CUDA_CHECK(cudaMalloc(&a, 1024 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&b, 1024 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&scales, 256 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&out, n_warps * sizeof(float)));

    fill_u32_kernel<<<1, 256>>>((uint32_t *) a, 1024);
    fill_u32_kernel<<<1, 256>>>((uint32_t *) b, 1024);
    fill_u32_kernel<<<1, 256>>>(scales, 256);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    fp4_mma_kernel<<<cfg.blocks, cfg.threads>>>(a, b, scales, out, 10);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    fp4_mma_kernel<<<cfg.blocks, cfg.threads>>>(a, b, scales, out, cfg.mma_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float ms = time_events(start, stop);

    const double dense_ops = (double) n_warps * (double) cfg.mma_iters * 2.0 * 16.0 * 8.0 * 64.0;
    std::printf("fp4_mma_mxf4nvf4:    %.3f TOPS  warps=%zu mma_iters=%" PRIu64 " time=%.3f ms\n",
                dense_ops / (ms / 1000.0) / 1.0e12, n_warps, cfg.mma_iters, ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(scales));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(a));
}

static void run_kq256_mma_bench(const bench_config & cfg, int cc) {
    if (cc < 1200 || cc >= 1300) {
        std::printf("kq256_mma_nvfp4:     skipped  reason=requires Blackwell sm_120/sm_121 device\n");
        return;
    }

    const size_t n_threads = (size_t) cfg.blocks * cfg.threads;
    const size_t n_warps = n_threads / 32;
    size_t n_blocks = floor_power_of_two(cfg.bytes / sizeof(block_nvfp4));
    if (n_blocks < 4096) {
        n_blocks = 4096;
    }

    block_nvfp4 * q = nullptr;
    block_nvfp4 * k = nullptr;
    float * out = nullptr;
    CUDA_CHECK(cudaMalloc(&q, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&k, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&out, n_warps * sizeof(float)));

    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(q, n_blocks);
    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(k, n_blocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    kq256_mma_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(q, k, n_blocks - 1, out, 10);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    kq256_mma_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(q, k, n_blocks - 1, out, cfg.mma_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float ms = time_events(start, stop);

    const double dense_ops = (double) n_warps * (double) cfg.mma_iters * 4.0 * 2.0 * 16.0 * 8.0 * 64.0;
    const double read_gb = (double) n_warps * (double) cfg.mma_iters * 4.0 * 2.0 * (double) sizeof(block_nvfp4) / 1.0e9;
    std::printf("kq256_mma_nvfp4:     %.3f TOPS  %.3f GB/s-read  blocks=%zu warps=%zu mma_iters=%" PRIu64 " time=%.3f ms\n",
                dense_ops / (ms / 1000.0) / 1.0e12, read_gb / (ms / 1000.0), n_blocks, n_warps, cfg.mma_iters, ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(k));
    CUDA_CHECK(cudaFree(q));
}

template <int dkq>
static void run_kq_staged_mma_bench(const bench_config & cfg, int cc, const char * label) {
    static_assert(dkq == 256 || dkq == 512, "unsupported staged KQ width");
    constexpr int nfrags = dkq / QK_NVFP4;

    if (cc < 1200 || cc >= 1300) {
        std::printf("%s: skipped  reason=requires Blackwell sm_120/sm_121 device\n", label);
        return;
    }

    const size_t n_threads = (size_t) cfg.blocks * cfg.threads;
    const size_t n_warps = n_threads / 32;
    const size_t stage_tile_size = (size_t) nfrags * 32u * sizeof(kq_mma_lane_frag);
    size_t n_tiles = floor_power_of_two(cfg.bytes / stage_tile_size);
    if (n_tiles < n_warps) {
        n_tiles = floor_power_of_two(n_warps);
    }
    if (n_tiles < 1) {
        n_tiles = 1;
    }

    const size_t n_blocks = floor_power_of_two(n_tiles * (size_t) nfrags);
    block_nvfp4 * q = nullptr;
    block_nvfp4 * k = nullptr;
    kq_mma_lane_frag * stage = nullptr;
    float * out = nullptr;
    CUDA_CHECK(cudaMalloc(&q, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&k, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&stage, n_tiles * stage_tile_size));
    CUDA_CHECK(cudaMalloc(&out, n_warps * sizeof(float)));

    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(q, n_blocks);
    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(k, n_blocks);
    fill_kq_mma_stage_kernel<nfrags><<<cfg.blocks, cfg.threads>>>(q, k, stage, n_tiles, n_blocks - 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    kq_mma_staged_nvfp4_kernel<nfrags><<<cfg.blocks, cfg.threads>>>(stage, n_tiles - 1, out, 10);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    kq_mma_staged_nvfp4_kernel<nfrags><<<cfg.blocks, cfg.threads>>>(stage, n_tiles - 1, out, cfg.mma_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float ms = time_events(start, stop);

    const double dense_ops = (double) n_warps * (double) cfg.mma_iters * (double) nfrags * 2.0 * 16.0 * 8.0 * 64.0;
    const double read_gb = (double) n_warps * (double) cfg.mma_iters * (double) nfrags * 32.0 * (double) sizeof(kq_mma_lane_frag) / 1.0e9;
    std::printf("%s: %.3f TOPS  %.3f GB/s-staged-read  tiles=%zu stage=%.3f MiB warps=%zu mma_iters=%" PRIu64 " time=%.3f ms\n",
                label,
                dense_ops / (ms / 1000.0) / 1.0e12,
                read_gb / (ms / 1000.0),
                n_tiles,
                (double) (n_tiles * stage_tile_size) / 1048576.0,
                n_warps,
                cfg.mma_iters,
                ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(stage));
    CUDA_CHECK(cudaFree(k));
    CUDA_CHECK(cudaFree(q));
}

template <int dkq>
static void run_kq_compact_mma_bench(const bench_config & cfg, int cc, const char * label) {
    static_assert(dkq == 256 || dkq == 512, "unsupported compact KQ width");
    constexpr int nfrags = dkq / QK_NVFP4;

    if (cc < 1200 || cc >= 1300) {
        std::printf("%s: skipped  reason=requires Blackwell sm_120/sm_121 device\n", label);
        return;
    }

    const size_t n_threads = (size_t) cfg.blocks * cfg.threads;
    const size_t n_warps = n_threads / 32;
    const size_t compact_tile_size = (size_t) nfrags * sizeof(kq_mma_compact_frag);
    size_t n_tiles = floor_power_of_two(cfg.bytes / compact_tile_size);
    if (n_tiles < n_warps) {
        n_tiles = floor_power_of_two(n_warps);
    }
    if (n_tiles < 1) {
        n_tiles = 1;
    }

    const size_t n_blocks = floor_power_of_two(n_tiles * (size_t) nfrags);
    block_nvfp4 * q = nullptr;
    block_nvfp4 * k = nullptr;
    kq_mma_compact_frag * stage = nullptr;
    float * out = nullptr;
    CUDA_CHECK(cudaMalloc(&q, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&k, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&stage, n_tiles * compact_tile_size));
    CUDA_CHECK(cudaMalloc(&out, n_warps * sizeof(float)));

    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(q, n_blocks);
    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(k, n_blocks);
    fill_kq_mma_compact_kernel<nfrags><<<cfg.blocks, cfg.threads>>>(q, k, stage, n_tiles, n_blocks - 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    kq_mma_compact_nvfp4_kernel<nfrags><<<cfg.blocks, cfg.threads>>>(stage, n_tiles - 1, out, 10);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    kq_mma_compact_nvfp4_kernel<nfrags><<<cfg.blocks, cfg.threads>>>(stage, n_tiles - 1, out, cfg.mma_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float ms = time_events(start, stop);

    const double dense_ops = (double) n_warps * (double) cfg.mma_iters * (double) nfrags * 2.0 * 16.0 * 8.0 * 64.0;
    const double compact_read_gb = (double) n_warps * (double) cfg.mma_iters * (double) nfrags * (double) sizeof(kq_mma_compact_frag) / 1.0e9;
    std::printf("%s: %.3f TOPS  %.3f GB/s-compact-stage-read  tiles=%zu stage=%.3f MiB warps=%zu mma_iters=%" PRIu64 " time=%.3f ms\n",
                label,
                dense_ops / (ms / 1000.0) / 1.0e12,
                compact_read_gb / (ms / 1000.0),
                n_tiles,
                (double) (n_tiles * compact_tile_size) / 1048576.0,
                n_warps,
                cfg.mma_iters,
                ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(stage));
    CUDA_CHECK(cudaFree(k));
    CUDA_CHECK(cudaFree(q));
}

template <int dkq>
static void run_kq_shared_mma_bench(const bench_config & cfg, int cc, const char * label) {
    static_assert(dkq == 256 || dkq == 512, "unsupported shared KQ width");
    constexpr int nfrags = dkq / QK_NVFP4;

    if (cc < 1200 || cc >= 1300) {
        std::printf("%s: skipped  reason=requires Blackwell sm_120/sm_121 device\n", label);
        return;
    }

    if (cfg.threads % 32 != 0) {
        std::printf("%s: skipped  reason=threads must be a multiple of 32\n", label);
        return;
    }

    const size_t n_threads = (size_t) cfg.blocks * cfg.threads;
    const size_t n_warps = n_threads / 32;
    size_t n_blocks = floor_power_of_two(cfg.bytes / sizeof(block_nvfp4));
    if (n_blocks < n_warps * (size_t) nfrags) {
        n_blocks = floor_power_of_two(n_warps * (size_t) nfrags);
    }
    if (n_blocks < 4096) {
        n_blocks = 4096;
    }

    const size_t shared_bytes = 2u * (size_t) (cfg.threads / 32) * (size_t) nfrags * sizeof(block_nvfp4);
    block_nvfp4 * q = nullptr;
    block_nvfp4 * k = nullptr;
    float * out = nullptr;
    CUDA_CHECK(cudaMalloc(&q, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&k, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&out, n_warps * sizeof(float)));

    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(q, n_blocks);
    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(k, n_blocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    kq_mma_shared_nvfp4_kernel<nfrags><<<cfg.blocks, cfg.threads, shared_bytes>>>(q, k, n_blocks - 1, out, 10);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    kq_mma_shared_nvfp4_kernel<nfrags><<<cfg.blocks, cfg.threads, shared_bytes>>>(q, k, n_blocks - 1, out, cfg.mma_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float ms = time_events(start, stop);

    const double dense_ops = (double) n_warps * (double) cfg.mma_iters * (double) nfrags * 2.0 * 16.0 * 8.0 * 64.0;
    const double compact_read_gb = (double) n_warps * (double) cfg.mma_iters * (double) nfrags * 2.0 * (double) sizeof(block_nvfp4) / 1.0e9;
    std::printf("%s: %.3f TOPS  %.3f GB/s-compact-read  blocks=%zu shared=%.3f KiB warps=%zu mma_iters=%" PRIu64 " time=%.3f ms\n",
                label,
                dense_ops / (ms / 1000.0) / 1.0e12,
                compact_read_gb / (ms / 1000.0),
                n_blocks,
                (double) shared_bytes / 1024.0,
                n_warps,
                cfg.mma_iters,
                ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(k));
    CUDA_CHECK(cudaFree(q));
}

template <int dkq>
static void run_kq_warp_mma_bench(const bench_config & cfg, int cc, const char * label) {
    static_assert(dkq == 256 || dkq == 512, "unsupported warp KQ width");
    constexpr int nfrags = dkq / QK_NVFP4;

    if (cc < 1200 || cc >= 1300) {
        std::printf("%s: skipped  reason=requires Blackwell sm_120/sm_121 device\n", label);
        return;
    }

    const size_t n_threads = (size_t) cfg.blocks * cfg.threads;
    const size_t n_warps = n_threads / 32;
    size_t n_blocks = floor_power_of_two(cfg.bytes / sizeof(block_nvfp4));
    if (n_blocks < n_warps * (size_t) nfrags) {
        n_blocks = floor_power_of_two(n_warps * (size_t) nfrags);
    }
    if (n_blocks < 4096) {
        n_blocks = 4096;
    }

    block_nvfp4 * q = nullptr;
    block_nvfp4 * k = nullptr;
    float * out = nullptr;
    CUDA_CHECK(cudaMalloc(&q, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&k, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&out, n_warps * sizeof(float)));

    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(q, n_blocks);
    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(k, n_blocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    kq_mma_warp_nvfp4_kernel<nfrags><<<cfg.blocks, cfg.threads>>>(q, k, n_blocks - 1, out, 10);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    kq_mma_warp_nvfp4_kernel<nfrags><<<cfg.blocks, cfg.threads>>>(q, k, n_blocks - 1, out, cfg.mma_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float ms = time_events(start, stop);

    const double dense_ops = (double) n_warps * (double) cfg.mma_iters * (double) nfrags * 2.0 * 16.0 * 8.0 * 64.0;
    const double compact_read_gb = (double) n_warps * (double) cfg.mma_iters * (double) nfrags * 2.0 * (double) sizeof(block_nvfp4) / 1.0e9;
    std::printf("%s: %.3f TOPS  %.3f GB/s-compact-read  blocks=%zu warps=%zu mma_iters=%" PRIu64 " time=%.3f ms\n",
                label,
                dense_ops / (ms / 1000.0) / 1.0e12,
                compact_read_gb / (ms / 1000.0),
                n_blocks,
                n_warps,
                cfg.mma_iters,
                ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(k));
    CUDA_CHECK(cudaFree(q));
}

template <int dkq>
static void run_kq_ldmatrix_mma_bench(const bench_config & cfg, int cc, const char * label) {
    static_assert(dkq == 256 || dkq == 512, "unsupported ldmatrix KQ width");
    constexpr int nfrags = dkq / QK_NVFP4;

    if (cc < 1200 || cc >= 1300) {
        std::printf("%s: skipped  reason=requires Blackwell sm_120/sm_121 device\n", label);
        return;
    }

    if (cfg.threads % 32 != 0) {
        std::printf("%s: skipped  reason=threads must be a multiple of 32\n", label);
        return;
    }

    const size_t n_threads = (size_t) cfg.blocks * cfg.threads;
    const size_t n_warps = n_threads / 32;
    size_t n_blocks = floor_power_of_two(cfg.bytes / sizeof(block_nvfp4));
    if (n_blocks < n_warps * (size_t) nfrags * (size_t) (KQ_MMA_A_ROWS + KQ_MMA_B_ROWS)) {
        n_blocks = floor_power_of_two(n_warps * (size_t) nfrags * (size_t) (KQ_MMA_A_ROWS + KQ_MMA_B_ROWS));
    }
    if (n_blocks < 4096) {
        n_blocks = 4096;
    }

    const size_t shared_bytes = (size_t) (cfg.threads / 32) * (size_t) KQ_MMA_SHARED_INTS * sizeof(int);

    block_nvfp4 * q = nullptr;
    block_nvfp4 * k = nullptr;
    float * out = nullptr;
    CUDA_CHECK(cudaMalloc(&q, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&k, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&out, n_warps * sizeof(float)));

    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(q, n_blocks);
    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(k, n_blocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    kq_mma_ldmatrix_nvfp4_kernel<nfrags><<<cfg.blocks, cfg.threads, shared_bytes>>>(q, k, n_blocks - 1, out, 10);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    kq_mma_ldmatrix_nvfp4_kernel<nfrags><<<cfg.blocks, cfg.threads, shared_bytes>>>(q, k, n_blocks - 1, out, cfg.mma_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float ms = time_events(start, stop);

    const double dense_ops = (double) n_warps * (double) cfg.mma_iters * (double) nfrags * 2.0 * 16.0 * 8.0 * 64.0;
    const double compact_read_gb = (double) n_warps * (double) cfg.mma_iters * (double) nfrags *
                                   (double) (KQ_MMA_A_ROWS + KQ_MMA_B_ROWS) * (double) sizeof(block_nvfp4) / 1.0e9;
    std::printf("%s: %.3f TOPS  %.3f GB/s-compact-read  blocks=%zu shared=%.3f KiB warps=%zu mma_iters=%" PRIu64 " time=%.3f ms\n",
                label,
                dense_ops / (ms / 1000.0) / 1.0e12,
                compact_read_gb / (ms / 1000.0),
                n_blocks,
                (double) shared_bytes / 1024.0,
                n_warps,
                cfg.mma_iters,
                ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(k));
    CUDA_CHECK(cudaFree(q));
}

template <int dkq, int k_reuse>
static void run_kq_ldmatrix_reuse_mma_bench(const bench_config & cfg, int cc, const char * label) {
    static_assert(dkq == 256 || dkq == 512, "unsupported ldmatrix KQ width");
    constexpr int nfrags = dkq / QK_NVFP4;

    if (cc < 1200 || cc >= 1300) {
        std::printf("%s: skipped  reason=requires Blackwell sm_120/sm_121 device\n", label);
        return;
    }

    if (cfg.threads % 32 != 0) {
        std::printf("%s: skipped  reason=threads must be a multiple of 32\n", label);
        return;
    }

    const size_t n_threads = (size_t) cfg.blocks * cfg.threads;
    const size_t n_warps = n_threads / 32;
    size_t n_blocks = floor_power_of_two(cfg.bytes / sizeof(block_nvfp4));
    const size_t min_blocks = n_warps * (size_t) nfrags * (size_t) (KQ_MMA_A_ROWS + k_reuse * KQ_MMA_B_ROWS);
    if (n_blocks < min_blocks) {
        n_blocks = floor_power_of_two(min_blocks);
    }
    if (n_blocks < 4096) {
        n_blocks = 4096;
    }

    const size_t shared_bytes = (size_t) (cfg.threads / 32) * (size_t) KQ_MMA_SHARED_INTS * sizeof(int);
    print_occupancy_line(cfg, label, kq_mma_ldmatrix_reuse_nvfp4_kernel<nfrags, k_reuse>, shared_bytes);

    block_nvfp4 * q = nullptr;
    block_nvfp4 * k = nullptr;
    float * out = nullptr;
    CUDA_CHECK(cudaMalloc(&q, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&k, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&out, n_warps * sizeof(float)));

    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(q, n_blocks);
    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(k, n_blocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    kq_mma_ldmatrix_reuse_nvfp4_kernel<nfrags, k_reuse><<<cfg.blocks, cfg.threads, shared_bytes>>>(q, k, n_blocks - 1, out, 10);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    kq_mma_ldmatrix_reuse_nvfp4_kernel<nfrags, k_reuse><<<cfg.blocks, cfg.threads, shared_bytes>>>(q, k, n_blocks - 1, out, cfg.mma_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float ms = time_events(start, stop);

    const double dense_ops = (double) n_warps * (double) cfg.mma_iters * (double) nfrags * (double) k_reuse * 2.0 * 16.0 * 8.0 * 64.0;
    const double dense_tops = dense_ops / (ms / 1000.0) / 1.0e12;
    const double compact_read_gb = (double) n_warps * (double) cfg.mma_iters * (double) nfrags *
                                   (double) (KQ_MMA_A_ROWS + k_reuse * KQ_MMA_B_ROWS) * (double) sizeof(block_nvfp4) / 1.0e9;
    std::printf("%s: %.3f TOPS  %.3f useful-mtp-TOPS  %.3f GB/s-compact-read  k_reuse=%d blocks=%zu shared=%.3f KiB warps=%zu mma_iters=%" PRIu64 " time=%.3f ms\n",
                label,
                dense_tops,
                dense_tops * useful_mtp_fraction(cfg),
                compact_read_gb / (ms / 1000.0),
                k_reuse,
                n_blocks,
                (double) shared_bytes / 1024.0,
                n_warps,
                cfg.mma_iters,
                ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(k));
    CUDA_CHECK(cudaFree(q));
}

template <int dkq, int k_reuse>
static void run_kq_ldmatrix_qreg_mma_bench(const bench_config & cfg, int cc, const char * label) {
    static_assert(dkq == 256 || dkq == 512, "unsupported ldmatrix KQ width");
    constexpr int nfrags = dkq / QK_NVFP4;

    if (cc < 1200 || cc >= 1300) {
        std::printf("%s: skipped  reason=requires Blackwell sm_120/sm_121 device\n", label);
        return;
    }

    if (cfg.threads % 32 != 0) {
        std::printf("%s: skipped  reason=threads must be a multiple of 32\n", label);
        return;
    }

    const size_t n_threads = (size_t) cfg.blocks * cfg.threads;
    const size_t n_warps = n_threads / 32;
    size_t n_blocks = floor_power_of_two(cfg.bytes / sizeof(block_nvfp4));
    const size_t min_blocks = n_warps * (size_t) nfrags * (size_t) (KQ_MMA_A_ROWS + k_reuse * KQ_MMA_B_ROWS);
    if (n_blocks < min_blocks) {
        n_blocks = floor_power_of_two(min_blocks);
    }
    if (n_blocks < 4096) {
        n_blocks = 4096;
    }

    const size_t shared_bytes = (size_t) (cfg.threads / 32) * (size_t) KQ_MMA_SHARED_INTS * sizeof(int);
    print_occupancy_line(cfg, label, kq_mma_ldmatrix_qreg_nvfp4_kernel<nfrags, k_reuse>, shared_bytes);

    block_nvfp4 * q = nullptr;
    block_nvfp4 * k = nullptr;
    float * out = nullptr;
    CUDA_CHECK(cudaMalloc(&q, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&k, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&out, n_warps * sizeof(float)));

    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(q, n_blocks);
    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(k, n_blocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    kq_mma_ldmatrix_qreg_nvfp4_kernel<nfrags, k_reuse><<<cfg.blocks, cfg.threads, shared_bytes>>>(q, k, n_blocks - 1, out, 10);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    kq_mma_ldmatrix_qreg_nvfp4_kernel<nfrags, k_reuse><<<cfg.blocks, cfg.threads, shared_bytes>>>(q, k, n_blocks - 1, out, cfg.mma_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float ms = time_events(start, stop);

    const double dense_ops = (double) n_warps * (double) cfg.mma_iters * (double) nfrags * (double) k_reuse * 2.0 * 16.0 * 8.0 * 64.0;
    const double dense_tops = dense_ops / (ms / 1000.0) / 1.0e12;
    const double compact_read_gb = (double) n_warps * (double) cfg.mma_iters * (double) nfrags *
                                   (double) (KQ_MMA_A_ROWS + k_reuse * KQ_MMA_B_ROWS) * (double) sizeof(block_nvfp4) / 1.0e9;
    std::printf("%s: %.3f TOPS  %.3f useful-mtp-TOPS  %.3f GB/s-compact-read  k_reuse=%d blocks=%zu shared=%.3f KiB warps=%zu mma_iters=%" PRIu64 " time=%.3f ms\n",
                label,
                dense_tops,
                dense_tops * useful_mtp_fraction(cfg),
                compact_read_gb / (ms / 1000.0),
                k_reuse,
                n_blocks,
                (double) shared_bytes / 1024.0,
                n_warps,
                cfg.mma_iters,
                ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(k));
    CUDA_CHECK(cudaFree(q));
}

template <int dkq, int k_reuse>
static void run_kq_ldmatrix_qreg_warpk_mma_bench(const bench_config & cfg, int cc, const char * label) {
    static_assert(dkq == 256 || dkq == 512, "unsupported ldmatrix KQ width");
    constexpr int nfrags = dkq / QK_NVFP4;

    if (cc < 1200 || cc >= 1300) {
        std::printf("%s: skipped  reason=requires Blackwell sm_120/sm_121 device\n", label);
        return;
    }

    if (cfg.threads % 32 != 0) {
        std::printf("%s: skipped  reason=threads must be a multiple of 32\n", label);
        return;
    }

    const size_t n_threads = (size_t) cfg.blocks * cfg.threads;
    const size_t n_warps = n_threads / 32;
    size_t n_blocks = floor_power_of_two(cfg.bytes / sizeof(block_nvfp4));
    const size_t min_blocks = n_warps * (size_t) nfrags * (size_t) (KQ_MMA_A_ROWS + k_reuse * KQ_MMA_B_ROWS);
    if (n_blocks < min_blocks) {
        n_blocks = floor_power_of_two(min_blocks);
    }
    if (n_blocks < 4096) {
        n_blocks = 4096;
    }

    const size_t shared_bytes = (size_t) (cfg.threads / 32) * (size_t) KQ_MMA_Q_SHARED_INTS * sizeof(int);
    print_occupancy_line(cfg, label, kq_mma_ldmatrix_qreg_warpk_nvfp4_kernel<nfrags, k_reuse>, shared_bytes);

    block_nvfp4 * q = nullptr;
    block_nvfp4 * k = nullptr;
    float * out = nullptr;
    CUDA_CHECK(cudaMalloc(&q, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&k, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&out, n_warps * sizeof(float)));

    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(q, n_blocks);
    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(k, n_blocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    kq_mma_ldmatrix_qreg_warpk_nvfp4_kernel<nfrags, k_reuse><<<cfg.blocks, cfg.threads, shared_bytes>>>(q, k, n_blocks - 1, out, 10);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    kq_mma_ldmatrix_qreg_warpk_nvfp4_kernel<nfrags, k_reuse><<<cfg.blocks, cfg.threads, shared_bytes>>>(q, k, n_blocks - 1, out, cfg.mma_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float ms = time_events(start, stop);

    const double dense_ops = (double) n_warps * (double) cfg.mma_iters * (double) nfrags * (double) k_reuse * 2.0 * 16.0 * 8.0 * 64.0;
    const double dense_tops = dense_ops / (ms / 1000.0) / 1.0e12;
    const double compact_read_gb = (double) n_warps * (double) cfg.mma_iters * (double) nfrags *
                                   (double) (KQ_MMA_A_ROWS + k_reuse * KQ_MMA_B_ROWS) * (double) sizeof(block_nvfp4) / 1.0e9;
    std::printf("%s: %.3f TOPS  %.3f useful-mtp-TOPS  %.3f GB/s-compact-read  k_reuse=%d blocks=%zu shared=%.3f KiB warps=%zu mma_iters=%" PRIu64 " time=%.3f ms\n",
                label,
                dense_tops,
                dense_tops * useful_mtp_fraction(cfg),
                compact_read_gb / (ms / 1000.0),
                k_reuse,
                n_blocks,
                (double) shared_bytes / 1024.0,
                n_warps,
                cfg.mma_iters,
                ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(k));
    CUDA_CHECK(cudaFree(q));
}

template <int dkq, int k_reuse, bool attentionish = false, bool inline_pv = false>
static void run_combined_kq_packedpv_budget_bench(const bench_config & cfg, int cc, const char * label) {
    static_assert(dkq == 256 || dkq == 512, "unsupported combined KQ/PV width");
    static_assert(k_reuse == 8, "combined probe currently models reuse 8");
    constexpr int nfrags = dkq / QK_NVFP4;
    constexpr int n_out_frags = dkq / 8;
    constexpr size_t stage_tile_entries = (size_t) n_out_frags * 32u;
    constexpr size_t stage_tile_size = stage_tile_entries * sizeof(pv_mma_b_lane_frag);

    if (cc < 1200 || cc >= 1300) {
        std::printf("%s: skipped  reason=requires Blackwell sm_120/sm_121 device\n", label);
        return;
    }

    if (cfg.threads % 32 != 0) {
        std::printf("%s: skipped  reason=threads must be a multiple of 32\n", label);
        return;
    }

    const size_t n_threads = (size_t) cfg.blocks * cfg.threads;
    const size_t n_warps = n_threads / 32;
    size_t n_tiles = floor_power_of_two(cfg.bytes / stage_tile_size);
    if (n_tiles < n_warps) {
        n_tiles = floor_power_of_two(n_warps);
    }
    if (n_tiles < 1) {
        n_tiles = 1;
    }

    const size_t stage_entries = n_tiles * stage_tile_entries;
    const size_t stage_blocks = n_tiles * (size_t) nfrags * 64u;
    const size_t min_kq_blocks = n_warps * (size_t) nfrags * (size_t) (KQ_MMA_A_ROWS + k_reuse * KQ_MMA_B_ROWS);
    const size_t cfg_blocks = floor_power_of_two(cfg.bytes / sizeof(block_nvfp4));
    size_t n_blocks = 4096;
    while (n_blocks < stage_blocks || n_blocks < min_kq_blocks || n_blocks < cfg_blocks) {
        n_blocks <<= 1;
    }

    const size_t shared_bytes = (size_t) (cfg.threads / 32) * (size_t) KQ_MMA_Q_SHARED_INTS * sizeof(int);
    print_occupancy_line(cfg, label, combined_kq_packedpv_budget_nvfp4_kernel<nfrags, k_reuse, attentionish, inline_pv>, shared_bytes);

    block_nvfp4 * q = nullptr;
    block_nvfp4 * k = nullptr;
    pv_mma_b_lane_frag * pv_stage = nullptr;
    uint32_t * pv_scales = nullptr;
    float * out = nullptr;
    CUDA_CHECK(cudaMalloc(&q, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&k, n_blocks * sizeof(block_nvfp4)));
    if (!inline_pv) {
        CUDA_CHECK(cudaMalloc(&pv_stage, stage_entries * sizeof(pv_mma_b_lane_frag)));
    }
    CUDA_CHECK(cudaMalloc(&pv_scales, 256 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&out, n_warps * sizeof(float)));

    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(q, n_blocks);
    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(k, n_blocks);
    fill_u32_kernel<<<cfg.blocks, cfg.threads>>>(pv_scales, 256);
    if (!inline_pv) {
        fill_pv_mma_b_stage_reuse_kernel<dkq><<<cfg.blocks, cfg.threads>>>(k, pv_stage, n_tiles, n_blocks - 1);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    combined_kq_packedpv_budget_nvfp4_kernel<nfrags, k_reuse, attentionish, inline_pv><<<cfg.blocks, cfg.threads, shared_bytes>>>(
        q, k, pv_stage, pv_scales, n_blocks - 1, n_tiles - 1, out, 10);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t transform_start;
    cudaEvent_t transform_stop;
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&transform_start));
    CUDA_CHECK(cudaEventCreate(&transform_stop));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float transform_ms = 0.0f;
    if (!inline_pv) {
        CUDA_CHECK(cudaEventRecord(transform_start));
        fill_pv_mma_b_stage_reuse_kernel<dkq><<<cfg.blocks, cfg.threads>>>(k, pv_stage, n_tiles, n_blocks - 1);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(transform_stop));
        transform_ms = time_events(transform_start, transform_stop);
    }

    CUDA_CHECK(cudaEventRecord(start));
    combined_kq_packedpv_budget_nvfp4_kernel<nfrags, k_reuse, attentionish, inline_pv><<<cfg.blocks, cfg.threads, shared_bytes>>>(
        q, k, pv_stage, pv_scales, n_blocks - 1, n_tiles - 1, out, cfg.mma_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float ms = time_events(start, stop);

    const double kq_ops = (double) n_warps * (double) cfg.mma_iters * (double) nfrags * (double) k_reuse * 2.0 * 16.0 * 8.0 * 64.0;
    const double pv_ops = (double) n_warps * (double) cfg.mma_iters * (double) n_out_frags * 2.0 * 16.0 * 8.0 * 64.0;
    const double total_ops = kq_ops + pv_ops;
    const double seconds = ms / 1000.0;
    const double kq_compact_read_gb = (double) n_warps * (double) cfg.mma_iters * (double) nfrags *
                                   (double) (KQ_MMA_A_ROWS + k_reuse * KQ_MMA_B_ROWS) * (double) sizeof(block_nvfp4) / 1.0e9;
    const double pv_inline_read_gb = inline_pv ? (double) n_warps * (double) cfg.mma_iters * (double) nfrags * 64.0 * (double) sizeof(block_nvfp4) / 1.0e9 : 0.0;
    const double stage_read_gb = inline_pv ? 0.0 : (double) n_warps * (double) cfg.mma_iters * (double) stage_tile_size / 1.0e9;
    const double transform_tile_s = (transform_ms / 1000.0) / (double) n_tiles;
    const double combined_tile_s = seconds / ((double) n_warps * (double) cfg.mma_iters);
    const double ops_per_tile = ((double) nfrags * (double) k_reuse + (double) n_out_frags) * 2.0 * 16.0 * 8.0 * 64.0;
    const double transform_equiv_combined_tiles = combined_tile_s > 0.0 ? transform_tile_s / combined_tile_s : 0.0;
    const double transform_inclusive_tops = ops_per_tile / (combined_tile_s + transform_tile_s) / 1.0e12;
    const double transform_compact_gb = (double) n_tiles * (double) nfrags * 64.0 * (double) sizeof(block_nvfp4) / 1.0e9;
    const double transform_stage_gb = (double) n_tiles * (double) stage_tile_size / 1.0e9;

    std::printf("%s: %.3f total-TOPS  %.3f useful-mtp-total-TOPS  %.3f transform-inclusive-TOPS  %.3f useful-mtp-transform-inclusive-TOPS  %.3f KQ-TOPS  %.3f packed-PV-TOPS  %.3f GB/s-kq-compact-read  %.3f GB/s-pv-inline-compact-read  %.3f GB/s-pv-stage-read  %.3f transform-equiv-combined-tiles  %.3f GB/s-transform-compact-read  %.3f GB/s-transform-stage-write  attentionish=%d inline_pv=%d shared=%.3f KiB stage=%.3f MiB tiles=%zu blocks=%zu warps=%zu mma_iters=%" PRIu64 " time=%.3f ms transform_time=%.3f ms\n",
                label,
                total_ops / seconds / 1.0e12,
                total_ops / seconds / 1.0e12 * useful_mtp_fraction(cfg),
                transform_inclusive_tops,
                transform_inclusive_tops * useful_mtp_fraction(cfg),
                kq_ops / seconds / 1.0e12,
                pv_ops / seconds / 1.0e12,
                kq_compact_read_gb / seconds,
                pv_inline_read_gb / seconds,
                stage_read_gb / seconds,
                transform_equiv_combined_tiles,
                inline_pv || transform_ms == 0.0f ? 0.0 : transform_compact_gb / (transform_ms / 1000.0),
                inline_pv || transform_ms == 0.0f ? 0.0 : transform_stage_gb / (transform_ms / 1000.0),
                attentionish ? 1 : 0,
                inline_pv ? 1 : 0,
                (double) shared_bytes / 1024.0,
                inline_pv ? 0.0 : (double) (stage_entries * sizeof(pv_mma_b_lane_frag)) / 1048576.0,
                n_tiles,
                n_blocks,
                n_warps,
                cfg.mma_iters,
                ms,
                transform_ms);

    CUDA_CHECK(cudaEventDestroy(transform_start));
    CUDA_CHECK(cudaEventDestroy(transform_stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(pv_scales));
    if (!inline_pv) {
        CUDA_CHECK(cudaFree(pv_stage));
    }
    CUDA_CHECK(cudaFree(k));
    CUDA_CHECK(cudaFree(q));
}

template <int dkq, int k_reuse, bool attentionish = true>
static void run_combined_kq_mixedpv_half_bench(const bench_config & cfg, int cc, const char * label) {
    static_assert(dkq == 512, "mixed-PV diagnostic currently models only D=512");
    static_assert(k_reuse == 8, "mixed-PV diagnostic currently models reuse 8");
    constexpr int nfrags = dkq / QK_NVFP4;
    constexpr int n_out_frags = dkq / 8;

    if (cc < 1200 || cc >= 1300) {
        std::printf("%s: skipped  reason=requires Blackwell sm_120/sm_121 device\n", label);
        return;
    }

    if (cfg.threads % 32 != 0) {
        std::printf("%s: skipped  reason=threads must be a multiple of 32\n", label);
        return;
    }

    const size_t n_threads = (size_t) cfg.blocks * cfg.threads;
    const size_t n_warps = n_threads / 32;
    const size_t min_kq_blocks = n_warps * (size_t) nfrags * (size_t) (KQ_MMA_A_ROWS + k_reuse * KQ_MMA_B_ROWS);
    const size_t min_pv_blocks = n_warps * (size_t) nfrags * 64u;
    const size_t cfg_blocks = floor_power_of_two(cfg.bytes / sizeof(block_nvfp4));
    size_t n_blocks = 4096;
    while (n_blocks < min_kq_blocks || n_blocks < min_pv_blocks || n_blocks < cfg_blocks) {
        n_blocks <<= 1;
    }

    const size_t shared_bytes = (size_t) (cfg.threads / 32) * (size_t) KQ_MMA_Q_SHARED_INTS * sizeof(int);
    print_occupancy_line(cfg, label, combined_kq_mixedpv_half_nvfp4_kernel<nfrags, k_reuse, attentionish>, shared_bytes);

    block_nvfp4 * q = nullptr;
    block_nvfp4 * k = nullptr;
    float * out = nullptr;
    CUDA_CHECK(cudaMalloc(&q, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&k, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&out, n_warps * sizeof(float)));

    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(q, n_blocks);
    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(k, n_blocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    combined_kq_mixedpv_half_nvfp4_kernel<nfrags, k_reuse, attentionish><<<cfg.blocks, cfg.threads, shared_bytes>>>(
        q, k, n_blocks - 1, out, 10);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    combined_kq_mixedpv_half_nvfp4_kernel<nfrags, k_reuse, attentionish><<<cfg.blocks, cfg.threads, shared_bytes>>>(
        q, k, n_blocks - 1, out, cfg.mma_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float ms = time_events(start, stop);

    const double kq_ops = (double) n_warps * (double) cfg.mma_iters * (double) nfrags * (double) k_reuse * 2.0 * 16.0 * 8.0 * 64.0;
    const double pv_ops = (double) n_warps * (double) cfg.mma_iters * (double) n_out_frags * 4.0 * 2.0 * 16.0 * 8.0 * 16.0;
    const double total_ops = kq_ops + pv_ops;
    const double seconds = ms / 1000.0;
    const double kq_compact_read_gb = (double) n_warps * (double) cfg.mma_iters * (double) nfrags *
                                   (double) (KQ_MMA_A_ROWS + k_reuse * KQ_MMA_B_ROWS) * (double) sizeof(block_nvfp4) / 1.0e9;
    const double pv_inline_read_gb = (double) n_warps * (double) cfg.mma_iters * (double) nfrags * 64.0 * (double) sizeof(block_nvfp4) / 1.0e9;
    const double pv_half_mma_count = (double) n_warps * (double) cfg.mma_iters * (double) n_out_frags * 4.0;

    std::printf("%s: %.3f total-TOPS  %.3f useful-mtp-total-TOPS  %.3f KQ-TOPS  %.3f mixed-PV-TOPS  %.3f GB/s-kq-compact-read  %.3f GB/s-pv-inline-compact-read  attentionish=%d half_mma_per_tile=%.0f shared=%.3f KiB blocks=%zu warps=%zu mma_iters=%" PRIu64 " time=%.3f ms\n",
                label,
                total_ops / seconds / 1.0e12,
                total_ops / seconds / 1.0e12 * useful_mtp_fraction(cfg),
                kq_ops / seconds / 1.0e12,
                pv_ops / seconds / 1.0e12,
                kq_compact_read_gb / seconds,
                pv_inline_read_gb / seconds,
                attentionish ? 1 : 0,
                pv_half_mma_count / ((double) n_warps * (double) cfg.mma_iters),
                (double) shared_bytes / 1024.0,
                n_blocks,
                n_warps,
                cfg.mma_iters,
                ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(k));
    CUDA_CHECK(cudaFree(q));
}

template <int dkq, int k_reuse, int staged_groups, bool attentionish = true>
static void run_combined_kq_mixedpv_stripmine_bench(const bench_config & cfg, int cc, const char * label) {
    static_assert(dkq == 512, "mixed-PV stripmine diagnostic currently models only D=512");
    static_assert(k_reuse == 8, "mixed-PV stripmine diagnostic currently models reuse 8");
    static_assert(staged_groups == 1 || staged_groups == 2 || staged_groups == 4 || staged_groups == 8, "unsupported staged group count");
    constexpr int nfrags = dkq / QK_NVFP4;
    constexpr int groups_per_frag = QK_NVFP4 / 8;
    constexpr int measured_pv_frags = nfrags * k_reuse * staged_groups;
    constexpr int full_pv_frags = nfrags * k_reuse * groups_per_frag;

    if (cc < 1200 || cc >= 1300) {
        std::printf("%s: skipped  reason=requires Blackwell sm_120/sm_121 device\n", label);
        return;
    }

    if (cfg.threads % 32 != 0) {
        std::printf("%s: skipped  reason=threads must be a multiple of 32\n", label);
        return;
    }

    const size_t n_threads = (size_t) cfg.blocks * cfg.threads;
    const size_t n_warps = n_threads / 32;
    const size_t min_blocks = n_warps * (size_t) nfrags * (size_t) (KQ_MMA_A_ROWS + k_reuse * KQ_MMA_B_ROWS);
    const size_t cfg_blocks = floor_power_of_two(cfg.bytes / sizeof(block_nvfp4));
    size_t n_blocks = 4096;
    while (n_blocks < min_blocks || n_blocks < cfg_blocks) {
        n_blocks <<= 1;
    }

    const size_t shared_bytes = (size_t) (cfg.threads / 32) * (size_t) KQ_MMA_Q_SHARED_INTS * sizeof(int);
    print_occupancy_line(cfg, label, combined_kq_mixedpv_stripmine_nvfp4_kernel<nfrags, k_reuse, staged_groups, attentionish>, shared_bytes);

    block_nvfp4 * q = nullptr;
    block_nvfp4 * k = nullptr;
    float * out = nullptr;
    CUDA_CHECK(cudaMalloc(&q, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&k, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&out, n_warps * sizeof(float)));

    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(q, n_blocks);
    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(k, n_blocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    combined_kq_mixedpv_stripmine_nvfp4_kernel<nfrags, k_reuse, staged_groups, attentionish><<<cfg.blocks, cfg.threads, shared_bytes>>>(
        q, k, n_blocks - 1, out, 10);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    combined_kq_mixedpv_stripmine_nvfp4_kernel<nfrags, k_reuse, staged_groups, attentionish><<<cfg.blocks, cfg.threads, shared_bytes>>>(
        q, k, n_blocks - 1, out, cfg.mma_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float ms = time_events(start, stop);

    const double kq_ops = (double) n_warps * (double) cfg.mma_iters * (double) nfrags * (double) k_reuse * 2.0 * 16.0 * 8.0 * 64.0;
    const double measured_pv_ops = (double) n_warps * (double) cfg.mma_iters * (double) measured_pv_frags * 4.0 * 2.0 * 16.0 * 8.0 * 16.0;
    const double full_pv_ops = (double) n_warps * (double) cfg.mma_iters * (double) full_pv_frags * 4.0 * 2.0 * 16.0 * 8.0 * 16.0;
    const double measured_ops = kq_ops + measured_pv_ops;
    const double seconds = ms / 1000.0;
    const double kq_compact_read_gb = (double) n_warps * (double) cfg.mma_iters * (double) nfrags *
                                   (double) (KQ_MMA_A_ROWS + k_reuse * KQ_MMA_B_ROWS) * (double) sizeof(block_nvfp4) / 1.0e9;
    const double pv_group_fraction = (double) measured_pv_frags / (double) full_pv_frags;

    std::printf("%s: %.3f measured-total-TOPS  %.3f useful-mtp-measured-total-TOPS  %.3f KQ-TOPS  %.3f measured-mixedPV-TOPS  %.3f projected-fullPV-TOPS-at-same-time  %.3f GB/s-kq-compact-read  %.3f GB/s-pv-extra-compact-read  pv_group_fraction=%.3f staged_groups=%d attentionish=%d shared=%.3f KiB blocks=%zu warps=%zu mma_iters=%" PRIu64 " time=%.3f ms\n",
                label,
                measured_ops / seconds / 1.0e12,
                measured_ops / seconds / 1.0e12 * useful_mtp_fraction(cfg),
                kq_ops / seconds / 1.0e12,
                measured_pv_ops / seconds / 1.0e12,
                full_pv_ops / seconds / 1.0e12,
                kq_compact_read_gb / seconds,
                0.0,
                pv_group_fraction,
                staged_groups,
                attentionish ? 1 : 0,
                (double) shared_bytes / 1024.0,
                n_blocks,
                n_warps,
                cfg.mma_iters,
                ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(k));
    CUDA_CHECK(cudaFree(q));
}

template <int dkq, int k_reuse, int staged_groups, bool attentionish = true>
static void run_combined_kq_mixedpv_smem_half_bench(const bench_config & cfg, int cc, const char * label) {
    static_assert(dkq == 512, "mixed-PV shared-stage diagnostic currently models only D=512");
    static_assert(k_reuse == 8, "mixed-PV shared-stage diagnostic currently models reuse 8");
    static_assert(staged_groups == 1 || staged_groups == 2 || staged_groups == 4, "unsupported staged group count");
    constexpr int nfrags = dkq / QK_NVFP4;
    constexpr int full_pv_frags = dkq / 8;
    constexpr int measured_pv_frags = nfrags * staged_groups;
    constexpr int mixed_stage_half2_per_row = staged_groups * 4;
    constexpr int mixed_stage_ints = KQ_MMA_B_ROWS * k_reuse * mixed_stage_half2_per_row;

    if (cc < 1200 || cc >= 1300) {
        std::printf("%s: skipped  reason=requires Blackwell sm_120/sm_121 device\n", label);
        return;
    }

    if (cfg.threads % 32 != 0) {
        std::printf("%s: skipped  reason=threads must be a multiple of 32\n", label);
        return;
    }

    const size_t n_threads = (size_t) cfg.blocks * cfg.threads;
    const size_t n_warps = n_threads / 32;
    const size_t min_blocks = n_warps * (size_t) nfrags * (size_t) (KQ_MMA_A_ROWS + k_reuse * KQ_MMA_B_ROWS);
    const size_t cfg_blocks = floor_power_of_two(cfg.bytes / sizeof(block_nvfp4));
    size_t n_blocks = 4096;
    while (n_blocks < min_blocks || n_blocks < cfg_blocks) {
        n_blocks <<= 1;
    }

    const size_t shared_bytes = (size_t) (cfg.threads / 32) * (size_t) (KQ_MMA_Q_SHARED_INTS + mixed_stage_ints) * sizeof(int);
    print_occupancy_line(cfg, label, combined_kq_mixedpv_smem_half_nvfp4_kernel<nfrags, k_reuse, staged_groups, attentionish>, shared_bytes);

    block_nvfp4 * q = nullptr;
    block_nvfp4 * k = nullptr;
    float * out = nullptr;
    CUDA_CHECK(cudaMalloc(&q, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&k, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&out, n_warps * sizeof(float)));

    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(q, n_blocks);
    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(k, n_blocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    combined_kq_mixedpv_smem_half_nvfp4_kernel<nfrags, k_reuse, staged_groups, attentionish><<<cfg.blocks, cfg.threads, shared_bytes>>>(
        q, k, n_blocks - 1, out, 10);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    combined_kq_mixedpv_smem_half_nvfp4_kernel<nfrags, k_reuse, staged_groups, attentionish><<<cfg.blocks, cfg.threads, shared_bytes>>>(
        q, k, n_blocks - 1, out, cfg.mma_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float ms = time_events(start, stop);

    const double kq_ops = (double) n_warps * (double) cfg.mma_iters * (double) nfrags * (double) k_reuse * 2.0 * 16.0 * 8.0 * 64.0;
    const double measured_pv_ops = (double) n_warps * (double) cfg.mma_iters * (double) measured_pv_frags * 4.0 * 2.0 * 16.0 * 8.0 * 16.0;
    const double full_pv_ops = (double) n_warps * (double) cfg.mma_iters * (double) full_pv_frags * 4.0 * 2.0 * 16.0 * 8.0 * 16.0;
    const double measured_ops = kq_ops + measured_pv_ops;
    const double seconds = ms / 1000.0;
    const double kq_compact_read_gb = (double) n_warps * (double) cfg.mma_iters * (double) nfrags *
                                   (double) (KQ_MMA_A_ROWS + k_reuse * KQ_MMA_B_ROWS) * (double) sizeof(block_nvfp4) / 1.0e9;
    const double shared_stage_bytes = (double) n_warps * (double) cfg.mma_iters * (double) nfrags *
                                      (double) mixed_stage_ints * (double) sizeof(int);
    const double pv_group_fraction = (double) measured_pv_frags / (double) full_pv_frags;

    std::printf("%s: %.3f measured-total-TOPS  %.3f useful-mtp-measured-total-TOPS  %.3f KQ-TOPS  %.3f measured-mixedPV-TOPS  %.3f projected-fullPV-TOPS-at-same-time  %.3f GB/s-kq-compact-read  %.3f GB/s-pv-extra-compact-read  %.3f GB/s-shared-stage-write  %.3f GB/s-shared-pv-read  pv_group_fraction=%.3f staged_groups=%d attentionish=%d mixed_pv_smem_groups=%d shared=%.3f KiB blocks=%zu warps=%zu mma_iters=%" PRIu64 " time=%.3f ms\n",
                label,
                measured_ops / seconds / 1.0e12,
                measured_ops / seconds / 1.0e12 * useful_mtp_fraction(cfg),
                kq_ops / seconds / 1.0e12,
                measured_pv_ops / seconds / 1.0e12,
                full_pv_ops / seconds / 1.0e12,
                kq_compact_read_gb / seconds,
                0.0,
                shared_stage_bytes / 1.0e9 / seconds,
                shared_stage_bytes / 1.0e9 / seconds,
                pv_group_fraction,
                staged_groups,
                attentionish ? 1 : 0,
                staged_groups,
                (double) shared_bytes / 1024.0,
                n_blocks,
                n_warps,
                cfg.mma_iters,
                ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(k));
    CUDA_CHECK(cudaFree(q));
}

template <int dkq, int k_reuse, int staged_groups, bool attentionish = true, bool consume_remaining_inline = false>
static void run_combined_kq_mixedpv_smem_compact_bench(const bench_config & cfg, int cc, const char * label) {
    static_assert(dkq == 512, "mixed-PV compact-stage diagnostic currently models only D=512");
    static_assert(k_reuse == 8, "mixed-PV compact-stage diagnostic currently models reuse 8");
    static_assert(staged_groups == 1 || staged_groups == 2 || staged_groups == 4 || staged_groups == 8, "unsupported staged group count");
    constexpr int nfrags = dkq / QK_NVFP4;
    constexpr int full_pv_frags = dkq / 8;
    constexpr int measured_pv_frags = consume_remaining_inline ? full_pv_frags : nfrags * staged_groups;
    constexpr int stage_word_ints = KQ_MMA_B_ROWS * k_reuse * staged_groups;
    constexpr int stage_scale_ints = KQ_MMA_B_ROWS * k_reuse;
    constexpr int stage_ints = stage_word_ints + stage_scale_ints;

    if (cc < 1200 || cc >= 1300) {
        std::printf("%s: skipped  reason=requires Blackwell sm_120/sm_121 device\n", label);
        return;
    }

    if (cfg.threads % 32 != 0) {
        std::printf("%s: skipped  reason=threads must be a multiple of 32\n", label);
        return;
    }

    const size_t n_threads = (size_t) cfg.blocks * cfg.threads;
    const size_t n_warps = n_threads / 32;
    const size_t min_blocks = n_warps * (size_t) nfrags * (size_t) (KQ_MMA_A_ROWS + k_reuse * KQ_MMA_B_ROWS);
    const size_t cfg_blocks = floor_power_of_two(cfg.bytes / sizeof(block_nvfp4));
    size_t n_blocks = 4096;
    while (n_blocks < min_blocks || n_blocks < cfg_blocks) {
        n_blocks <<= 1;
    }

    const size_t shared_bytes = (size_t) (cfg.threads / 32) * (size_t) (KQ_MMA_Q_SHARED_INTS + stage_ints) * sizeof(int);
    print_occupancy_line(cfg, label, combined_kq_mixedpv_smem_compact_nvfp4_kernel<nfrags, k_reuse, staged_groups, attentionish, consume_remaining_inline>, shared_bytes);

    block_nvfp4 * q = nullptr;
    block_nvfp4 * k = nullptr;
    float * out = nullptr;
    CUDA_CHECK(cudaMalloc(&q, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&k, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&out, n_warps * sizeof(float)));

    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(q, n_blocks);
    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(k, n_blocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    combined_kq_mixedpv_smem_compact_nvfp4_kernel<nfrags, k_reuse, staged_groups, attentionish, consume_remaining_inline><<<cfg.blocks, cfg.threads, shared_bytes>>>(
        q, k, n_blocks - 1, out, 10);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    combined_kq_mixedpv_smem_compact_nvfp4_kernel<nfrags, k_reuse, staged_groups, attentionish, consume_remaining_inline><<<cfg.blocks, cfg.threads, shared_bytes>>>(
        q, k, n_blocks - 1, out, cfg.mma_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float ms = time_events(start, stop);

    const double kq_ops = (double) n_warps * (double) cfg.mma_iters * (double) nfrags * (double) k_reuse * 2.0 * 16.0 * 8.0 * 64.0;
    const double measured_pv_ops = (double) n_warps * (double) cfg.mma_iters * (double) measured_pv_frags * 4.0 * 2.0 * 16.0 * 8.0 * 16.0;
    const double full_pv_ops = (double) n_warps * (double) cfg.mma_iters * (double) full_pv_frags * 4.0 * 2.0 * 16.0 * 8.0 * 16.0;
    const double measured_ops = kq_ops + measured_pv_ops;
    const double seconds = ms / 1000.0;
    const double kq_compact_read_gb = (double) n_warps * (double) cfg.mma_iters * (double) nfrags *
                                   (double) (KQ_MMA_A_ROWS + k_reuse * KQ_MMA_B_ROWS) * (double) sizeof(block_nvfp4) / 1.0e9;
    const double pv_extra_compact_read_gb = consume_remaining_inline ?
                                            (double) n_warps * (double) cfg.mma_iters * (double) nfrags * 64.0 * (double) sizeof(block_nvfp4) / 1.0e9 :
                                            0.0;
    const double shared_stage_bytes = (double) n_warps * (double) cfg.mma_iters * (double) nfrags *
                                      (double) stage_ints * (double) sizeof(int);
    const double pv_group_fraction = (double) measured_pv_frags / (double) full_pv_frags;

    std::printf("%s: %.3f measured-total-TOPS  %.3f useful-mtp-measured-total-TOPS  %.3f KQ-TOPS  %.3f measured-mixedPV-TOPS  %.3f projected-fullPV-TOPS-at-same-time  %.3f GB/s-kq-compact-read  %.3f GB/s-pv-extra-compact-read  %.3f GB/s-shared-stage-write  %.3f GB/s-shared-pv-read  pv_group_fraction=%.3f staged_groups=%d attentionish=%d mixed_pv_compact_smem_groups=%d inline_remainder=%d shared=%.3f KiB blocks=%zu warps=%zu mma_iters=%" PRIu64 " time=%.3f ms\n",
                label,
                measured_ops / seconds / 1.0e12,
                measured_ops / seconds / 1.0e12 * useful_mtp_fraction(cfg),
                kq_ops / seconds / 1.0e12,
                measured_pv_ops / seconds / 1.0e12,
                full_pv_ops / seconds / 1.0e12,
                kq_compact_read_gb / seconds,
                pv_extra_compact_read_gb / seconds,
                shared_stage_bytes / 1.0e9 / seconds,
                shared_stage_bytes / 1.0e9 / seconds,
                pv_group_fraction,
                staged_groups,
                attentionish ? 1 : 0,
                staged_groups,
                consume_remaining_inline ? 1 : 0,
                (double) shared_bytes / 1024.0,
                n_blocks,
                n_warps,
                cfg.mma_iters,
                ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(k));
    CUDA_CHECK(cudaFree(q));
}

template <int dkq, int k_reuse, bool attentionish = true>
static void run_combined_kq_kreusepv_budget_bench(const bench_config & cfg, int cc, const char * label) {
    static_assert(dkq == 512, "K-reuse PV diagnostic currently models only D=512");
    static_assert(k_reuse == 8, "K-reuse PV diagnostic currently models reuse 8");
    constexpr int nfrags = dkq / QK_NVFP4;
    constexpr int n_out_frags = dkq / 8;

    if (cc < 1200 || cc >= 1300) {
        std::printf("%s: skipped  reason=requires Blackwell sm_120/sm_121 device\n", label);
        return;
    }

    if (cfg.threads % 32 != 0) {
        std::printf("%s: skipped  reason=threads must be a multiple of 32\n", label);
        return;
    }

    const size_t n_threads = (size_t) cfg.blocks * cfg.threads;
    const size_t n_warps = n_threads / 32;
    const size_t min_blocks = n_warps * (size_t) nfrags * (size_t) (KQ_MMA_A_ROWS + k_reuse * KQ_MMA_B_ROWS);
    const size_t cfg_blocks = floor_power_of_two(cfg.bytes / sizeof(block_nvfp4));
    size_t n_blocks = 4096;
    while (n_blocks < min_blocks || n_blocks < cfg_blocks) {
        n_blocks <<= 1;
    }

    const size_t shared_bytes = (size_t) (cfg.threads / 32) * (size_t) KQ_MMA_Q_SHARED_INTS * sizeof(int);
    print_occupancy_line(cfg, label, combined_kq_kreusepv_budget_nvfp4_kernel<nfrags, k_reuse, attentionish>, shared_bytes);

    block_nvfp4 * q = nullptr;
    block_nvfp4 * k = nullptr;
    uint32_t * pv_scales = nullptr;
    float * out = nullptr;
    CUDA_CHECK(cudaMalloc(&q, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&k, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&pv_scales, 256 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&out, n_warps * sizeof(float)));

    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(q, n_blocks);
    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(k, n_blocks);
    fill_u32_kernel<<<cfg.blocks, cfg.threads>>>(pv_scales, 256);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    combined_kq_kreusepv_budget_nvfp4_kernel<nfrags, k_reuse, attentionish><<<cfg.blocks, cfg.threads, shared_bytes>>>(
        q, k, pv_scales, n_blocks - 1, out, 10);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    combined_kq_kreusepv_budget_nvfp4_kernel<nfrags, k_reuse, attentionish><<<cfg.blocks, cfg.threads, shared_bytes>>>(
        q, k, pv_scales, n_blocks - 1, out, cfg.mma_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float ms = time_events(start, stop);

    const double kq_ops = (double) n_warps * (double) cfg.mma_iters * (double) nfrags * (double) k_reuse * 2.0 * 16.0 * 8.0 * 64.0;
    const double pv_ops = (double) n_warps * (double) cfg.mma_iters * (double) n_out_frags * 2.0 * 16.0 * 8.0 * 64.0;
    const double total_ops = kq_ops + pv_ops;
    const double seconds = ms / 1000.0;
    const double kq_compact_read_gb = (double) n_warps * (double) cfg.mma_iters * (double) nfrags *
                                   (double) (KQ_MMA_A_ROWS + k_reuse * KQ_MMA_B_ROWS) * (double) sizeof(block_nvfp4) / 1.0e9;

    std::printf("%s: %.3f total-TOPS  %.3f useful-mtp-total-TOPS  %.3f KQ-TOPS  %.3f packed-PV-TOPS  %.3f GB/s-kq-compact-read  %.3f GB/s-pv-extra-compact-read  attentionish=%d k_reuse_pv=1 shared=%.3f KiB blocks=%zu warps=%zu mma_iters=%" PRIu64 " time=%.3f ms\n",
                label,
                total_ops / seconds / 1.0e12,
                total_ops / seconds / 1.0e12 * useful_mtp_fraction(cfg),
                kq_ops / seconds / 1.0e12,
                pv_ops / seconds / 1.0e12,
                kq_compact_read_gb / seconds,
                0.0,
                attentionish ? 1 : 0,
                (double) shared_bytes / 1024.0,
                n_blocks,
                n_warps,
                cfg.mma_iters,
                ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(pv_scales));
    CUDA_CHECK(cudaFree(k));
    CUDA_CHECK(cudaFree(q));
}

// Experimental only: narrows the K-reuse PV live set to one 8-dim group per fragment.
// It still stalls the low-iteration local smoke, so it remains opt-in.
template <int nfrags, int k_reuse, bool attentionish>
__global__ void combined_kq_kreusepv_subtile_nvfp4_kernel(
        const block_nvfp4 * q,
        const block_nvfp4 * k,
        const uint32_t *    pv_scales,
        size_t              block_mask,
        float *             out,
        uint64_t            iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    static_assert(nfrags == 8, "K-reuse PV sub-tile diagnostic currently models only D=512");
    static_assert(k_reuse == 8, "K-reuse PV sub-tile diagnostic currently models reuse 8");
    extern __shared__ int smem_i[];

    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warp = ((int) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const unsigned mask = 0xffffffffu;

    int * warp_s = smem_i + warp_in_block * KQ_MMA_Q_SHARED_INTS;
    int * q_s = warp_s;
    uint32_t * q_sc = reinterpret_cast<uint32_t *>(q_s + KQ_MMA_A_ROWS * KQ_MMA_A_STRIDE);

    float kq0 = 0.0f;
    float kq1 = 0.0f;
    float kq2 = 0.0f;
    float kq3 = 0.0f;
    float pv0 = 0.0f;
    float pv1 = 0.0f;
    float pv2 = 0.0f;
    float pv3 = 0.0f;
    float row_m0 = -64.0f;
    float row_m1 = -64.0f;
    float row_l0 = 0.0f;
    float row_l1 = 0.0f;
    float out0 = 0.0f;
    float out1 = 0.0f;
    float out2 = 0.0f;
    float out3 = 0.0f;

    for (uint64_t i = 0; i < iters; ++i) {
        const int pax0 = (int) (0x11111111u + (uint32_t) lane);
        const int pax1 = (int) (0x22222222u + (uint32_t) lane);
        const int pax2 = (int) (0x33333333u + (uint32_t) lane);
        const int pax3 = (int) (0x44444444u + (uint32_t) lane);
        const uint32_t p_scale = pv_scales[lane & 255];

#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const size_t q_base = ((size_t) warp * (size_t) nfrags * (size_t) KQ_MMA_A_ROWS) +
                                  ((size_t) frag * (size_t) KQ_MMA_A_ROWS) +
                                  ((size_t) i * (size_t) gridDim.x * (size_t) nfrags * (size_t) KQ_MMA_A_ROWS);

            if (lane < KQ_MMA_A_ROWS) {
                const block_nvfp4 q_blk = q[(q_base + (size_t) lane) & block_mask];
                const uint32_t * q_qs = reinterpret_cast<const uint32_t *>(q_blk.qs);
#pragma unroll
                for (int c = 0; c < KQ_MMA_WORDS; ++c) {
                    q_s[lane * KQ_MMA_A_STRIDE + c] = (int) q_qs[c];
                }
                q_sc[lane] = ((uint32_t) q_blk.d[0]) | ((uint32_t) q_blk.d[1] << 8) | ((uint32_t) q_blk.d[2] << 16) | ((uint32_t) q_blk.d[3] << 24);
            }

            __syncwarp(mask);

            int ax0;
            int ax1;
            int ax2;
            int ax3;
            const int * a_ptr = q_s + (lane % KQ_MMA_A_ROWS) * KQ_MMA_A_STRIDE + (lane / KQ_MMA_A_ROWS) * 4;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                : "=r"(ax0), "=r"(ax1), "=r"(ax2), "=r"(ax3)
                : "l"(a_ptr));

            const int tidx_a = lane / 4 + (lane % 2) * 8;
            const uint32_t q_scale = q_sc[tidx_a];
            uint32_t pv_b0 = 0;
            uint32_t pv_b1 = 0;
            uint32_t pv_scale = 0;

#pragma unroll
            for (int kr = 0; kr < k_reuse; ++kr) {
                const size_t k_base = ((size_t) warp * (size_t) nfrags * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) frag * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) kr * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) i * (size_t) gridDim.x * (size_t) nfrags * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS);

                uint32_t kv0 = 0;
                uint32_t kv1 = 0;
                uint32_t kv2 = 0;
                uint32_t kv3 = 0;
                uint32_t kv4 = 0;
                uint32_t kv5 = 0;
                uint32_t kv6 = 0;
                uint32_t kv7 = 0;
                uint32_t k_scale_row = 0;

                if (lane < KQ_MMA_B_ROWS) {
                    const block_nvfp4 k_blk = k[(k_base + (size_t) lane) & block_mask];
                    const uint32_t * k_qs = reinterpret_cast<const uint32_t *>(k_blk.qs);
                    kv0 = k_qs[0];
                    kv1 = k_qs[1];
                    kv2 = k_qs[2];
                    kv3 = k_qs[3];
                    kv4 = k_qs[4];
                    kv5 = k_qs[5];
                    kv6 = k_qs[6];
                    kv7 = k_qs[7];
                    k_scale_row = ((uint32_t) k_blk.d[0]) | ((uint32_t) k_blk.d[1] << 8) | ((uint32_t) k_blk.d[2] << 16) | ((uint32_t) k_blk.d[3] << 24);
                }

                const int pv_row0 = 2 * lane + 0;
                const int pv_row1 = 2 * lane + 1;
                const int pv_kr0 = pv_row0 >> 3;
                const int pv_kr1 = pv_row1 >> 3;
                const int pv_src0 = pv_row0 & 7;
                const int pv_src1 = pv_row1 & 7;

                if (kr == pv_kr0) {
                    pv_b0 = __shfl_sync(mask, kv0, pv_src0);
                    pv_scale = __shfl_sync(mask, k_scale_row, pv_src0);
                }
                if (kr == pv_kr1) {
                    pv_b1 = __shfl_sync(mask, kv0, pv_src1);
                }

                const int b_i = lane / 4;
                const int b_j0 = lane % 4;
                const int row_v0 = (int) __shfl_sync(mask, kv0, b_i);
                const int row_v1 = (int) __shfl_sync(mask, kv1, b_i);
                const int row_v2 = (int) __shfl_sync(mask, kv2, b_i);
                const int row_v3 = (int) __shfl_sync(mask, kv3, b_i);
                const int row_v4 = (int) __shfl_sync(mask, kv4, b_i);
                const int row_v5 = (int) __shfl_sync(mask, kv5, b_i);
                const int row_v6 = (int) __shfl_sync(mask, kv6, b_i);
                const int row_v7 = (int) __shfl_sync(mask, kv7, b_i);

                const int bx0 = b_j0 == 0 ? row_v0 : (b_j0 == 1 ? row_v1 : (b_j0 == 2 ? row_v2 : row_v3));
                const int bx1 = b_j0 == 0 ? row_v4 : (b_j0 == 1 ? row_v5 : (b_j0 == 2 ? row_v6 : row_v7));
                const uint32_t k_scale = __shfl_sync(mask, k_scale_row, b_i);

                asm volatile(
                    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                    "%10, {0, 0}, %11, {0, 0};"
                    : "+f"(kq0), "+f"(kq1), "+f"(kq2), "+f"(kq3)
                    : "r"(ax0), "r"(ax1), "r"(ax2), "r"(ax3), "r"(bx0), "r"(bx1), "r"(q_scale), "r"(k_scale));

                if constexpr (attentionish) {
                    const int q_row = lane & (KQ_MMA_A_ROWS - 1);
                    const int key_pos = frag * k_reuse + kr;
                    const int base_pos = (int) (i & 1023u);
                    const bool causal = key_pos <= base_pos + q_row;
                    const bool swa = key_pos + 1024 >= base_pos + q_row;

                    const float score0 = (causal && swa) ? kq0 * 0.000244140625f : -64.0f;
                    const float score1 = (causal && swa) ? kq2 * 0.000244140625f : -64.0f;
                    const float next_m0 = fmaxf(row_m0, score0);
                    const float next_m1 = fmaxf(row_m1, score1);
                    const float alpha0 = row_l0 == 0.0f ? 0.0f : exp2f(row_m0 - next_m0);
                    const float alpha1 = row_l1 == 0.0f ? 0.0f : exp2f(row_m1 - next_m1);
                    const float beta0 = exp2f(score0 - next_m0);
                    const float beta1 = exp2f(score1 - next_m1);
                    row_l0 = row_l0 * alpha0 + beta0;
                    row_l1 = row_l1 * alpha1 + beta1;
                    row_m0 = next_m0;
                    row_m1 = next_m1;
                } else {
                    row_m0 = fmaxf(row_m0, kq0);
                    row_m1 = fmaxf(row_m1, kq2);
                    row_l0 += fabsf(kq1) * 0.000001f + 1.0f;
                    row_l1 += fabsf(kq3) * 0.000001f + 1.0f;
                }
            }

            asm volatile(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                "%10, {0, 0}, %11, {0, 0};"
                : "+f"(pv0), "+f"(pv1), "+f"(pv2), "+f"(pv3)
                : "r"(pax0), "r"(pax1), "r"(pax2), "r"(pax3), "r"((int) pv_b0), "r"((int) pv_b1), "r"(p_scale), "r"(pv_scale));

            const float mix = (float) (frag + 1) * 0.000001f;
            out0 += pv0 * mix;
            out1 += pv1 * mix;
            out2 += pv2 * mix;
            out3 += pv3 * mix;
        }
    }

    if (lane == 0) {
        out[warp] = kq0 + kq1 + kq2 + kq3 + pv0 + pv1 + pv2 + pv3 +
                    row_m0 + row_m1 + row_l0 + row_l1 + out0 + out1 + out2 + out3;
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = 0.0f;
    }
    (void) q;
    (void) k;
    (void) pv_scales;
    (void) block_mask;
    (void) out;
    (void) iters;
#endif
}

template <int dkq, int k_reuse, bool attentionish = true>
static void run_combined_kq_kreusepv_subtile_bench(const bench_config & cfg, int cc, const char * label) {
    static_assert(dkq == 512, "K-reuse PV sub-tile diagnostic currently models only D=512");
    static_assert(k_reuse == 8, "K-reuse PV sub-tile diagnostic currently models reuse 8");
    constexpr int nfrags = dkq / QK_NVFP4;
    constexpr int measured_pv_frags = nfrags;
    constexpr int full_pv_frags = dkq / 8;

    if (cc < 1200 || cc >= 1300) {
        std::printf("%s: skipped  reason=requires Blackwell sm_120/sm_121 device\n", label);
        return;
    }

    const size_t n_threads = (size_t) cfg.blocks * cfg.threads;
    const size_t n_warps = n_threads / 32;
    const size_t min_blocks = n_warps * (size_t) nfrags * (size_t) (KQ_MMA_A_ROWS + k_reuse * KQ_MMA_B_ROWS);
    const size_t cfg_blocks = floor_power_of_two(cfg.bytes / sizeof(block_nvfp4));
    size_t n_blocks = 4096;
    while (n_blocks < min_blocks || n_blocks < cfg_blocks) {
        n_blocks <<= 1;
    }

    const size_t shared_bytes = (size_t) (cfg.threads / 32) * (size_t) KQ_MMA_Q_SHARED_INTS * sizeof(int);
    print_occupancy_line(cfg, label, combined_kq_kreusepv_subtile_nvfp4_kernel<nfrags, k_reuse, attentionish>, shared_bytes);

    block_nvfp4 * q = nullptr;
    block_nvfp4 * k = nullptr;
    uint32_t * pv_scales = nullptr;
    float * out = nullptr;
    CUDA_CHECK(cudaMalloc(&q, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&k, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&pv_scales, 256 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&out, n_warps * sizeof(float)));

    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(q, n_blocks);
    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(k, n_blocks);
    fill_u32_kernel<<<cfg.blocks, cfg.threads>>>(pv_scales, 256);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    combined_kq_kreusepv_subtile_nvfp4_kernel<nfrags, k_reuse, attentionish><<<cfg.blocks, cfg.threads, shared_bytes>>>(
        q, k, pv_scales, n_blocks - 1, out, 10);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    combined_kq_kreusepv_subtile_nvfp4_kernel<nfrags, k_reuse, attentionish><<<cfg.blocks, cfg.threads, shared_bytes>>>(
        q, k, pv_scales, n_blocks - 1, out, cfg.mma_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float ms = time_events(start, stop);

    const double kq_ops = (double) n_warps * (double) cfg.mma_iters * (double) nfrags * (double) k_reuse * 2.0 * 16.0 * 8.0 * 64.0;
    const double measured_pv_ops = (double) n_warps * (double) cfg.mma_iters * (double) measured_pv_frags * 2.0 * 16.0 * 8.0 * 64.0;
    const double full_pv_ops = (double) n_warps * (double) cfg.mma_iters * (double) full_pv_frags * 2.0 * 16.0 * 8.0 * 64.0;
    const double measured_ops = kq_ops + measured_pv_ops;
    const double seconds = ms / 1000.0;
    const double kq_compact_read_gb = (double) n_warps * (double) cfg.mma_iters * (double) nfrags *
                                   (double) (KQ_MMA_A_ROWS + k_reuse * KQ_MMA_B_ROWS) * (double) sizeof(block_nvfp4) / 1.0e9;
    const double pv_group_fraction = (double) measured_pv_frags / (double) full_pv_frags;

    std::printf("%s: %.3f measured-total-TOPS  %.3f useful-mtp-measured-total-TOPS  %.3f KQ-TOPS  %.3f measured-subPV-TOPS  %.3f projected-fullPV-TOPS-at-same-time  %.3f GB/s-kq-compact-read  %.3f GB/s-pv-extra-compact-read  pv_group_fraction=%.3f attentionish=%d k_reuse_subtile=1 shared=%.3f KiB blocks=%zu warps=%zu mma_iters=%" PRIu64 " time=%.3f ms\n",
                label,
                measured_ops / seconds / 1.0e12,
                measured_ops / seconds / 1.0e12 * useful_mtp_fraction(cfg),
                kq_ops / seconds / 1.0e12,
                measured_pv_ops / seconds / 1.0e12,
                full_pv_ops / seconds / 1.0e12,
                kq_compact_read_gb / seconds,
                0.0,
                pv_group_fraction,
                attentionish ? 1 : 0,
                (double) shared_bytes / 1024.0,
                n_blocks,
                n_warps,
                cfg.mma_iters,
                ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(pv_scales));
    CUDA_CHECK(cudaFree(k));
    CUDA_CHECK(cudaFree(q));
}

template <int nfrags, int k_reuse, bool attentionish>
__global__ void combined_kq_kreusepv_smem_nvfp4_kernel(
        const block_nvfp4 * q,
        const block_nvfp4 * k,
        const uint32_t *    pv_scales,
        size_t              block_mask,
        float *             out,
        uint64_t            iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    static_assert(nfrags == 8, "K-reuse PV shared-stage diagnostic currently models only D=512");
    static_assert(k_reuse == 8, "K-reuse PV shared-stage diagnostic currently models reuse 8");
    extern __shared__ int smem_i[];

    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warp = ((int) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const unsigned mask = 0xffffffffu;

    int * warp_s = smem_i + warp_in_block * (KQ_MMA_Q_SHARED_INTS + KREUSE_STAGE_INTS);
    int * q_s = warp_s;
    uint32_t * q_sc = reinterpret_cast<uint32_t *>(q_s + KQ_MMA_A_ROWS * KQ_MMA_A_STRIDE);
    int * k_stage = warp_s + KQ_MMA_Q_SHARED_INTS;
    uint32_t * k_stage_sc = reinterpret_cast<uint32_t *>(k_stage + KREUSE_STAGE_WORD_INTS);

    float kq0 = 0.0f;
    float kq1 = 0.0f;
    float kq2 = 0.0f;
    float kq3 = 0.0f;
    float pv0 = 0.0f;
    float pv1 = 0.0f;
    float pv2 = 0.0f;
    float pv3 = 0.0f;
    float row_m0 = -64.0f;
    float row_m1 = -64.0f;
    float row_l0 = 0.0f;
    float row_l1 = 0.0f;
    float out0 = 0.0f;
    float out1 = 0.0f;
    float out2 = 0.0f;
    float out3 = 0.0f;
    float out4 = 0.0f;
    float out5 = 0.0f;
    float out6 = 0.0f;
    float out7 = 0.0f;

    for (uint64_t i = 0; i < iters; ++i) {
        const int pax0 = (int) (0x11111111u + (uint32_t) lane);
        const int pax1 = (int) (0x22222222u + (uint32_t) lane);
        const int pax2 = (int) (0x33333333u + (uint32_t) lane);
        const int pax3 = (int) (0x44444444u + (uint32_t) lane);
        const uint32_t p_scale = pv_scales[lane & 255];

        const auto accumulate_pv = [&](int out_frag, int bx0, int bx1, uint32_t b_scale) {
            asm volatile(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                "%10, {0, 0}, %11, {0, 0};"
                : "+f"(pv0), "+f"(pv1), "+f"(pv2), "+f"(pv3)
                : "r"(pax0), "r"(pax1), "r"(pax2), "r"(pax3), "r"(bx0), "r"(bx1), "r"(p_scale), "r"(b_scale));

            float mix = (float) (out_frag + 1) * 0.000001f;
            if constexpr (attentionish) {
                const float norm = 1.0f / (row_l0 + row_l1 + 0.000001f);
                mix *= norm;
            }
            out0 += pv0 * mix;
            out1 += pv1 * mix;
            out2 += pv2 * mix;
            out3 += pv3 * mix;
            out4 += (pv0 + row_m0) * mix;
            out5 += (pv1 + row_l0) * mix;
            out6 += (pv2 + row_m1) * mix;
            out7 += (pv3 + row_l1) * mix;
        };

#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const size_t q_base = ((size_t) warp * (size_t) nfrags * (size_t) KQ_MMA_A_ROWS) +
                                  ((size_t) frag * (size_t) KQ_MMA_A_ROWS) +
                                  ((size_t) i * (size_t) gridDim.x * (size_t) nfrags * (size_t) KQ_MMA_A_ROWS);

            if (lane < KQ_MMA_A_ROWS) {
                const block_nvfp4 q_blk = q[(q_base + (size_t) lane) & block_mask];
                const uint32_t * q_qs = reinterpret_cast<const uint32_t *>(q_blk.qs);
#pragma unroll
                for (int c = 0; c < KQ_MMA_WORDS; ++c) {
                    q_s[lane * KQ_MMA_A_STRIDE + c] = (int) q_qs[c];
                }
                q_sc[lane] = ((uint32_t) q_blk.d[0]) | ((uint32_t) q_blk.d[1] << 8) | ((uint32_t) q_blk.d[2] << 16) | ((uint32_t) q_blk.d[3] << 24);
            }

            __syncwarp(mask);

            int ax0;
            int ax1;
            int ax2;
            int ax3;
            const int * a_ptr = q_s + (lane % KQ_MMA_A_ROWS) * KQ_MMA_A_STRIDE + (lane / KQ_MMA_A_ROWS) * 4;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                : "=r"(ax0), "=r"(ax1), "=r"(ax2), "=r"(ax3)
                : "l"(a_ptr));

            const int tidx_a = lane / 4 + (lane % 2) * 8;
            const uint32_t q_scale = q_sc[tidx_a];

#pragma unroll
            for (int kr = 0; kr < k_reuse; ++kr) {
                const size_t k_base = ((size_t) warp * (size_t) nfrags * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) frag * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) kr * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) i * (size_t) gridDim.x * (size_t) nfrags * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS);

                uint32_t kv0 = 0;
                uint32_t kv1 = 0;
                uint32_t kv2 = 0;
                uint32_t kv3 = 0;
                uint32_t kv4 = 0;
                uint32_t kv5 = 0;
                uint32_t kv6 = 0;
                uint32_t kv7 = 0;
                uint32_t k_scale_row = 0;

                if (lane < KQ_MMA_B_ROWS) {
                    const block_nvfp4 k_blk = k[(k_base + (size_t) lane) & block_mask];
                    const uint32_t * k_qs = reinterpret_cast<const uint32_t *>(k_blk.qs);
                    kv0 = k_qs[0];
                    kv1 = k_qs[1];
                    kv2 = k_qs[2];
                    kv3 = k_qs[3];
                    kv4 = k_qs[4];
                    kv5 = k_qs[5];
                    kv6 = k_qs[6];
                    kv7 = k_qs[7];
                    k_scale_row = ((uint32_t) k_blk.d[0]) | ((uint32_t) k_blk.d[1] << 8) | ((uint32_t) k_blk.d[2] << 16) | ((uint32_t) k_blk.d[3] << 24);

                    const int stage_row = kr * KQ_MMA_B_ROWS + lane;
                    k_stage[stage_row * KQ_MMA_WORDS + 0] = (int) kv0;
                    k_stage[stage_row * KQ_MMA_WORDS + 1] = (int) kv1;
                    k_stage[stage_row * KQ_MMA_WORDS + 2] = (int) kv2;
                    k_stage[stage_row * KQ_MMA_WORDS + 3] = (int) kv3;
                    k_stage[stage_row * KQ_MMA_WORDS + 4] = (int) kv4;
                    k_stage[stage_row * KQ_MMA_WORDS + 5] = (int) kv5;
                    k_stage[stage_row * KQ_MMA_WORDS + 6] = (int) kv6;
                    k_stage[stage_row * KQ_MMA_WORDS + 7] = (int) kv7;
                    k_stage_sc[stage_row] = k_scale_row;
                }

                const int b_i = lane / 4;
                const int b_j0 = lane % 4;
                const int row_v0 = (int) __shfl_sync(mask, kv0, b_i);
                const int row_v1 = (int) __shfl_sync(mask, kv1, b_i);
                const int row_v2 = (int) __shfl_sync(mask, kv2, b_i);
                const int row_v3 = (int) __shfl_sync(mask, kv3, b_i);
                const int row_v4 = (int) __shfl_sync(mask, kv4, b_i);
                const int row_v5 = (int) __shfl_sync(mask, kv5, b_i);
                const int row_v6 = (int) __shfl_sync(mask, kv6, b_i);
                const int row_v7 = (int) __shfl_sync(mask, kv7, b_i);

                const int bx0 = b_j0 == 0 ? row_v0 : (b_j0 == 1 ? row_v1 : (b_j0 == 2 ? row_v2 : row_v3));
                const int bx1 = b_j0 == 0 ? row_v4 : (b_j0 == 1 ? row_v5 : (b_j0 == 2 ? row_v6 : row_v7));
                const uint32_t k_scale = __shfl_sync(mask, k_scale_row, b_i);

                asm volatile(
                    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                    "%10, {0, 0}, %11, {0, 0};"
                    : "+f"(kq0), "+f"(kq1), "+f"(kq2), "+f"(kq3)
                    : "r"(ax0), "r"(ax1), "r"(ax2), "r"(ax3), "r"(bx0), "r"(bx1), "r"(q_scale), "r"(k_scale));

                if constexpr (attentionish) {
                    const int q_row = lane & (KQ_MMA_A_ROWS - 1);
                    const int key_pos = frag * k_reuse + kr;
                    const int base_pos = (int) (i & 1023u);
                    const bool causal = key_pos <= base_pos + q_row;
                    const bool swa = key_pos + 1024 >= base_pos + q_row;

                    const float score0 = (causal && swa) ? kq0 * 0.000244140625f : -64.0f;
                    const float score1 = (causal && swa) ? kq2 * 0.000244140625f : -64.0f;
                    const float next_m0 = fmaxf(row_m0, score0);
                    const float next_m1 = fmaxf(row_m1, score1);
                    const float alpha0 = row_l0 == 0.0f ? 0.0f : exp2f(row_m0 - next_m0);
                    const float alpha1 = row_l1 == 0.0f ? 0.0f : exp2f(row_m1 - next_m1);
                    const float beta0 = exp2f(score0 - next_m0);
                    const float beta1 = exp2f(score1 - next_m1);
                    row_l0 = row_l0 * alpha0 + beta0;
                    row_l1 = row_l1 * alpha1 + beta1;
                    row_m0 = next_m0;
                    row_m1 = next_m1;
                } else {
                    row_m0 = fmaxf(row_m0, kq0);
                    row_m1 = fmaxf(row_m1, kq2);
                    row_l0 += fabsf(kq1) * 0.000001f + 1.0f;
                    row_l1 += fabsf(kq3) * 0.000001f + 1.0f;
                }
            }

            __syncwarp(mask);

#pragma unroll
            for (int group = 0; group < KQ_MMA_WORDS; ++group) {
                const int row0 = 2 * lane + 0;
                const int row1 = 2 * lane + 1;
                const int bx0 = k_stage[row0 * KQ_MMA_WORDS + group];
                const int bx1 = k_stage[row1 * KQ_MMA_WORDS + group];
                const uint32_t b_scale = k_stage_sc[row0];
                accumulate_pv(frag * KQ_MMA_WORDS + group, bx0, bx1, b_scale);
            }
        }
    }

    if (lane == 0) {
        out[warp] = kq0 + kq1 + kq2 + kq3 + pv0 + pv1 + pv2 + pv3 +
                    row_m0 + row_m1 + row_l0 + row_l1 +
                    out0 + out1 + out2 + out3 + out4 + out5 + out6 + out7;
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = 0.0f;
    }
    (void) q;
    (void) k;
    (void) pv_scales;
    (void) block_mask;
    (void) out;
    (void) iters;
#endif
}

template <int dkq, int k_reuse, bool attentionish = true>
static void run_combined_kq_kreusepv_smem_bench(const bench_config & cfg, int cc, const char * label) {
    static_assert(dkq == 512, "K-reuse PV shared-stage diagnostic currently models only D=512");
    static_assert(k_reuse == 8, "K-reuse PV shared-stage diagnostic currently models reuse 8");
    constexpr int nfrags = dkq / QK_NVFP4;
    constexpr int n_out_frags = dkq / 8;

    if (cc < 1200 || cc >= 1300) {
        std::printf("%s: skipped  reason=requires Blackwell sm_120/sm_121 device\n", label);
        return;
    }

    const size_t n_threads = (size_t) cfg.blocks * cfg.threads;
    const size_t n_warps = n_threads / 32;
    const size_t min_blocks = n_warps * (size_t) nfrags * (size_t) (KQ_MMA_A_ROWS + k_reuse * KQ_MMA_B_ROWS);
    const size_t cfg_blocks = floor_power_of_two(cfg.bytes / sizeof(block_nvfp4));
    size_t n_blocks = 4096;
    while (n_blocks < min_blocks || n_blocks < cfg_blocks) {
        n_blocks <<= 1;
    }

    const size_t shared_bytes = (size_t) (cfg.threads / 32) * (size_t) (KQ_MMA_Q_SHARED_INTS + KREUSE_STAGE_INTS) * sizeof(int);
    print_occupancy_line(cfg, label, combined_kq_kreusepv_smem_nvfp4_kernel<nfrags, k_reuse, attentionish>, shared_bytes);

    block_nvfp4 * q = nullptr;
    block_nvfp4 * k = nullptr;
    uint32_t * pv_scales = nullptr;
    float * out = nullptr;
    CUDA_CHECK(cudaMalloc(&q, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&k, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&pv_scales, 256 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&out, n_warps * sizeof(float)));

    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(q, n_blocks);
    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(k, n_blocks);
    fill_u32_kernel<<<cfg.blocks, cfg.threads>>>(pv_scales, 256);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    combined_kq_kreusepv_smem_nvfp4_kernel<nfrags, k_reuse, attentionish><<<cfg.blocks, cfg.threads, shared_bytes>>>(
        q, k, pv_scales, n_blocks - 1, out, 10);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    combined_kq_kreusepv_smem_nvfp4_kernel<nfrags, k_reuse, attentionish><<<cfg.blocks, cfg.threads, shared_bytes>>>(
        q, k, pv_scales, n_blocks - 1, out, cfg.mma_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float ms = time_events(start, stop);

    const double kq_ops = (double) n_warps * (double) cfg.mma_iters * (double) nfrags * (double) k_reuse * 2.0 * 16.0 * 8.0 * 64.0;
    const double pv_ops = (double) n_warps * (double) cfg.mma_iters * (double) n_out_frags * 2.0 * 16.0 * 8.0 * 64.0;
    const double total_ops = kq_ops + pv_ops;
    const double seconds = ms / 1000.0;
    const double kq_compact_read_gb = (double) n_warps * (double) cfg.mma_iters * (double) nfrags *
                                   (double) (KQ_MMA_A_ROWS + k_reuse * KQ_MMA_B_ROWS) * (double) sizeof(block_nvfp4) / 1.0e9;
    const double shared_stage_bytes = (double) n_warps * (double) cfg.mma_iters * (double) nfrags *
                                      (double) (KREUSE_STAGE_WORD_INTS + KREUSE_STAGE_SCALE_INTS) * (double) sizeof(int);
    const double shared_pv_read_bytes = shared_stage_bytes;

    std::printf("%s: %.3f total-TOPS  %.3f useful-mtp-total-TOPS  %.3f KQ-TOPS  %.3f packed-PV-TOPS  %.3f GB/s-kq-compact-read  %.3f GB/s-pv-extra-compact-read  %.3f GB/s-shared-stage-write  %.3f GB/s-shared-pv-read  attentionish=%d k_reuse_smem=1 shared=%.3f KiB blocks=%zu warps=%zu mma_iters=%" PRIu64 " time=%.3f ms\n",
                label,
                total_ops / seconds / 1.0e12,
                total_ops / seconds / 1.0e12 * useful_mtp_fraction(cfg),
                kq_ops / seconds / 1.0e12,
                pv_ops / seconds / 1.0e12,
                kq_compact_read_gb / seconds,
                0.0,
                shared_stage_bytes / 1.0e9 / seconds,
                shared_pv_read_bytes / 1.0e9 / seconds,
                attentionish ? 1 : 0,
                (double) shared_bytes / 1024.0,
                n_blocks,
                n_warps,
                cfg.mma_iters,
                ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(pv_scales));
    CUDA_CHECK(cudaFree(k));
    CUDA_CHECK(cudaFree(q));
}

template <int nfrags, int k_reuse, int staged_groups, bool attentionish>
__global__ void combined_kq_kreusepv_smem_groups_nvfp4_kernel(
        const block_nvfp4 * q,
        const block_nvfp4 * k,
        const uint32_t *    pv_scales,
        size_t              block_mask,
        float *             out,
        uint64_t            iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    static_assert(nfrags == 8, "K-reuse grouped shared-stage diagnostic currently models only D=512");
    static_assert(k_reuse == 8, "K-reuse grouped shared-stage diagnostic currently models reuse 8");
    static_assert(staged_groups == 1 || staged_groups == 2 || staged_groups == 4, "unsupported staged group count");
    extern __shared__ int smem_i[];

    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warp = ((int) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const unsigned mask = 0xffffffffu;

    constexpr int stage_word_ints = KQ_MMA_B_ROWS * 8 * staged_groups;
    constexpr int stage_ints = stage_word_ints + KREUSE_STAGE_SCALE_INTS;

    int * warp_s = smem_i + warp_in_block * (KQ_MMA_Q_SHARED_INTS + stage_ints);
    int * q_s = warp_s;
    uint32_t * q_sc = reinterpret_cast<uint32_t *>(q_s + KQ_MMA_A_ROWS * KQ_MMA_A_STRIDE);
    int * k_stage = warp_s + KQ_MMA_Q_SHARED_INTS;
    uint32_t * k_stage_sc = reinterpret_cast<uint32_t *>(k_stage + stage_word_ints);

    float kq0 = 0.0f;
    float kq1 = 0.0f;
    float kq2 = 0.0f;
    float kq3 = 0.0f;
    float pv0 = 0.0f;
    float pv1 = 0.0f;
    float pv2 = 0.0f;
    float pv3 = 0.0f;
    float row_m0 = -64.0f;
    float row_m1 = -64.0f;
    float row_l0 = 0.0f;
    float row_l1 = 0.0f;
    float out0 = 0.0f;
    float out1 = 0.0f;
    float out2 = 0.0f;
    float out3 = 0.0f;

    for (uint64_t i = 0; i < iters; ++i) {
        const int pax0 = (int) (0x11111111u + (uint32_t) lane);
        const int pax1 = (int) (0x22222222u + (uint32_t) lane);
        const int pax2 = (int) (0x33333333u + (uint32_t) lane);
        const int pax3 = (int) (0x44444444u + (uint32_t) lane);
        const uint32_t p_scale = pv_scales[lane & 255];

#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const size_t q_base = ((size_t) warp * (size_t) nfrags * (size_t) KQ_MMA_A_ROWS) +
                                  ((size_t) frag * (size_t) KQ_MMA_A_ROWS) +
                                  ((size_t) i * (size_t) gridDim.x * (size_t) nfrags * (size_t) KQ_MMA_A_ROWS);

            if (lane < KQ_MMA_A_ROWS) {
                const block_nvfp4 q_blk = q[(q_base + (size_t) lane) & block_mask];
                const uint32_t * q_qs = reinterpret_cast<const uint32_t *>(q_blk.qs);
#pragma unroll
                for (int c = 0; c < KQ_MMA_WORDS; ++c) {
                    q_s[lane * KQ_MMA_A_STRIDE + c] = (int) q_qs[c];
                }
                q_sc[lane] = ((uint32_t) q_blk.d[0]) | ((uint32_t) q_blk.d[1] << 8) | ((uint32_t) q_blk.d[2] << 16) | ((uint32_t) q_blk.d[3] << 24);
            }

            __syncwarp(mask);

            int ax0;
            int ax1;
            int ax2;
            int ax3;
            const int * a_ptr = q_s + (lane % KQ_MMA_A_ROWS) * KQ_MMA_A_STRIDE + (lane / KQ_MMA_A_ROWS) * 4;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                : "=r"(ax0), "=r"(ax1), "=r"(ax2), "=r"(ax3)
                : "l"(a_ptr));

            const int tidx_a = lane / 4 + (lane % 2) * 8;
            const uint32_t q_scale = q_sc[tidx_a];

#pragma unroll
            for (int kr = 0; kr < k_reuse; ++kr) {
                const size_t k_base = ((size_t) warp * (size_t) nfrags * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) frag * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) kr * (size_t) KQ_MMA_B_ROWS) +
                                      ((size_t) i * (size_t) gridDim.x * (size_t) nfrags * (size_t) k_reuse * (size_t) KQ_MMA_B_ROWS);

                uint32_t kv0 = 0;
                uint32_t kv1 = 0;
                uint32_t kv2 = 0;
                uint32_t kv3 = 0;
                uint32_t kv4 = 0;
                uint32_t kv5 = 0;
                uint32_t kv6 = 0;
                uint32_t kv7 = 0;
                uint32_t k_scale_row = 0;

                if (lane < KQ_MMA_B_ROWS) {
                    const block_nvfp4 k_blk = k[(k_base + (size_t) lane) & block_mask];
                    const uint32_t * k_qs = reinterpret_cast<const uint32_t *>(k_blk.qs);
                    kv0 = k_qs[0];
                    kv1 = k_qs[1];
                    kv2 = k_qs[2];
                    kv3 = k_qs[3];
                    kv4 = k_qs[4];
                    kv5 = k_qs[5];
                    kv6 = k_qs[6];
                    kv7 = k_qs[7];
                    k_scale_row = ((uint32_t) k_blk.d[0]) | ((uint32_t) k_blk.d[1] << 8) | ((uint32_t) k_blk.d[2] << 16) | ((uint32_t) k_blk.d[3] << 24);

                    const int stage_row = kr * KQ_MMA_B_ROWS + lane;
#pragma unroll
                    for (int group = 0; group < staged_groups; ++group) {
                        const uint32_t kv =
                            group == 0 ? kv0 :
                            group == 1 ? kv1 :
                            group == 2 ? kv2 : kv3;
                        k_stage[stage_row * staged_groups + group] = (int) kv;
                    }
                    k_stage_sc[stage_row] = k_scale_row;
                }

                const int b_i = lane / 4;
                const int b_j0 = lane % 4;
                const int row_v0 = (int) __shfl_sync(mask, kv0, b_i);
                const int row_v1 = (int) __shfl_sync(mask, kv1, b_i);
                const int row_v2 = (int) __shfl_sync(mask, kv2, b_i);
                const int row_v3 = (int) __shfl_sync(mask, kv3, b_i);
                const int row_v4 = (int) __shfl_sync(mask, kv4, b_i);
                const int row_v5 = (int) __shfl_sync(mask, kv5, b_i);
                const int row_v6 = (int) __shfl_sync(mask, kv6, b_i);
                const int row_v7 = (int) __shfl_sync(mask, kv7, b_i);

                const int bx0 = b_j0 == 0 ? row_v0 : (b_j0 == 1 ? row_v1 : (b_j0 == 2 ? row_v2 : row_v3));
                const int bx1 = b_j0 == 0 ? row_v4 : (b_j0 == 1 ? row_v5 : (b_j0 == 2 ? row_v6 : row_v7));
                const uint32_t k_scale = __shfl_sync(mask, k_scale_row, b_i);

                asm volatile(
                    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                    "%10, {0, 0}, %11, {0, 0};"
                    : "+f"(kq0), "+f"(kq1), "+f"(kq2), "+f"(kq3)
                    : "r"(ax0), "r"(ax1), "r"(ax2), "r"(ax3), "r"(bx0), "r"(bx1), "r"(q_scale), "r"(k_scale));

                if constexpr (attentionish) {
                    const int q_row = lane & (KQ_MMA_A_ROWS - 1);
                    const int key_pos = frag * k_reuse + kr;
                    const int base_pos = (int) (i & 1023u);
                    const bool causal = key_pos <= base_pos + q_row;
                    const bool swa = key_pos + 1024 >= base_pos + q_row;

                    const float score0 = (causal && swa) ? kq0 * 0.000244140625f : -64.0f;
                    const float score1 = (causal && swa) ? kq2 * 0.000244140625f : -64.0f;
                    const float next_m0 = fmaxf(row_m0, score0);
                    const float next_m1 = fmaxf(row_m1, score1);
                    const float alpha0 = row_l0 == 0.0f ? 0.0f : exp2f(row_m0 - next_m0);
                    const float alpha1 = row_l1 == 0.0f ? 0.0f : exp2f(row_m1 - next_m1);
                    const float beta0 = exp2f(score0 - next_m0);
                    const float beta1 = exp2f(score1 - next_m1);
                    row_l0 = row_l0 * alpha0 + beta0;
                    row_l1 = row_l1 * alpha1 + beta1;
                    row_m0 = next_m0;
                    row_m1 = next_m1;
                } else {
                    row_m0 = fmaxf(row_m0, kq0);
                    row_m1 = fmaxf(row_m1, kq2);
                    row_l0 += fabsf(kq1) * 0.000001f + 1.0f;
                    row_l1 += fabsf(kq3) * 0.000001f + 1.0f;
                }
            }

            __syncwarp(mask);

            const int row0 = 2 * lane + 0;
            const int row1 = 2 * lane + 1;
            const uint32_t b_scale = k_stage_sc[row0];
#pragma unroll
            for (int group = 0; group < staged_groups; ++group) {
                const int bx0 = k_stage[row0 * staged_groups + group];
                const int bx1 = k_stage[row1 * staged_groups + group];
                asm volatile(
                    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                    "%10, {0, 0}, %11, {0, 0};"
                    : "+f"(pv0), "+f"(pv1), "+f"(pv2), "+f"(pv3)
                    : "r"(pax0), "r"(pax1), "r"(pax2), "r"(pax3), "r"(bx0), "r"(bx1), "r"(p_scale), "r"(b_scale));

                const float mix = (float) (frag * staged_groups + group + 1) * 0.000001f;
                out0 += pv0 * mix;
                out1 += pv1 * mix;
                out2 += pv2 * mix;
                out3 += pv3 * mix;
            }
        }
    }

    if (lane == 0) {
        out[warp] = kq0 + kq1 + kq2 + kq3 + pv0 + pv1 + pv2 + pv3 +
                    row_m0 + row_m1 + row_l0 + row_l1 + out0 + out1 + out2 + out3;
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = 0.0f;
    }
    (void) q;
    (void) k;
    (void) pv_scales;
    (void) block_mask;
    (void) out;
    (void) iters;
#endif
}

template <int dkq, int k_reuse, int staged_groups, bool attentionish = true>
static void run_combined_kq_kreusepv_smem_groups_bench(const bench_config & cfg, int cc, const char * label) {
    static_assert(dkq == 512, "K-reuse grouped shared-stage diagnostic currently models only D=512");
    static_assert(k_reuse == 8, "K-reuse grouped shared-stage diagnostic currently models reuse 8");
    static_assert(staged_groups == 1 || staged_groups == 2 || staged_groups == 4, "unsupported staged group count");
    constexpr int nfrags = dkq / QK_NVFP4;
    constexpr int measured_pv_frags = nfrags * staged_groups;
    constexpr int full_pv_frags = dkq / 8;
    constexpr int stage_word_ints = KQ_MMA_B_ROWS * 8 * staged_groups;
    constexpr int stage_ints = stage_word_ints + KREUSE_STAGE_SCALE_INTS;

    if (cc < 1200 || cc >= 1300) {
        std::printf("%s: skipped  reason=requires Blackwell sm_120/sm_121 device\n", label);
        return;
    }

    const size_t n_threads = (size_t) cfg.blocks * cfg.threads;
    const size_t n_warps = n_threads / 32;
    const size_t min_blocks = n_warps * (size_t) nfrags * (size_t) (KQ_MMA_A_ROWS + k_reuse * KQ_MMA_B_ROWS);
    const size_t cfg_blocks = floor_power_of_two(cfg.bytes / sizeof(block_nvfp4));
    size_t n_blocks = 4096;
    while (n_blocks < min_blocks || n_blocks < cfg_blocks) {
        n_blocks <<= 1;
    }

    const size_t shared_bytes = (size_t) (cfg.threads / 32) * (size_t) (KQ_MMA_Q_SHARED_INTS + stage_ints) * sizeof(int);
    print_occupancy_line(cfg, label, combined_kq_kreusepv_smem_groups_nvfp4_kernel<nfrags, k_reuse, staged_groups, attentionish>, shared_bytes);

    block_nvfp4 * q = nullptr;
    block_nvfp4 * k = nullptr;
    uint32_t * pv_scales = nullptr;
    float * out = nullptr;
    CUDA_CHECK(cudaMalloc(&q, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&k, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&pv_scales, 256 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&out, n_warps * sizeof(float)));

    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(q, n_blocks);
    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(k, n_blocks);
    fill_u32_kernel<<<cfg.blocks, cfg.threads>>>(pv_scales, 256);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    combined_kq_kreusepv_smem_groups_nvfp4_kernel<nfrags, k_reuse, staged_groups, attentionish><<<cfg.blocks, cfg.threads, shared_bytes>>>(
        q, k, pv_scales, n_blocks - 1, out, 10);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    combined_kq_kreusepv_smem_groups_nvfp4_kernel<nfrags, k_reuse, staged_groups, attentionish><<<cfg.blocks, cfg.threads, shared_bytes>>>(
        q, k, pv_scales, n_blocks - 1, out, cfg.mma_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float ms = time_events(start, stop);

    const double kq_ops = (double) n_warps * (double) cfg.mma_iters * (double) nfrags * (double) k_reuse * 2.0 * 16.0 * 8.0 * 64.0;
    const double measured_pv_ops = (double) n_warps * (double) cfg.mma_iters * (double) measured_pv_frags * 2.0 * 16.0 * 8.0 * 64.0;
    const double full_pv_ops = (double) n_warps * (double) cfg.mma_iters * (double) full_pv_frags * 2.0 * 16.0 * 8.0 * 64.0;
    const double measured_ops = kq_ops + measured_pv_ops;
    const double seconds = ms / 1000.0;
    const double kq_compact_read_gb = (double) n_warps * (double) cfg.mma_iters * (double) nfrags *
                                   (double) (KQ_MMA_A_ROWS + k_reuse * KQ_MMA_B_ROWS) * (double) sizeof(block_nvfp4) / 1.0e9;
    const double shared_stage_bytes = (double) n_warps * (double) cfg.mma_iters * (double) nfrags *
                                      (double) (stage_word_ints + KREUSE_STAGE_SCALE_INTS) * (double) sizeof(int);
    const double pv_group_fraction = (double) measured_pv_frags / (double) full_pv_frags;

    std::printf("%s: %.3f measured-total-TOPS  %.3f useful-mtp-measured-total-TOPS  %.3f KQ-TOPS  %.3f measured-subPV-TOPS  %.3f projected-fullPV-TOPS-at-same-time  %.3f GB/s-kq-compact-read  %.3f GB/s-pv-extra-compact-read  %.3f GB/s-shared-stage-write  %.3f GB/s-shared-pv-read  pv_group_fraction=%.3f staged_groups=%d attentionish=%d k_reuse_smem_groups=%d shared=%.3f KiB blocks=%zu warps=%zu mma_iters=%" PRIu64 " time=%.3f ms\n",
                label,
                measured_ops / seconds / 1.0e12,
                measured_ops / seconds / 1.0e12 * useful_mtp_fraction(cfg),
                kq_ops / seconds / 1.0e12,
                measured_pv_ops / seconds / 1.0e12,
                full_pv_ops / seconds / 1.0e12,
                kq_compact_read_gb / seconds,
                0.0,
                shared_stage_bytes / 1.0e9 / seconds,
                shared_stage_bytes / 1.0e9 / seconds,
                pv_group_fraction,
                staged_groups,
                attentionish ? 1 : 0,
                staged_groups,
                (double) shared_bytes / 1024.0,
                n_blocks,
                n_warps,
                cfg.mma_iters,
                ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(pv_scales));
    CUDA_CHECK(cudaFree(k));
    CUDA_CHECK(cudaFree(q));
}

template <int dkq, int k_reuse>
static void run_kq_ldmatrix_qreg_kpair_mma_bench(const bench_config & cfg, int cc, const char * label) {
    static_assert(dkq == 256 || dkq == 512, "unsupported ldmatrix KQ width");
    constexpr int nfrags = dkq / QK_NVFP4;

    if (cc < 1200 || cc >= 1300) {
        std::printf("%s: skipped  reason=requires Blackwell sm_120/sm_121 device\n", label);
        return;
    }

    if (cfg.threads % 32 != 0) {
        std::printf("%s: skipped  reason=threads must be a multiple of 32\n", label);
        return;
    }

    const size_t n_threads = (size_t) cfg.blocks * cfg.threads;
    const size_t n_warps = n_threads / 32;
    size_t n_blocks = floor_power_of_two(cfg.bytes / sizeof(block_nvfp4));
    const size_t min_blocks = n_warps * (size_t) nfrags * (size_t) (KQ_MMA_A_ROWS + k_reuse * KQ_MMA_B_ROWS);
    if (n_blocks < min_blocks) {
        n_blocks = floor_power_of_two(min_blocks);
    }
    if (n_blocks < 4096) {
        n_blocks = 4096;
    }

    const size_t shared_bytes = (size_t) (cfg.threads / 32) * (size_t) KQ_MMA_Q_SHARED_INTS * sizeof(int);
    print_occupancy_line(cfg, label, kq_mma_ldmatrix_qreg_kpair_nvfp4_kernel<nfrags, k_reuse>, shared_bytes);

    block_nvfp4 * q = nullptr;
    block_nvfp4 * k = nullptr;
    float * out = nullptr;
    CUDA_CHECK(cudaMalloc(&q, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&k, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&out, n_warps * sizeof(float)));

    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(q, n_blocks);
    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(k, n_blocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    kq_mma_ldmatrix_qreg_kpair_nvfp4_kernel<nfrags, k_reuse><<<cfg.blocks, cfg.threads, shared_bytes>>>(q, k, n_blocks - 1, out, 10);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    kq_mma_ldmatrix_qreg_kpair_nvfp4_kernel<nfrags, k_reuse><<<cfg.blocks, cfg.threads, shared_bytes>>>(q, k, n_blocks - 1, out, cfg.mma_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float ms = time_events(start, stop);

    const double dense_ops = (double) n_warps * (double) cfg.mma_iters * (double) nfrags * (double) k_reuse * 2.0 * 16.0 * 8.0 * 64.0;
    const double dense_tops = dense_ops / (ms / 1000.0) / 1.0e12;
    const double compact_read_gb = (double) n_warps * (double) cfg.mma_iters * (double) nfrags *
                                   (double) (KQ_MMA_A_ROWS + k_reuse * KQ_MMA_B_ROWS) * (double) sizeof(block_nvfp4) / 1.0e9;
    std::printf("%s: %.3f TOPS  %.3f useful-mtp-TOPS  %.3f GB/s-compact-read  k_reuse=%d blocks=%zu shared=%.3f KiB warps=%zu mma_iters=%" PRIu64 " time=%.3f ms\n",
                label,
                dense_tops,
                dense_tops * useful_mtp_fraction(cfg),
                compact_read_gb / (ms / 1000.0),
                k_reuse,
                n_blocks,
                (double) shared_bytes / 1024.0,
                n_warps,
                cfg.mma_iters,
                ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(k));
    CUDA_CHECK(cudaFree(q));
}

template <int dkq, int v_rows = PV_TILE_ROWS>
static void run_pv_dequant_v_eq_k_bench(const bench_config & cfg, int cc, const char * label) {
    static_assert(dkq == 256 || dkq == 512, "unsupported PV width");
    static_assert(v_rows == 2 || v_rows == 4 || v_rows == PV_TILE_ROWS, "unsupported PV tile rows");
    constexpr int nfrags = dkq / QK_NVFP4;
    constexpr int blocks_per_tile = v_rows * nfrags;

    if (cc < 1200 || cc >= 1300) {
        std::printf("%s: skipped  reason=requires Blackwell sm_120/sm_121 device\n", label);
        return;
    }

    if (cfg.threads % 32 != 0) {
        std::printf("%s: skipped  reason=threads must be a multiple of 32\n", label);
        return;
    }

    const size_t n_threads = (size_t) cfg.blocks * cfg.threads;
    const size_t n_warps = n_threads / 32;
    size_t n_blocks = floor_power_of_two(cfg.bytes / sizeof(block_nvfp4));
    const size_t min_blocks = n_warps * (size_t) blocks_per_tile;
    if (n_blocks < min_blocks) {
        n_blocks = floor_power_of_two(min_blocks);
    }
    if (n_blocks < 4096) {
        n_blocks = 4096;
    }

    print_occupancy_line(cfg, label, pv_dequant_v_eq_k_nvfp4_kernel<dkq, v_rows>, 0);

    block_nvfp4 * v = nullptr;
    float * out = nullptr;
    CUDA_CHECK(cudaMalloc(&v, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&out, n_warps * sizeof(float)));

    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(v, n_blocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    pv_dequant_v_eq_k_nvfp4_kernel<dkq, v_rows><<<cfg.blocks, cfg.threads>>>(v, n_blocks - 1, out, 10);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    pv_dequant_v_eq_k_nvfp4_kernel<dkq, v_rows><<<cfg.blocks, cfg.threads>>>(v, n_blocks - 1, out, cfg.mma_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float ms = time_events(start, stop);

    const double compact_read_gb = (double) n_warps * (double) cfg.mma_iters * (double) blocks_per_tile * (double) sizeof(block_nvfp4) / 1.0e9;
    const double model_pv_ops = (double) n_warps * (double) cfg.mma_iters * 2.0 * (double) KQ_MMA_A_ROWS * (double) v_rows * (double) dkq;
    const double model_pv_tops = model_pv_ops / (ms / 1000.0) / 1.0e12;
    const size_t tile_bytes = (size_t) blocks_per_tile * sizeof(block_nvfp4);
    std::printf("%s: %.3f GB/s-v-eq-k-dequant-read  %.3f model-pv-TOPS  %.3f useful-mtp-model-pv-TOPS  tile_bytes=%zu v_rows=%d blocks=%zu warps=%zu mma_iters=%" PRIu64 " time=%.3f ms\n",
                label,
                compact_read_gb / (ms / 1000.0),
                model_pv_tops,
                model_pv_tops * useful_mtp_fraction(cfg),
                tile_bytes,
                v_rows,
                n_blocks,
                n_warps,
                cfg.mma_iters,
                ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(v));
}

template <int dkq>
static void run_pv_fp4_mma_native_bench(const bench_config & cfg, int cc, const char * label) {
    static_assert(dkq == 256 || dkq == 512, "unsupported native FP4 PV width");
    constexpr int n_out_frags = dkq / 8;

    if (cc < 1200 || cc >= 1300) {
        std::printf("%s: skipped  reason=requires Blackwell sm_120/sm_121 device\n", label);
        return;
    }

    const size_t n_threads = (size_t) cfg.blocks * cfg.threads;
    const size_t n_warps = n_threads / 32;
    int * a = nullptr;
    int * b = nullptr;
    uint32_t * scales = nullptr;
    float * out = nullptr;
    CUDA_CHECK(cudaMalloc(&a, 1024 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&b, 4096 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&scales, 256 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&out, n_warps * sizeof(float)));

    fill_u32_kernel<<<cfg.blocks, cfg.threads>>>((uint32_t *) a, 1024);
    fill_u32_kernel<<<cfg.blocks, cfg.threads>>>((uint32_t *) b, 4096);
    fill_u32_kernel<<<cfg.blocks, cfg.threads>>>(scales, 256);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    pv_fp4_mma_native_kernel<dkq><<<cfg.blocks, cfg.threads>>>(a, b, scales, out, 10);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    pv_fp4_mma_native_kernel<dkq><<<cfg.blocks, cfg.threads>>>(a, b, scales, out, cfg.mma_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float ms = time_events(start, stop);

    const double dense_ops = (double) n_warps * (double) cfg.mma_iters * (double) n_out_frags * 2.0 * 16.0 * 8.0 * 64.0;
    const double dense_tops = dense_ops / (ms / 1000.0) / 1.0e12;
    std::printf("%s: %.3f native-fp4-pv-TOPS  %.3f useful-mtp-native-fp4-pv-TOPS  n_out_frags=%d warps=%zu mma_iters=%" PRIu64 " time=%.3f ms\n",
                label,
                dense_tops,
                dense_tops * useful_mtp_fraction(cfg),
                n_out_frags,
                n_warps,
                cfg.mma_iters,
                ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(scales));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(a));
}

template <int dkq>
static void run_pv_fp4_mma_compact_b_bench(const bench_config & cfg, int cc, const char * label) {
    static_assert(dkq == 256 || dkq == 512, "unsupported compact FP4 PV width");
    constexpr int nfrags = dkq / QK_NVFP4;
    constexpr int n_out_frags = dkq / 8;

    if (cc < 1200 || cc >= 1300) {
        std::printf("%s: skipped  reason=requires Blackwell sm_120/sm_121 device\n", label);
        return;
    }

    if (cfg.threads % 32 != 0) {
        std::printf("%s: skipped  reason=threads must be a multiple of 32\n", label);
        return;
    }

    const size_t n_threads = (size_t) cfg.blocks * cfg.threads;
    const size_t n_warps = n_threads / 32;
    size_t n_blocks = floor_power_of_two(cfg.bytes / sizeof(block_nvfp4));
    const size_t min_blocks = n_warps * (size_t) nfrags * 64u;
    if (n_blocks < min_blocks) {
        n_blocks = floor_power_of_two(min_blocks);
    }
    if (n_blocks < 4096) {
        n_blocks = 4096;
    }

    int * a = nullptr;
    block_nvfp4 * v = nullptr;
    uint32_t * scales = nullptr;
    float * out = nullptr;
    CUDA_CHECK(cudaMalloc(&a, 1024 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&v, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&scales, 256 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&out, n_warps * sizeof(float)));

    fill_u32_kernel<<<cfg.blocks, cfg.threads>>>((uint32_t *) a, 1024);
    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(v, n_blocks);
    fill_u32_kernel<<<cfg.blocks, cfg.threads>>>(scales, 256);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    pv_fp4_mma_compact_b_kernel<dkq><<<cfg.blocks, cfg.threads>>>(a, v, scales, n_blocks - 1, out, 10);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    pv_fp4_mma_compact_b_kernel<dkq><<<cfg.blocks, cfg.threads>>>(a, v, scales, n_blocks - 1, out, cfg.mma_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float ms = time_events(start, stop);

    const double dense_ops = (double) n_warps * (double) cfg.mma_iters * (double) n_out_frags * 2.0 * 16.0 * 8.0 * 64.0;
    const double dense_tops = dense_ops / (ms / 1000.0) / 1.0e12;
    const double compact_read_gb = (double) n_warps * (double) cfg.mma_iters * (double) nfrags * 64.0 * (double) sizeof(block_nvfp4) / 1.0e9;
    std::printf("%s: %.3f compact-fp4-pv-TOPS  %.3f useful-mtp-compact-fp4-pv-TOPS  %.3f GB/s-compact-v-read  n_out_frags=%d blocks=%zu warps=%zu mma_iters=%" PRIu64 " time=%.3f ms\n",
                label,
                dense_tops,
                dense_tops * useful_mtp_fraction(cfg),
                compact_read_gb / (ms / 1000.0),
                n_out_frags,
                n_blocks,
                n_warps,
                cfg.mma_iters,
                ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(scales));
    CUDA_CHECK(cudaFree(v));
    CUDA_CHECK(cudaFree(a));
}

template <int dkq>
static void run_pv_fp4_mma_packed_b_bench(const bench_config & cfg, int cc, const char * label) {
    static_assert(dkq == 256 || dkq == 512, "unsupported packed FP4 PV width");
    constexpr int nfrags = dkq / QK_NVFP4;
    constexpr int n_out_frags = dkq / 8;
    constexpr size_t stage_tile_entries = (size_t) n_out_frags * 32u;
    constexpr size_t stage_tile_size = stage_tile_entries * sizeof(pv_mma_b_lane_frag);

    if (cc < 1200 || cc >= 1300) {
        std::printf("%s: skipped  reason=requires Blackwell sm_120/sm_121 device\n", label);
        return;
    }

    if (cfg.threads % 32 != 0) {
        std::printf("%s: skipped  reason=threads must be a multiple of 32\n", label);
        return;
    }

    const size_t n_threads = (size_t) cfg.blocks * cfg.threads;
    const size_t n_warps = n_threads / 32;
    size_t n_tiles = floor_power_of_two(cfg.bytes / stage_tile_size);
    if (n_tiles < n_warps) {
        n_tiles = floor_power_of_two(n_warps);
    }
    if (n_tiles < 1) {
        n_tiles = 1;
    }

    const size_t n_blocks = n_tiles * (size_t) nfrags * 64u;
    const size_t stage_entries = n_tiles * stage_tile_entries;

    print_occupancy_line(cfg, label, pv_fp4_mma_packed_b_kernel<dkq>, 0);

    int * a = nullptr;
    block_nvfp4 * v = nullptr;
    pv_mma_b_lane_frag * stage = nullptr;
    uint32_t * scales = nullptr;
    float * out = nullptr;
    CUDA_CHECK(cudaMalloc(&a, 1024 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&v, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&stage, stage_entries * sizeof(pv_mma_b_lane_frag)));
    CUDA_CHECK(cudaMalloc(&scales, 256 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&out, n_warps * sizeof(float)));

    fill_u32_kernel<<<cfg.blocks, cfg.threads>>>((uint32_t *) a, 1024);
    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(v, n_blocks);
    fill_u32_kernel<<<cfg.blocks, cfg.threads>>>(scales, 256);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    fill_pv_mma_b_stage_kernel<dkq><<<cfg.blocks, cfg.threads>>>(v, stage, n_tiles, n_blocks - 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    fill_pv_mma_b_stage_reuse_kernel<dkq><<<cfg.blocks, cfg.threads>>>(v, stage, n_tiles, n_blocks - 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    pv_fp4_mma_packed_b_kernel<dkq><<<cfg.blocks, cfg.threads>>>(a, stage, scales, n_tiles - 1, out, 10);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t fill_start;
    cudaEvent_t fill_stop;
    cudaEvent_t fill_reuse_start;
    cudaEvent_t fill_reuse_stop;
    cudaEvent_t mma_start;
    cudaEvent_t mma_stop;
    CUDA_CHECK(cudaEventCreate(&fill_start));
    CUDA_CHECK(cudaEventCreate(&fill_stop));
    CUDA_CHECK(cudaEventCreate(&fill_reuse_start));
    CUDA_CHECK(cudaEventCreate(&fill_reuse_stop));
    CUDA_CHECK(cudaEventCreate(&mma_start));
    CUDA_CHECK(cudaEventCreate(&mma_stop));

    CUDA_CHECK(cudaEventRecord(fill_start));
    fill_pv_mma_b_stage_kernel<dkq><<<cfg.blocks, cfg.threads>>>(v, stage, n_tiles, n_blocks - 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(fill_stop));
    const float fill_ms = time_events(fill_start, fill_stop);

    CUDA_CHECK(cudaEventRecord(fill_reuse_start));
    fill_pv_mma_b_stage_reuse_kernel<dkq><<<cfg.blocks, cfg.threads>>>(v, stage, n_tiles, n_blocks - 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(fill_reuse_stop));
    const float fill_reuse_ms = time_events(fill_reuse_start, fill_reuse_stop);

    CUDA_CHECK(cudaEventRecord(mma_start));
    pv_fp4_mma_packed_b_kernel<dkq><<<cfg.blocks, cfg.threads>>>(a, stage, scales, n_tiles - 1, out, cfg.mma_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(mma_stop));
    const float mma_ms = time_events(mma_start, mma_stop);

    const double dense_ops = (double) n_warps * (double) cfg.mma_iters * (double) n_out_frags * 2.0 * 16.0 * 8.0 * 64.0;
    const double dense_tops = dense_ops / (mma_ms / 1000.0) / 1.0e12;
    const double compact_src_gb = (double) n_tiles * (double) nfrags * 64.0 * (double) sizeof(block_nvfp4) / 1.0e9;
    const double stage_write_gb = (double) n_tiles * (double) stage_tile_size / 1.0e9;
    const double stage_read_gb = (double) n_warps * (double) cfg.mma_iters * (double) stage_tile_size / 1.0e9;
    const double transform_tile_s = (fill_ms / 1000.0) / (double) n_tiles;
    const double transform_reuse_tile_s = (fill_reuse_ms / 1000.0) / (double) n_tiles;
    const double consume_tile_s = (mma_ms / 1000.0) / ((double) n_warps * (double) cfg.mma_iters);
    const double ops_per_tile = (double) n_out_frags * 2.0 * 16.0 * 8.0 * 64.0;
    const double transform_equiv_consumes = consume_tile_s > 0.0 ? transform_tile_s / consume_tile_s : 0.0;
    const double transform_reuse_equiv_consumes = consume_tile_s > 0.0 ? transform_reuse_tile_s / consume_tile_s : 0.0;
    const auto effective_tops = [&](double reuse) {
        return (reuse * ops_per_tile) / (transform_reuse_tile_s + reuse * consume_tile_s) / 1.0e12;
    };

    std::printf("%s_transform_naive: %.3f GB/s-compact-read  %.3f GB/s-stage-write  transform_equiv_consumes=%.3f  tiles=%zu compact_tile=%zu stage_tile=%zu expansion=%.3fx time=%.3f ms\n",
                label,
                compact_src_gb / (fill_ms / 1000.0),
                stage_write_gb / (fill_ms / 1000.0),
                transform_equiv_consumes,
                n_tiles,
                (size_t) nfrags * 64u * sizeof(block_nvfp4),
                stage_tile_size,
                (double) stage_tile_size / ((double) nfrags * 64.0 * (double) sizeof(block_nvfp4)),
                fill_ms);

    std::printf("%s_transform_reuse: %.3f GB/s-compact-read  %.3f GB/s-stage-write  transform_equiv_consumes=%.3f  source_read_reduction=%.1fx time=%.3f ms\n",
                label,
                compact_src_gb / (fill_reuse_ms / 1000.0),
                stage_write_gb / (fill_reuse_ms / 1000.0),
                transform_reuse_equiv_consumes,
                (double) (QK_NVFP4 / 8),
                fill_reuse_ms);

    std::printf("%s_consume: %.3f packed-fp4-pv-TOPS  %.3f useful-mtp-packed-fp4-pv-TOPS  %.3f GB/s-stage-read  reuse_transform_equiv_consumes=%.3f effective_TOPS_reuse1=%.3f reuse4=%.3f reuse8=%.3f n_out_frags=%d warps=%zu mma_iters=%" PRIu64 " time=%.3f ms\n",
                label,
                dense_tops,
                dense_tops * useful_mtp_fraction(cfg),
                stage_read_gb / (mma_ms / 1000.0),
                transform_reuse_equiv_consumes,
                effective_tops(1.0),
                effective_tops(4.0),
                effective_tops(8.0),
                n_out_frags,
                n_warps,
                cfg.mma_iters,
                mma_ms);

    CUDA_CHECK(cudaEventDestroy(fill_start));
    CUDA_CHECK(cudaEventDestroy(fill_stop));
    CUDA_CHECK(cudaEventDestroy(fill_reuse_start));
    CUDA_CHECK(cudaEventDestroy(fill_reuse_stop));
    CUDA_CHECK(cudaEventDestroy(mma_start));
    CUDA_CHECK(cudaEventDestroy(mma_stop));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(scales));
    CUDA_CHECK(cudaFree(stage));
    CUDA_CHECK(cudaFree(v));
    CUDA_CHECK(cudaFree(a));
}

template <int dkq>
static void run_pv_mixed_half_mma_packed_b_bench(const bench_config & cfg, int cc, const char * label) {
    static_assert(dkq == 256 || dkq == 512, "unsupported packed mixed PV width");
    constexpr int n_out_frags = dkq / 8;
    constexpr size_t stage_tile_entries = (size_t) n_out_frags * 4u * 32u;
    constexpr size_t stage_tile_size = stage_tile_entries * sizeof(mixed_pv_b_lane_frag);

    if (cc < 1200 || cc >= 1300) {
        std::printf("%s: skipped  reason=requires Blackwell sm_120/sm_121 device\n", label);
        return;
    }

    if (cfg.threads % 32 != 0) {
        std::printf("%s: skipped  reason=threads must be a multiple of 32\n", label);
        return;
    }

    const size_t n_threads = (size_t) cfg.blocks * cfg.threads;
    const size_t n_warps = n_threads / 32;
    size_t n_tiles = floor_power_of_two(cfg.bytes / stage_tile_size);
    if (n_tiles < n_warps) {
        n_tiles = floor_power_of_two(n_warps);
    }
    if (n_tiles < 1) {
        n_tiles = 1;
    }

    const size_t stage_entries = n_tiles * stage_tile_entries;

    print_occupancy_line(cfg, label, pv_mixed_half_mma_packed_b_kernel<dkq>, 0);

    mixed_pv_b_lane_frag * stage = nullptr;
    float * out = nullptr;
    CUDA_CHECK(cudaMalloc(&stage, stage_entries * sizeof(mixed_pv_b_lane_frag)));
    CUDA_CHECK(cudaMalloc(&out, n_warps * sizeof(float)));

    fill_mixed_pv_b_stage_kernel<<<cfg.blocks, cfg.threads>>>(stage, stage_entries);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    pv_mixed_half_mma_packed_b_kernel<dkq><<<cfg.blocks, cfg.threads>>>(stage, n_tiles - 1, out, 10);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    pv_mixed_half_mma_packed_b_kernel<dkq><<<cfg.blocks, cfg.threads>>>(stage, n_tiles - 1, out, cfg.mma_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float ms = time_events(start, stop);

    const double dense_ops = (double) n_warps * (double) cfg.mma_iters * (double) n_out_frags * 4.0 * 2.0 * 16.0 * 8.0 * 16.0;
    const double dense_tops = dense_ops / (ms / 1000.0) / 1.0e12;
    const double stage_read_gb = (double) n_warps * (double) cfg.mma_iters * (double) stage_tile_size / 1.0e9;

    std::printf("%s_consume: %.3f packed-mixed-pv-TOPS  %.3f useful-mtp-packed-mixed-pv-TOPS  %.3f GB/s-stage-read  half_mma_per_tile=%d n_out_frags=%d stage_tile=%zu warps=%zu mma_iters=%" PRIu64 " time=%.3f ms\n",
                label,
                dense_tops,
                dense_tops * useful_mtp_fraction(cfg),
                stage_read_gb / (ms / 1000.0),
                n_out_frags * 4,
                n_out_frags,
                stage_tile_size,
                n_warps,
                cfg.mma_iters,
                ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(stage));
}

template <int dkq>
static void run_pv_mixed_half_mma_transform_bench(const bench_config & cfg, int cc, const char * label) {
    static_assert(dkq == 256 || dkq == 512, "unsupported packed mixed PV width");
    constexpr int nfrags = dkq / QK_NVFP4;
    constexpr int n_out_frags = dkq / 8;
    constexpr size_t stage_tile_entries = (size_t) n_out_frags * 4u * 32u;
    constexpr size_t stage_tile_size = stage_tile_entries * sizeof(mixed_pv_b_lane_frag);

    if (cc < 1200 || cc >= 1300) {
        std::printf("%s: skipped  reason=requires Blackwell sm_120/sm_121 device\n", label);
        return;
    }

    if (cfg.threads % 32 != 0) {
        std::printf("%s: skipped  reason=threads must be a multiple of 32\n", label);
        return;
    }

    const size_t n_threads = (size_t) cfg.blocks * cfg.threads;
    const size_t n_warps = n_threads / 32;
    size_t n_tiles = floor_power_of_two(cfg.bytes / stage_tile_size);
    if (n_tiles < n_warps) {
        n_tiles = floor_power_of_two(n_warps);
    }
    if (n_tiles < 1) {
        n_tiles = 1;
    }

    const size_t n_blocks = n_tiles * (size_t) nfrags * 64u;
    const size_t stage_entries = n_tiles * stage_tile_entries;

    block_nvfp4 * v = nullptr;
    mixed_pv_b_lane_frag * stage = nullptr;
    float * out = nullptr;
    CUDA_CHECK(cudaMalloc(&v, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&stage, stage_entries * sizeof(mixed_pv_b_lane_frag)));
    CUDA_CHECK(cudaMalloc(&out, n_warps * sizeof(float)));

    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(v, n_blocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    fill_mixed_pv_b_stage_from_nvfp4_reuse_kernel<dkq><<<cfg.blocks, cfg.threads>>>(v, stage, n_tiles, n_blocks - 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    pv_mixed_half_mma_packed_b_kernel<dkq><<<cfg.blocks, cfg.threads>>>(stage, n_tiles - 1, out, 10);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t transform_start;
    cudaEvent_t transform_stop;
    cudaEvent_t consume_start;
    cudaEvent_t consume_stop;
    CUDA_CHECK(cudaEventCreate(&transform_start));
    CUDA_CHECK(cudaEventCreate(&transform_stop));
    CUDA_CHECK(cudaEventCreate(&consume_start));
    CUDA_CHECK(cudaEventCreate(&consume_stop));

    CUDA_CHECK(cudaEventRecord(transform_start));
    fill_mixed_pv_b_stage_from_nvfp4_reuse_kernel<dkq><<<cfg.blocks, cfg.threads>>>(v, stage, n_tiles, n_blocks - 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(transform_stop));
    const float transform_ms = time_events(transform_start, transform_stop);

    CUDA_CHECK(cudaEventRecord(consume_start));
    pv_mixed_half_mma_packed_b_kernel<dkq><<<cfg.blocks, cfg.threads>>>(stage, n_tiles - 1, out, cfg.mma_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(consume_stop));
    const float consume_ms = time_events(consume_start, consume_stop);

    const double dense_ops = (double) n_warps * (double) cfg.mma_iters * (double) n_out_frags * 4.0 * 2.0 * 16.0 * 8.0 * 16.0;
    const double consume_tops = dense_ops / (consume_ms / 1000.0) / 1.0e12;
    const double compact_src_gb = (double) n_tiles * (double) nfrags * 64.0 * (double) sizeof(block_nvfp4) / 1.0e9;
    const double stage_write_gb = (double) n_tiles * (double) stage_tile_size / 1.0e9;
    const double stage_read_gb = (double) n_warps * (double) cfg.mma_iters * (double) stage_tile_size / 1.0e9;
    const double transform_tile_s = (transform_ms / 1000.0) / (double) n_tiles;
    const double consume_tile_s = (consume_ms / 1000.0) / ((double) n_warps * (double) cfg.mma_iters);
    const double ops_per_tile = (double) n_out_frags * 4.0 * 2.0 * 16.0 * 8.0 * 16.0;
    const double transform_equiv_consumes = consume_tile_s > 0.0 ? transform_tile_s / consume_tile_s : 0.0;
    const auto effective_tops = [&](double reuse) {
        return (reuse * ops_per_tile) / (transform_tile_s + reuse * consume_tile_s) / 1.0e12;
    };

    std::printf("%s_transform: %.3f GB/s-compact-read  %.3f GB/s-stage-write  transform_equiv_consumes=%.3f  tiles=%zu compact_tile=%zu stage_tile=%zu expansion=%.3fx time=%.3f ms\n",
                label,
                compact_src_gb / (transform_ms / 1000.0),
                stage_write_gb / (transform_ms / 1000.0),
                transform_equiv_consumes,
                n_tiles,
                (size_t) nfrags * 64u * sizeof(block_nvfp4),
                stage_tile_size,
                (double) stage_tile_size / ((double) nfrags * 64.0 * (double) sizeof(block_nvfp4)),
                transform_ms);

    std::printf("%s_consume: %.3f packed-mixed-pv-TOPS  %.3f useful-mtp-packed-mixed-pv-TOPS  %.3f GB/s-stage-read  effective_TOPS_reuse1=%.3f reuse4=%.3f reuse8=%.3f half_mma_per_tile=%d n_out_frags=%d warps=%zu mma_iters=%" PRIu64 " time=%.3f ms\n",
                label,
                consume_tops,
                consume_tops * useful_mtp_fraction(cfg),
                stage_read_gb / (consume_ms / 1000.0),
                effective_tops(1.0),
                effective_tops(4.0),
                effective_tops(8.0),
                n_out_frags * 4,
                n_out_frags,
                n_warps,
                cfg.mma_iters,
                consume_ms);

    CUDA_CHECK(cudaEventDestroy(transform_start));
    CUDA_CHECK(cudaEventDestroy(transform_stop));
    CUDA_CHECK(cudaEventDestroy(consume_start));
    CUDA_CHECK(cudaEventDestroy(consume_stop));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(stage));
    CUDA_CHECK(cudaFree(v));
}

template <int dkq, int staged_groups>
static void run_pv_mixed_half_mma_transform_groups_bench(const bench_config & cfg, int cc, const char * label) {
    static_assert(dkq == 256 || dkq == 512, "unsupported partial packed mixed PV width");
    static_assert(staged_groups == 1 || staged_groups == 2 || staged_groups == 4, "unsupported staged group count");
    constexpr int nfrags = dkq / QK_NVFP4;
    constexpr int groups_per_frag = QK_NVFP4 / 8;
    constexpr int measured_out_frags = nfrags * staged_groups;
    constexpr int full_out_frags = dkq / 8;
    constexpr size_t stage_tile_entries = (size_t) measured_out_frags * 4u * 32u;
    constexpr size_t stage_tile_size = stage_tile_entries * sizeof(mixed_pv_b_lane_frag);

    if (cc < 1200 || cc >= 1300) {
        std::printf("%s: skipped  reason=requires Blackwell sm_120/sm_121 device\n", label);
        return;
    }

    if (cfg.threads % 32 != 0) {
        std::printf("%s: skipped  reason=threads must be a multiple of 32\n", label);
        return;
    }

    const size_t n_threads = (size_t) cfg.blocks * cfg.threads;
    const size_t n_warps = n_threads / 32;
    size_t n_tiles = floor_power_of_two(cfg.bytes / stage_tile_size);
    if (n_tiles < n_warps) {
        n_tiles = floor_power_of_two(n_warps);
    }
    if (n_tiles < 1) {
        n_tiles = 1;
    }

    const size_t n_blocks = n_tiles * (size_t) nfrags * 64u;
    const size_t stage_entries = n_tiles * stage_tile_entries;

    block_nvfp4 * v = nullptr;
    mixed_pv_b_lane_frag * stage = nullptr;
    float * out = nullptr;
    CUDA_CHECK(cudaMalloc(&v, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&stage, stage_entries * sizeof(mixed_pv_b_lane_frag)));
    CUDA_CHECK(cudaMalloc(&out, n_warps * sizeof(float)));

    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(v, n_blocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    fill_mixed_pv_b_stage_from_nvfp4_groups_kernel<dkq, staged_groups><<<cfg.blocks, cfg.threads>>>(v, stage, n_tiles, n_blocks - 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    pv_mixed_half_mma_packed_b_groups_kernel<dkq, staged_groups><<<cfg.blocks, cfg.threads>>>(stage, n_tiles - 1, out, 10);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t transform_start;
    cudaEvent_t transform_stop;
    cudaEvent_t consume_start;
    cudaEvent_t consume_stop;
    CUDA_CHECK(cudaEventCreate(&transform_start));
    CUDA_CHECK(cudaEventCreate(&transform_stop));
    CUDA_CHECK(cudaEventCreate(&consume_start));
    CUDA_CHECK(cudaEventCreate(&consume_stop));

    CUDA_CHECK(cudaEventRecord(transform_start));
    fill_mixed_pv_b_stage_from_nvfp4_groups_kernel<dkq, staged_groups><<<cfg.blocks, cfg.threads>>>(v, stage, n_tiles, n_blocks - 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(transform_stop));
    const float transform_ms = time_events(transform_start, transform_stop);

    CUDA_CHECK(cudaEventRecord(consume_start));
    pv_mixed_half_mma_packed_b_groups_kernel<dkq, staged_groups><<<cfg.blocks, cfg.threads>>>(stage, n_tiles - 1, out, cfg.mma_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(consume_stop));
    const float consume_ms = time_events(consume_start, consume_stop);

    const double measured_ops = (double) n_warps * (double) cfg.mma_iters * (double) measured_out_frags * 4.0 * 2.0 * 16.0 * 8.0 * 16.0;
    const double full_ops_at_same_time = (double) n_warps * (double) cfg.mma_iters * (double) full_out_frags * 4.0 * 2.0 * 16.0 * 8.0 * 16.0;
    const double consume_tops = measured_ops / (consume_ms / 1000.0) / 1.0e12;
    const double compact_src_gb = (double) n_tiles * (double) nfrags * 64.0 * (double) sizeof(block_nvfp4) / 1.0e9;
    const double stage_write_gb = (double) n_tiles * (double) stage_tile_size / 1.0e9;
    const double stage_read_gb = (double) n_warps * (double) cfg.mma_iters * (double) stage_tile_size / 1.0e9;
    const double transform_tile_s = (transform_ms / 1000.0) / (double) n_tiles;
    const double consume_tile_s = (consume_ms / 1000.0) / ((double) n_warps * (double) cfg.mma_iters);
    const double ops_per_tile = (double) measured_out_frags * 4.0 * 2.0 * 16.0 * 8.0 * 16.0;
    const double transform_equiv_consumes = consume_tile_s > 0.0 ? transform_tile_s / consume_tile_s : 0.0;
    const auto effective_tops = [&](double reuse) {
        return (reuse * ops_per_tile) / (transform_tile_s + reuse * consume_tile_s) / 1.0e12;
    };

    std::printf("%s_transform: %.3f GB/s-compact-read  %.3f GB/s-stage-write  transform_equiv_consumes=%.3f  tiles=%zu compact_tile=%zu stage_tile=%zu expansion=%.3fx pv_group_fraction=%.3f staged_groups=%d time=%.3f ms\n",
                label,
                compact_src_gb / (transform_ms / 1000.0),
                stage_write_gb / (transform_ms / 1000.0),
                transform_equiv_consumes,
                n_tiles,
                (size_t) nfrags * 64u * sizeof(block_nvfp4),
                stage_tile_size,
                (double) stage_tile_size / ((double) nfrags * 64.0 * (double) sizeof(block_nvfp4)),
                (double) staged_groups / (double) groups_per_frag,
                staged_groups,
                transform_ms);

    std::printf("%s_consume: %.3f measured-packed-mixed-pv-TOPS  %.3f useful-mtp-measured-packed-mixed-pv-TOPS  %.3f projected-fullPV-TOPS-at-same-time  %.3f GB/s-stage-read  effective_measured_TOPS_reuse1=%.3f reuse4=%.3f reuse8=%.3f half_mma_per_tile=%d measured_out_frags=%d warps=%zu mma_iters=%" PRIu64 " time=%.3f ms\n",
                label,
                consume_tops,
                consume_tops * useful_mtp_fraction(cfg),
                full_ops_at_same_time / (consume_ms / 1000.0) / 1.0e12,
                stage_read_gb / (consume_ms / 1000.0),
                effective_tops(1.0),
                effective_tops(4.0),
                effective_tops(8.0),
                measured_out_frags * 4,
                measured_out_frags,
                n_warps,
                cfg.mma_iters,
                consume_ms);

    CUDA_CHECK(cudaEventDestroy(transform_start));
    CUDA_CHECK(cudaEventDestroy(transform_stop));
    CUDA_CHECK(cudaEventDestroy(consume_start));
    CUDA_CHECK(cudaEventDestroy(consume_stop));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(stage));
    CUDA_CHECK(cudaFree(v));
}

template <int dkq, int staged_groups>
static void run_pv_mixed_half_partial_bench(const bench_config & cfg, int cc, const char * label) {
    static_assert(dkq == 256 || dkq == 512, "unsupported partial mixed PV width");
    static_assert(staged_groups == 1 || staged_groups == 2 || staged_groups == 4, "unsupported staged group count");
    constexpr int nfrags = dkq / QK_NVFP4;
    constexpr int groups_per_frag = QK_NVFP4 / 8;
    constexpr int measured_out_frags = nfrags * staged_groups;
    constexpr int full_out_frags = nfrags * groups_per_frag;

    if (cc < 1200 || cc >= 1300) {
        std::printf("%s: skipped  reason=requires Blackwell sm_120/sm_121 device\n", label);
        return;
    }

    if (cfg.threads % 32 != 0) {
        std::printf("%s: skipped  reason=threads must be a multiple of 32\n", label);
        return;
    }

    const size_t n_threads = (size_t) cfg.blocks * cfg.threads;
    const size_t n_warps = n_threads / 32;
    const size_t min_blocks = n_warps * (size_t) nfrags * 64u;
    size_t n_blocks = floor_power_of_two(cfg.bytes / sizeof(block_nvfp4));
    if (n_blocks < min_blocks) {
        n_blocks = ceil_power_of_two(min_blocks);
    }
    if (n_blocks < 4096) {
        n_blocks = 4096;
    }

    print_occupancy_line(cfg, label, pv_mixed_half_partial_nvfp4_kernel<dkq, staged_groups>, 0);

    block_nvfp4 * v = nullptr;
    float * out = nullptr;
    CUDA_CHECK(cudaMalloc(&v, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&out, n_warps * sizeof(float)));

    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(v, n_blocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    pv_mixed_half_partial_nvfp4_kernel<dkq, staged_groups><<<cfg.blocks, cfg.threads>>>(v, n_blocks - 1, out, 10);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    pv_mixed_half_partial_nvfp4_kernel<dkq, staged_groups><<<cfg.blocks, cfg.threads>>>(v, n_blocks - 1, out, cfg.mma_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float ms = time_events(start, stop);

    const double measured_ops = (double) n_warps * (double) cfg.mma_iters * (double) measured_out_frags * 4.0 * 2.0 * 16.0 * 8.0 * 16.0;
    const double full_ops = (double) n_warps * (double) cfg.mma_iters * (double) full_out_frags * 4.0 * 2.0 * 16.0 * 8.0 * 16.0;
    const double measured_tops = measured_ops / (ms / 1000.0) / 1.0e12;
    const double projected_full_tops = full_ops / (ms / 1000.0) / 1.0e12;
    const double compact_read_gb = (double) n_warps * (double) cfg.mma_iters * (double) nfrags * 64.0 * (double) sizeof(block_nvfp4) / 1.0e9;

    std::printf("%s: %.3f measured-mixedPV-TOPS  %.3f useful-mtp-measured-mixedPV-TOPS  %.3f projected-fullPV-TOPS-at-same-time  %.3f GB/s-compact-read  pv_group_fraction=%.3f staged_groups=%d full_row_read=1 nfrags=%d warps=%zu blocks=%zu mma_iters=%" PRIu64 " time=%.3f ms\n",
                label,
                measured_tops,
                measured_tops * useful_mtp_fraction(cfg),
                projected_full_tops,
                compact_read_gb / (ms / 1000.0),
                (double) staged_groups / (double) groups_per_frag,
                staged_groups,
                nfrags,
                n_warps,
                n_blocks,
                cfg.mma_iters,
                ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(v));
}

template <int dkq, int v_rows = PV_TILE_ROWS>
static void run_pv_stage_half2_v_eq_k_bench(const bench_config & cfg, int cc, const char * label) {
    static_assert(dkq == 256 || dkq == 512, "unsupported PV width");
    static_assert(v_rows == 2 || v_rows == 4 || v_rows == PV_TILE_ROWS, "unsupported PV tile rows");
    constexpr int nfrags = dkq / QK_NVFP4;
    constexpr int blocks_per_tile = v_rows * nfrags;
    constexpr int half2_per_tile = blocks_per_tile * (QK_NVFP4 / 2);

    if (cc < 1200 || cc >= 1300) {
        std::printf("%s: skipped  reason=requires Blackwell sm_120/sm_121 device\n", label);
        return;
    }

    if (cfg.threads % 32 != 0) {
        std::printf("%s: skipped  reason=threads must be a multiple of 32\n", label);
        return;
    }

    const size_t n_threads = (size_t) cfg.blocks * cfg.threads;
    const size_t n_warps = n_threads / 32;
    const size_t warps_per_block = (size_t) cfg.threads / 32;
    size_t n_blocks = floor_power_of_two(cfg.bytes / sizeof(block_nvfp4));
    const size_t min_blocks = n_warps * (size_t) blocks_per_tile;
    if (n_blocks < min_blocks) {
        n_blocks = floor_power_of_two(min_blocks);
    }
    if (n_blocks < 4096) {
        n_blocks = 4096;
    }

    const size_t shared_bytes = warps_per_block * (size_t) half2_per_tile * sizeof(__half2);
    if (shared_bytes > 48u * 1024u) {
        CUDA_CHECK(cudaFuncSetAttribute(
            pv_stage_half2_v_eq_k_nvfp4_kernel<dkq, v_rows>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int) shared_bytes));
    }
    print_occupancy_line(cfg, label, pv_stage_half2_v_eq_k_nvfp4_kernel<dkq, v_rows>, shared_bytes);

    block_nvfp4 * v = nullptr;
    float * out = nullptr;
    CUDA_CHECK(cudaMalloc(&v, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&out, n_warps * sizeof(float)));

    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(v, n_blocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    pv_stage_half2_v_eq_k_nvfp4_kernel<dkq, v_rows><<<cfg.blocks, cfg.threads, shared_bytes>>>(v, n_blocks - 1, out, 10);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    pv_stage_half2_v_eq_k_nvfp4_kernel<dkq, v_rows><<<cfg.blocks, cfg.threads, shared_bytes>>>(v, n_blocks - 1, out, cfg.mma_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float ms = time_events(start, stop);

    const double compact_read_gb = (double) n_warps * (double) cfg.mma_iters * (double) blocks_per_tile * (double) sizeof(block_nvfp4) / 1.0e9;
    const double shared_stage_gb = (double) n_warps * (double) cfg.mma_iters * (double) half2_per_tile * (double) sizeof(__half2) * 2.0 / 1.0e9;
    const double model_pv_ops = (double) n_warps * (double) cfg.mma_iters * 2.0 * (double) KQ_MMA_A_ROWS * (double) v_rows * (double) dkq;
    const double model_pv_tops = model_pv_ops / (ms / 1000.0) / 1.0e12;
    const size_t tile_bytes = (size_t) blocks_per_tile * sizeof(block_nvfp4);
    std::printf("%s: %.3f GB/s-compact-read  %.3f GB/s-shared-stage-rw  %.3f model-pv-TOPS  %.3f useful-mtp-model-pv-TOPS  tile_bytes=%zu v_rows=%d half2_tile=%d shared=%.3f KiB blocks=%zu warps=%zu mma_iters=%" PRIu64 " time=%.3f ms\n",
                label,
                compact_read_gb / (ms / 1000.0),
                shared_stage_gb / (ms / 1000.0),
                model_pv_tops,
                model_pv_tops * useful_mtp_fraction(cfg),
                tile_bytes,
                v_rows,
                half2_per_tile,
                (double) shared_bytes / 1024.0,
                n_blocks,
                n_warps,
                cfg.mma_iters,
                ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(v));
}

template <int dkq, int k_reuse>
static void run_fused_kq_pv_v_eq_k_bench(const bench_config & cfg, int cc, const char * label) {
    static_assert(dkq == 256 || dkq == 512, "unsupported fused KQ/PV width");
    constexpr int nfrags = dkq / QK_NVFP4;

    if (cc < 1200 || cc >= 1300) {
        std::printf("%s: skipped  reason=requires Blackwell sm_120/sm_121 device\n", label);
        return;
    }

    if (cfg.threads % 32 != 0) {
        std::printf("%s: skipped  reason=threads must be a multiple of 32\n", label);
        return;
    }

    const size_t n_threads = (size_t) cfg.blocks * cfg.threads;
    const size_t n_warps = n_threads / 32;
    size_t n_blocks = floor_power_of_two(cfg.bytes / sizeof(block_nvfp4));
    const size_t min_blocks = n_warps * (size_t) nfrags * (size_t) (KQ_MMA_A_ROWS + k_reuse * KQ_MMA_B_ROWS);
    if (n_blocks < min_blocks) {
        n_blocks = floor_power_of_two(min_blocks);
    }
    if (n_blocks < 4096) {
        n_blocks = 4096;
    }

    const size_t shared_bytes = (size_t) (cfg.threads / 32) * (size_t) KQ_MMA_Q_SHARED_INTS * sizeof(int);
    print_occupancy_line(cfg, label, fused_kq_pv_v_eq_k_qreg_warpk_nvfp4_kernel<nfrags, k_reuse>, shared_bytes);

    block_nvfp4 * q = nullptr;
    block_nvfp4 * k = nullptr;
    float * out = nullptr;
    CUDA_CHECK(cudaMalloc(&q, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&k, n_blocks * sizeof(block_nvfp4)));
    CUDA_CHECK(cudaMalloc(&out, n_warps * sizeof(float)));

    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(q, n_blocks);
    fill_nvfp4_kernel<<<cfg.blocks, cfg.threads>>>(k, n_blocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    fused_kq_pv_v_eq_k_qreg_warpk_nvfp4_kernel<nfrags, k_reuse><<<cfg.blocks, cfg.threads, shared_bytes>>>(q, k, n_blocks - 1, out, 10);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    fused_kq_pv_v_eq_k_qreg_warpk_nvfp4_kernel<nfrags, k_reuse><<<cfg.blocks, cfg.threads, shared_bytes>>>(q, k, n_blocks - 1, out, cfg.mma_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float ms = time_events(start, stop);

    const double kq_ops = (double) n_warps * (double) cfg.mma_iters * (double) nfrags * (double) k_reuse * 2.0 * 16.0 * 8.0 * 64.0;
    const double pv_model_ops = kq_ops;
    const double kq_tops = kq_ops / (ms / 1000.0) / 1.0e12;
    const double pv_model_tops = pv_model_ops / (ms / 1000.0) / 1.0e12;
    const double fused_model_tops = kq_tops + pv_model_tops;
    const double compact_read_gb = (double) n_warps * (double) cfg.mma_iters * (double) nfrags *
                                   (double) (KQ_MMA_A_ROWS + k_reuse * KQ_MMA_B_ROWS) * (double) sizeof(block_nvfp4) / 1.0e9;
    std::printf("%s: %.3f KQ-TOPS  %.3f model-PV-TOPS  %.3f fused-model-TOPS  %.3f useful-mtp-fused-model-TOPS  %.3f GB/s-single-k-eq-v-read  k_reuse=%d blocks=%zu shared=%.3f KiB warps=%zu mma_iters=%" PRIu64 " time=%.3f ms\n",
                label,
                kq_tops,
                pv_model_tops,
                fused_model_tops,
                fused_model_tops * useful_mtp_fraction(cfg),
                compact_read_gb / (ms / 1000.0),
                k_reuse,
                n_blocks,
                (double) shared_bytes / 1024.0,
                n_warps,
                cfg.mma_iters,
                ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(k));
    CUDA_CHECK(cudaFree(q));
}

int main(int argc, char ** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

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
    cfg.max_threads_per_sm = prop.maxThreadsPerMultiProcessor;

    std::printf("device:               %d %s\n", cfg.device, prop.name);
    std::printf("compute_capability:   sm_%d%d cc=%d\n", prop.major, prop.minor, cc);
    std::printf("spark_target:         %s\n", cc == 1210 ? "yes" : "no");
    std::printf("blackwell_fp4_target: %s\n", cc >= 1200 && cc < 1300 ? "yes" : "no");
    std::printf("sms:                  %d\n", prop.multiProcessorCount);
    std::printf("config:               blocks=%d threads=%d mtp_rows=%d useful_m16=%.1f%% bytes=%zu seconds=%d mma_iters=%" PRIu64 "\n",
                cfg.blocks, cfg.threads, cfg.mtp_rows, 100.0 * useful_mtp_fraction(cfg), cfg.bytes, cfg.seconds, cfg.mma_iters);
    print_gemma4_global_memory_model();

    if (!cfg.combined_only) {
        run_u4_read_bench(cfg);
        run_nvfp4_read_bench(cfg);
        run_fp4_mma_bench(cfg, cc);
        run_kq256_mma_bench(cfg, cc);
        run_kq_staged_mma_bench<256>(cfg, cc, "kq256_mma_staged");
        run_kq_staged_mma_bench<512>(cfg, cc, "kq512_mma_staged");
        run_kq_compact_mma_bench<256>(cfg, cc, "kq256_mma_compact");
        run_kq_compact_mma_bench<512>(cfg, cc, "kq512_mma_compact");
        run_kq_shared_mma_bench<256>(cfg, cc, "kq256_mma_shared");
        run_kq_shared_mma_bench<512>(cfg, cc, "kq512_mma_shared");
        run_kq_warp_mma_bench<256>(cfg, cc, "kq256_mma_warp");
        run_kq_warp_mma_bench<512>(cfg, cc, "kq512_mma_warp");
        run_kq_ldmatrix_mma_bench<256>(cfg, cc, "kq256_mma_ldmatrix");
        run_kq_ldmatrix_mma_bench<512>(cfg, cc, "kq512_mma_ldmatrix");
        run_kq_ldmatrix_reuse_mma_bench<256, 8>(cfg, cc, "kq256_mma_ldmatrix_reuse8");
        run_kq_ldmatrix_reuse_mma_bench<512, 8>(cfg, cc, "kq512_mma_ldmatrix_reuse8");
        run_kq_ldmatrix_qreg_mma_bench<256, 8>(cfg, cc, "kq256_mma_ldmatrix_qreg8");
        run_kq_ldmatrix_qreg_mma_bench<512, 8>(cfg, cc, "kq512_mma_ldmatrix_qreg8");
        run_kq_ldmatrix_qreg_warpk_mma_bench<256, 8>(cfg, cc, "kq256_mma_ldmatrix_qreg_warpk8");
    }
    run_kq_ldmatrix_qreg_warpk_mma_bench<512, 8>(cfg, cc, "kq512_mma_ldmatrix_qreg_warpk8");
    if (!cfg.combined_only) {
        run_combined_kq_packedpv_budget_bench<256, 8>(cfg, cc, "combined256_kq_packedpv_budget");
    }
    run_combined_kq_packedpv_budget_bench<512, 8>(cfg, cc, "combined512_kq_packedpv_budget");
    run_combined_kq_packedpv_budget_bench<512, 8, true>(cfg, cc, "combined512_kq_packedpv_attentionish");
    run_combined_kq_packedpv_budget_bench<512, 8, true, true>(cfg, cc, "combined512_kq_inlinepv_attentionish");
    if (cfg.mixed_pv) {
        run_combined_kq_mixedpv_half_bench<512, 8, true>(cfg, cc, "combined512_kq_mixedpv_half_attentionish");
    }
    if (cfg.mixed_pv_stripmine_groups == 1) {
        run_combined_kq_mixedpv_stripmine_bench<512, 8, 1, true>(cfg, cc, "combined512_kq_mixedpv_stripmine_group1_attentionish");
    } else if (cfg.mixed_pv_stripmine_groups == 2) {
        run_combined_kq_mixedpv_stripmine_bench<512, 8, 2, true>(cfg, cc, "combined512_kq_mixedpv_stripmine_group2_attentionish");
    } else if (cfg.mixed_pv_stripmine_groups == 4) {
        run_combined_kq_mixedpv_stripmine_bench<512, 8, 4, true>(cfg, cc, "combined512_kq_mixedpv_stripmine_group4_attentionish");
    } else if (cfg.mixed_pv_stripmine_groups == 8) {
        run_combined_kq_mixedpv_stripmine_bench<512, 8, 8, true>(cfg, cc, "combined512_kq_mixedpv_stripmine_group8_attentionish");
    }
    if (cfg.mixed_pv_prepacked) {
        run_pv_mixed_half_mma_packed_b_bench<512>(cfg, cc, "pv512_mixed_half_mma_packed_b_stage");
    }
    if (cfg.mixed_pv_transform) {
        run_pv_mixed_half_mma_transform_bench<512>(cfg, cc, "pv512_mixed_half_mma_transform");
    }
    if (cfg.mixed_pv_transform_groups == 1) {
        run_pv_mixed_half_mma_transform_groups_bench<512, 1>(cfg, cc, "pv512_mixed_half_mma_transform_group1");
    } else if (cfg.mixed_pv_transform_groups == 2) {
        run_pv_mixed_half_mma_transform_groups_bench<512, 2>(cfg, cc, "pv512_mixed_half_mma_transform_group2");
    } else if (cfg.mixed_pv_transform_groups == 4) {
        run_pv_mixed_half_mma_transform_groups_bench<512, 4>(cfg, cc, "pv512_mixed_half_mma_transform_group4");
    }
    if (cfg.mixed_pv_partial_groups == 1) {
        run_pv_mixed_half_partial_bench<512, 1>(cfg, cc, "pv512_mixed_half_partial_group1");
    } else if (cfg.mixed_pv_partial_groups == 2) {
        run_pv_mixed_half_partial_bench<512, 2>(cfg, cc, "pv512_mixed_half_partial_group2");
    } else if (cfg.mixed_pv_partial_groups == 4) {
        run_pv_mixed_half_partial_bench<512, 4>(cfg, cc, "pv512_mixed_half_partial_group4");
    }
    if (cfg.mixed_pv_smem_groups == 1) {
        run_combined_kq_mixedpv_smem_half_bench<512, 8, 1, true>(cfg, cc, "combined512_kq_mixedpv_smem_group1_attentionish");
    } else if (cfg.mixed_pv_smem_groups == 2) {
        run_combined_kq_mixedpv_smem_half_bench<512, 8, 2, true>(cfg, cc, "combined512_kq_mixedpv_smem_group2_attentionish");
    } else if (cfg.mixed_pv_smem_groups == 4) {
        run_combined_kq_mixedpv_smem_half_bench<512, 8, 4, true>(cfg, cc, "combined512_kq_mixedpv_smem_group4_attentionish");
    }
    if (cfg.mixed_pv_compact_smem_groups == 1) {
        run_combined_kq_mixedpv_smem_compact_bench<512, 8, 1, true>(cfg, cc, "combined512_kq_mixedpv_compact_smem_group1_attentionish");
    } else if (cfg.mixed_pv_compact_smem_groups == 2) {
        run_combined_kq_mixedpv_smem_compact_bench<512, 8, 2, true>(cfg, cc, "combined512_kq_mixedpv_compact_smem_group2_attentionish");
    } else if (cfg.mixed_pv_compact_smem_groups == 4) {
        run_combined_kq_mixedpv_smem_compact_bench<512, 8, 4, true>(cfg, cc, "combined512_kq_mixedpv_compact_smem_group4_attentionish");
    } else if (cfg.mixed_pv_compact_smem_groups == 8) {
        run_combined_kq_mixedpv_smem_compact_bench<512, 8, 8, true>(cfg, cc, "combined512_kq_mixedpv_compact_smem_group8_attentionish");
    }
    if (cfg.mixed_pv_rolling_groups == 1) {
        run_combined_kq_mixedpv_smem_compact_bench<512, 8, 1, true, true>(cfg, cc, "combined512_kq_mixedpv_rolling_group1_attentionish");
    } else if (cfg.mixed_pv_rolling_groups == 2) {
        run_combined_kq_mixedpv_smem_compact_bench<512, 8, 2, true, true>(cfg, cc, "combined512_kq_mixedpv_rolling_group2_attentionish");
    } else if (cfg.mixed_pv_rolling_groups == 4) {
        run_combined_kq_mixedpv_smem_compact_bench<512, 8, 4, true, true>(cfg, cc, "combined512_kq_mixedpv_rolling_group4_attentionish");
    }
    if (cfg.kreuse_subtile) {
        run_combined_kq_kreusepv_subtile_bench<512, 8, true>(cfg, cc, "combined512_kq_kreusepv_subtile_attentionish");
    }
    if (cfg.kreuse_smem) {
        run_combined_kq_kreusepv_smem_bench<512, 8, true>(cfg, cc, "combined512_kq_kreusepv_smem_attentionish");
    }
    if (cfg.kreuse_smem_groups == 1) {
        run_combined_kq_kreusepv_smem_groups_bench<512, 8, 1, true>(cfg, cc, "combined512_kq_kreusepv_smem_group1_attentionish");
    } else if (cfg.kreuse_smem_groups == 2) {
        run_combined_kq_kreusepv_smem_groups_bench<512, 8, 2, true>(cfg, cc, "combined512_kq_kreusepv_smem_group2_attentionish");
    } else if (cfg.kreuse_smem_groups == 4) {
        run_combined_kq_kreusepv_smem_groups_bench<512, 8, 4, true>(cfg, cc, "combined512_kq_kreusepv_smem_group4_attentionish");
    }
    if (!cfg.combined_only) {
        run_kq_ldmatrix_qreg_kpair_mma_bench<256, 8>(cfg, cc, "kq256_mma_ldmatrix_qreg_kpair8");
        run_kq_ldmatrix_qreg_kpair_mma_bench<512, 8>(cfg, cc, "kq512_mma_ldmatrix_qreg_kpair8");
        run_pv_dequant_v_eq_k_bench<256>(cfg, cc, "pv256_dequant_v_eq_k_nvfp4");
        run_pv_dequant_v_eq_k_bench<512>(cfg, cc, "pv512_dequant_v_eq_k_nvfp4");
        run_pv_fp4_mma_native_bench<256>(cfg, cc, "pv256_fp4_mma_native_lower_bound");
        run_pv_fp4_mma_native_bench<512>(cfg, cc, "pv512_fp4_mma_native_lower_bound");
        run_pv_fp4_mma_compact_b_bench<256>(cfg, cc, "pv256_fp4_mma_compact_b");
        run_pv_fp4_mma_compact_b_bench<512>(cfg, cc, "pv512_fp4_mma_compact_b");
        run_pv_fp4_mma_packed_b_bench<256>(cfg, cc, "pv256_fp4_mma_packed_b_stage");
        run_pv_fp4_mma_packed_b_bench<512>(cfg, cc, "pv512_fp4_mma_packed_b_stage");
        run_pv_stage_half2_v_eq_k_bench<256>(cfg, cc, "pv256_stage_half2_v_eq_k_nvfp4");
        run_pv_stage_half2_v_eq_k_bench<512>(cfg, cc, "pv512_stage_half2_v_eq_k_nvfp4");
        run_pv_stage_half2_v_eq_k_bench<512, 4>(cfg, cc, "pv512_stage_half2_v4_v_eq_k_nvfp4");
        run_pv_stage_half2_v_eq_k_bench<512, 2>(cfg, cc, "pv512_stage_half2_v2_v_eq_k_nvfp4");
        run_fused_kq_pv_v_eq_k_bench<256, 8>(cfg, cc, "fused256_kq_scalarpv_v_eq_k_qreg_warpk8");
        run_fused_kq_pv_v_eq_k_bench<512, 8>(cfg, cc, "fused512_kq_scalarpv_v_eq_k_qreg_warpk8");
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}
