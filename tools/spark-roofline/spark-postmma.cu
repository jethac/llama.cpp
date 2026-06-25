#include <cuda_runtime.h>
#include <cuda_fp16.h>

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
    int      blocks    = 0;
    int      threads   = 128;
    int      mtp_rows  = 4;
    int      groups    = 1;
    uint64_t iters     = 20000;
};

static void print_usage(const char * exe) {
    std::printf(
        "usage: %s [options]\n"
        "\n"
        "Small phase-separated FP4-KQ + synthetic FP16-PV issue probe.\n"
        "\n"
        "options:\n"
        "  --device N     CUDA device id (default: 0)\n"
        "  --blocks N     CUDA blocks (default: 4 * SM count)\n"
        "  --threads N    CUDA threads per block (default: 128)\n"
        "  --mtp-rows N   useful MTP verification rows in an m16 tile (default: 4)\n"
        "  --groups N     synthetic mixed-PV groups per key tile (1,2,4,8; default: 1)\n"
        "  --iters N      loop iterations per warp (default: 20000)\n"
        "  --help         print this help\n",
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
        } else if (std::strcmp(arg, "--groups") == 0) {
            cfg.groups = (int) parse_u64(require_value(arg), arg);
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
    if (cfg.groups != 1 && cfg.groups != 2 && cfg.groups != 4 && cfg.groups != 8) {
        std::fprintf(stderr, "invalid --groups value: %d (expected 1, 2, 4, or 8)\n", cfg.groups);
        std::exit(1);
    }

    return cfg;
}

static __device__ __forceinline__ uint32_t half2_bits(__half2 v) {
    union {
        __half2  h;
        uint32_t u;
    } cvt;
    cvt.h = v;
    return cvt.u;
}

__global__ void postmma_issue_kernel(
        int       groups,
        float *   out,
        uint64_t  iters) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300
    const int lane = threadIdx.x & 31;
    const int warp = ((int) blockIdx.x * blockDim.x + threadIdx.x) >> 5;

    const int ax0 = (int) (0x11111111u + (uint32_t) lane);
    const int ax1 = (int) (0x22222222u + (uint32_t) lane);
    const int ax2 = (int) (0x33333333u + (uint32_t) lane);
    const int ax3 = (int) (0x44444444u + (uint32_t) lane);
    const int bx0 = (int) (0x55555555u + (uint32_t) lane);
    const int bx1 = (int) (0x66666666u + (uint32_t) lane);
    const uint32_t q_scale = 0x38383838u;
    const uint32_t k_scale = 0x39393939u;

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
#pragma unroll
        for (int tile = 0; tile < 64; ++tile) {
            asm volatile(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                "%10, {0, 0}, %11, {0, 0};"
                : "+f"(kq0), "+f"(kq1), "+f"(kq2), "+f"(kq3)
                : "r"(ax0), "r"(ax1), "r"(ax2), "r"(ax3), "r"(bx0), "r"(bx1), "r"(q_scale), "r"(k_scale));

            const int q_row = lane & 15;
            const int base_pos = (int) (i & 1023u);
            const bool causal = tile <= base_pos + q_row;
            const bool swa = tile + 1024 >= base_pos + q_row;
            const float score0 = (causal && swa) ? kq0 * 0.000244140625f : -64.0f;
            const float score1 = (causal && swa) ? kq2 * 0.000244140625f : -64.0f;
            const float next_m0 = fmaxf(row_m0, score0);
            const float next_m1 = fmaxf(row_m1, score1);
            const float alpha0 = row_l0 == 0.0f ? 0.0f : exp2f(row_m0 - next_m0);
            const float alpha1 = row_l1 == 0.0f ? 0.0f : exp2f(row_m1 - next_m1);
            row_l0 = row_l0 * alpha0 + exp2f(score0 - next_m0);
            row_l1 = row_l1 * alpha1 + exp2f(score1 - next_m1);
            row_m0 = next_m0;
            row_m1 = next_m1;
        }

        const float p_base = 1.0f / (row_l0 + row_l1 + 1.0f);
        const uint32_t pax0 = half2_bits(__floats2half2_rn(p_base, p_base * 0.9375f));
        const uint32_t pax1 = half2_bits(__floats2half2_rn(p_base * 0.875f, p_base * 0.8125f));
        const uint32_t pax2 = half2_bits(__floats2half2_rn(p_base * 0.75f, p_base * 0.6875f));
        const uint32_t pax3 = half2_bits(__floats2half2_rn(p_base * 0.625f, p_base * 0.5625f));

        for (int tile = 0; tile < 64; ++tile) {
            for (int group = 0; group < groups; ++group) {
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const uint32_t pv_bx0 = half2_bits(__floats2half2_rn(
                        0.03125f * (float) (1 + ((lane + group + kk) & 7)),
                        0.02930f * (float) (1 + ((lane + tile + kk) & 7))));
                    const uint32_t pv_bx1 = half2_bits(__floats2half2_rn(
                        0.02734f * (float) (1 + ((lane + tile + kk) & 7)),
                        0.02539f * (float) (1 + ((lane + group) & 7))));
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                        : "+f"(pv0), "+f"(pv1), "+f"(pv2), "+f"(pv3)
                        : "r"(pax0), "r"(pax1), "r"(pax2), "r"(pax3), "r"(pv_bx0), "r"(pv_bx1));
                }

                const float mix = (float) (tile * 8 + group + 1) * 0.000001f / (row_l0 + row_l1 + 0.000001f);
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
    (void) groups;
    (void) out;
    (void) iters;
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
    std::printf("config:               blocks=%d threads=%d mtp_rows=%d useful_m16=%.1f%% groups=%d iters=%" PRIu64 "\n",
                cfg.blocks, cfg.threads, cfg.mtp_rows, 100.0 * useful_mtp_fraction(cfg), cfg.groups, cfg.iters);

    if (cc < 1200 || cc >= 1300) {
        std::printf("postmma_issue: skipped reason=requires Blackwell sm_120/sm_121 device\n");
        return 0;
    }

    const int active_blocks_per_sm = [] (const bench_config & c) {
        int active = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, postmma_issue_kernel, c.threads, 0));
        return active;
    }(cfg);
    const int active_warps_per_sm = active_blocks_per_sm * (cfg.threads / 32);
    const double occupancy = prop.maxThreadsPerMultiProcessor > 0 ?
        (double) active_blocks_per_sm * (double) cfg.threads / (double) prop.maxThreadsPerMultiProcessor : 0.0;
    std::printf("postmma_issue occupancy: active_blocks_per_sm=%d active_warps_per_sm=%d occupancy=%.1f%% shared=0.000 KiB\n",
                active_blocks_per_sm, active_warps_per_sm, 100.0 * occupancy);

    const size_t n_threads = (size_t) cfg.blocks * (size_t) cfg.threads;
    const size_t n_warps = n_threads / 32;
    float * out = nullptr;
    CUDA_CHECK(cudaMalloc(&out, n_warps * sizeof(float)));

    postmma_issue_kernel<<<cfg.blocks, cfg.threads>>>(cfg.groups, out, 10);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    postmma_issue_kernel<<<cfg.blocks, cfg.threads>>>(cfg.groups, out, cfg.iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    const float ms = time_events(start, stop);

    constexpr int kq_tiles = 64;
    constexpr int full_groups = 8;
    const double kq_ops = (double) n_warps * (double) cfg.iters * (double) kq_tiles * 2.0 * 16.0 * 8.0 * 64.0;
    const double measured_pv_ops = (double) n_warps * (double) cfg.iters * (double) kq_tiles * (double) cfg.groups * 4.0 * 2.0 * 16.0 * 8.0 * 16.0;
    const double full_pv_ops = (double) n_warps * (double) cfg.iters * (double) kq_tiles * (double) full_groups * 4.0 * 2.0 * 16.0 * 8.0 * 16.0;
    const double measured_ops = kq_ops + measured_pv_ops;
    const double seconds = ms / 1000.0;

    std::printf("postmma_issue: %.3f measured-total-TOPS  %.3f useful-mtp-measured-total-TOPS  %.3f KQ-issue-TOPS  %.3f measured-mixedPV-issue-TOPS  %.3f projected-fullPV-issue-TOPS-at-same-time  pv_group_fraction=%.3f groups=%d warps=%zu iters=%" PRIu64 " time=%.3f ms\n",
                measured_ops / seconds / 1.0e12,
                measured_ops / seconds / 1.0e12 * useful_mtp_fraction(cfg),
                kq_ops / seconds / 1.0e12,
                measured_pv_ops / seconds / 1.0e12,
                full_pv_ops / seconds / 1.0e12,
                (double) cfg.groups / (double) full_groups,
                cfg.groups,
                n_warps,
                cfg.iters,
                ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}
