#pragma once

#include "common.cuh"

#include <cstdint>

// Experimental Blackwell-only execution layout for NVFP4 KV cache side buffers.
// This is a runtime CUDA layout, not a GGUF/file-format layout.

static constexpr int GGML_CUDA_NVFP4_VX_KV_TILE       = QK_NVFP4;
static constexpr int GGML_CUDA_NVFP4_VX_COLS          = 8;
static constexpr int GGML_CUDA_NVFP4_VX_WORDS_PER_COL = QK_NVFP4 / 8;
static constexpr int GGML_CUDA_NVFP4_VX_QS_WORDS      =
    GGML_CUDA_NVFP4_VX_COLS * GGML_CUDA_NVFP4_VX_WORDS_PER_COL;

struct alignas(16) ggml_cuda_nvfp4_vx_tile {
    uint32_t scale[GGML_CUDA_NVFP4_VX_COLS];
    int32_t  qs[GGML_CUDA_NVFP4_VX_QS_WORDS];
};

static_assert(QK_NVFP4 == 64, "NVFP4 V execution layout assumes 64-value FP4 blocks");
static_assert(QK_NVFP4_SUB == 16, "NVFP4 V execution layout assumes 16-value scale groups");
static_assert(sizeof(ggml_cuda_nvfp4_vx_tile) == 288, "unexpected NVFP4 V execution tile size");
static constexpr size_t GGML_CUDA_NVFP4_VX_TILE_BYTES = sizeof(ggml_cuda_nvfp4_vx_tile);

struct ggml_cuda_nvfp4_vx_layout {
    ggml_cuda_nvfp4_vx_tile * tiles;

    int64_t n_stream;
    int64_t n_kv_heads;
    int64_t kv_size;
    int64_t v_head_dim;
    int64_t kv_tile_count;
    int64_t col_tile_count;
};

static __host__ __device__ __forceinline__ int64_t ggml_cuda_nvfp4_vx_kv_tile_count(const int64_t kv_size) {
    return (kv_size + GGML_CUDA_NVFP4_VX_KV_TILE - 1) / GGML_CUDA_NVFP4_VX_KV_TILE;
}

static __host__ __device__ __forceinline__ int64_t ggml_cuda_nvfp4_vx_col_tile_count(const int64_t v_head_dim) {
    return v_head_dim / GGML_CUDA_NVFP4_VX_COLS;
}

static __host__ __device__ __forceinline__ size_t ggml_cuda_nvfp4_vx_tile_count(
        const int64_t n_stream,
        const int64_t n_kv_heads,
        const int64_t kv_size,
        const int64_t v_head_dim) {
    return (size_t) n_stream * n_kv_heads *
        ggml_cuda_nvfp4_vx_col_tile_count(v_head_dim) *
        ggml_cuda_nvfp4_vx_kv_tile_count(kv_size);
}

static __host__ __device__ __forceinline__ int64_t ggml_cuda_nvfp4_vx_tile_id(
        const ggml_cuda_nvfp4_vx_layout & layout,
        const int64_t stream,
        const int64_t kv_head,
        const int64_t col_tile,
        const int64_t kv_tile) {
    return (((stream * layout.n_kv_heads + kv_head)
                * layout.col_tile_count + col_tile)
                * layout.kv_tile_count + kv_tile);
}

static __host__ __device__ __forceinline__ ggml_cuda_nvfp4_vx_tile * ggml_cuda_nvfp4_vx_tile_ptr(
        const ggml_cuda_nvfp4_vx_layout & layout,
        const int64_t stream,
        const int64_t kv_head,
        const int64_t col_tile,
        const int64_t kv_tile) {
    return layout.tiles + ggml_cuda_nvfp4_vx_tile_id(layout, stream, kv_head, col_tile, kv_tile);
}

static __device__ __forceinline__ float ggml_cuda_nvfp4_vx_dequant_tile_value(
        const ggml_cuda_nvfp4_vx_tile & tile,
        const int col,
        const int k) {
    const int sub = k / QK_NVFP4_SUB;
    const int off = k % QK_NVFP4_SUB;
    const int j = off % (QK_NVFP4_SUB / 2);
    const int word = 2 * sub + j / 4;
    const int byte_shift = 8 * (j % 4);
    const uint32_t packed = (uint32_t) tile.qs[col * GGML_CUDA_NVFP4_VX_WORDS_PER_COL + word];
    const uint8_t q4 = (packed >> byte_shift >> (off >= QK_NVFP4_SUB / 2 ? 4 : 0)) & 0x0f;
    const uint8_t scale = (tile.scale[col] >> (8 * sub)) & 0xff;
    return ggml_cuda_ue4m3_to_fp32(scale) * kvalues_mxfp4[q4];
}

static __device__ __forceinline__ float ggml_cuda_nvfp4_vx_dequant_value(
        const ggml_cuda_nvfp4_vx_layout & layout,
        const int64_t stream,
        const int64_t kv_head,
        const int64_t kv_row,
        const int64_t v_col) {
    const int64_t kv_tile = kv_row / GGML_CUDA_NVFP4_VX_KV_TILE;
    const int64_t col_tile = v_col / GGML_CUDA_NVFP4_VX_COLS;
    const int k = (int) (kv_row - kv_tile * GGML_CUDA_NVFP4_VX_KV_TILE);
    const int col = (int) (v_col - col_tile * GGML_CUDA_NVFP4_VX_COLS);
    const ggml_cuda_nvfp4_vx_tile * tile =
        ggml_cuda_nvfp4_vx_tile_ptr(layout, stream, kv_head, col_tile, kv_tile);
    return ggml_cuda_nvfp4_vx_dequant_tile_value(*tile, col, k);
}

#if defined(GGML_CUDA_NVFP4_KV_EXEC_LAYOUT)

bool ggml_cuda_nvfp4_vx_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * V);

bool ggml_cuda_nvfp4_vx_find(
    ggml_backend_cuda_context & ctx, const ggml_tensor * V, ggml_cuda_nvfp4_vx_layout * layout);

bool ggml_cuda_nvfp4_vx_after_set_rows(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);

bool ggml_cuda_nvfp4_vx_has_direct_updates(ggml_backend_cuda_context & ctx, const ggml_tensor * V);

bool ggml_cuda_nvfp4_vx_validate(
    ggml_backend_cuda_context & ctx, const ggml_tensor * V, float * max_abs_error, float * mean_abs_error);

void ggml_cuda_nvfp4_vx_clear_context(ggml_backend_cuda_context & ctx);

#endif // defined(GGML_CUDA_NVFP4_KV_EXEC_LAYOUT)
