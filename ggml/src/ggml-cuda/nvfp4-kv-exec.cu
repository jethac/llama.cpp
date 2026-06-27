#include "nvfp4-kv-exec.cuh"

#if defined(GGML_CUDA_NVFP4_KV_EXEC_LAYOUT)

#include <cstring>
#include <mutex>
#include <unordered_map>

struct ggml_cuda_nvfp4_vx_entry {
    ggml_cuda_nvfp4_vx_layout layout = {};
    size_t tile_count = 0;
    int device = -1;
    bool direct_updates = false;
};

struct ggml_cuda_nvfp4_kx_entry {
    ggml_cuda_nvfp4_kx_layout layout = {};
    size_t tile_count = 0;
    int device = -1;
    bool direct_updates = false;
};

static std::mutex g_nvfp4_vx_mutex;
static std::unordered_map<const ggml_backend_cuda_context *,
    std::unordered_map<const void *, ggml_cuda_nvfp4_vx_entry>> g_nvfp4_vx_registry;

static std::mutex g_nvfp4_kx_mutex;
static std::unordered_map<const ggml_backend_cuda_context *,
    std::unordered_map<const void *, ggml_cuda_nvfp4_kx_entry>> g_nvfp4_kx_registry;

static const ggml_tensor * ggml_cuda_nvfp4_vx_base_tensor(const ggml_tensor * t) {
    while (t != nullptr && t->view_src != nullptr) {
        t = t->view_src;
    }
    return t;
}

static const void * ggml_cuda_nvfp4_vx_key(const ggml_tensor * t) {
    const ggml_tensor * base = ggml_cuda_nvfp4_vx_base_tensor(t);
    return base != nullptr ? base->data : nullptr;
}

static const ggml_tensor * ggml_cuda_nvfp4_kx_base_tensor(const ggml_tensor * t) {
    while (t != nullptr && t->view_src != nullptr) {
        t = t->view_src;
    }
    return t;
}

static const void * ggml_cuda_nvfp4_kx_key(const ggml_tensor * t) {
    const ggml_tensor * base = ggml_cuda_nvfp4_kx_base_tensor(t);
    return base != nullptr ? base->data : nullptr;
}

static bool ggml_cuda_nvfp4_vx_supported_view(const ggml_tensor * V) {
    if (V == nullptr || V->type != GGML_TYPE_NVFP4) {
        return false;
    }

    if (V->ne[0] != 256 && V->ne[0] != 512) {
        return false;
    }

    if (V->ne[1] <= 0 || V->ne[2] <= 0 || V->ne[3] <= 0) {
        return false;
    }

    if (V->ne[0] % GGML_CUDA_NVFP4_VX_COLS != 0) {
        return false;
    }

    const ggml_tensor * base = ggml_cuda_nvfp4_vx_base_tensor(V);
    if (base == nullptr || base->data == nullptr || base->type != GGML_TYPE_NVFP4) {
        return false;
    }

    return true;
}

static bool ggml_cuda_nvfp4_kx_supported_view(const ggml_tensor * K) {
    if (K == nullptr || K->type != GGML_TYPE_NVFP4) {
        return false;
    }

    if (K->ne[0] != 256 && K->ne[0] != 512) {
        return false;
    }

    if (K->ne[1] <= 0 || K->ne[2] <= 0 || K->ne[3] <= 0) {
        return false;
    }

    if (K->ne[0] % GGML_CUDA_NVFP4_KX_HEAD_TILE != 0) {
        return false;
    }

    const ggml_tensor * base = ggml_cuda_nvfp4_kx_base_tensor(K);
    if (base == nullptr || base->data == nullptr || base->type != GGML_TYPE_NVFP4) {
        return false;
    }

    return true;
}

static int64_t ggml_cuda_nvfp4_vx_stride_blocks(const ggml_tensor * t, const int dim) {
    GGML_ASSERT(t->nb[dim] % sizeof(block_nvfp4) == 0);
    return t->nb[dim] / sizeof(block_nvfp4);
}

static int64_t ggml_cuda_nvfp4_kx_stride_blocks(const ggml_tensor * t, const int dim) {
    GGML_ASSERT(t->nb[dim] % sizeof(block_nvfp4) == 0);
    return t->nb[dim] / sizeof(block_nvfp4);
}

static __device__ __forceinline__ float ggml_cuda_nvfp4_vx_dequant_row_value(
        const block_nvfp4 * row,
        const int col) {
    const block_nvfp4 & blk = row[col / QK_NVFP4];
    const int i = col % QK_NVFP4;
    const int sub = i / QK_NVFP4_SUB;
    const int j = i % (QK_NVFP4_SUB / 2);
    const uint8_t q = blk.qs[sub * (QK_NVFP4_SUB / 2) + j];
    const uint8_t q4 = (i % QK_NVFP4_SUB) < (QK_NVFP4_SUB / 2) ? (q & 0x0f) : (q >> 4);
    return ggml_cuda_ue4m3_to_fp32(blk.d[sub]) * kvalues_mxfp4[q4];
}

#if defined(BLACKWELL_MMA_AVAILABLE)
static __device__ __forceinline__ float ggml_cuda_nvfp4_vx_quant_error(
        const float * vals,
        const uint8_t scale_code) {
    const float scale = ggml_cuda_ue4m3_to_fp32(scale_code);
    const float inv_scale = scale > 0.0f ? 0.5f / scale : 0.0f;
    float err = 0.0f;

#pragma unroll
    for (int j = 0; j < QK_NVFP4_SUB; ++j) {
        const uint8_t q = ggml_cuda_float_to_fp4_e2m1(vals[j], inv_scale);
        const float dq = scale * kvalues_mxfp4[q];
        const float d = vals[j] - dq;
        err += d * d;
    }

    return err;
}

static __device__ __forceinline__ uint8_t ggml_cuda_nvfp4_vx_best_scale_code(
        const float * vals,
        const float amax) {
    if (amax == 0.0f) {
        return 0;
    }

    uint8_t best_code = ggml_cuda_fp32_to_ue4m3(amax / 6.0f);
    float best_err = ggml_cuda_nvfp4_vx_quant_error(vals, best_code);

    for (int code = 1; code < 256; ++code) {
        const float err = ggml_cuda_nvfp4_vx_quant_error(vals, (uint8_t) code);
        if (err < best_err) {
            best_err = err;
            best_code = (uint8_t) code;
        }
    }

    return best_code;
}

static __device__ __forceinline__ void ggml_cuda_nvfp4_vx_quantize_64_words(
        const float * vals,
        int * qs_words,
        uint32_t & scale_word) {
    scale_word = 0;

#pragma unroll
    for (int sub = 0; sub < QK_NVFP4 / QK_NVFP4_SUB; ++sub) {
        const float * xb = vals + sub * QK_NVFP4_SUB;

        float amax = 0.0f;
#pragma unroll
        for (int j = 0; j < QK_NVFP4_SUB; ++j) {
            amax = fmaxf(amax, fabsf(xb[j]));
        }

        const uint8_t scale_code = ggml_cuda_nvfp4_vx_best_scale_code(xb, amax);
        const float scale = ggml_cuda_ue4m3_to_fp32(scale_code);
        const float inv_scale = scale > 0.0f ? 0.5f / scale : 0.0f;
        scale_word |= (uint32_t) scale_code << (8 * sub);

        uint32_t q0 = 0;
        uint32_t q1 = 0;
#pragma unroll
        for (int k = 0; k < QK_NVFP4_SUB / 4; ++k) {
            q0 |= (uint32_t) ggml_cuda_float_to_fp4_e2m1(xb[k +  0], inv_scale) << (8 * k);
            q0 |= (uint32_t) ggml_cuda_float_to_fp4_e2m1(xb[k +  8], inv_scale) << (8 * k + 4);
            q1 |= (uint32_t) ggml_cuda_float_to_fp4_e2m1(xb[k +  4], inv_scale) << (8 * k);
            q1 |= (uint32_t) ggml_cuda_float_to_fp4_e2m1(xb[k + 12], inv_scale) << (8 * k + 4);
        }

        qs_words[2 * sub + 0] = (int) q0;
        qs_words[2 * sub + 1] = (int) q1;
    }
}
#endif // defined(BLACKWELL_MMA_AVAILABLE)

static __device__ __forceinline__ void ggml_cuda_nvfp4_vx_write_col(
        ggml_cuda_nvfp4_vx_tile * out,
        const int col,
        const float * vals) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    int qs_words[GGML_CUDA_NVFP4_VX_WORDS_PER_COL];
    uint32_t scale_word;
    ggml_cuda_nvfp4_vx_quantize_64_words(vals, qs_words, scale_word);

    out->scale[col] = scale_word;
#pragma unroll
    for (int word = 0; word < GGML_CUDA_NVFP4_VX_WORDS_PER_COL; ++word) {
        out->qs[col * GGML_CUDA_NVFP4_VX_WORDS_PER_COL + word] = qs_words[word];
    }
#else
    GGML_UNUSED_VARS(out, col, vals);
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}

static __device__ __forceinline__ uint32_t ggml_cuda_nvfp4_kx_scale_word(const block_nvfp4 & blk) {
    return (uint32_t) blk.d[0] | ((uint32_t) blk.d[1] << 8) | ((uint32_t) blk.d[2] << 16) | ((uint32_t) blk.d[3] << 24);
}

static __device__ __forceinline__ uint32_t ggml_cuda_nvfp4_kx_qs_word(const block_nvfp4 & blk, const int word) {
    const int j = 4 * word;
    return (uint32_t) blk.qs[j + 0] |
        ((uint32_t) blk.qs[j + 1] << 8) |
        ((uint32_t) blk.qs[j + 2] << 16) |
        ((uint32_t) blk.qs[j + 3] << 24);
}

static __device__ __forceinline__ void ggml_cuda_nvfp4_kx_write_row_copy(
        ggml_cuda_nvfp4_kx_tile * out,
        const int row,
        const block_nvfp4 & blk) {
    out->scale[row] = ggml_cuda_nvfp4_kx_scale_word(blk);
#pragma unroll
    for (int word = 0; word < GGML_CUDA_NVFP4_KX_WORDS_PER_ROW; ++word) {
        out->qs[row * GGML_CUDA_NVFP4_KX_WORDS_PER_ROW + word] = (int32_t) ggml_cuda_nvfp4_kx_qs_word(blk, word);
    }
}

static __device__ __forceinline__ void ggml_cuda_nvfp4_kx_write_row(
        ggml_cuda_nvfp4_kx_tile * out,
        const int row,
        const float * vals) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    int qs_words[GGML_CUDA_NVFP4_KX_WORDS_PER_ROW];
    uint32_t scale_word;
    ggml_cuda_nvfp4_vx_quantize_64_words(vals, qs_words, scale_word);

    out->scale[row] = scale_word;
#pragma unroll
    for (int word = 0; word < GGML_CUDA_NVFP4_KX_WORDS_PER_ROW; ++word) {
        out->qs[row * GGML_CUDA_NVFP4_KX_WORDS_PER_ROW + word] = qs_words[word];
    }
#else
    GGML_UNUSED_VARS(out, row, vals);
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}

static __global__ void ggml_cuda_nvfp4_vx_rebuild_kernel(
        ggml_cuda_nvfp4_vx_layout layout,
        const block_nvfp4 * V,
        int64_t v_stride_row,
        int64_t v_stride_head,
        int64_t v_stride_seq,
        int64_t ne_kv_rows) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int64_t tile_id = (int64_t) blockIdx.x;
    const int64_t total = (int64_t) ggml_cuda_nvfp4_vx_tile_count(
        layout.n_stream, layout.n_kv_heads, layout.kv_size, layout.v_head_dim);

    if (tile_id >= total) {
        return;
    }

    int64_t rem = tile_id;
    const int64_t kv_tile = rem % layout.kv_tile_count;
    rem /= layout.kv_tile_count;
    const int64_t col_tile = rem % layout.col_tile_count;
    rem /= layout.col_tile_count;
    const int64_t kv_head = rem % layout.n_kv_heads;
    rem /= layout.n_kv_heads;
    const int64_t stream = rem;

    ggml_cuda_nvfp4_vx_tile * out = layout.tiles + tile_id;
    const int64_t kv_start = kv_tile * GGML_CUDA_NVFP4_VX_KV_TILE;
    const int64_t col_base = col_tile * GGML_CUDA_NVFP4_VX_COLS;

    if (threadIdx.x == 0) {
#pragma unroll
        for (int col = 0; col < GGML_CUDA_NVFP4_VX_COLS; ++col) {
            float vals[QK_NVFP4];

#pragma unroll
            for (int k = 0; k < QK_NVFP4; ++k) {
                const int64_t kv_row = kv_start + k;
                const int64_t v_col = col_base + col;

                if (kv_row < ne_kv_rows && v_col < layout.v_head_dim) {
                    const block_nvfp4 * row = V +
                        stream  * v_stride_seq +
                        kv_head * v_stride_head +
                        kv_row  * v_stride_row;
                    vals[k] = ggml_cuda_nvfp4_vx_dequant_row_value(row, (int) v_col);
                } else {
                    vals[k] = 0.0f;
                }
            }

            ggml_cuda_nvfp4_vx_write_col(out, col, vals);
        }
    }
#else
    GGML_UNUSED_VARS(layout, V, v_stride_row, v_stride_head, v_stride_seq, ne_kv_rows);
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}

static __global__ void ggml_cuda_nvfp4_kx_rebuild_kernel(
        ggml_cuda_nvfp4_kx_layout layout,
        const block_nvfp4 * K,
        int64_t k_stride_row,
        int64_t k_stride_head,
        int64_t k_stride_seq,
        int64_t ne_kv_rows) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int64_t tile_id = (int64_t) blockIdx.x;
    const int64_t total = (int64_t) ggml_cuda_nvfp4_kx_tile_count(
        layout.n_stream, layout.n_kv_heads, layout.kv_size, layout.k_head_dim);

    if (tile_id >= total || threadIdx.x != 0) {
        return;
    }

    int64_t rem = tile_id;
    const int64_t kv_tile = rem % layout.kv_tile_count;
    rem /= layout.kv_tile_count;
    const int64_t frag = rem % layout.frag_count;
    rem /= layout.frag_count;
    const int64_t kv_head = rem % layout.n_kv_heads;
    rem /= layout.n_kv_heads;
    const int64_t stream = rem;

    ggml_cuda_nvfp4_kx_tile * out = layout.tiles + tile_id;
    const int64_t kv_start = kv_tile * GGML_CUDA_NVFP4_KX_KV_ROWS;

#pragma unroll
    for (int row = 0; row < GGML_CUDA_NVFP4_KX_KV_ROWS; ++row) {
        const int64_t kv_row = kv_start + row;
        if (kv_row < ne_kv_rows) {
            const block_nvfp4 * k_row = K +
                stream  * k_stride_seq +
                kv_head * k_stride_head +
                kv_row  * k_stride_row;
            ggml_cuda_nvfp4_kx_write_row_copy(out, row, k_row[frag]);
        } else {
            float vals[QK_NVFP4] = {};
            ggml_cuda_nvfp4_kx_write_row(out, row, vals);
        }
    }
#else
    GGML_UNUSED_VARS(layout, K, k_stride_row, k_stride_head, k_stride_seq, ne_kv_rows);
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}

template <typename idx_t>
static __global__ void ggml_cuda_nvfp4_vx_update_dirty_tiles_f32_kernel(
        ggml_cuda_nvfp4_vx_layout layout,
        const ggml_cuda_nvfp4_vx_tile * old_tiles,
        const float * src0,
        const idx_t * row_ids,
        int64_t row_count,
        int64_t row_id_stride,
        int64_t src0_stride_row,
        int64_t logical_v_head_dim) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    int64_t rem = (int64_t) blockIdx.x;
    const int64_t kv_tile = rem % layout.kv_tile_count;
    rem /= layout.kv_tile_count;
    const int64_t col_tile = rem % layout.col_tile_count;
    rem /= layout.col_tile_count;
    const int64_t kv_head = rem;
    if (kv_head >= layout.n_kv_heads) {
        return;
    }

    const int64_t kv_start = kv_tile * GGML_CUDA_NVFP4_VX_KV_TILE;
    const int64_t kv_end = kv_start + GGML_CUDA_NVFP4_VX_KV_TILE;
    bool dirty = false;
    for (int64_t row = 0; row < row_count; ++row) {
        const int64_t dst_row = (int64_t) row_ids[row * row_id_stride];
        dirty = dirty || (dst_row >= kv_start && dst_row < kv_end);
    }
    if (!dirty || threadIdx.x != 0) {
        return;
    }

    ggml_cuda_nvfp4_vx_tile * out = ggml_cuda_nvfp4_vx_tile_ptr(layout, 0, kv_head, col_tile, kv_tile);
    const ggml_cuda_nvfp4_vx_tile old = *out;
    const int64_t col_base = col_tile * GGML_CUDA_NVFP4_VX_COLS;

#pragma unroll
    for (int col = 0; col < GGML_CUDA_NVFP4_VX_COLS; ++col) {
        float vals[QK_NVFP4];

#pragma unroll
        for (int k = 0; k < QK_NVFP4; ++k) {
            const int64_t kv_row = kv_start + k;
            const int64_t v_col = col_base + col;

            float val = 0.0f;
            if (kv_row < layout.kv_size && v_col < layout.v_head_dim) {
                val = ggml_cuda_nvfp4_vx_dequant_tile_value(old, col, k);
                for (int64_t row = 0; row < row_count; ++row) {
                    const int64_t dst_row = (int64_t) row_ids[row * row_id_stride];
                    if (dst_row == kv_row) {
                        const int64_t logical_col = kv_head * logical_v_head_dim + v_col;
                        val = src0[row * src0_stride_row + logical_col];
                        break;
                    }
                }
            }

            vals[k] = val;
        }

        ggml_cuda_nvfp4_vx_write_col(out, col, vals);
    }

    GGML_UNUSED(old_tiles);
#else
    GGML_UNUSED_VARS(layout, old_tiles, src0, row_ids, row_count, row_id_stride, src0_stride_row, logical_v_head_dim);
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}

template <typename idx_t>
static __global__ void ggml_cuda_nvfp4_kx_update_dirty_tiles_f32_kernel(
        ggml_cuda_nvfp4_kx_layout layout,
        const float * src0,
        const idx_t * row_ids,
        int64_t row_count,
        int64_t row_id_stride,
        int64_t src0_stride_row,
        int64_t logical_k_head_dim) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    int64_t rem = (int64_t) blockIdx.x;
    const int64_t kv_tile = rem % layout.kv_tile_count;
    rem /= layout.kv_tile_count;
    const int64_t frag = rem % layout.frag_count;
    rem /= layout.frag_count;
    const int64_t kv_head = rem;
    if (kv_head >= layout.n_kv_heads || threadIdx.x != 0) {
        return;
    }

    const int64_t kv_start = kv_tile * GGML_CUDA_NVFP4_KX_KV_ROWS;
    const int64_t kv_end = kv_start + GGML_CUDA_NVFP4_KX_KV_ROWS;
    bool dirty = false;
    for (int64_t row = 0; row < row_count; ++row) {
        const int64_t dst_row = (int64_t) row_ids[row * row_id_stride];
        dirty = dirty || (dst_row >= kv_start && dst_row < kv_end);
    }
    if (!dirty) {
        return;
    }

    ggml_cuda_nvfp4_kx_tile * out = ggml_cuda_nvfp4_kx_tile_ptr(layout, 0, kv_head, frag, kv_tile);
    const ggml_cuda_nvfp4_kx_tile old = *out;
    const int64_t frag_col_base = frag * GGML_CUDA_NVFP4_KX_HEAD_TILE;

#pragma unroll
    for (int row = 0; row < GGML_CUDA_NVFP4_KX_KV_ROWS; ++row) {
        const int64_t kv_row = kv_start + row;
        float vals[QK_NVFP4];

#pragma unroll
        for (int k = 0; k < QK_NVFP4; ++k) {
            const int64_t k_col = frag_col_base + k;
            float val = 0.0f;
            if (kv_row < layout.kv_size && k_col < layout.k_head_dim) {
                val = ggml_cuda_nvfp4_kx_dequant_tile_value(old, row, k);
                for (int64_t src_row = 0; src_row < row_count; ++src_row) {
                    const int64_t dst_row = (int64_t) row_ids[src_row * row_id_stride];
                    if (dst_row == kv_row) {
                        const int64_t logical_col = kv_head * logical_k_head_dim + k_col;
                        val = src0[src_row * src0_stride_row + logical_col];
                        break;
                    }
                }
            }

            vals[k] = val;
        }

        ggml_cuda_nvfp4_kx_write_row(out, row, vals);
    }
#else
    GGML_UNUSED_VARS(layout, src0, row_ids, row_count, row_id_stride, src0_stride_row, logical_k_head_dim);
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}

template <typename idx_t>
static __global__ void ggml_cuda_nvfp4_vx_update_dirty_tiles_kernel(
        ggml_cuda_nvfp4_vx_layout layout,
        const block_nvfp4 * V,
        const idx_t * row_ids,
        int64_t row_count,
        int64_t row_id_stride,
        int64_t v_stride_row,
        int64_t logical_v_head_dim) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    int64_t rem = (int64_t) blockIdx.x;
    const int64_t kv_tile = rem % layout.kv_tile_count;
    rem /= layout.kv_tile_count;
    const int64_t col_tile = rem % layout.col_tile_count;
    rem /= layout.col_tile_count;
    const int64_t kv_head = rem;
    if (kv_head >= layout.n_kv_heads) {
        return;
    }

    const int64_t kv_start = kv_tile * GGML_CUDA_NVFP4_VX_KV_TILE;
    bool dirty = false;
    for (int64_t row = 0; row < row_count; ++row) {
        const int64_t dst_row = row_ids[row * row_id_stride];
        dirty = dirty || (dst_row >= kv_start && dst_row < kv_start + GGML_CUDA_NVFP4_VX_KV_TILE);
    }
    if (!dirty) {
        return;
    }

    ggml_cuda_nvfp4_vx_tile * out = ggml_cuda_nvfp4_vx_tile_ptr(layout, 0, kv_head, col_tile, kv_tile);
    const int64_t col_base = col_tile * GGML_CUDA_NVFP4_VX_COLS;
    const int64_t head_col_base = kv_head * layout.v_head_dim;

    if (threadIdx.x == 0) {
#pragma unroll
        for (int col = 0; col < GGML_CUDA_NVFP4_VX_COLS; ++col) {
            float vals[QK_NVFP4];

#pragma unroll
            for (int k = 0; k < QK_NVFP4; ++k) {
                const int64_t kv_row = kv_start + k;
                const int64_t v_col = head_col_base + col_base + col;

                if (kv_row < layout.kv_size && v_col < logical_v_head_dim) {
                    const block_nvfp4 * row = V + kv_row * v_stride_row;
                    vals[k] = ggml_cuda_nvfp4_vx_dequant_row_value(row, (int) v_col);
                } else {
                    vals[k] = 0.0f;
                }
            }

            ggml_cuda_nvfp4_vx_write_col(out, col, vals);
        }
    }
#else
    GGML_UNUSED_VARS(layout, V, row_ids, row_count, row_id_stride, v_stride_row, logical_v_head_dim);
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}

template <typename idx_t>
static __global__ void ggml_cuda_nvfp4_kx_update_dirty_tiles_kernel(
        ggml_cuda_nvfp4_kx_layout layout,
        const block_nvfp4 * K,
        const idx_t * row_ids,
        int64_t row_count,
        int64_t row_id_stride,
        int64_t k_stride_row,
        int64_t logical_k_head_dim) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    int64_t rem = (int64_t) blockIdx.x;
    const int64_t kv_tile = rem % layout.kv_tile_count;
    rem /= layout.kv_tile_count;
    const int64_t frag = rem % layout.frag_count;
    rem /= layout.frag_count;
    const int64_t kv_head = rem;
    if (kv_head >= layout.n_kv_heads || threadIdx.x != 0) {
        return;
    }

    const int64_t kv_start = kv_tile * GGML_CUDA_NVFP4_KX_KV_ROWS;
    bool dirty = false;
    for (int64_t row = 0; row < row_count; ++row) {
        const int64_t dst_row = row_ids[row * row_id_stride];
        dirty = dirty || (dst_row >= kv_start && dst_row < kv_start + GGML_CUDA_NVFP4_KX_KV_ROWS);
    }
    if (!dirty) {
        return;
    }

    ggml_cuda_nvfp4_kx_tile * out = ggml_cuda_nvfp4_kx_tile_ptr(layout, 0, kv_head, frag, kv_tile);
    const int64_t head_col_base = kv_head * layout.k_head_dim;

#pragma unroll
    for (int row = 0; row < GGML_CUDA_NVFP4_KX_KV_ROWS; ++row) {
        const int64_t kv_row = kv_start + row;
        const int64_t k_col = head_col_base + frag * GGML_CUDA_NVFP4_KX_HEAD_TILE;

        if (kv_row < layout.kv_size && k_col < logical_k_head_dim) {
            const block_nvfp4 * k_row = K + kv_row * k_stride_row;
            ggml_cuda_nvfp4_kx_write_row_copy(out, row, k_row[head_col_base / QK_NVFP4 + frag]);
        } else {
            float vals[QK_NVFP4] = {};
            ggml_cuda_nvfp4_kx_write_row(out, row, vals);
        }
    }
#else
    GGML_UNUSED_VARS(layout, K, row_ids, row_count, row_id_stride, k_stride_row, logical_k_head_dim);
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}

static __global__ void ggml_cuda_nvfp4_vx_validate_kernel(
        ggml_cuda_nvfp4_vx_layout layout,
        const block_nvfp4 * V,
        int64_t v_stride_row,
        int64_t v_stride_head,
        int64_t v_stride_seq,
        int64_t ne_kv_rows,
        uint32_t * max_error_bits,
        float * sum_error) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int64_t total = layout.n_stream * layout.n_kv_heads * ne_kv_rows * layout.v_head_dim;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (int64_t) blockDim.x * gridDim.x) {
        int64_t rem = i;
        const int64_t v_col = rem % layout.v_head_dim;
        rem /= layout.v_head_dim;
        const int64_t kv_row = rem % ne_kv_rows;
        rem /= ne_kv_rows;
        const int64_t kv_head = rem % layout.n_kv_heads;
        rem /= layout.n_kv_heads;
        const int64_t stream = rem;

        const block_nvfp4 * row = V + stream * v_stride_seq + kv_head * v_stride_head + kv_row * v_stride_row;
        const float logical = ggml_cuda_nvfp4_vx_dequant_row_value(row, (int) v_col);
        const float vx = ggml_cuda_nvfp4_vx_dequant_value(layout, stream, kv_head, kv_row, v_col);
        const float err = fabsf(logical - vx);

        atomicAdd(sum_error, err);
        atomicMax(max_error_bits, __float_as_uint(err));
    }
#else
    GGML_UNUSED_VARS(layout, V, v_stride_row, v_stride_head, v_stride_seq, ne_kv_rows, max_error_bits, sum_error);
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}

static __global__ void ggml_cuda_nvfp4_kx_validate_kernel(
        ggml_cuda_nvfp4_kx_layout layout,
        const block_nvfp4 * K,
        int64_t k_stride_row,
        int64_t k_stride_head,
        int64_t k_stride_seq,
        int64_t ne_kv_rows,
        uint32_t * max_error_bits,
        float * sum_error) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int64_t total = layout.n_stream * layout.n_kv_heads * ne_kv_rows * layout.k_head_dim;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (int64_t) blockDim.x * gridDim.x) {
        int64_t rem = i;
        const int64_t k_col = rem % layout.k_head_dim;
        rem /= layout.k_head_dim;
        const int64_t kv_row = rem % ne_kv_rows;
        rem /= ne_kv_rows;
        const int64_t kv_head = rem % layout.n_kv_heads;
        rem /= layout.n_kv_heads;
        const int64_t stream = rem;

        const block_nvfp4 * row = K + stream * k_stride_seq + kv_head * k_stride_head + kv_row * k_stride_row;
        const float logical = ggml_cuda_nvfp4_vx_dequant_row_value(row, (int) k_col);
        const float kx = ggml_cuda_nvfp4_kx_dequant_value(layout, stream, kv_head, kv_row, k_col);
        const float err = fabsf(logical - kx);

        atomicAdd(sum_error, err);
        atomicMax(max_error_bits, __float_as_uint(err));
    }
#else
    GGML_UNUSED_VARS(layout, K, k_stride_row, k_stride_head, k_stride_seq, ne_kv_rows, max_error_bits, sum_error);
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}

static ggml_cuda_nvfp4_vx_layout ggml_cuda_nvfp4_vx_get_or_create(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * V) {
    const void * key = ggml_cuda_nvfp4_vx_key(V);
    GGML_ASSERT(key != nullptr);

    const int device = ctx.device;
    const int64_t n_stream = V->ne[3];
    const int64_t n_kv_heads = V->ne[2];
    const int64_t kv_size = V->ne[1];
    const int64_t v_head_dim = V->ne[0];
    const int64_t kv_tile_count = ggml_cuda_nvfp4_vx_kv_tile_count(kv_size);
    const int64_t col_tile_count = ggml_cuda_nvfp4_vx_col_tile_count(v_head_dim);
    const size_t tile_count = ggml_cuda_nvfp4_vx_tile_count(n_stream, n_kv_heads, kv_size, v_head_dim);
    const size_t bytes = tile_count * GGML_CUDA_NVFP4_VX_TILE_BYTES;

    std::lock_guard<std::mutex> lock(g_nvfp4_vx_mutex);
    ggml_cuda_nvfp4_vx_entry & entry = g_nvfp4_vx_registry[&ctx][key];
    const bool needs_alloc =
        entry.layout.tiles == nullptr ||
        entry.device != device ||
        entry.tile_count != tile_count ||
        entry.layout.n_stream != n_stream ||
        entry.layout.n_kv_heads != n_kv_heads ||
        entry.layout.kv_size != kv_size ||
        entry.layout.v_head_dim != v_head_dim;

    if (needs_alloc) {
        if (entry.layout.tiles != nullptr) {
            CUDA_CHECK(cudaFree(entry.layout.tiles));
        }

        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaMalloc((void **) &entry.layout.tiles, bytes));
        CUDA_CHECK(cudaMemsetAsync(entry.layout.tiles, 0, bytes, ctx.stream()));

        entry.device = device;
        entry.tile_count = tile_count;
        entry.layout.n_stream = n_stream;
        entry.layout.n_kv_heads = n_kv_heads;
        entry.layout.kv_size = kv_size;
        entry.layout.v_head_dim = v_head_dim;
        entry.layout.kv_tile_count = kv_tile_count;
        entry.layout.col_tile_count = col_tile_count;
        entry.direct_updates = false;
    }

    return entry.layout;
}

static ggml_cuda_nvfp4_kx_layout ggml_cuda_nvfp4_kx_get_or_create(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * K) {
    const void * key = ggml_cuda_nvfp4_kx_key(K);
    GGML_ASSERT(key != nullptr);

    const int device = ctx.device;
    const int64_t n_stream = K->ne[3];
    const int64_t n_kv_heads = K->ne[2];
    const int64_t kv_size = K->ne[1];
    const int64_t k_head_dim = K->ne[0];
    const int64_t kv_tile_count = ggml_cuda_nvfp4_kx_kv_tile_count(kv_size);
    const int64_t frag_count = ggml_cuda_nvfp4_kx_frag_count(k_head_dim);
    const size_t tile_count = ggml_cuda_nvfp4_kx_tile_count(n_stream, n_kv_heads, kv_size, k_head_dim);
    const size_t bytes = tile_count * GGML_CUDA_NVFP4_KX_TILE_BYTES;

    std::lock_guard<std::mutex> lock(g_nvfp4_kx_mutex);
    ggml_cuda_nvfp4_kx_entry & entry = g_nvfp4_kx_registry[&ctx][key];
    const bool needs_alloc =
        entry.layout.tiles == nullptr ||
        entry.device != device ||
        entry.tile_count != tile_count ||
        entry.layout.n_stream != n_stream ||
        entry.layout.n_kv_heads != n_kv_heads ||
        entry.layout.kv_size != kv_size ||
        entry.layout.k_head_dim != k_head_dim;

    if (needs_alloc) {
        if (entry.layout.tiles != nullptr) {
            CUDA_CHECK(cudaFree(entry.layout.tiles));
        }

        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaMalloc((void **) &entry.layout.tiles, bytes));
        CUDA_CHECK(cudaMemsetAsync(entry.layout.tiles, 0, bytes, ctx.stream()));

        entry.device = device;
        entry.tile_count = tile_count;
        entry.layout.n_stream = n_stream;
        entry.layout.n_kv_heads = n_kv_heads;
        entry.layout.kv_size = kv_size;
        entry.layout.k_head_dim = k_head_dim;
        entry.layout.kv_tile_count = kv_tile_count;
        entry.layout.frag_count = frag_count;
        entry.direct_updates = false;
    }

    return entry.layout;
}

static bool ggml_cuda_nvfp4_vx_logical_set_rows_supported(
        const ggml_tensor * dst,
        const ggml_cuda_nvfp4_vx_layout & layout) {
    if (dst == nullptr || dst->type != GGML_TYPE_NVFP4 || dst->src[0] == nullptr || dst->src[1] == nullptr) {
        return false;
    }

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    if (src0->ne[1] <= 0 || src0->ne[2] * src0->ne[3] != 1) {
        return false;
    }

    if (src0->ne[0] != dst->ne[0]) {
        return false;
    }

    if (src1->ne[0] < src0->ne[1]) {
        return false;
    }

    if (src1->type != GGML_TYPE_I64 && src1->type != GGML_TYPE_I32) {
        return false;
    }

    if (dst->ne[0] != layout.n_kv_heads * layout.v_head_dim ||
            dst->ne[1] < layout.kv_size ||
            dst->ne[2] * dst->ne[3] != layout.n_stream ||
            layout.n_stream != 1) {
        return false;
    }

    return true;
}

static bool ggml_cuda_nvfp4_kx_logical_set_rows_supported(
        const ggml_tensor * dst,
        const ggml_cuda_nvfp4_kx_layout & layout) {
    if (dst == nullptr || dst->type != GGML_TYPE_NVFP4 || dst->src[0] == nullptr || dst->src[1] == nullptr) {
        return false;
    }

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    if (src0->ne[1] <= 0 || src0->ne[2] * src0->ne[3] != 1) {
        return false;
    }

    if (src0->ne[0] != dst->ne[0]) {
        return false;
    }

    if (src1->ne[0] < src0->ne[1]) {
        return false;
    }

    if (src1->type != GGML_TYPE_I64 && src1->type != GGML_TYPE_I32) {
        return false;
    }

    // SET_ROWS row ids are global cache-row ids. This dirty-tile updater only
    // handles the single-stream layout where global and local row ids coincide.
    if (dst->ne[0] != layout.n_kv_heads * layout.k_head_dim ||
            dst->ne[1] < layout.kv_size ||
            dst->ne[2] * dst->ne[3] != layout.n_stream ||
            layout.n_stream != 1) {
        return false;
    }

    return true;
}

static bool ggml_cuda_nvfp4_vx_f32_set_rows_supported(
        const ggml_tensor * dst,
        const ggml_cuda_nvfp4_vx_layout & layout) {
    if (!ggml_cuda_nvfp4_vx_logical_set_rows_supported(dst, layout)) {
        return false;
    }

    const ggml_tensor * src0 = dst->src[0];
    if (src0->type != GGML_TYPE_F32 || src0->nb[0] != sizeof(float)) {
        return false;
    }

    if (src0->nb[1] % sizeof(float) != 0) {
        return false;
    }

    return true;
}

static bool ggml_cuda_nvfp4_kx_f32_set_rows_supported(
        const ggml_tensor * dst,
        const ggml_cuda_nvfp4_kx_layout & layout) {
    // Kx is currently an execution copy of the logical NVFP4 K cache. Update
    // it from the already-written logical cache so rebuilds and dirty-tile
    // updates have identical quantization semantics.
    GGML_UNUSED_VARS(dst, layout);
    return false;
}

static void ggml_cuda_nvfp4_vx_mark_direct_updates(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * V) {
    const void * key = ggml_cuda_nvfp4_vx_key(V);
    if (key == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_nvfp4_vx_mutex);
    auto ctx_it = g_nvfp4_vx_registry.find(&ctx);
    if (ctx_it == g_nvfp4_vx_registry.end()) {
        return;
    }

    auto entry_it = ctx_it->second.find(key);
    if (entry_it == ctx_it->second.end()) {
        return;
    }

    entry_it->second.direct_updates = true;
}

static void ggml_cuda_nvfp4_kx_mark_direct_updates(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * K) {
    const void * key = ggml_cuda_nvfp4_kx_key(K);
    if (key == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_nvfp4_kx_mutex);
    auto ctx_it = g_nvfp4_kx_registry.find(&ctx);
    if (ctx_it == g_nvfp4_kx_registry.end()) {
        return;
    }

    auto entry_it = ctx_it->second.find(key);
    if (entry_it == ctx_it->second.end()) {
        return;
    }

    entry_it->second.direct_updates = true;
}

bool ggml_cuda_nvfp4_vx_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * V) {
    if (!ggml_cuda_nvfp4_vx_supported_view(V)) {
        return false;
    }

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!blackwell_mma_available(cc)) {
        return false;
    }

    const ggml_cuda_nvfp4_vx_layout layout = ggml_cuda_nvfp4_vx_get_or_create(ctx, V);

    const int64_t v_stride_row  = ggml_cuda_nvfp4_vx_stride_blocks(V, 1);
    const int64_t v_stride_head = ggml_cuda_nvfp4_vx_stride_blocks(V, 2);
    const int64_t v_stride_seq  = ggml_cuda_nvfp4_vx_stride_blocks(V, 3);

    const dim3 block(WARP_SIZE, 1, 1);
    const dim3 grid((uint32_t) layout.n_stream * layout.n_kv_heads * layout.col_tile_count * layout.kv_tile_count, 1, 1);
    ggml_cuda_nvfp4_vx_rebuild_kernel<<<grid, block, 0, ctx.stream()>>>(
        layout, (const block_nvfp4 *) V->data, v_stride_row, v_stride_head, v_stride_seq, V->ne[1]);
    CUDA_CHECK(cudaGetLastError());

    {
        const void * key = ggml_cuda_nvfp4_vx_key(V);
        std::lock_guard<std::mutex> lock(g_nvfp4_vx_mutex);
        auto ctx_it = g_nvfp4_vx_registry.find(&ctx);
        if (ctx_it != g_nvfp4_vx_registry.end()) {
            auto entry_it = ctx_it->second.find(key);
            if (entry_it != ctx_it->second.end()) {
                entry_it->second.direct_updates = false;
            }
        }
    }

    return true;
}

bool ggml_cuda_nvfp4_kx_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * K) {
    if (!ggml_cuda_nvfp4_kx_supported_view(K)) {
        return false;
    }

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!blackwell_mma_available(cc)) {
        return false;
    }

    const ggml_cuda_nvfp4_kx_layout layout = ggml_cuda_nvfp4_kx_get_or_create(ctx, K);

    const int64_t k_stride_row  = ggml_cuda_nvfp4_kx_stride_blocks(K, 1);
    const int64_t k_stride_head = ggml_cuda_nvfp4_kx_stride_blocks(K, 2);
    const int64_t k_stride_seq  = ggml_cuda_nvfp4_kx_stride_blocks(K, 3);

    const dim3 block(WARP_SIZE, 1, 1);
    const dim3 grid((uint32_t) layout.n_stream * layout.n_kv_heads * layout.frag_count * layout.kv_tile_count, 1, 1);
    ggml_cuda_nvfp4_kx_rebuild_kernel<<<grid, block, 0, ctx.stream()>>>(
        layout, (const block_nvfp4 *) K->data, k_stride_row, k_stride_head, k_stride_seq, K->ne[1]);
    CUDA_CHECK(cudaGetLastError());

    {
        const void * key = ggml_cuda_nvfp4_kx_key(K);
        std::lock_guard<std::mutex> lock(g_nvfp4_kx_mutex);
        auto ctx_it = g_nvfp4_kx_registry.find(&ctx);
        if (ctx_it != g_nvfp4_kx_registry.end()) {
            auto entry_it = ctx_it->second.find(key);
            if (entry_it != ctx_it->second.end()) {
                entry_it->second.direct_updates = false;
            }
        }
    }

    return true;
}

template <typename idx_t>
static bool ggml_cuda_nvfp4_vx_after_set_rows_f32_t(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * dst,
        const ggml_cuda_nvfp4_vx_layout & layout) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const int64_t row_count = src0->ne[1];
    const int64_t row_id_stride = src1->nb[0] / sizeof(idx_t);
    const int64_t src0_stride_row = src0->nb[1] / sizeof(float);
    const int64_t logical_v_head_dim = dst->ne[0] / layout.n_kv_heads;

    const dim3 block(WARP_SIZE, 1, 1);
    const dim3 grid((uint32_t) (layout.n_kv_heads * layout.col_tile_count * layout.kv_tile_count), 1, 1);
    ggml_cuda_nvfp4_vx_update_dirty_tiles_f32_kernel<idx_t><<<grid, block, 0, ctx.stream()>>>(
        layout,
        layout.tiles,
        (const float *) src0->data,
        (const idx_t *) src1->data,
        row_count,
        row_id_stride,
        src0_stride_row,
        logical_v_head_dim);
    CUDA_CHECK(cudaGetLastError());
    ggml_cuda_nvfp4_vx_mark_direct_updates(ctx, dst);
    return true;
}

template <typename idx_t>
static bool ggml_cuda_nvfp4_kx_after_set_rows_f32_t(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * dst,
        const ggml_cuda_nvfp4_kx_layout & layout) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const int64_t row_count = src0->ne[1];
    const int64_t row_id_stride = src1->nb[0] / sizeof(idx_t);
    const int64_t src0_stride_row = src0->nb[1] / sizeof(float);
    const int64_t logical_k_head_dim = dst->ne[0] / layout.n_kv_heads;

    const dim3 block(WARP_SIZE, 1, 1);
    const dim3 grid((uint32_t) (layout.n_kv_heads * layout.frag_count * layout.kv_tile_count), 1, 1);
    ggml_cuda_nvfp4_kx_update_dirty_tiles_f32_kernel<idx_t><<<grid, block, 0, ctx.stream()>>>(
        layout,
        (const float *) src0->data,
        (const idx_t *) src1->data,
        row_count,
        row_id_stride,
        src0_stride_row,
        logical_k_head_dim);
    CUDA_CHECK(cudaGetLastError());
    ggml_cuda_nvfp4_kx_mark_direct_updates(ctx, dst);
    return true;
}

template <typename idx_t>
static bool ggml_cuda_nvfp4_vx_after_set_rows_t(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * dst,
        const ggml_cuda_nvfp4_vx_layout & layout) {
    const ggml_tensor * src1 = dst->src[1];

    const int64_t v_stride_row = ggml_cuda_nvfp4_vx_stride_blocks(dst, 1);
    const int64_t logical_v_head_dim = dst->ne[0];
    const int64_t row_count = dst->src[0]->ne[1];
    const int64_t row_id_stride = src1->nb[0] / sizeof(idx_t);

    const dim3 block(WARP_SIZE, 1, 1);
    const dim3 grid((uint32_t) (layout.n_kv_heads * layout.col_tile_count * layout.kv_tile_count), 1, 1);
    ggml_cuda_nvfp4_vx_update_dirty_tiles_kernel<idx_t><<<grid, block, 0, ctx.stream()>>>(
        layout,
        (const block_nvfp4 *) dst->data,
        (const idx_t *) src1->data,
        row_count,
        row_id_stride,
        v_stride_row,
        logical_v_head_dim);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

template <typename idx_t>
static bool ggml_cuda_nvfp4_kx_after_set_rows_t(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * dst,
        const ggml_cuda_nvfp4_kx_layout & layout) {
    const ggml_tensor * src1 = dst->src[1];

    const int64_t k_stride_row = ggml_cuda_nvfp4_kx_stride_blocks(dst, 1);
    const int64_t logical_k_head_dim = dst->ne[0];
    const int64_t row_count = dst->src[0]->ne[1];
    const int64_t row_id_stride = src1->nb[0] / sizeof(idx_t);

    const dim3 block(WARP_SIZE, 1, 1);
    const dim3 grid((uint32_t) (layout.n_kv_heads * layout.frag_count * layout.kv_tile_count), 1, 1);
    ggml_cuda_nvfp4_kx_update_dirty_tiles_kernel<idx_t><<<grid, block, 0, ctx.stream()>>>(
        layout,
        (const block_nvfp4 *) dst->data,
        (const idx_t *) src1->data,
        row_count,
        row_id_stride,
        k_stride_row,
        logical_k_head_dim);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_nvfp4_vx_after_set_rows(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    ggml_cuda_nvfp4_vx_layout layout = {};
    if (!ggml_cuda_nvfp4_vx_find(ctx, dst, &layout)) {
        return false;
    }

    if (!ggml_cuda_nvfp4_vx_logical_set_rows_supported(dst, layout)) {
        return false;
    }

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!blackwell_mma_available(cc)) {
        return false;
    }

    if (ggml_cuda_nvfp4_vx_f32_set_rows_supported(dst, layout)) {
        if (dst->src[1]->type == GGML_TYPE_I64) {
            return ggml_cuda_nvfp4_vx_after_set_rows_f32_t<int64_t>(ctx, dst, layout);
        }

        return ggml_cuda_nvfp4_vx_after_set_rows_f32_t<int32_t>(ctx, dst, layout);
    }

    if (dst->src[1]->type == GGML_TYPE_I64) {
        return ggml_cuda_nvfp4_vx_after_set_rows_t<int64_t>(ctx, dst, layout);
    }

    return ggml_cuda_nvfp4_vx_after_set_rows_t<int32_t>(ctx, dst, layout);
}

bool ggml_cuda_nvfp4_kx_after_set_rows(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    ggml_cuda_nvfp4_kx_layout layout = {};
    if (!ggml_cuda_nvfp4_kx_find(ctx, dst, &layout)) {
        return false;
    }

    if (!ggml_cuda_nvfp4_kx_logical_set_rows_supported(dst, layout)) {
        return false;
    }

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!blackwell_mma_available(cc)) {
        return false;
    }

    if (ggml_cuda_nvfp4_kx_f32_set_rows_supported(dst, layout)) {
        if (dst->src[1]->type == GGML_TYPE_I64) {
            return ggml_cuda_nvfp4_kx_after_set_rows_f32_t<int64_t>(ctx, dst, layout);
        }

        return ggml_cuda_nvfp4_kx_after_set_rows_f32_t<int32_t>(ctx, dst, layout);
    }

    if (dst->src[1]->type == GGML_TYPE_I64) {
        return ggml_cuda_nvfp4_kx_after_set_rows_t<int64_t>(ctx, dst, layout);
    }

    return ggml_cuda_nvfp4_kx_after_set_rows_t<int32_t>(ctx, dst, layout);
}

bool ggml_cuda_nvfp4_vx_find(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * V,
        ggml_cuda_nvfp4_vx_layout * layout) {
    if (layout == nullptr) {
        return false;
    }

    const void * key = ggml_cuda_nvfp4_vx_key(V);
    if (key == nullptr) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_nvfp4_vx_mutex);
    auto ctx_it = g_nvfp4_vx_registry.find(&ctx);
    if (ctx_it == g_nvfp4_vx_registry.end()) {
        return false;
    }

    auto entry_it = ctx_it->second.find(key);
    if (entry_it == ctx_it->second.end()) {
        return false;
    }

    *layout = entry_it->second.layout;
    return true;
}

bool ggml_cuda_nvfp4_kx_find(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * K,
        ggml_cuda_nvfp4_kx_layout * layout) {
    if (layout == nullptr) {
        return false;
    }

    const void * key = ggml_cuda_nvfp4_kx_key(K);
    if (key == nullptr) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_nvfp4_kx_mutex);
    auto ctx_it = g_nvfp4_kx_registry.find(&ctx);
    if (ctx_it == g_nvfp4_kx_registry.end()) {
        return false;
    }

    auto entry_it = ctx_it->second.find(key);
    if (entry_it == ctx_it->second.end()) {
        return false;
    }

    *layout = entry_it->second.layout;
    return true;
}

bool ggml_cuda_nvfp4_vx_has_direct_updates(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * V) {
    const void * key = ggml_cuda_nvfp4_vx_key(V);
    if (key == nullptr) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_nvfp4_vx_mutex);
    auto ctx_it = g_nvfp4_vx_registry.find(&ctx);
    if (ctx_it == g_nvfp4_vx_registry.end()) {
        return false;
    }

    auto entry_it = ctx_it->second.find(key);
    if (entry_it == ctx_it->second.end()) {
        return false;
    }

    return entry_it->second.direct_updates;
}

bool ggml_cuda_nvfp4_kx_has_direct_updates(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * K) {
    const void * key = ggml_cuda_nvfp4_kx_key(K);
    if (key == nullptr) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_nvfp4_kx_mutex);
    auto ctx_it = g_nvfp4_kx_registry.find(&ctx);
    if (ctx_it == g_nvfp4_kx_registry.end()) {
        return false;
    }

    auto entry_it = ctx_it->second.find(key);
    if (entry_it == ctx_it->second.end()) {
        return false;
    }

    return entry_it->second.direct_updates;
}

bool ggml_cuda_nvfp4_vx_validate(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * V,
        float * max_abs_error,
        float * mean_abs_error) {
    if (max_abs_error != nullptr) {
        *max_abs_error = 0.0f;
    }
    if (mean_abs_error != nullptr) {
        *mean_abs_error = 0.0f;
    }

    if (!ggml_cuda_nvfp4_vx_supported_view(V)) {
        return false;
    }

    ggml_cuda_nvfp4_vx_layout layout = {};
    if (!ggml_cuda_nvfp4_vx_find(ctx, V, &layout)) {
        return false;
    }

    const int64_t value_count = layout.n_stream * layout.n_kv_heads * V->ne[1] * layout.v_head_dim;
    if (value_count <= 0) {
        return false;
    }

    uint32_t * max_error_bits_d = nullptr;
    float * sum_error_d = nullptr;
    CUDA_CHECK(cudaMalloc((void **) &max_error_bits_d, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void **) &sum_error_d, sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(max_error_bits_d, 0, sizeof(uint32_t), ctx.stream()));
    CUDA_CHECK(cudaMemsetAsync(sum_error_d, 0, sizeof(float), ctx.stream()));

    const int64_t v_stride_row  = ggml_cuda_nvfp4_vx_stride_blocks(V, 1);
    const int64_t v_stride_head = ggml_cuda_nvfp4_vx_stride_blocks(V, 2);
    const int64_t v_stride_seq  = ggml_cuda_nvfp4_vx_stride_blocks(V, 3);

    const dim3 block(256, 1, 1);
    int64_t grid_x = (value_count + block.x - 1) / block.x;
    grid_x = grid_x < 65535 ? grid_x : 65535;
    const dim3 grid((uint32_t) grid_x, 1, 1);
    ggml_cuda_nvfp4_vx_validate_kernel<<<grid, block, 0, ctx.stream()>>>(
        layout, (const block_nvfp4 *) V->data, v_stride_row, v_stride_head, v_stride_seq, V->ne[1],
        max_error_bits_d, sum_error_d);
    CUDA_CHECK(cudaGetLastError());

    uint32_t max_error_bits_h = 0;
    float sum_error_h = 0.0f;
    CUDA_CHECK(cudaMemcpyAsync(&max_error_bits_h, max_error_bits_d, sizeof(max_error_bits_h), cudaMemcpyDeviceToHost, ctx.stream()));
    CUDA_CHECK(cudaMemcpyAsync(&sum_error_h, sum_error_d, sizeof(sum_error_h), cudaMemcpyDeviceToHost, ctx.stream()));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    CUDA_CHECK(cudaFree(max_error_bits_d));
    CUDA_CHECK(cudaFree(sum_error_d));

    if (max_abs_error != nullptr) {
        std::memcpy(max_abs_error, &max_error_bits_h, sizeof(*max_abs_error));
    }
    if (mean_abs_error != nullptr) {
        *mean_abs_error = sum_error_h / (float) value_count;
    }

    return true;
}

bool ggml_cuda_nvfp4_kx_validate(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * K,
        float * max_abs_error,
        float * mean_abs_error) {
    if (max_abs_error != nullptr) {
        *max_abs_error = 0.0f;
    }
    if (mean_abs_error != nullptr) {
        *mean_abs_error = 0.0f;
    }

    if (!ggml_cuda_nvfp4_kx_supported_view(K)) {
        return false;
    }

    ggml_cuda_nvfp4_kx_layout layout = {};
    if (!ggml_cuda_nvfp4_kx_find(ctx, K, &layout)) {
        return false;
    }

    const int64_t value_count = layout.n_stream * layout.n_kv_heads * K->ne[1] * layout.k_head_dim;
    if (value_count <= 0) {
        return false;
    }

    uint32_t * max_error_bits_d = nullptr;
    float * sum_error_d = nullptr;
    CUDA_CHECK(cudaMalloc((void **) &max_error_bits_d, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void **) &sum_error_d, sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(max_error_bits_d, 0, sizeof(uint32_t), ctx.stream()));
    CUDA_CHECK(cudaMemsetAsync(sum_error_d, 0, sizeof(float), ctx.stream()));

    const int64_t k_stride_row  = ggml_cuda_nvfp4_kx_stride_blocks(K, 1);
    const int64_t k_stride_head = ggml_cuda_nvfp4_kx_stride_blocks(K, 2);
    const int64_t k_stride_seq  = ggml_cuda_nvfp4_kx_stride_blocks(K, 3);

    const dim3 block(256, 1, 1);
    int64_t grid_x = (value_count + block.x - 1) / block.x;
    grid_x = grid_x < 65535 ? grid_x : 65535;
    const dim3 grid((uint32_t) grid_x, 1, 1);
    ggml_cuda_nvfp4_kx_validate_kernel<<<grid, block, 0, ctx.stream()>>>(
        layout, (const block_nvfp4 *) K->data, k_stride_row, k_stride_head, k_stride_seq, K->ne[1],
        max_error_bits_d, sum_error_d);
    CUDA_CHECK(cudaGetLastError());

    uint32_t max_error_bits_h = 0;
    float sum_error_h = 0.0f;
    CUDA_CHECK(cudaMemcpyAsync(&max_error_bits_h, max_error_bits_d, sizeof(max_error_bits_h), cudaMemcpyDeviceToHost, ctx.stream()));
    CUDA_CHECK(cudaMemcpyAsync(&sum_error_h, sum_error_d, sizeof(sum_error_h), cudaMemcpyDeviceToHost, ctx.stream()));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    CUDA_CHECK(cudaFree(max_error_bits_d));
    CUDA_CHECK(cudaFree(sum_error_d));

    if (max_abs_error != nullptr) {
        std::memcpy(max_abs_error, &max_error_bits_h, sizeof(*max_abs_error));
    }
    if (mean_abs_error != nullptr) {
        *mean_abs_error = sum_error_h / (float) value_count;
    }

    return true;
}

void ggml_cuda_nvfp4_vx_clear_context(ggml_backend_cuda_context & ctx) {
    std::lock_guard<std::mutex> lock(g_nvfp4_vx_mutex);
    auto ctx_it = g_nvfp4_vx_registry.find(&ctx);
    if (ctx_it == g_nvfp4_vx_registry.end()) {
        return;
    }

    ggml_cuda_set_device(ctx.device);
    for (auto & it : ctx_it->second) {
        ggml_cuda_nvfp4_vx_entry & entry = it.second;
        if (entry.layout.tiles != nullptr) {
            CUDA_CHECK(cudaFree(entry.layout.tiles));
            entry.layout.tiles = nullptr;
        }
    }

    g_nvfp4_vx_registry.erase(ctx_it);
}

void ggml_cuda_nvfp4_kx_clear_context(ggml_backend_cuda_context & ctx) {
    std::lock_guard<std::mutex> lock(g_nvfp4_kx_mutex);
    auto ctx_it = g_nvfp4_kx_registry.find(&ctx);
    if (ctx_it == g_nvfp4_kx_registry.end()) {
        return;
    }

    ggml_cuda_set_device(ctx.device);
    for (auto & it : ctx_it->second) {
        ggml_cuda_nvfp4_kx_entry & entry = it.second;
        if (entry.layout.tiles != nullptr) {
            CUDA_CHECK(cudaFree(entry.layout.tiles));
            entry.layout.tiles = nullptr;
        }
    }

    g_nvfp4_kx_registry.erase(ctx_it);
}

void ggml_cuda_nvfp4_kv_exec_clear_context(ggml_backend_cuda_context & ctx) {
    ggml_cuda_nvfp4_vx_clear_context(ctx);
    ggml_cuda_nvfp4_kx_clear_context(ctx);
}

#endif // defined(GGML_CUDA_NVFP4_KV_EXEC_LAYOUT)
