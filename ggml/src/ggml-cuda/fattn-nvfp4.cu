#include "fattn-nvfp4.cuh"

#include "cpy-utils.cuh"
#include "fattn-common.cuh"
#include "mma.cuh"
#if defined(GGML_CUDA_NVFP4_KV_EXEC_LAYOUT)
#include "nvfp4-kv-exec.cuh"
#endif

#include <cmath>
#include <cstring>

#if defined(GGML_CUDA_NVFP4_FA)
#define GGML_CUDA_NVFP4_FA_MMA
#define GGML_CUDA_NVFP4_FA_MMA_MULTIWARP
#if !defined(GGML_CUDA_NVFP4_FA_SCHEDULE_FUSED)
#define GGML_CUDA_NVFP4_FA_MMA_SPLIT_KV
#endif
#if defined(GGML_CUDA_NVFP4_FA_SCHEDULE_DECOUPLED)
#define GGML_CUDA_NVFP4_FA_MMA_DECOUPLED
#elif defined(GGML_CUDA_NVFP4_FA_SCHEDULE_TWOPASS)
#define GGML_CUDA_NVFP4_FA_MMA_TWOPASS
#endif
#endif

#if defined(GGML_CUDA_NVFP4_FA)
#define GGML_CUDA_NVFP4_FA_ACTIVE_SPLIT_KV_ROWS GGML_CUDA_NVFP4_FA_SPLIT_KV_ROWS
#endif

#if defined(GGML_CUDA_NVFP4_FA)
#define GGML_CUDA_NVFP4_FA_ACTIVE_DECOUPLED_WINDOW_SPLITS GGML_CUDA_NVFP4_FA_DECOUPLED_WINDOW_SPLITS
#endif

static constexpr size_t FATTN_NVFP4_LUT_SCALE_COUNT = 256;
static constexpr size_t FATTN_NVFP4_LUT_FP4_COUNT   = 256;
static constexpr size_t FATTN_NVFP4_LUT_SIZE        = FATTN_NVFP4_LUT_SCALE_COUNT * FATTN_NVFP4_LUT_FP4_COUNT;
static constexpr int    FATTN_NVFP4_MTP4_ROWS       = 4;
static constexpr int    FATTN_NVFP4_MAX_HEAD_DIM    = 512;
static constexpr int    FATTN_NVFP4_MAX_NFRAG       = FATTN_NVFP4_MAX_HEAD_DIM / QK_NVFP4;
static constexpr int    FATTN_NVFP4_PV_COL_TILE     = 8;
static constexpr int    FATTN_NVFP4_PV_COL_TILES_PER_CTA = 8;
static constexpr int    FATTN_NVFP4_MAX_NCOL_TILE   = FATTN_NVFP4_MAX_HEAD_DIM / FATTN_NVFP4_PV_COL_TILE;
#ifdef GGML_CUDA_NVFP4_FA_MMA_MULTIWARP
static constexpr int    FATTN_NVFP4_TC_KQ_WARPS     = 1;
static constexpr int    FATTN_NVFP4_TC_PV_WARPS     = 8;
static constexpr int    FATTN_NVFP4_TC_WARPS        = FATTN_NVFP4_TC_KQ_WARPS + FATTN_NVFP4_TC_PV_WARPS;
static constexpr int    FATTN_NVFP4_TC_THREADS      = FATTN_NVFP4_TC_WARPS * WARP_SIZE;
#ifdef GGML_CUDA_NVFP4_FA_MMA_SPLIT_KV
static constexpr int    FATTN_NVFP4_SPLIT_KV_ROWS   = GGML_CUDA_NVFP4_FA_ACTIVE_SPLIT_KV_ROWS;
#endif // GGML_CUDA_NVFP4_FA_MMA_SPLIT_KV
#endif // GGML_CUDA_NVFP4_FA_MMA_MULTIWARP
static_assert(FATTN_NVFP4_MAX_NCOL_TILE % FATTN_NVFP4_PV_COL_TILES_PER_CTA == 0,
    "PV tile grouping must divide the maximum column tile count");

struct fattn_nvfp4_mtp4_params {
    const float       * Q;
    const block_nvfp4 * K;
    const block_nvfp4 * V;
    const half        * mask;
    float             * dst;
    const uint32_t    * v_lut;
#if defined(GGML_CUDA_NVFP4_FA_MMA) && defined(GGML_CUDA_NVFP4_FA_MMA_MULTIWARP)
    float             * split_partial;
    float             * split_prob;
    float2            * split_meta;
    int64_t             kv_split_size;
    int64_t             kv_split_count;
    int64_t             kv_split_base;
    int64_t             kv_split_active_count;
#if defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_REQUANT)
    int              * native_pv_v_qs;
    uint32_t         * native_pv_v_scale;
    int64_t            native_pv_nkv_tile;
#if defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_VX) && defined(GGML_CUDA_NVFP4_KV_EXEC_LAYOUT)
    ggml_cuda_nvfp4_vx_layout native_pv_vx_layout;
    bool                       native_pv_use_vx;
#endif // defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_VX) && defined(GGML_CUDA_NVFP4_KV_EXEC_LAYOUT)
#endif // defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_REQUANT)
#endif // defined(GGML_CUDA_NVFP4_FA_MMA) && defined(GGML_CUDA_NVFP4_FA_MMA_MULTIWARP)

    float scale;

    int64_t ne_q_rows;
    int64_t ne_q_heads;
    int64_t ne_kv_rows;
    int64_t ne_kv_heads;
    int64_t ne_seqs;
    int64_t gqa_ratio;
    int64_t k_head_dim;
    int64_t v_head_dim;

    int64_t q_stride_row;
    int64_t q_stride_head;
    int64_t q_stride_seq;
    int64_t k_stride_row;
    int64_t k_stride_head;
    int64_t k_stride_seq;
    int64_t v_stride_row;
    int64_t v_stride_head;
    int64_t v_stride_seq;
    int64_t mask_stride_row;
    int64_t mask_stride_seq;
    int64_t dst_stride_row;
    int64_t dst_stride_head;
    int64_t dst_stride_seq;
};

static int64_t ggml_cuda_fattn_nvfp4_stride_elems(const ggml_tensor * t, const int dim, const size_t elem_size) {
    GGML_ASSERT(t->nb[dim] % (int64_t) elem_size == 0);
    return t->nb[dim] / (int64_t) elem_size;
}

static fattn_nvfp4_mtp4_params ggml_cuda_fattn_nvfp4_mtp4_make_params(const ggml_tensor * dst, const uint32_t * v_lut) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    GGML_ASSERT(Q != nullptr);
    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);
    GGML_ASSERT(mask != nullptr);

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    fattn_nvfp4_mtp4_params params = {};
    params.Q     = (const float *) Q->data;
    params.K     = (const block_nvfp4 *) K->data;
    params.V     = (const block_nvfp4 *) V->data;
    params.mask  = (const half *) mask->data;
    params.dst   = (float *) dst->data;
    params.v_lut = v_lut;
#if defined(GGML_CUDA_NVFP4_FA_MMA) && defined(GGML_CUDA_NVFP4_FA_MMA_MULTIWARP)
    params.split_partial = nullptr;
    params.split_prob    = nullptr;
    params.split_meta    = nullptr;
    params.kv_split_size = 0;
    params.kv_split_count = 1;
    params.kv_split_base = 0;
    params.kv_split_active_count = 1;
#if defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_REQUANT)
    params.native_pv_v_qs = nullptr;
    params.native_pv_v_scale = nullptr;
    params.native_pv_nkv_tile = 0;
#if defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_VX) && defined(GGML_CUDA_NVFP4_KV_EXEC_LAYOUT)
    params.native_pv_vx_layout = {};
    params.native_pv_use_vx = false;
#endif // defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_VX) && defined(GGML_CUDA_NVFP4_KV_EXEC_LAYOUT)
#endif // defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_REQUANT)
#endif // defined(GGML_CUDA_NVFP4_FA_MMA) && defined(GGML_CUDA_NVFP4_FA_MMA_MULTIWARP)
    params.scale = scale;

    params.ne_q_rows  = Q->ne[1];
    params.ne_q_heads = Q->ne[2];
    params.ne_kv_rows = K->ne[1];
    params.ne_kv_heads = K->ne[2];
    params.ne_seqs    = Q->ne[3];
    params.gqa_ratio  = Q->ne[2] / K->ne[2];
    params.k_head_dim = Q->ne[0];
    params.v_head_dim = V->ne[0];

    params.q_stride_row  = ggml_cuda_fattn_nvfp4_stride_elems(Q, 1, sizeof(float));
    params.q_stride_head = ggml_cuda_fattn_nvfp4_stride_elems(Q, 2, sizeof(float));
    params.q_stride_seq  = ggml_cuda_fattn_nvfp4_stride_elems(Q, 3, sizeof(float));

    params.k_stride_row  = ggml_cuda_fattn_nvfp4_stride_elems(K, 1, sizeof(block_nvfp4));
    params.k_stride_head = ggml_cuda_fattn_nvfp4_stride_elems(K, 2, sizeof(block_nvfp4));
    params.k_stride_seq  = ggml_cuda_fattn_nvfp4_stride_elems(K, 3, sizeof(block_nvfp4));

    params.v_stride_row  = ggml_cuda_fattn_nvfp4_stride_elems(V, 1, sizeof(block_nvfp4));
    params.v_stride_head = ggml_cuda_fattn_nvfp4_stride_elems(V, 2, sizeof(block_nvfp4));
    params.v_stride_seq  = ggml_cuda_fattn_nvfp4_stride_elems(V, 3, sizeof(block_nvfp4));

    params.mask_stride_row = ggml_cuda_fattn_nvfp4_stride_elems(mask, 1, sizeof(half));
    params.mask_stride_seq = mask->ne[3] == 1 ? 0 : ggml_cuda_fattn_nvfp4_stride_elems(mask, 3, sizeof(half));

    params.dst_stride_head = ggml_cuda_fattn_nvfp4_stride_elems(dst, 1, sizeof(float));
    params.dst_stride_row  = ggml_cuda_fattn_nvfp4_stride_elems(dst, 2, sizeof(float));
    params.dst_stride_seq  = ggml_cuda_fattn_nvfp4_stride_elems(dst, 3, sizeof(float));

    return params;
}

static dim3 ggml_cuda_fattn_nvfp4_mtp4_blocks(const fattn_nvfp4_mtp4_params & params) {
#if defined(GGML_CUDA_NVFP4_FA_MMA) && defined(GGML_CUDA_NVFP4_FA_MMA_MULTIWARP)
    const int64_t grid_z = params.ne_seqs * params.kv_split_count;
#else
    const int64_t grid_z = params.ne_seqs;
#endif

    return dim3(
        (uint32_t) ((params.ne_q_rows + FATTN_NVFP4_MTP4_ROWS - 1) / FATTN_NVFP4_MTP4_ROWS),
        (uint32_t) params.ne_q_heads,
        (uint32_t) grid_z);
}

#if defined(GGML_CUDA_NVFP4_FA_MMA) && \
    !defined(GGML_CUDA_NVFP4_FA_MMA_SCALAR_PV) && \
    !defined(GGML_CUDA_NVFP4_FA_MMA_MULTIWARP)
static int ggml_cuda_fattn_nvfp4_ncol_group(const int64_t head_dim) {
    GGML_ASSERT(head_dim % (FATTN_NVFP4_PV_COL_TILE * FATTN_NVFP4_PV_COL_TILES_PER_CTA) == 0);
    GGML_ASSERT(head_dim <= FATTN_NVFP4_MAX_HEAD_DIM);
    return (int) (head_dim / FATTN_NVFP4_PV_COL_TILE / FATTN_NVFP4_PV_COL_TILES_PER_CTA);
}
#endif // defined(GGML_CUDA_NVFP4_FA_MMA) && !defined(GGML_CUDA_NVFP4_FA_MMA_SCALAR_PV) && !defined(GGML_CUDA_NVFP4_FA_MMA_MULTIWARP)

static __device__ __forceinline__ uint32_t ggml_cuda_fattn_nvfp4_half2_bits(const half2 v) {
    union {
        half2    h;
        uint32_t u;
    } tmp;
    tmp.h = v;
    return tmp.u;
}

static __device__ __forceinline__ uint32_t ggml_cuda_fattn_nvfp4_block_scale(const block_nvfp4 & b) {
    return *reinterpret_cast<const uint32_t *>(b.d);
}

#ifdef GGML_CUDA_NVFP4_FA_MMA
static __device__ __forceinline__ block_nvfp4 ggml_cuda_fattn_nvfp4_quantize_q_frag(
        const float * q,
        const int     frag) {
    float q_frag[QK_NVFP4];

#pragma unroll
    for (int i = 0; i < QK_NVFP4; ++i) {
        q_frag[i] = q[frag * QK_NVFP4 + i];
    }

    block_nvfp4 out;
    quantize_f32_nvfp4_block(q_frag, &out);
    return out;
}
#endif // GGML_CUDA_NVFP4_FA_MMA

#ifdef GGML_CUDA_NVFP4_FA_MMA
struct fattn_nvfp4_softmax_state {
    float kq_max;
    float rowsum;
};

struct fattn_nvfp4_pv_state {
    float pv[4];
};

static __device__ __forceinline__ fattn_nvfp4_softmax_state ggml_cuda_fattn_nvfp4_softmax_init() {
    return {
        -3.402823466e+38F,
        0.0f,
    };
}

static __device__ __forceinline__ fattn_nvfp4_pv_state ggml_cuda_fattn_nvfp4_pv_init() {
    return {
        {0.0f, 0.0f, 0.0f, 0.0f},
    };
}
#endif // GGML_CUDA_NVFP4_FA_MMA

static __device__ __forceinline__ float ggml_cuda_fattn_nvfp4_mask_value(
        const fattn_nvfp4_mtp4_params & params,
        const int64_t                   q_row,
        const int64_t                   kv_row,
        const int64_t                   seq) {
    return __half2float(params.mask[
        q_row * params.mask_stride_row +
        seq   * params.mask_stride_seq +
        kv_row]);
}

static __device__ __forceinline__ float ggml_cuda_fattn_nvfp4_dequant(
        const block_nvfp4 & b,
        const int           i) {
    const int sub = i / QK_NVFP4_SUB;
    const int j   = i % QK_NVFP4_SUB;
    const uint8_t q = b.qs[sub * (QK_NVFP4_SUB / 2) + (j % (QK_NVFP4_SUB / 2))];
    const uint8_t qc = j < QK_NVFP4_SUB / 2 ? q & 0x0f : q >> 4;
    return ggml_cuda_ue4m3_to_fp32(b.d[sub]) * kvalues_mxfp4[qc];
}

#ifdef GGML_CUDA_NVFP4_FA_MMA_SCALAR_KQ
static __device__ __forceinline__ float ggml_cuda_fattn_nvfp4_dequant_packed(
        const int *    qs_words,
        const uint32_t scale_words,
        const int      i) {
    const int sub      = i / QK_NVFP4_SUB;
    const int j        = i % QK_NVFP4_SUB;
    const int byte_idx = sub * (QK_NVFP4_SUB / 2) + (j % (QK_NVFP4_SUB / 2));
    const uint32_t q_word = (uint32_t) qs_words[byte_idx / 4];
    const uint8_t q = (uint8_t) ((q_word >> (8 * (byte_idx % 4))) & 0xff);
    const uint8_t qc = j < QK_NVFP4_SUB / 2 ? q & 0x0f : q >> 4;
    const uint8_t scale = (uint8_t) ((scale_words >> (8 * sub)) & 0xff);
    return ggml_cuda_ue4m3_to_fp32(scale) * kvalues_mxfp4[qc];
}
#endif // GGML_CUDA_NVFP4_FA_MMA_SCALAR_KQ

static __device__ __forceinline__ float ggml_cuda_fattn_nvfp4_dequant_row_value(
        const block_nvfp4 * row,
        const int           col) {
    return ggml_cuda_fattn_nvfp4_dequant(row[col / QK_NVFP4], col % QK_NVFP4);
}

#ifdef GGML_CUDA_NVFP4_FA_MMA
static __device__ __forceinline__ half2 ggml_cuda_fattn_nvfp4_half2_from_bits(const uint32_t bits) {
    union {
        uint32_t u;
        half2    h;
    } tmp;
    tmp.u = bits;
    return tmp.h;
}

static __device__ __forceinline__ uint32_t ggml_cuda_fattn_nvfp4_lookup_row_value_half_bits(
        const uint32_t *    lut,
        const block_nvfp4 * row,
        const int           col) {
    const block_nvfp4 & b = row[col / QK_NVFP4];
    const int i           = col % QK_NVFP4;
    const int sub         = i / QK_NVFP4_SUB;
    const int j           = i % QK_NVFP4_SUB;
    const int byte_idx    = sub * (QK_NVFP4_SUB / 2) + (j % (QK_NVFP4_SUB / 2));
    const uint32_t scale  = (uint32_t) b.d[sub] << 8;
    const uint32_t pair   = lut[scale | (uint32_t) b.qs[byte_idx]];
    return j < QK_NVFP4_SUB / 2 ? (pair & 0x0000ffffu) : (pair >> 16);
}

static __device__ __forceinline__ half2 ggml_cuda_fattn_nvfp4_lookup_row_value_low_half2(
        const uint32_t *    lut,
        const block_nvfp4 * row,
        const int           col) {
    return ggml_cuda_fattn_nvfp4_half2_from_bits(ggml_cuda_fattn_nvfp4_lookup_row_value_half_bits(lut, row, col));
}

#ifdef GGML_CUDA_NVFP4_FA_MMA_MULTIWARP
static __device__ __forceinline__ half2 ggml_cuda_fattn_nvfp4_lookup_row_pair_half2(
        const uint32_t *    lut,
        const block_nvfp4 * row0,
        const block_nvfp4 * row1,
        const int           col) {
    const uint32_t low  = ggml_cuda_fattn_nvfp4_lookup_row_value_half_bits(lut, row0, col);
    const uint32_t high = ggml_cuda_fattn_nvfp4_lookup_row_value_half_bits(lut, row1, col);
    return ggml_cuda_fattn_nvfp4_half2_from_bits(low | (high << 16));
}
#endif // GGML_CUDA_NVFP4_FA_MMA_MULTIWARP
#endif // GGML_CUDA_NVFP4_FA_MMA

static __device__ __forceinline__ float ggml_cuda_fattn_nvfp4_dot_q_k(
        const float *       q,
        const block_nvfp4 * k,
        const int64_t       head_dim) {
    float sum = 0.0f;

    const int64_t nfrag = head_dim / QK_NVFP4;

#pragma unroll
    for (int64_t frag = 0; frag < nfrag; ++frag) {
#pragma unroll
        for (int i = 0; i < QK_NVFP4; ++i) {
            sum += q[frag * QK_NVFP4 + i] * ggml_cuda_fattn_nvfp4_dequant(k[frag], i);
        }
    }

    return sum;
}

#ifdef GGML_CUDA_NVFP4_FA_MMA
static __device__ __forceinline__ void ggml_cuda_fattn_nvfp4_rescale_pv_fragment(
        fattn_nvfp4_pv_state &      state,
        const float                 scale_old) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        state.pv[i] *= scale_old;
    }
}

static __device__ __forceinline__ float ggml_cuda_fattn_nvfp4_online_softmax_prepare(
        fattn_nvfp4_softmax_state & state,
        const float                 score,
        float &                     scale_old) {
    if (isinf(score) && score < 0.0f) {
        scale_old = 1.0f;
        return 0.0f;
    }

    const float kq_max_new = fmaxf(state.kq_max, score);
    scale_old = state.rowsum == 0.0f ? 0.0f : expf(state.kq_max - kq_max_new);
    const float scale_new = score - kq_max_new >= SOFTMAX_FTZ_THRESHOLD ? expf(score - kq_max_new) : 0.0f;

    state.rowsum = state.rowsum * scale_old + scale_new;
    state.kq_max = kq_max_new;

    return scale_new;
}
#endif // GGML_CUDA_NVFP4_FA_MMA

static __device__ __forceinline__ void ggml_cuda_fattn_nvfp4_kq_mma(
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
#ifdef BLACKWELL_MMA_AVAILABLE
    ggml_cuda_mma::tile<16, 8, int>   A = {};
    ggml_cuda_mma::tile<8, 8, int>    B = {};
    ggml_cuda_mma::tile<16, 8, float> D = {};

    A.x[0] = ax0;
    A.x[1] = ax1;
    A.x[2] = ax2;
    A.x[3] = ax3;
    B.x[0] = bx0;
    B.x[1] = bx1;
    D.x[0] = d0;
    D.x[1] = d1;
    D.x[2] = d2;
    D.x[3] = d3;

    ggml_cuda_mma::mma_block_scaled_fp4<GGML_TYPE_NVFP4>(D, A, B, q_scale, k_scale);

    d0 = D.x[0];
    d1 = D.x[1];
    d2 = D.x[2];
    d3 = D.x[3];
#else
    GGML_UNUSED_VARS(ax0, ax1, ax2, ax3, bx0, bx1, q_scale, k_scale, d0, d1, d2, d3);
#endif // BLACKWELL_MMA_AVAILABLE
}

static __device__ __forceinline__ void ggml_cuda_fattn_nvfp4_pv_mma(
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
#ifdef TURING_MMA_AVAILABLE
    ggml_cuda_mma::tile<16, 8, half2>  A = {};
    ggml_cuda_mma::tile<8, 8, half2>   B = {};
    ggml_cuda_mma::tile<16, 8, float>  D = {};

    reinterpret_cast<uint32_t *>(A.x)[0] = ax0;
    reinterpret_cast<uint32_t *>(A.x)[1] = ax1;
    reinterpret_cast<uint32_t *>(A.x)[2] = ax2;
    reinterpret_cast<uint32_t *>(A.x)[3] = ax3;
    reinterpret_cast<uint32_t *>(B.x)[0] = bx0;
    reinterpret_cast<uint32_t *>(B.x)[1] = bx1;
    D.x[0] = d0;
    D.x[1] = d1;
    D.x[2] = d2;
    D.x[3] = d3;

    ggml_cuda_mma::mma(D, A, B);

    d0 = D.x[0];
    d1 = D.x[1];
    d2 = D.x[2];
    d3 = D.x[3];
#else
    GGML_UNUSED_VARS(ax0, ax1, ax2, ax3, bx0, bx1, d0, d1, d2, d3);
#endif // TURING_MMA_AVAILABLE
}

#ifdef GGML_CUDA_NVFP4_FA_MMA_MULTIWARP
static __device__ __forceinline__ int ggml_cuda_fattn_nvfp4_pv_a_i(const int l, const int lane) {
    return ((l % 2) * 8) + (lane / 4);
}

static __device__ __forceinline__ int ggml_cuda_fattn_nvfp4_pv_a_j(const int l, const int lane) {
    return ((l / 2) * 4) + (lane % 4);
}

static __device__ __forceinline__ int ggml_cuda_fattn_nvfp4_pv_b_i(const int /*l*/, const int lane) {
    return lane / 4;
}

static __device__ __forceinline__ int ggml_cuda_fattn_nvfp4_pv_b_j(const int l, const int lane) {
    return (l * 4) + (lane % 4);
}

static __device__ __forceinline__ int ggml_cuda_fattn_nvfp4_pv_c_i(const int l, const int lane) {
    return ((l / 2) * 8) + (lane / 4);
}

static __device__ __forceinline__ int ggml_cuda_fattn_nvfp4_pv_c_j(const int l, const int lane) {
    return ((lane % 4) * 2) + (l % 2);
}
#endif // GGML_CUDA_NVFP4_FA_MMA_MULTIWARP

#if defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_REQUANT)
static __device__ __forceinline__ float ggml_cuda_fattn_nvfp4_quant_error_16(
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

static __device__ __forceinline__ uint8_t ggml_cuda_fattn_nvfp4_best_scale_code_16(
        const float * vals,
        const float   amax) {
    if (amax == 0.0f) {
        return 0;
    }

    uint8_t best_code = ggml_cuda_fp32_to_ue4m3(amax / 6.0f);
    float best_err = ggml_cuda_fattn_nvfp4_quant_error_16(vals, best_code);

#pragma unroll 1
    for (int code = 1; code < 256; ++code) {
        const float err = ggml_cuda_fattn_nvfp4_quant_error_16(vals, (uint8_t) code);
        if (err < best_err) {
            best_err = err;
            best_code = (uint8_t) code;
        }
    }

    return best_code;
}

static __device__ __forceinline__ void ggml_cuda_fattn_nvfp4_quantize_64_words(
        const float * vals,
        int *         qs_words,
        uint32_t &    scale_word) {
    static_assert(QK_NVFP4 == 64, "unexpected NVFP4 block size");
    static_assert(QK_NVFP4_SUB == 16, "unexpected NVFP4 sub-block size");

    scale_word = 0;

#pragma unroll
    for (int sub = 0; sub < QK_NVFP4 / QK_NVFP4_SUB; ++sub) {
        float amax = 0.0f;
#pragma unroll
        for (int j = 0; j < QK_NVFP4_SUB; ++j) {
            amax = fmaxf(amax, fabsf(vals[sub * QK_NVFP4_SUB + j]));
        }

        const uint8_t sf = ggml_cuda_fattn_nvfp4_best_scale_code_16(vals + sub * QK_NVFP4_SUB, amax);
        scale_word |= (uint32_t) sf << (8 * sub);

        const float d = ggml_cuda_ue4m3_to_fp32(sf);
        const float inv_scale = d > 0.0f ? 0.5f / d : 0.0f;

        uint32_t packed[2] = {};
#pragma unroll
        for (int j = 0; j < QK_NVFP4_SUB / 2; ++j) {
            const uint8_t q0 = ggml_cuda_float_to_fp4_e2m1(vals[sub * QK_NVFP4_SUB + j], inv_scale);
            const uint8_t q1 = ggml_cuda_float_to_fp4_e2m1(vals[sub * QK_NVFP4_SUB + j + QK_NVFP4_SUB / 2], inv_scale);
            const uint32_t byte = (uint32_t) (q0 | (q1 << 4));
            packed[j / 4] |= byte << (8 * (j % 4));
        }
        qs_words[2 * sub + 0] = (int) packed[0];
        qs_words[2 * sub + 1] = (int) packed[1];
    }
}

static __device__ __forceinline__ int ggml_cuda_fattn_nvfp4_native_pv_a_i(const int l, const int lane) {
    return ((l / 2) * 8) + (lane / 4);
}

static __device__ __forceinline__ int ggml_cuda_fattn_nvfp4_native_pv_a_j(const int l, const int lane) {
    return ((lane % 4) * 2) + (l % 2);
}

static __device__ __forceinline__ int ggml_cuda_fattn_nvfp4_native_pv_b_i(const int /*l*/, const int lane) {
    return lane / 4;
}

static __device__ __forceinline__ int ggml_cuda_fattn_nvfp4_native_pv_b_j(const int l, const int lane) {
    return (l * 4) + (lane % 4);
}
#endif // defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_REQUANT)

#ifdef GGML_CUDA_NVFP4_FA_MMA
static __device__ __forceinline__ void ggml_cuda_fattn_nvfp4_store_pv_fragment(
        float *                           dst,
        const fattn_nvfp4_softmax_state & softmax,
        const fattn_nvfp4_pv_state &      pv,
        const int                         col_base,
        const int                         row_in_mma,
        const int64_t                     head_dim) {
    if (softmax.rowsum == 0.0f) {
        return;
    }

    using tile_C = ggml_cuda_mma::tile<16, 8, float>;

#pragma unroll
    for (int l = 0; l < tile_C::ne; ++l) {
        const int col = col_base + tile_C::get_j(l);
        if (tile_C::get_i(l) == row_in_mma && col < head_dim) {
            dst[col] = pv.pv[l] / softmax.rowsum;
        }
    }
}
#endif // GGML_CUDA_NVFP4_FA_MMA

static __global__ void fattn_nvfp4_fill_lut(uint32_t * lut) {
    const size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t) gridDim.x * blockDim.x;

    for (size_t i = tid; i < FATTN_NVFP4_LUT_SIZE; i += stride) {
        const uint8_t scale = (uint8_t) (i >> 8);
        const uint8_t q     = (uint8_t) (i & 0xff);
        const float d = ggml_cuda_ue4m3_to_fp32(scale);
        const float lo = d * kvalues_mxfp4[q & 0x0f];
        const float hi = d * kvalues_mxfp4[q >> 4];
        lut[i] = ggml_cuda_fattn_nvfp4_half2_bits(make_half2(lo, hi));
    }
}

#ifdef GGML_CUDA_NVFP4_FA_MMA
__global__ void fattn_nvfp4_mtp4_kernel(const fattn_nvfp4_mtp4_params params) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int tid  = threadIdx.x;

#if defined(GGML_CUDA_NVFP4_FA_MMA) && !defined(GGML_CUDA_NVFP4_FA_MMA_SCALAR_PV)
    const int ncol_group = (int) (params.v_head_dim / FATTN_NVFP4_PV_COL_TILE / FATTN_NVFP4_PV_COL_TILES_PER_CTA);
    const int64_t q_row_block = (int64_t) blockIdx.x / ncol_group;
    const int     pv_col_group = (int) blockIdx.x % ncol_group;
#else
    const int64_t q_row_block = (int64_t) blockIdx.x;
    [[maybe_unused]] const int pv_col_group = 0;
#endif // defined(GGML_CUDA_NVFP4_FA_MMA) && !defined(GGML_CUDA_NVFP4_FA_MMA_SCALAR_PV)

    const int64_t q_row_base = q_row_block * FATTN_NVFP4_MTP4_ROWS;
    const int64_t q_head     = (int64_t) blockIdx.y;
    const int64_t seq        = (int64_t) blockIdx.z;
    const int64_t kv_head    = q_head / params.gqa_ratio;

    if (tid >= WARP_SIZE || q_head >= params.ne_q_heads || seq >= params.ne_seqs || kv_head >= params.ne_kv_heads) {
        return;
    }

    using tile_A = ggml_cuda_mma::tile<16, 8, int>;
    using tile_B = ggml_cuda_mma::tile< 8, 8, int>;
    using tile_C = ggml_cuda_mma::tile<16, 8, float>;

    const int nfrag = (int) (params.k_head_dim / QK_NVFP4);

    __shared__ int      q_tile[FATTN_NVFP4_MTP4_ROWS][FATTN_NVFP4_MAX_NFRAG][tile_A::J];
    __shared__ uint32_t q_tile_scale[FATTN_NVFP4_MTP4_ROWS][FATTN_NVFP4_MAX_NFRAG];
    __shared__ int      kq_a_tile[tile_A::I * tile_A::J];
    __shared__ int      kq_b_tile[tile_B::I * tile_B::J];
    __shared__ uint32_t kq_a_scale[tile_A::I];
    __shared__ uint32_t kq_b_scale[tile_B::I];

#if defined(GGML_CUDA_NVFP4_FA_MMA) && !defined(GGML_CUDA_NVFP4_FA_MMA_SCALAR_PV)
    using tile_PV_A = ggml_cuda_mma::tile<16, 8, half2>;
    using tile_PV_B = ggml_cuda_mma::tile< 8, 8, half2>;
    __shared__ half2 pv_a_tile[tile_PV_A::I * tile_PV_A::J];
    __shared__ half2 pv_b_tile[tile_PV_B::I * tile_PV_B::J];
#endif // defined(GGML_CUDA_NVFP4_FA_MMA) && !defined(GGML_CUDA_NVFP4_FA_MMA_SCALAR_PV)

    for (int q_idx = tid; q_idx < FATTN_NVFP4_MTP4_ROWS * nfrag; q_idx += WARP_SIZE) {
        const int row  = q_idx / nfrag;
        const int frag = q_idx % nfrag;

        const int64_t q_row = q_row_base + row;
        if (q_row >= params.ne_q_rows) {
            q_tile_scale[row][frag] = 0;
#pragma unroll
            for (int k = 0; k < tile_A::J; ++k) {
                q_tile[row][frag][k] = 0;
            }
            continue;
        }

        const float * q_ptr = params.Q +
            q_row  * params.q_stride_row +
            q_head * params.q_stride_head +
            seq    * params.q_stride_seq;
        const block_nvfp4 q_blk = ggml_cuda_fattn_nvfp4_quantize_q_frag(q_ptr, frag);
        const uint32_t * q_qs = reinterpret_cast<const uint32_t *>(q_blk.qs);

#pragma unroll
        for (int k = 0; k < tile_A::J; ++k) {
            q_tile[row][frag][k] = (int) q_qs[k];
        }
        q_tile_scale[row][frag] = ggml_cuda_fattn_nvfp4_block_scale(q_blk);
    }
    __syncwarp();

#ifdef GGML_CUDA_NVFP4_FA_MMA_SCALAR_PV
    static_assert(FATTN_NVFP4_MAX_HEAD_DIM % WARP_SIZE == 0, "scalar PV diagnostic assumes one warp covers the head");

    float kq_max[FATTN_NVFP4_MTP4_ROWS];
    float rowsum[FATTN_NVFP4_MTP4_ROWS];
    float pv_scalar[FATTN_NVFP4_MTP4_ROWS][FATTN_NVFP4_MAX_HEAD_DIM / WARP_SIZE];

#pragma unroll
    for (int row = 0; row < FATTN_NVFP4_MTP4_ROWS; ++row) {
        kq_max[row] = -3.402823466e+38F;
        rowsum[row] = 0.0f;
        for (int col_slot = 0; col_slot < params.v_head_dim / WARP_SIZE; ++col_slot) {
            pv_scalar[row][col_slot] = 0.0f;
        }
    }
#elif defined(GGML_CUDA_NVFP4_FA_MMA)
    fattn_nvfp4_softmax_state softmax_state[FATTN_NVFP4_MTP4_ROWS];
    fattn_nvfp4_pv_state      pv_state[FATTN_NVFP4_MTP4_ROWS][FATTN_NVFP4_PV_COL_TILES_PER_CTA];

#pragma unroll
    for (int row = 0; row < FATTN_NVFP4_MTP4_ROWS; ++row) {
        softmax_state[row] = ggml_cuda_fattn_nvfp4_softmax_init();
#pragma unroll
        for (int tile = 0; tile < FATTN_NVFP4_PV_COL_TILES_PER_CTA; ++tile) {
            pv_state[row][tile] = ggml_cuda_fattn_nvfp4_pv_init();
        }
    }
#endif // GGML_CUDA_NVFP4_FA_MMA_SCALAR_PV

#pragma unroll 1
    for (int64_t kv_row = 0; kv_row < params.ne_kv_rows; ++kv_row) {
        const block_nvfp4 * k_ptr = params.K +
            kv_head * params.k_stride_head +
            seq     * params.k_stride_seq +
            kv_row  * params.k_stride_row;

        tile_C kq_tile = {};

        for (int q_frag = 0; q_frag < nfrag; ++q_frag) {
            const block_nvfp4 k_blk = k_ptr[q_frag];

            const uint32_t * k_qs = reinterpret_cast<const uint32_t *>(k_blk.qs);
            const uint32_t k_scale = ggml_cuda_fattn_nvfp4_block_scale(k_blk);

            for (int i = tid; i < tile_A::I * tile_A::J; i += WARP_SIZE) {
                kq_a_tile[i] = 0;
            }
            for (int i = tid; i < tile_B::I * tile_B::J; i += WARP_SIZE) {
                kq_b_tile[i] = 0;
            }
            for (int i = tid; i < tile_A::I; i += WARP_SIZE) {
                kq_a_scale[i] = 0;
            }
            for (int i = tid; i < tile_B::I; i += WARP_SIZE) {
                kq_b_scale[i] = k_scale;
            }
            __syncwarp();

            for (int i = tid; i < FATTN_NVFP4_MTP4_ROWS * tile_A::J; i += WARP_SIZE) {
                const int row = i / tile_A::J;
                const int k   = i % tile_A::J;
                kq_a_tile[row * tile_A::J + k] = q_tile[row][q_frag][k];
            }
            for (int i = tid; i < tile_B::J; i += WARP_SIZE) {
                kq_b_tile[i] = (int) k_qs[i];
            }
            if (tid < FATTN_NVFP4_MTP4_ROWS) {
                kq_a_scale[tid] = q_tile_scale[tid][q_frag];
            }
            __syncwarp();

            tile_A A;
            tile_B B;
            ggml_cuda_mma::load_ldmatrix(A, kq_a_tile, tile_A::J);
            ggml_cuda_mma::load_generic(B, kq_b_tile, tile_B::J);

            const int tidx_A = threadIdx.x / 4 + (threadIdx.x % 2) * 8;
            const int tidx_B = threadIdx.x / 4;
            ggml_cuda_mma::mma_block_scaled_fp4<GGML_TYPE_NVFP4>(
                kq_tile, A, B, kq_a_scale[tidx_A], kq_b_scale[tidx_B]);
        }

        float kq_score[FATTN_NVFP4_MTP4_ROWS] = {};

#pragma unroll
        for (int row = 0; row < FATTN_NVFP4_MTP4_ROWS; ++row) {
            float local_score = 0.0f;
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                if (tile_C::get_i(l) == row && tile_C::get_j(l) == 0) {
                    local_score += kq_tile.x[l];
                }
            }

#pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                local_score += __shfl_down_sync(0xffffffff, local_score, offset);
            }
            kq_score[row] = __shfl_sync(0xffffffff, local_score, 0);
        }

#ifdef GGML_CUDA_NVFP4_FA_MMA_SCALAR_KQ
#pragma unroll
        for (int row = 0; row < FATTN_NVFP4_MTP4_ROWS; ++row) {
            float local_score = 0.0f;

            for (int col = tid; col < params.k_head_dim; col += WARP_SIZE) {
                const int frag = col / QK_NVFP4;
                const int i    = col % QK_NVFP4;
                const float qv = ggml_cuda_fattn_nvfp4_dequant_packed(q_tile[row][frag], q_tile_scale[row][frag], i);
                const float kv = ggml_cuda_fattn_nvfp4_dequant_row_value(k_ptr, col);
                local_score += qv * kv;
            }

#pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                local_score += __shfl_down_sync(0xffffffff, local_score, offset);
            }
            kq_score[row] = __shfl_sync(0xffffffff, local_score, 0);
        }
#endif // GGML_CUDA_NVFP4_FA_MMA_SCALAR_KQ

#if defined(GGML_CUDA_NVFP4_FA_MMA_SCALAR_PV) || defined(GGML_CUDA_NVFP4_FA_MMA)
        float p[FATTN_NVFP4_MTP4_ROWS] = {};
#endif // defined(GGML_CUDA_NVFP4_FA_MMA_SCALAR_PV) || defined(GGML_CUDA_NVFP4_FA_MMA)

#pragma unroll
        for (int row = 0; row < FATTN_NVFP4_MTP4_ROWS; ++row) {
            const int64_t q_row = q_row_base + row;
            if (q_row >= params.ne_q_rows) {
                continue;
            }

#if defined(GGML_CUDA_NVFP4_FA_MMA_SCALAR_PV) || defined(GGML_CUDA_NVFP4_FA_MMA)
            const float mask = ggml_cuda_fattn_nvfp4_mask_value(params, q_row, kv_row, seq);
            const float score = kq_score[row] * params.scale + mask;
#ifdef GGML_CUDA_NVFP4_FA_MMA_SCALAR_PV
            const float kq_max_new = fmaxf(kq_max[row], score);
            const float scale_old = rowsum[row] == 0.0f ? 0.0f : expf(kq_max[row] - kq_max_new);
            const float scale_new = score - kq_max_new >= SOFTMAX_FTZ_THRESHOLD ? expf(score - kq_max_new) : 0.0f;

            for (int col_slot = 0; col_slot < params.v_head_dim / WARP_SIZE; ++col_slot) {
                pv_scalar[row][col_slot] *= scale_old;
            }
            rowsum[row] = rowsum[row] * scale_old + scale_new;
            kq_max[row] = kq_max_new;
            p[row] = scale_new;
#elif defined(GGML_CUDA_NVFP4_FA_MMA)
            float scale_old = 1.0f;
            p[row] = ggml_cuda_fattn_nvfp4_online_softmax_prepare(softmax_state[row], score, scale_old);

#pragma unroll
            for (int tile = 0; tile < FATTN_NVFP4_PV_COL_TILES_PER_CTA; ++tile) {
                ggml_cuda_fattn_nvfp4_rescale_pv_fragment(pv_state[row][tile], scale_old);
            }
#endif // GGML_CUDA_NVFP4_FA_MMA_SCALAR_PV
#endif // defined(GGML_CUDA_NVFP4_FA_MMA_SCALAR_PV) || defined(GGML_CUDA_NVFP4_FA_MMA)
        }

#ifdef GGML_CUDA_NVFP4_FA_MMA_SCALAR_PV
        const block_nvfp4 * v_ptr = params.V +
            kv_head * params.v_stride_head +
            seq     * params.v_stride_seq +
            kv_row  * params.v_stride_row;

#pragma unroll
        for (int row = 0; row < FATTN_NVFP4_MTP4_ROWS; ++row) {
            if (q_row_base + row >= params.ne_q_rows) {
                continue;
            }

            for (int col_slot = 0; col_slot < params.v_head_dim / WARP_SIZE; ++col_slot) {
                const int col = tid + col_slot * WARP_SIZE;
                pv_scalar[row][col_slot] += p[row] * ggml_cuda_fattn_nvfp4_dequant_row_value(v_ptr, col);
            }
        }
#elif defined(GGML_CUDA_NVFP4_FA_MMA)
        const block_nvfp4 * v_ptr = params.V +
            kv_head * params.v_stride_head +
            seq     * params.v_stride_seq +
            kv_row  * params.v_stride_row;

#pragma unroll
        for (int tile = 0; tile < FATTN_NVFP4_PV_COL_TILES_PER_CTA; ++tile) {
        const int col_base =
            (pv_col_group * FATTN_NVFP4_PV_COL_TILES_PER_CTA + tile) * FATTN_NVFP4_PV_COL_TILE;

        for (int i = tid; i < tile_PV_A::I * tile_PV_A::J; i += WARP_SIZE) {
            pv_a_tile[i] = __floats2half2_rn(0.0f, 0.0f);
        }
        for (int i = tid; i < tile_PV_B::I * tile_PV_B::J; i += WARP_SIZE) {
            pv_b_tile[i] = __floats2half2_rn(0.0f, 0.0f);
        }
        __syncwarp();

        if (tid < FATTN_NVFP4_MTP4_ROWS && q_row_base + tid < params.ne_q_rows) {
            pv_a_tile[tid * tile_PV_A::J] = __floats2half2_rn(p[tid], 0.0f);
        }
        if (tid < FATTN_NVFP4_PV_COL_TILE) {
            pv_b_tile[tid * tile_PV_B::J] =
                ggml_cuda_fattn_nvfp4_lookup_row_value_low_half2(params.v_lut, v_ptr, col_base + tid);
        }
        __syncwarp();

        tile_PV_A A;
        tile_PV_B B;
        tile_C C;
        ggml_cuda_mma::load_generic(A, pv_a_tile, tile_PV_A::J);
        ggml_cuda_mma::load_generic(B, pv_b_tile, tile_PV_B::J);

#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
            const int row = tile_C::get_i(l);
            C.x[l] = row < FATTN_NVFP4_MTP4_ROWS ? pv_state[row][tile].pv[l] : 0.0f;
        }

        ggml_cuda_mma::mma(C, A, B);

#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
            const int row = tile_C::get_i(l);
            if (row < FATTN_NVFP4_MTP4_ROWS) {
                pv_state[row][tile].pv[l] = C.x[l];
            }
        }
        }
#endif // GGML_CUDA_NVFP4_FA_MMA_SCALAR_PV
    }

#ifdef GGML_CUDA_NVFP4_FA_MMA
#ifdef GGML_CUDA_NVFP4_FA_MMA_SCALAR_PV
#pragma unroll
    for (int row = 0; row < FATTN_NVFP4_MTP4_ROWS; ++row) {
        const int64_t q_row = q_row_base + row;
        if (q_row >= params.ne_q_rows) {
            continue;
        }

        float * dst_ptr = params.dst +
            q_row  * params.dst_stride_row +
            q_head * params.dst_stride_head +
            seq    * params.dst_stride_seq;

        for (int col_slot = 0; col_slot < params.v_head_dim / WARP_SIZE; ++col_slot) {
            const int col = tid + col_slot * WARP_SIZE;
            dst_ptr[col] = rowsum[row] == 0.0f ? 0.0f : pv_scalar[row][col_slot] / rowsum[row];
        }
    }
#else
    for (int row = 0; row < FATTN_NVFP4_MTP4_ROWS; ++row) {
        if (q_row_base + row >= params.ne_q_rows) {
            continue;
        }

        float * dst_ptr = params.dst +
            (q_row_base + row) * params.dst_stride_row +
            q_head            * params.dst_stride_head +
            seq               * params.dst_stride_seq;

#pragma unroll
        for (int tile = 0; tile < FATTN_NVFP4_PV_COL_TILES_PER_CTA; ++tile) {
            const int col_base =
                (pv_col_group * FATTN_NVFP4_PV_COL_TILES_PER_CTA + tile) * FATTN_NVFP4_PV_COL_TILE;
            ggml_cuda_fattn_nvfp4_store_pv_fragment(
                dst_ptr,
                softmax_state[row],
                pv_state[row][tile],
                col_base,
                row,
                params.v_head_dim);
        }
    }
#endif // GGML_CUDA_NVFP4_FA_MMA_SCALAR_PV
#endif // GGML_CUDA_NVFP4_FA_MMA
#else
    GGML_UNUSED(params);
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}
#endif // GGML_CUDA_NVFP4_FA_MMA

#if defined(GGML_CUDA_NVFP4_FA_MMA) && defined(GGML_CUDA_NVFP4_FA_MMA_MULTIWARP)
__global__ void fattn_nvfp4_mtp4_multiwarp_kernel(const fattn_nvfp4_mtp4_params params) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane    = tid & (WARP_SIZE - 1);

    const int64_t q_row_block = (int64_t) blockIdx.x;
    const int64_t q_row_base  = q_row_block * FATTN_NVFP4_MTP4_ROWS;
    const int64_t q_head      = (int64_t) blockIdx.y;
#ifdef GGML_CUDA_NVFP4_FA_MMA_SPLIT_KV
    const int64_t seq         = (int64_t) blockIdx.z / params.kv_split_count;
    const int64_t kv_split    = (int64_t) blockIdx.z - seq * params.kv_split_count;
    const int64_t kv_start    = kv_split * params.kv_split_size;
    const int64_t kv_end      = min(params.ne_kv_rows, kv_start + params.kv_split_size);
#else
    const int64_t seq         = (int64_t) blockIdx.z;
    const int64_t kv_split    = 0;
    const int64_t kv_start    = 0;
    const int64_t kv_end      = params.ne_kv_rows;
#endif // GGML_CUDA_NVFP4_FA_MMA_SPLIT_KV
    const int64_t kv_head     = q_head / params.gqa_ratio;

    if (q_head >= params.ne_q_heads || seq >= params.ne_seqs || kv_head >= params.ne_kv_heads || kv_start >= kv_end) {
        return;
    }

    using tile_A    = ggml_cuda_mma::tile<16, 8, int>;
    using tile_B    = ggml_cuda_mma::tile< 8, 8, int>;
    using tile_C    = ggml_cuda_mma::tile<16, 8, float>;
    using tile_PV_A = ggml_cuda_mma::tile<16, 8, half2>;
    using tile_PV_B = ggml_cuda_mma::tile< 8, 8, half2>;

    const int nfrag     = (int) (params.k_head_dim / QK_NVFP4);
    const int ncol_tile = (int) (params.v_head_dim / FATTN_NVFP4_PV_COL_TILE);

    __shared__ int      q_tile[FATTN_NVFP4_MTP4_ROWS][FATTN_NVFP4_MAX_NFRAG][tile_A::J];
    __shared__ uint32_t q_tile_scale[FATTN_NVFP4_MTP4_ROWS][FATTN_NVFP4_MAX_NFRAG];
    __shared__ int      kq_a_tile[tile_A::I * tile_A::J];
    __shared__ int      kq_b_tile[tile_B::I * tile_B::J];
    __shared__ uint32_t kq_a_scale[tile_A::I];
    __shared__ uint32_t kq_b_scale[tile_B::I];
    __shared__ float    smem_kq_max[FATTN_NVFP4_MTP4_ROWS];
    __shared__ float    smem_rowsum[FATTN_NVFP4_MTP4_ROWS];
    __shared__ float    smem_p[FATTN_NVFP4_MTP4_ROWS];
    __shared__ float    smem_scale_old[FATTN_NVFP4_MTP4_ROWS];
    __shared__ float    smem_pair_p0[FATTN_NVFP4_MTP4_ROWS];
    __shared__ float    smem_pair_scale_old[FATTN_NVFP4_MTP4_ROWS];
    __shared__ float    pv_accum[FATTN_NVFP4_MAX_NCOL_TILE][WARP_SIZE][tile_C::ne];

    for (int q_idx = tid; q_idx < FATTN_NVFP4_MTP4_ROWS * nfrag; q_idx += FATTN_NVFP4_TC_THREADS) {
        const int row  = q_idx / nfrag;
        const int frag = q_idx % nfrag;

        const int64_t q_row = q_row_base + row;
        if (q_row >= params.ne_q_rows) {
            q_tile_scale[row][frag] = 0;
#pragma unroll
            for (int k = 0; k < tile_A::J; ++k) {
                q_tile[row][frag][k] = 0;
            }
            continue;
        }

        const float * q_ptr = params.Q +
            q_row  * params.q_stride_row +
            q_head * params.q_stride_head +
            seq    * params.q_stride_seq;
        const block_nvfp4 q_blk = ggml_cuda_fattn_nvfp4_quantize_q_frag(q_ptr, frag);
        const uint32_t * q_qs = reinterpret_cast<const uint32_t *>(q_blk.qs);

#pragma unroll
        for (int k = 0; k < tile_A::J; ++k) {
            q_tile[row][frag][k] = (int) q_qs[k];
        }
        q_tile_scale[row][frag] = ggml_cuda_fattn_nvfp4_block_scale(q_blk);
    }

    for (int i = tid; i < FATTN_NVFP4_MTP4_ROWS; i += FATTN_NVFP4_TC_THREADS) {
        smem_kq_max[i]    = -3.402823466e+38F;
        smem_rowsum[i]    = 0.0f;
        smem_p[i]         = 0.0f;
        smem_scale_old[i] = 1.0f;
        smem_pair_p0[i]   = 0.0f;
        smem_pair_scale_old[i] = 1.0f;
    }

    for (int i = tid; i < FATTN_NVFP4_MAX_NCOL_TILE * WARP_SIZE * tile_C::ne; i += FATTN_NVFP4_TC_THREADS) {
        reinterpret_cast<float *>(pv_accum)[i] = 0.0f;
    }
    __syncthreads();

#pragma unroll 1
    for (int64_t kv_row = kv_start; kv_row < kv_end; ++kv_row) {
        const block_nvfp4 * k_ptr = params.K +
            kv_head * params.k_stride_head +
            seq     * params.k_stride_seq +
            kv_row  * params.k_stride_row;

        if (warp_id == 0) {
            tile_C kq_tile = {};

            for (int q_frag = 0; q_frag < nfrag; ++q_frag) {
                const block_nvfp4 k_blk = k_ptr[q_frag];

                const uint32_t * k_qs = reinterpret_cast<const uint32_t *>(k_blk.qs);
                const uint32_t k_scale = ggml_cuda_fattn_nvfp4_block_scale(k_blk);

                for (int i = lane; i < tile_A::I * tile_A::J; i += WARP_SIZE) {
                    kq_a_tile[i] = 0;
                }
                for (int i = lane; i < tile_B::I * tile_B::J; i += WARP_SIZE) {
                    kq_b_tile[i] = 0;
                }
                for (int i = lane; i < tile_A::I; i += WARP_SIZE) {
                    kq_a_scale[i] = 0;
                }
                for (int i = lane; i < tile_B::I; i += WARP_SIZE) {
                    kq_b_scale[i] = k_scale;
                }
                __syncwarp();

                for (int i = lane; i < FATTN_NVFP4_MTP4_ROWS * tile_A::J; i += WARP_SIZE) {
                    const int row = i / tile_A::J;
                    const int k   = i % tile_A::J;
                    kq_a_tile[row * tile_A::J + k] = q_tile[row][q_frag][k];
                }
                for (int i = lane; i < tile_B::J; i += WARP_SIZE) {
                    kq_b_tile[i] = (int) k_qs[i];
                }
                if (lane < FATTN_NVFP4_MTP4_ROWS) {
                    kq_a_scale[lane] = q_tile_scale[lane][q_frag];
                }
                __syncwarp();

                tile_A A;
                tile_B B;
                ggml_cuda_mma::load_ldmatrix(A, kq_a_tile, tile_A::J);
                ggml_cuda_mma::load_generic(B, kq_b_tile, tile_B::J);

                const int tidx_A = threadIdx.x / 4 + (threadIdx.x % 2) * 8;
                const int tidx_B = threadIdx.x / 4;
                ggml_cuda_mma::mma_block_scaled_fp4<GGML_TYPE_NVFP4>(
                    kq_tile, A, B, kq_a_scale[tidx_A], kq_b_scale[tidx_B]);
            }

            float kq_score[FATTN_NVFP4_MTP4_ROWS] = {};

#pragma unroll
            for (int row = 0; row < FATTN_NVFP4_MTP4_ROWS; ++row) {
                float local_score = 0.0f;
#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    if (tile_C::get_i(l) == row && tile_C::get_j(l) == 0) {
                        local_score += kq_tile.x[l];
                    }
                }

#pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                    local_score += __shfl_down_sync(0xffffffff, local_score, offset);
                }
                kq_score[row] = __shfl_sync(0xffffffff, local_score, 0);
            }

#ifdef GGML_CUDA_NVFP4_FA_MMA_SCALAR_KQ
#pragma unroll
            for (int row = 0; row < FATTN_NVFP4_MTP4_ROWS; ++row) {
                float local_score = 0.0f;

                for (int col = lane; col < params.k_head_dim; col += WARP_SIZE) {
                    const int frag = col / QK_NVFP4;
                    const int i    = col % QK_NVFP4;
                    const float qv = ggml_cuda_fattn_nvfp4_dequant_packed(q_tile[row][frag], q_tile_scale[row][frag], i);
                    const float kv = ggml_cuda_fattn_nvfp4_dequant_row_value(k_ptr, col);
                    local_score += qv * kv;
                }

#pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                    local_score += __shfl_down_sync(0xffffffff, local_score, offset);
                }
                kq_score[row] = __shfl_sync(0xffffffff, local_score, 0);
            }
#endif // GGML_CUDA_NVFP4_FA_MMA_SCALAR_KQ

            if (lane < FATTN_NVFP4_MTP4_ROWS) {
                const int row = lane;
                const int64_t q_row = q_row_base + row;
                if (q_row >= params.ne_q_rows) {
                    smem_p[row] = 0.0f;
                    smem_scale_old[row] = 1.0f;
                } else {
                    const float mask = ggml_cuda_fattn_nvfp4_mask_value(params, q_row, kv_row, seq);
                    const float score = kq_score[row] * params.scale + mask;
                    if (isinf(score) && score < 0.0f) {
                        smem_p[row] = 0.0f;
                        smem_scale_old[row] = 1.0f;
                    } else {
                        const float kq_max_new = fmaxf(smem_kq_max[row], score);
                        const float scale_old = smem_rowsum[row] == 0.0f ? 0.0f : expf(smem_kq_max[row] - kq_max_new);
                        const float scale_new = score - kq_max_new >= SOFTMAX_FTZ_THRESHOLD ? expf(score - kq_max_new) : 0.0f;

                        smem_rowsum[row] = smem_rowsum[row] * scale_old + scale_new;
                        smem_kq_max[row] = kq_max_new;
                        smem_p[row] = scale_new;
                        smem_scale_old[row] = scale_old;
                    }
                }

                if ((kv_row & 1) == 0) {
                    smem_pair_p0[row] = smem_p[row];
                    smem_pair_scale_old[row] = smem_scale_old[row];
                }
            }
        }
        __syncthreads();

        if ((kv_row & 1) != 0 && warp_id >= FATTN_NVFP4_TC_KQ_WARPS && warp_id < FATTN_NVFP4_TC_WARPS) {
            const block_nvfp4 * v_ptr0 = params.V +
                kv_head       * params.v_stride_head +
                seq           * params.v_stride_seq +
                (kv_row - 1)  * params.v_stride_row;
            const block_nvfp4 * v_ptr = params.V +
                kv_head * params.v_stride_head +
                seq     * params.v_stride_seq +
                kv_row  * params.v_stride_row;
            const int pv_warp = warp_id - FATTN_NVFP4_TC_KQ_WARPS;

            for (int tile_idx = pv_warp; tile_idx < ncol_tile; tile_idx += FATTN_NVFP4_TC_PV_WARPS) {
                const int col_base = tile_idx * FATTN_NVFP4_PV_COL_TILE;

                tile_PV_A A;
                tile_PV_B B;
                tile_C C;

#pragma unroll
                for (int l = 0; l < tile_PV_A::ne; ++l) {
                    const int row = ggml_cuda_fattn_nvfp4_pv_a_i(l, lane);
                    const int col = ggml_cuda_fattn_nvfp4_pv_a_j(l, lane);
                    A.x[l] = row < FATTN_NVFP4_MTP4_ROWS && col == 0 ?
                        __floats2half2_rn(smem_pair_p0[row] * smem_scale_old[row], smem_p[row]) :
                        __floats2half2_rn(0.0f, 0.0f);
                }

#pragma unroll
                for (int l = 0; l < tile_PV_B::ne; ++l) {
                    const int row = ggml_cuda_fattn_nvfp4_pv_b_i(l, lane);
                    const int col = ggml_cuda_fattn_nvfp4_pv_b_j(l, lane);
                    B.x[l] = row < FATTN_NVFP4_PV_COL_TILE && col == 0 ?
                        ggml_cuda_fattn_nvfp4_lookup_row_pair_half2(params.v_lut, v_ptr0, v_ptr, col_base + row) :
                        __floats2half2_rn(0.0f, 0.0f);
                }

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int row = ggml_cuda_fattn_nvfp4_pv_c_i(l, lane);
                    C.x[l] = row < FATTN_NVFP4_MTP4_ROWS ?
                        pv_accum[tile_idx][lane][l] * smem_pair_scale_old[row] * smem_scale_old[row] :
                        0.0f;
                }

                ggml_cuda_mma::mma(C, A, B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int row = ggml_cuda_fattn_nvfp4_pv_c_i(l, lane);
                    if (row < FATTN_NVFP4_MTP4_ROWS) {
                        pv_accum[tile_idx][lane][l] = C.x[l];
                    }
                }
            }
        }
        if ((kv_row & 1) != 0) {
            __syncthreads();
        }
    }

    if (((kv_end - 1) & 1) == 0 && warp_id >= FATTN_NVFP4_TC_KQ_WARPS && warp_id < FATTN_NVFP4_TC_WARPS) {
        const block_nvfp4 * v_ptr = params.V +
            kv_head * params.v_stride_head +
            seq     * params.v_stride_seq +
            (kv_end - 1) * params.v_stride_row;
        const int pv_warp = warp_id - FATTN_NVFP4_TC_KQ_WARPS;

        for (int tile_idx = pv_warp; tile_idx < ncol_tile; tile_idx += FATTN_NVFP4_TC_PV_WARPS) {
            const int col_base = tile_idx * FATTN_NVFP4_PV_COL_TILE;

            tile_PV_A A;
            tile_PV_B B;
            tile_C C;

#pragma unroll
            for (int l = 0; l < tile_PV_A::ne; ++l) {
                const int row = ggml_cuda_fattn_nvfp4_pv_a_i(l, lane);
                const int col = ggml_cuda_fattn_nvfp4_pv_a_j(l, lane);
                A.x[l] = row < FATTN_NVFP4_MTP4_ROWS && col == 0 ?
                    __floats2half2_rn(smem_pair_p0[row], 0.0f) :
                    __floats2half2_rn(0.0f, 0.0f);
            }

#pragma unroll
            for (int l = 0; l < tile_PV_B::ne; ++l) {
                const int row = ggml_cuda_fattn_nvfp4_pv_b_i(l, lane);
                const int col = ggml_cuda_fattn_nvfp4_pv_b_j(l, lane);
                B.x[l] = row < FATTN_NVFP4_PV_COL_TILE && col == 0 ?
                    ggml_cuda_fattn_nvfp4_lookup_row_value_low_half2(params.v_lut, v_ptr, col_base + row) :
                    __floats2half2_rn(0.0f, 0.0f);
            }

#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int row = ggml_cuda_fattn_nvfp4_pv_c_i(l, lane);
                C.x[l] = row < FATTN_NVFP4_MTP4_ROWS ?
                    pv_accum[tile_idx][lane][l] * smem_pair_scale_old[row] :
                    0.0f;
            }

            ggml_cuda_mma::mma(C, A, B);

#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int row = ggml_cuda_fattn_nvfp4_pv_c_i(l, lane);
                if (row < FATTN_NVFP4_MTP4_ROWS) {
                    pv_accum[tile_idx][lane][l] = C.x[l];
                }
            }
        }
    }
    __syncthreads();

    if (warp_id >= FATTN_NVFP4_TC_KQ_WARPS && warp_id < FATTN_NVFP4_TC_WARPS) {
        const int pv_warp = warp_id - FATTN_NVFP4_TC_KQ_WARPS;

        for (int tile_idx = pv_warp; tile_idx < ncol_tile; tile_idx += FATTN_NVFP4_TC_PV_WARPS) {
            const int col_base = tile_idx * FATTN_NVFP4_PV_COL_TILE;

#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int row = ggml_cuda_fattn_nvfp4_pv_c_i(l, lane);
                const int col = col_base + ggml_cuda_fattn_nvfp4_pv_c_j(l, lane);
                const int64_t q_row = q_row_base + row;
                if (row < FATTN_NVFP4_MTP4_ROWS && q_row < params.ne_q_rows && col < params.v_head_dim && smem_rowsum[row] != 0.0f) {
#ifdef GGML_CUDA_NVFP4_FA_MMA_SPLIT_KV
                    if (params.split_partial != nullptr) {
                        const int64_t partial_idx =
                            ((((kv_split * params.ne_seqs + seq) * params.ne_q_heads + q_head) * params.ne_q_rows + q_row)
                                * params.v_head_dim + col);
                        params.split_partial[partial_idx] = pv_accum[tile_idx][lane][l];
                    } else
#endif // GGML_CUDA_NVFP4_FA_MMA_SPLIT_KV
                    {
                    float * dst_ptr = params.dst +
                        q_row  * params.dst_stride_row +
                        q_head * params.dst_stride_head +
                        seq    * params.dst_stride_seq;
                    dst_ptr[col] = pv_accum[tile_idx][lane][l] / smem_rowsum[row];
                    }
                }
            }
        }
    }

#ifdef GGML_CUDA_NVFP4_FA_MMA_SPLIT_KV
    if (params.split_meta != nullptr && warp_id == 0 && lane < FATTN_NVFP4_MTP4_ROWS) {
        const int row = lane;
        const int64_t q_row = q_row_base + row;
        if (q_row < params.ne_q_rows) {
            const int64_t meta_idx =
                ((kv_split * params.ne_seqs + seq) * params.ne_q_heads + q_head) * params.ne_q_rows + q_row;
            params.split_meta[meta_idx] = make_float2(smem_kq_max[row], smem_rowsum[row]);
        }
    }
#endif // GGML_CUDA_NVFP4_FA_MMA_SPLIT_KV
#else
    GGML_UNUSED(params);
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}
#endif // defined(GGML_CUDA_NVFP4_FA_MMA) && defined(GGML_CUDA_NVFP4_FA_MMA_MULTIWARP)

#if defined(GGML_CUDA_NVFP4_FA_MMA) && defined(GGML_CUDA_NVFP4_FA_MMA_MULTIWARP) && \
    defined(GGML_CUDA_NVFP4_FA_MMA_SPLIT_KV) && defined(GGML_CUDA_NVFP4_FA_MMA_TWOPASS)
__global__ void fattn_nvfp4_mtp4_multiwarp_twopass_kernel(const fattn_nvfp4_mtp4_params params) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane    = tid & (WARP_SIZE - 1);

    const int64_t q_row_block = (int64_t) blockIdx.x;
    const int64_t q_row_base  = q_row_block * FATTN_NVFP4_MTP4_ROWS;
    const int64_t q_head      = (int64_t) blockIdx.y;
    const int64_t seq         = (int64_t) blockIdx.z / params.kv_split_count;
    const int64_t kv_split    = (int64_t) blockIdx.z - seq * params.kv_split_count;
    const int64_t kv_start    = kv_split * params.kv_split_size;
    const int64_t kv_end      = min(params.ne_kv_rows, kv_start + params.kv_split_size);
    const int64_t kv_count    = kv_end - kv_start;
    const int64_t kv_head     = q_head / params.gqa_ratio;

    if (q_head >= params.ne_q_heads || seq >= params.ne_seqs || kv_head >= params.ne_kv_heads || kv_start >= kv_end) {
        return;
    }

    using tile_A    = ggml_cuda_mma::tile<16, 8, int>;
    using tile_B    = ggml_cuda_mma::tile< 8, 8, int>;
    using tile_C    = ggml_cuda_mma::tile<16, 8, float>;
    using tile_PV_A = ggml_cuda_mma::tile<16, 8, half2>;
    using tile_PV_B = ggml_cuda_mma::tile< 8, 8, half2>;

    const int nfrag     = (int) (params.k_head_dim / QK_NVFP4);
    const int ncol_tile = (int) (params.v_head_dim / FATTN_NVFP4_PV_COL_TILE);

    __shared__ int      q_tile[FATTN_NVFP4_MTP4_ROWS][FATTN_NVFP4_MAX_NFRAG][tile_A::J];
    __shared__ uint32_t q_tile_scale[FATTN_NVFP4_MTP4_ROWS][FATTN_NVFP4_MAX_NFRAG];
    __shared__ int      kq_a_tile[tile_A::I * tile_A::J];
    __shared__ int      kq_b_tile[tile_B::I * tile_B::J];
    __shared__ uint32_t kq_a_scale[tile_A::I];
    __shared__ uint32_t kq_b_scale[tile_B::I];
    __shared__ float    score_table[FATTN_NVFP4_SPLIT_KV_ROWS][FATTN_NVFP4_MTP4_ROWS];
    __shared__ float    p_table[FATTN_NVFP4_SPLIT_KV_ROWS][FATTN_NVFP4_MTP4_ROWS];
    __shared__ float    smem_kq_max[FATTN_NVFP4_MTP4_ROWS];
    __shared__ float    smem_rowsum[FATTN_NVFP4_MTP4_ROWS];
    __shared__ float    pv_accum[FATTN_NVFP4_MAX_NCOL_TILE][WARP_SIZE][tile_C::ne];

    for (int q_idx = tid; q_idx < FATTN_NVFP4_MTP4_ROWS * nfrag; q_idx += FATTN_NVFP4_TC_THREADS) {
        const int row  = q_idx / nfrag;
        const int frag = q_idx % nfrag;

        const int64_t q_row = q_row_base + row;
        if (q_row >= params.ne_q_rows) {
            q_tile_scale[row][frag] = 0;
#pragma unroll
            for (int k = 0; k < tile_A::J; ++k) {
                q_tile[row][frag][k] = 0;
            }
            continue;
        }

        const float * q_ptr = params.Q +
            q_row  * params.q_stride_row +
            q_head * params.q_stride_head +
            seq    * params.q_stride_seq;
        const block_nvfp4 q_blk = ggml_cuda_fattn_nvfp4_quantize_q_frag(q_ptr, frag);
        const uint32_t * q_qs = reinterpret_cast<const uint32_t *>(q_blk.qs);

#pragma unroll
        for (int k = 0; k < tile_A::J; ++k) {
            q_tile[row][frag][k] = (int) q_qs[k];
        }
        q_tile_scale[row][frag] = ggml_cuda_fattn_nvfp4_block_scale(q_blk);
    }

    for (int i = tid; i < FATTN_NVFP4_MTP4_ROWS; i += FATTN_NVFP4_TC_THREADS) {
        smem_kq_max[i] = -INFINITY;
        smem_rowsum[i] = 0.0f;
    }

    for (int i = tid; i < FATTN_NVFP4_MAX_NCOL_TILE * WARP_SIZE * tile_C::ne; i += FATTN_NVFP4_TC_THREADS) {
        reinterpret_cast<float *>(pv_accum)[i] = 0.0f;
    }
    __syncthreads();

    if (warp_id == 0) {
#pragma unroll 1
        for (int64_t kv_local = 0; kv_local < kv_count; ++kv_local) {
            const int64_t kv_row = kv_start + kv_local;
            const block_nvfp4 * k_ptr = params.K +
                kv_head * params.k_stride_head +
                seq     * params.k_stride_seq +
                kv_row  * params.k_stride_row;

            tile_C kq_tile = {};

            for (int q_frag = 0; q_frag < nfrag; ++q_frag) {
                const block_nvfp4 k_blk = k_ptr[q_frag];

                const uint32_t * k_qs = reinterpret_cast<const uint32_t *>(k_blk.qs);
                const uint32_t k_scale = ggml_cuda_fattn_nvfp4_block_scale(k_blk);

                for (int i = lane; i < tile_A::I * tile_A::J; i += WARP_SIZE) {
                    kq_a_tile[i] = 0;
                }
                for (int i = lane; i < tile_B::I * tile_B::J; i += WARP_SIZE) {
                    kq_b_tile[i] = 0;
                }
                for (int i = lane; i < tile_A::I; i += WARP_SIZE) {
                    kq_a_scale[i] = 0;
                }
                for (int i = lane; i < tile_B::I; i += WARP_SIZE) {
                    kq_b_scale[i] = k_scale;
                }
                __syncwarp();

                for (int i = lane; i < FATTN_NVFP4_MTP4_ROWS * tile_A::J; i += WARP_SIZE) {
                    const int row = i / tile_A::J;
                    const int k   = i % tile_A::J;
                    kq_a_tile[row * tile_A::J + k] = q_tile[row][q_frag][k];
                }
                for (int i = lane; i < tile_B::J; i += WARP_SIZE) {
                    kq_b_tile[i] = (int) k_qs[i];
                }
                if (lane < FATTN_NVFP4_MTP4_ROWS) {
                    kq_a_scale[lane] = q_tile_scale[lane][q_frag];
                }
                __syncwarp();

                tile_A A;
                tile_B B;
                ggml_cuda_mma::load_ldmatrix(A, kq_a_tile, tile_A::J);
                ggml_cuda_mma::load_generic(B, kq_b_tile, tile_B::J);

                const int tidx_A = threadIdx.x / 4 + (threadIdx.x % 2) * 8;
                const int tidx_B = threadIdx.x / 4;
                ggml_cuda_mma::mma_block_scaled_fp4<GGML_TYPE_NVFP4>(
                    kq_tile, A, B, kq_a_scale[tidx_A], kq_b_scale[tidx_B]);
            }

            float kq_score[FATTN_NVFP4_MTP4_ROWS] = {};

#pragma unroll
            for (int row = 0; row < FATTN_NVFP4_MTP4_ROWS; ++row) {
                float local_score = 0.0f;
#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    if (tile_C::get_i(l) == row && tile_C::get_j(l) == 0) {
                        local_score += kq_tile.x[l];
                    }
                }

#pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                    local_score += __shfl_down_sync(0xffffffff, local_score, offset);
                }
                kq_score[row] = __shfl_sync(0xffffffff, local_score, 0);
            }

#ifdef GGML_CUDA_NVFP4_FA_MMA_SCALAR_KQ
#pragma unroll
            for (int row = 0; row < FATTN_NVFP4_MTP4_ROWS; ++row) {
                float local_score = 0.0f;

                for (int col = lane; col < params.k_head_dim; col += WARP_SIZE) {
                    const int frag = col / QK_NVFP4;
                    const int i    = col % QK_NVFP4;
                    const float qv = ggml_cuda_fattn_nvfp4_dequant_packed(q_tile[row][frag], q_tile_scale[row][frag], i);
                    const float kv = ggml_cuda_fattn_nvfp4_dequant_row_value(k_ptr, col);
                    local_score += qv * kv;
                }

#pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                    local_score += __shfl_down_sync(0xffffffff, local_score, offset);
                }
                kq_score[row] = __shfl_sync(0xffffffff, local_score, 0);
            }
#endif // GGML_CUDA_NVFP4_FA_MMA_SCALAR_KQ

            if (lane < FATTN_NVFP4_MTP4_ROWS) {
                const int row = lane;
                const int64_t q_row = q_row_base + row;
                float score = -INFINITY;
                if (q_row < params.ne_q_rows) {
                    const float mask = ggml_cuda_fattn_nvfp4_mask_value(params, q_row, kv_row, seq);
                    score = kq_score[row] * params.scale + mask;
                    if (isinf(score) && score < 0.0f) {
                        score = -INFINITY;
                    }
                    smem_kq_max[row] = fmaxf(smem_kq_max[row], score);
                }
                score_table[kv_local][row] = score;
            }
        }

        if (lane < FATTN_NVFP4_MTP4_ROWS) {
            const int row = lane;
            float rowsum = 0.0f;

#pragma unroll 1
            for (int64_t kv_local = 0; kv_local < kv_count; ++kv_local) {
                const float score = score_table[kv_local][row];
                const float p = score - smem_kq_max[row] >= SOFTMAX_FTZ_THRESHOLD ?
                    expf(score - smem_kq_max[row]) : 0.0f;
                p_table[kv_local][row] = p;
                rowsum += p;
            }

            smem_rowsum[row] = rowsum;
        }
    }
    __syncthreads();

    if (warp_id >= FATTN_NVFP4_TC_KQ_WARPS && warp_id < FATTN_NVFP4_TC_WARPS) {
        const int pv_warp = warp_id - FATTN_NVFP4_TC_KQ_WARPS;

#pragma unroll 1
        for (int64_t kv_local = 0; kv_local < kv_count; kv_local += 2) {
            const int64_t kv_row0 = kv_start + kv_local;
            const int64_t kv_row1 = kv_row0 + 1;
            const block_nvfp4 * v_ptr0 = params.V +
                kv_head * params.v_stride_head +
                seq     * params.v_stride_seq +
                kv_row0 * params.v_stride_row;
            const block_nvfp4 * v_ptr1 = kv_row1 < kv_end ? params.V +
                kv_head * params.v_stride_head +
                seq     * params.v_stride_seq +
                kv_row1 * params.v_stride_row : nullptr;

            for (int tile_idx = pv_warp; tile_idx < ncol_tile; tile_idx += FATTN_NVFP4_TC_PV_WARPS) {
                const int col_base = tile_idx * FATTN_NVFP4_PV_COL_TILE;

                tile_PV_A A;
                tile_PV_B B;
                tile_C C;

#pragma unroll
                for (int l = 0; l < tile_PV_A::ne; ++l) {
                    const int row = ggml_cuda_fattn_nvfp4_pv_a_i(l, lane);
                    const int col = ggml_cuda_fattn_nvfp4_pv_a_j(l, lane);
                    A.x[l] = row < FATTN_NVFP4_MTP4_ROWS && col == 0 ?
                        __floats2half2_rn(
                            p_table[kv_local][row],
                            kv_local + 1 < kv_count ? p_table[kv_local + 1][row] : 0.0f) :
                        __floats2half2_rn(0.0f, 0.0f);
                }

#pragma unroll
                for (int l = 0; l < tile_PV_B::ne; ++l) {
                    const int row = ggml_cuda_fattn_nvfp4_pv_b_i(l, lane);
                    const int col = ggml_cuda_fattn_nvfp4_pv_b_j(l, lane);
                    B.x[l] = row < FATTN_NVFP4_PV_COL_TILE && col == 0 ?
                        (v_ptr1 != nullptr ?
                            ggml_cuda_fattn_nvfp4_lookup_row_pair_half2(params.v_lut, v_ptr0, v_ptr1, col_base + row) :
                            ggml_cuda_fattn_nvfp4_lookup_row_value_low_half2(params.v_lut, v_ptr0, col_base + row)) :
                        __floats2half2_rn(0.0f, 0.0f);
                }

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    C.x[l] = pv_accum[tile_idx][lane][l];
                }

                ggml_cuda_mma::mma(C, A, B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int row = ggml_cuda_fattn_nvfp4_pv_c_i(l, lane);
                    if (row < FATTN_NVFP4_MTP4_ROWS) {
                        pv_accum[tile_idx][lane][l] = C.x[l];
                    }
                }
            }
        }
    }
    __syncthreads();

    if (warp_id >= FATTN_NVFP4_TC_KQ_WARPS && warp_id < FATTN_NVFP4_TC_WARPS) {
        const int pv_warp = warp_id - FATTN_NVFP4_TC_KQ_WARPS;

        for (int tile_idx = pv_warp; tile_idx < ncol_tile; tile_idx += FATTN_NVFP4_TC_PV_WARPS) {
            const int col_base = tile_idx * FATTN_NVFP4_PV_COL_TILE;

#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int row = ggml_cuda_fattn_nvfp4_pv_c_i(l, lane);
                const int col = col_base + ggml_cuda_fattn_nvfp4_pv_c_j(l, lane);
                const int64_t q_row = q_row_base + row;
                if (row < FATTN_NVFP4_MTP4_ROWS && q_row < params.ne_q_rows && col < params.v_head_dim && smem_rowsum[row] != 0.0f) {
                    const int64_t partial_idx =
                        ((((kv_split * params.ne_seqs + seq) * params.ne_q_heads + q_head) * params.ne_q_rows + q_row)
                            * params.v_head_dim + col);
                    params.split_partial[partial_idx] = pv_accum[tile_idx][lane][l];
                }
            }
        }
    }

    if (params.split_meta != nullptr && warp_id == 0 && lane < FATTN_NVFP4_MTP4_ROWS) {
        const int row = lane;
        const int64_t q_row = q_row_base + row;
        if (q_row < params.ne_q_rows) {
            const int64_t meta_idx =
                ((kv_split * params.ne_seqs + seq) * params.ne_q_heads + q_head) * params.ne_q_rows + q_row;
            params.split_meta[meta_idx] = make_float2(smem_kq_max[row], smem_rowsum[row]);
        }
    }
#else
    GGML_UNUSED(params);
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}
#endif // defined(GGML_CUDA_NVFP4_FA_MMA) && defined(GGML_CUDA_NVFP4_FA_MMA_MULTIWARP) && defined(GGML_CUDA_NVFP4_FA_MMA_SPLIT_KV) && defined(GGML_CUDA_NVFP4_FA_MMA_TWOPASS)

#if defined(GGML_CUDA_NVFP4_FA_MMA) && defined(GGML_CUDA_NVFP4_FA_MMA_MULTIWARP) && \
    defined(GGML_CUDA_NVFP4_FA_MMA_SPLIT_KV) && defined(GGML_CUDA_NVFP4_FA_MMA_DECOUPLED)
static __device__ __forceinline__ int64_t ggml_cuda_fattn_nvfp4_split_prob_idx(
        const fattn_nvfp4_mtp4_params & params,
        const int64_t                   kv_split,
        const int64_t                   seq,
        const int64_t                   q_head,
        const int64_t                   q_row,
        const int64_t                   kv_local) {
#if defined(GGML_CUDA_NVFP4_FA_ACTIVE_DECOUPLED_WINDOW_SPLITS)
    const int64_t prob_split = kv_split - params.kv_split_base;
#else
    const int64_t prob_split = kv_split;
#endif // defined(GGML_CUDA_NVFP4_FA_ACTIVE_DECOUPLED_WINDOW_SPLITS)
    return ((((prob_split * params.ne_seqs + seq) * params.ne_q_heads + q_head) * params.ne_q_rows + q_row)
        * params.kv_split_size + kv_local);
}

#if defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_REQUANT)
__global__ void fattn_nvfp4_mtp4_prepack_v_kernel(const fattn_nvfp4_mtp4_params params) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    using tile_PV_B = ggml_cuda_mma::tile<8, 8, int>;

    const int64_t ncol_tile = params.v_head_dim / FATTN_NVFP4_PV_COL_TILE;
    const int64_t tile_id   = (int64_t) blockIdx.x;
    const int64_t total     =
        params.kv_split_active_count * params.ne_seqs * params.ne_kv_heads * ncol_tile * params.native_pv_nkv_tile;

    if (tile_id >= total || params.native_pv_v_qs == nullptr || params.native_pv_v_scale == nullptr) {
        return;
    }

    int64_t rem = tile_id;
    const int64_t kv_tile = rem % params.native_pv_nkv_tile;
    rem /= params.native_pv_nkv_tile;
    const int64_t col_tile = rem % ncol_tile;
    rem /= ncol_tile;
    const int64_t kv_head = rem % params.ne_kv_heads;
    rem /= params.ne_kv_heads;
    const int64_t seq = rem % params.ne_seqs;
    rem /= params.ne_seqs;
    const int64_t kv_split_local = rem;

    const int64_t kv_start = (params.kv_split_base + kv_split_local) * params.kv_split_size + kv_tile * QK_NVFP4;
    const int64_t col_base = col_tile * FATTN_NVFP4_PV_COL_TILE;

    int * const      qs_out    = params.native_pv_v_qs    + tile_id * (tile_PV_B::I * tile_PV_B::J);
    uint32_t * const scale_out = params.native_pv_v_scale + tile_id * tile_PV_B::I;

    if (threadIdx.x == 0) {
#pragma unroll
        for (int col = 0; col < FATTN_NVFP4_PV_COL_TILE; ++col) {
            float vals[QK_NVFP4];

#pragma unroll
            for (int k = 0; k < QK_NVFP4; ++k) {
                const int64_t kv_row = kv_start + k;
                const int64_t v_col  = col_base + col;
                if (kv_row < params.ne_kv_rows && v_col < params.v_head_dim) {
                    const block_nvfp4 * v_ptr = params.V +
                        kv_head * params.v_stride_head +
                        seq     * params.v_stride_seq +
                        kv_row  * params.v_stride_row;
                    vals[k] = ggml_cuda_fattn_nvfp4_dequant_row_value(v_ptr, v_col);
                } else {
                    vals[k] = 0.0f;
                }
            }

            int qs_words[QK_NVFP4 / 8];
            uint32_t scale_word;
            ggml_cuda_fattn_nvfp4_quantize_64_words(vals, qs_words, scale_word);

#pragma unroll
            for (int word = 0; word < QK_NVFP4 / 8; ++word) {
                qs_out[col * tile_PV_B::J + word] = qs_words[word];
            }
            scale_out[col] = scale_word;
        }
    }
#else
    GGML_UNUSED(params);
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}
#endif // defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_REQUANT)

__global__ void fattn_nvfp4_mtp4_split_kq_prob_kernel(const fattn_nvfp4_mtp4_params params) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    const int64_t q_row_block = (int64_t) blockIdx.x;
    const int64_t q_row_base  = q_row_block * FATTN_NVFP4_MTP4_ROWS;
    const int64_t q_head      = (int64_t) blockIdx.y;
#if defined(GGML_CUDA_NVFP4_FA_ACTIVE_DECOUPLED_WINDOW_SPLITS)
    const int64_t seq         = (int64_t) blockIdx.z / params.kv_split_active_count;
    const int64_t kv_split    = params.kv_split_base + (int64_t) blockIdx.z - seq * params.kv_split_active_count;
#else
    const int64_t seq         = (int64_t) blockIdx.z / params.kv_split_count;
    const int64_t kv_split    = (int64_t) blockIdx.z - seq * params.kv_split_count;
#endif // defined(GGML_CUDA_NVFP4_FA_ACTIVE_DECOUPLED_WINDOW_SPLITS)
    const int64_t kv_start    = kv_split * params.kv_split_size;
    const int64_t kv_end      = min(params.ne_kv_rows, kv_start + params.kv_split_size);
    const int64_t kv_count    = kv_end - kv_start;
    const int64_t kv_head     = q_head / params.gqa_ratio;

    if (q_head >= params.ne_q_heads || seq >= params.ne_seqs || kv_head >= params.ne_kv_heads || kv_start >= kv_end ||
            params.split_prob == nullptr || params.split_meta == nullptr) {
        return;
    }

    using tile_A = ggml_cuda_mma::tile<16, 8, int>;
    using tile_B = ggml_cuda_mma::tile< 8, 8, int>;
    using tile_C = ggml_cuda_mma::tile<16, 8, float>;

    const int nfrag = (int) (params.k_head_dim / QK_NVFP4);

    __shared__ int      q_tile[FATTN_NVFP4_MTP4_ROWS][FATTN_NVFP4_MAX_NFRAG][tile_A::J];
    __shared__ uint32_t q_tile_scale[FATTN_NVFP4_MTP4_ROWS][FATTN_NVFP4_MAX_NFRAG];
    __shared__ int      kq_a_tile[tile_A::I * tile_A::J];
    __shared__ int      kq_b_tile[tile_B::I * tile_B::J];
    __shared__ uint32_t kq_a_scale[tile_A::I];
    __shared__ uint32_t kq_b_scale[tile_B::I];
    __shared__ float    score_table[FATTN_NVFP4_SPLIT_KV_ROWS][FATTN_NVFP4_MTP4_ROWS];
    __shared__ float    smem_kq_max[FATTN_NVFP4_MTP4_ROWS];

    for (int q_idx = lane; q_idx < FATTN_NVFP4_MTP4_ROWS * nfrag; q_idx += WARP_SIZE) {
        const int row  = q_idx / nfrag;
        const int frag = q_idx % nfrag;

        const int64_t q_row = q_row_base + row;
        if (q_row >= params.ne_q_rows) {
            q_tile_scale[row][frag] = 0;
#pragma unroll
            for (int k = 0; k < tile_A::J; ++k) {
                q_tile[row][frag][k] = 0;
            }
            continue;
        }

        const float * q_ptr = params.Q +
            q_row  * params.q_stride_row +
            q_head * params.q_stride_head +
            seq    * params.q_stride_seq;
        const block_nvfp4 q_blk = ggml_cuda_fattn_nvfp4_quantize_q_frag(q_ptr, frag);
        const uint32_t * q_qs = reinterpret_cast<const uint32_t *>(q_blk.qs);

#pragma unroll
        for (int k = 0; k < tile_A::J; ++k) {
            q_tile[row][frag][k] = (int) q_qs[k];
        }
        q_tile_scale[row][frag] = ggml_cuda_fattn_nvfp4_block_scale(q_blk);
    }

    if (lane < FATTN_NVFP4_MTP4_ROWS) {
        smem_kq_max[lane] = -INFINITY;
    }
    __syncwarp();

#pragma unroll 1
    for (int64_t kv_local = 0; kv_local < kv_count; ++kv_local) {
        const int64_t kv_row = kv_start + kv_local;
        const block_nvfp4 * k_ptr = params.K +
            kv_head * params.k_stride_head +
            seq     * params.k_stride_seq +
            kv_row  * params.k_stride_row;

        tile_C kq_tile = {};

        for (int q_frag = 0; q_frag < nfrag; ++q_frag) {
            const block_nvfp4 k_blk = k_ptr[q_frag];

            const uint32_t * k_qs = reinterpret_cast<const uint32_t *>(k_blk.qs);
            const uint32_t k_scale = ggml_cuda_fattn_nvfp4_block_scale(k_blk);

            for (int i = lane; i < tile_A::I * tile_A::J; i += WARP_SIZE) {
                kq_a_tile[i] = 0;
            }
            for (int i = lane; i < tile_B::I * tile_B::J; i += WARP_SIZE) {
                kq_b_tile[i] = 0;
            }
            for (int i = lane; i < tile_A::I; i += WARP_SIZE) {
                kq_a_scale[i] = 0;
            }
            for (int i = lane; i < tile_B::I; i += WARP_SIZE) {
                kq_b_scale[i] = k_scale;
            }
            __syncwarp();

            for (int i = lane; i < FATTN_NVFP4_MTP4_ROWS * tile_A::J; i += WARP_SIZE) {
                const int row = i / tile_A::J;
                const int k   = i % tile_A::J;
                kq_a_tile[row * tile_A::J + k] = q_tile[row][q_frag][k];
            }
            for (int i = lane; i < tile_B::J; i += WARP_SIZE) {
                kq_b_tile[i] = (int) k_qs[i];
            }
            if (lane < FATTN_NVFP4_MTP4_ROWS) {
                kq_a_scale[lane] = q_tile_scale[lane][q_frag];
            }
            __syncwarp();

            tile_A A;
            tile_B B;
            ggml_cuda_mma::load_ldmatrix(A, kq_a_tile, tile_A::J);
            ggml_cuda_mma::load_generic(B, kq_b_tile, tile_B::J);

            const int tidx_A = threadIdx.x / 4 + (threadIdx.x % 2) * 8;
            const int tidx_B = threadIdx.x / 4;
            ggml_cuda_mma::mma_block_scaled_fp4<GGML_TYPE_NVFP4>(
                kq_tile, A, B, kq_a_scale[tidx_A], kq_b_scale[tidx_B]);
        }

        float kq_score[FATTN_NVFP4_MTP4_ROWS] = {};

#pragma unroll
        for (int row = 0; row < FATTN_NVFP4_MTP4_ROWS; ++row) {
            float local_score = 0.0f;
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                if (tile_C::get_i(l) == row && tile_C::get_j(l) == 0) {
                    local_score += kq_tile.x[l];
                }
            }

#pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                local_score += __shfl_down_sync(0xffffffff, local_score, offset);
            }
            kq_score[row] = __shfl_sync(0xffffffff, local_score, 0);
        }

#ifdef GGML_CUDA_NVFP4_FA_MMA_SCALAR_KQ
#pragma unroll
        for (int row = 0; row < FATTN_NVFP4_MTP4_ROWS; ++row) {
            float local_score = 0.0f;

            for (int col = lane; col < params.k_head_dim; col += WARP_SIZE) {
                const int frag = col / QK_NVFP4;
                const int i    = col % QK_NVFP4;
                const float qv = ggml_cuda_fattn_nvfp4_dequant_packed(q_tile[row][frag], q_tile_scale[row][frag], i);
                const float kv = ggml_cuda_fattn_nvfp4_dequant_row_value(k_ptr, col);
                local_score += qv * kv;
            }

#pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                local_score += __shfl_down_sync(0xffffffff, local_score, offset);
            }
            kq_score[row] = __shfl_sync(0xffffffff, local_score, 0);
        }
#endif // GGML_CUDA_NVFP4_FA_MMA_SCALAR_KQ

        if (lane < FATTN_NVFP4_MTP4_ROWS) {
            const int row = lane;
            const int64_t q_row = q_row_base + row;
            float score = -INFINITY;
            if (q_row < params.ne_q_rows) {
                const float mask = ggml_cuda_fattn_nvfp4_mask_value(params, q_row, kv_row, seq);
                score = kq_score[row] * params.scale + mask;
                if (isinf(score) && score < 0.0f) {
                    score = -INFINITY;
                }
                smem_kq_max[row] = fmaxf(smem_kq_max[row], score);
            }
            score_table[kv_local][row] = score;
        }
    }

    if (lane < FATTN_NVFP4_MTP4_ROWS) {
        const int row = lane;
        const int64_t q_row = q_row_base + row;
        float rowsum = 0.0f;

#pragma unroll 1
        for (int64_t kv_local = 0; kv_local < kv_count; ++kv_local) {
            const float score = score_table[kv_local][row];
            const float p = score - smem_kq_max[row] >= SOFTMAX_FTZ_THRESHOLD ?
                expf(score - smem_kq_max[row]) : 0.0f;
            if (q_row < params.ne_q_rows) {
                params.split_prob[ggml_cuda_fattn_nvfp4_split_prob_idx(params, kv_split, seq, q_head, q_row, kv_local)] = p;
            }
            rowsum += p;
        }

        if (q_row < params.ne_q_rows) {
            const int64_t meta_idx =
                ((kv_split * params.ne_seqs + seq) * params.ne_q_heads + q_head) * params.ne_q_rows + q_row;
            params.split_meta[meta_idx] = make_float2(smem_kq_max[row], rowsum);
        }
    }
#else
    GGML_UNUSED(params);
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}

__global__ void fattn_nvfp4_mtp4_split_pv_from_prob_kernel(const fattn_nvfp4_mtp4_params params) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane    = tid & (WARP_SIZE - 1);

    const int64_t q_row_block = (int64_t) blockIdx.x;
    const int64_t q_row_base  = q_row_block * FATTN_NVFP4_MTP4_ROWS;
    const int64_t q_head      = (int64_t) blockIdx.y;
#if defined(GGML_CUDA_NVFP4_FA_ACTIVE_DECOUPLED_WINDOW_SPLITS)
    const int64_t seq         = (int64_t) blockIdx.z / params.kv_split_active_count;
    const int64_t kv_split    = params.kv_split_base + (int64_t) blockIdx.z - seq * params.kv_split_active_count;
#else
    const int64_t seq         = (int64_t) blockIdx.z / params.kv_split_count;
    const int64_t kv_split    = (int64_t) blockIdx.z - seq * params.kv_split_count;
#endif // defined(GGML_CUDA_NVFP4_FA_ACTIVE_DECOUPLED_WINDOW_SPLITS)
    const int64_t kv_start    = kv_split * params.kv_split_size;
    const int64_t kv_end      = min(params.ne_kv_rows, kv_start + params.kv_split_size);
    const int64_t kv_count    = kv_end - kv_start;
    const int64_t kv_head     = q_head / params.gqa_ratio;

    if (q_head >= params.ne_q_heads || seq >= params.ne_seqs || kv_head >= params.ne_kv_heads || kv_start >= kv_end ||
            params.split_prob == nullptr || params.split_partial == nullptr) {
        return;
    }

#if defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_REQUANT)
    if ((params.native_pv_v_qs == nullptr || params.native_pv_v_scale == nullptr)
#if defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_VX) && defined(GGML_CUDA_NVFP4_KV_EXEC_LAYOUT)
            && !params.native_pv_use_vx
#endif // defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_VX) && defined(GGML_CUDA_NVFP4_KV_EXEC_LAYOUT)
            ) {
        return;
    }
#endif // defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_REQUANT)

    using tile_C = ggml_cuda_mma::tile<16, 8, float>;

#if defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_REQUANT)
    using tile_PV_A = ggml_cuda_mma::tile<16, 8, int>;
    using tile_PV_B = ggml_cuda_mma::tile< 8, 8, int>;

    __shared__ int native_pv_a_qs[FATTN_NVFP4_TC_PV_WARPS][tile_PV_A::I * tile_PV_A::J];
    __shared__ uint32_t native_pv_a_scale[FATTN_NVFP4_TC_PV_WARPS][tile_PV_A::I];
#else
    using tile_PV_A = ggml_cuda_mma::tile<16, 8, half2>;
    using tile_PV_B = ggml_cuda_mma::tile< 8, 8, half2>;
#endif // defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_REQUANT)

    const int ncol_tile = (int) (params.v_head_dim / FATTN_NVFP4_PV_COL_TILE);

    for (int tile_idx = warp_id; tile_idx < ncol_tile; tile_idx += FATTN_NVFP4_TC_PV_WARPS) {
        const int col_base = tile_idx * FATTN_NVFP4_PV_COL_TILE;
        tile_C C = {};

#if defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_REQUANT)
#pragma unroll 1
        for (int64_t kv_local = 0; kv_local < kv_count; kv_local += QK_NVFP4) {
            int * const      a_qs    = native_pv_a_qs[warp_id];
            uint32_t * const a_scale = native_pv_a_scale[warp_id];

            const int64_t native_pv_split   = kv_split - params.kv_split_base;
            const int64_t native_pv_kv_tile = kv_local / QK_NVFP4;
            const int * b_qs = nullptr;
            const uint32_t * b_scale = nullptr;
#if defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_VX) && defined(GGML_CUDA_NVFP4_KV_EXEC_LAYOUT)
            if (params.native_pv_use_vx) {
                const int64_t vx_kv_tile = (kv_split * params.kv_split_size + kv_local) / QK_NVFP4;
                const ggml_cuda_nvfp4_vx_tile * vx_tile =
                    ggml_cuda_nvfp4_vx_tile_ptr(params.native_pv_vx_layout, seq, kv_head, tile_idx, vx_kv_tile);
                b_qs = vx_tile->qs;
                b_scale = vx_tile->scale;
            } else
#endif // defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_VX) && defined(GGML_CUDA_NVFP4_KV_EXEC_LAYOUT)
            {
                const int64_t native_pv_tile_id =
                    ((((native_pv_split * params.ne_seqs + seq) * params.ne_kv_heads + kv_head) * ncol_tile + tile_idx)
                        * params.native_pv_nkv_tile + native_pv_kv_tile);
                b_qs = params.native_pv_v_qs + native_pv_tile_id * (tile_PV_B::I * tile_PV_B::J);
                b_scale = params.native_pv_v_scale + native_pv_tile_id * tile_PV_B::I;
            }

            for (int i = lane; i < tile_PV_A::I * tile_PV_A::J; i += WARP_SIZE) {
                a_qs[i] = 0;
            }
            for (int i = lane; i < tile_PV_A::I; i += WARP_SIZE) {
                a_scale[i] = 0;
            }

            __syncwarp();

            if (lane == 0) {
#pragma unroll
                for (int row = 0; row < FATTN_NVFP4_MTP4_ROWS; ++row) {
                    float vals[QK_NVFP4];

#pragma unroll
                    for (int k = 0; k < QK_NVFP4; ++k) {
                        const int64_t q_row = q_row_base + row;
                        vals[k] = q_row < params.ne_q_rows && kv_local + k < kv_count ?
                            params.split_prob[ggml_cuda_fattn_nvfp4_split_prob_idx(params, kv_split, seq, q_head, q_row, kv_local + k)] :
                            0.0f;
                    }

                    int qs_words[QK_NVFP4 / 8];
                    uint32_t scale_word;
                    ggml_cuda_fattn_nvfp4_quantize_64_words(vals, qs_words, scale_word);

#pragma unroll
                    for (int word = 0; word < QK_NVFP4 / 8; ++word) {
                        a_qs[row * tile_PV_A::J + word] = qs_words[word];
                    }
                    a_scale[row] = scale_word;
                }
            }

            __syncwarp();

            tile_PV_A A;
            tile_PV_B B;
            tile_C C_chunk = {};

#pragma unroll
            for (int l = 0; l < tile_PV_A::ne; ++l) {
                const int row = ggml_cuda_fattn_nvfp4_native_pv_a_i(l, lane);
                const int col = ggml_cuda_fattn_nvfp4_native_pv_a_j(l, lane);
                A.x[l] = a_qs[row * tile_PV_A::J + col];
            }

#pragma unroll
            for (int l = 0; l < tile_PV_B::ne; ++l) {
                const int row = ggml_cuda_fattn_nvfp4_native_pv_b_i(l, lane);
                const int col = ggml_cuda_fattn_nvfp4_native_pv_b_j(l, lane);
                B.x[l] = b_qs[row * tile_PV_B::J + col];
            }

            const int tidx_A = lane / 4 + (lane % 2) * 8;
            const int tidx_B = lane / 4;
            ggml_cuda_mma::mma_block_scaled_fp4<GGML_TYPE_NVFP4>(C_chunk, A, B, a_scale[tidx_A], b_scale[tidx_B]);

#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                C.x[l] += C_chunk.x[l];
            }
        }
#else
#pragma unroll 1
        for (int64_t kv_local = 0; kv_local < kv_count; kv_local += 2) {
            const int64_t kv_row0 = kv_start + kv_local;
            const int64_t kv_row1 = kv_row0 + 1;
            const block_nvfp4 * v_ptr0 = params.V +
                kv_head * params.v_stride_head +
                seq     * params.v_stride_seq +
                kv_row0 * params.v_stride_row;
            const block_nvfp4 * v_ptr1 = kv_row1 < kv_end ? params.V +
                kv_head * params.v_stride_head +
                seq     * params.v_stride_seq +
                kv_row1 * params.v_stride_row : nullptr;

            tile_PV_A A;
            tile_PV_B B;

#pragma unroll
            for (int l = 0; l < tile_PV_A::ne; ++l) {
                const int row = ggml_cuda_fattn_nvfp4_pv_a_i(l, lane);
                const int col = ggml_cuda_fattn_nvfp4_pv_a_j(l, lane);
                if (row < FATTN_NVFP4_MTP4_ROWS && col == 0) {
                    const int64_t q_row = q_row_base + row;
                    const float p0 = q_row < params.ne_q_rows ?
                        params.split_prob[ggml_cuda_fattn_nvfp4_split_prob_idx(params, kv_split, seq, q_head, q_row, kv_local)] :
                        0.0f;
                    const float p1 = q_row < params.ne_q_rows && kv_local + 1 < kv_count ?
                        params.split_prob[ggml_cuda_fattn_nvfp4_split_prob_idx(params, kv_split, seq, q_head, q_row, kv_local + 1)] :
                        0.0f;
                    A.x[l] = __floats2half2_rn(p0, p1);
                } else {
                    A.x[l] = __floats2half2_rn(0.0f, 0.0f);
                }
            }

#pragma unroll
            for (int l = 0; l < tile_PV_B::ne; ++l) {
                const int row = ggml_cuda_fattn_nvfp4_pv_b_i(l, lane);
                const int col = ggml_cuda_fattn_nvfp4_pv_b_j(l, lane);
                B.x[l] = row < FATTN_NVFP4_PV_COL_TILE && col == 0 ?
                    (v_ptr1 != nullptr ?
                        ggml_cuda_fattn_nvfp4_lookup_row_pair_half2(params.v_lut, v_ptr0, v_ptr1, col_base + row) :
                        ggml_cuda_fattn_nvfp4_lookup_row_value_low_half2(params.v_lut, v_ptr0, col_base + row)) :
                    __floats2half2_rn(0.0f, 0.0f);
            }

            ggml_cuda_mma::mma(C, A, B);
        }
#endif // defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_REQUANT)

#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
            const int row = ggml_cuda_fattn_nvfp4_pv_c_i(l, lane);
            const int col = col_base + ggml_cuda_fattn_nvfp4_pv_c_j(l, lane);
            const int64_t q_row = q_row_base + row;
            if (row < FATTN_NVFP4_MTP4_ROWS && q_row < params.ne_q_rows && col < params.v_head_dim) {
                const int64_t partial_idx =
                    ((((kv_split * params.ne_seqs + seq) * params.ne_q_heads + q_head) * params.ne_q_rows + q_row)
                        * params.v_head_dim + col);
                params.split_partial[partial_idx] = C.x[l];
            }
        }
    }
#else
    GGML_UNUSED(params);
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}
#endif // defined(GGML_CUDA_NVFP4_FA_MMA) && defined(GGML_CUDA_NVFP4_FA_MMA_MULTIWARP) && defined(GGML_CUDA_NVFP4_FA_MMA_SPLIT_KV) && defined(GGML_CUDA_NVFP4_FA_MMA_DECOUPLED)

#if defined(GGML_CUDA_NVFP4_FA_MMA) && defined(GGML_CUDA_NVFP4_FA_MMA_MULTIWARP) && defined(GGML_CUDA_NVFP4_FA_MMA_SPLIT_KV)
__global__ void fattn_nvfp4_mtp4_split_kv_combine_kernel(const fattn_nvfp4_mtp4_params params) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = params.ne_seqs * params.ne_q_heads * params.ne_q_rows * params.v_head_dim;
    if (idx >= total) {
        return;
    }

    int64_t rem = idx;
    const int64_t col = rem % params.v_head_dim;
    rem /= params.v_head_dim;
    const int64_t q_row = rem % params.ne_q_rows;
    rem /= params.ne_q_rows;
    const int64_t q_head = rem % params.ne_q_heads;
    rem /= params.ne_q_heads;
    const int64_t seq = rem;

    float acc = 0.0f;
    float max_val = -3.402823466e+38F;
    float rowsum = 0.0f;

    for (int64_t split = 0; split < params.kv_split_count; ++split) {
        const int64_t meta_idx =
            ((split * params.ne_seqs + seq) * params.ne_q_heads + q_head) * params.ne_q_rows + q_row;
        const float2 meta = params.split_meta[meta_idx];
        if (meta.y == 0.0f) {
            continue;
        }

        const int64_t partial_idx =
            ((((split * params.ne_seqs + seq) * params.ne_q_heads + q_head) * params.ne_q_rows + q_row)
                * params.v_head_dim + col);
        const float partial = params.split_partial[partial_idx];

        const float max_new = fmaxf(max_val, meta.x);
        const float scale_acc = max_val - max_new >= SOFTMAX_FTZ_THRESHOLD ? expf(max_val - max_new) : 0.0f;
        const float scale_add = meta.x   - max_new >= SOFTMAX_FTZ_THRESHOLD ? expf(meta.x   - max_new) : 0.0f;

        acc = acc * scale_acc + partial * scale_add;
        rowsum = rowsum * scale_acc + meta.y * scale_add;
        max_val = max_new;
    }

    float * dst_ptr = params.dst +
        q_row  * params.dst_stride_row +
        q_head * params.dst_stride_head +
        seq    * params.dst_stride_seq;
    dst_ptr[col] = rowsum == 0.0f ? 0.0f : acc / rowsum;
#else
    GGML_UNUSED(params);
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}
#endif // defined(GGML_CUDA_NVFP4_FA_MMA) && defined(GGML_CUDA_NVFP4_FA_MMA_MULTIWARP) && defined(GGML_CUDA_NVFP4_FA_MMA_SPLIT_KV)

__global__ void fattn_nvfp4_mtp4_scalar_correctness_kernel(const fattn_nvfp4_mtp4_params params) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int col = threadIdx.x;

    const int64_t q_row_base = (int64_t) blockIdx.x * FATTN_NVFP4_MTP4_ROWS;
    const int64_t q_head     = (int64_t) blockIdx.y;
    const int64_t seq        = (int64_t) blockIdx.z;
    const int64_t kv_head    = q_head / params.gqa_ratio;

    if (col >= params.v_head_dim || q_head >= params.ne_q_heads || seq >= params.ne_seqs || kv_head >= params.ne_kv_heads) {
        return;
    }

#pragma unroll
    for (int row = 0; row < FATTN_NVFP4_MTP4_ROWS; ++row) {
        const int64_t q_row = q_row_base + row;
        if (q_row >= params.ne_q_rows) {
            continue;
        }

        const float * q_ptr = params.Q +
            q_row  * params.q_stride_row +
            q_head * params.q_stride_head +
            seq    * params.q_stride_seq;
        float * dst_ptr = params.dst +
            q_row  * params.dst_stride_row +
            q_head * params.dst_stride_head +
            seq    * params.dst_stride_seq;

        float kq_max = -3.402823466e+38F;
        float rowsum = 0.0f;
        float pv     = 0.0f;

#pragma unroll 1
        for (int64_t kv_row = 0; kv_row < params.ne_kv_rows; ++kv_row) {
            const block_nvfp4 * k_ptr = params.K +
                kv_head * params.k_stride_head +
                seq     * params.k_stride_seq +
                kv_row  * params.k_stride_row;
            const block_nvfp4 * v_ptr = params.V +
                kv_head * params.v_stride_head +
                seq     * params.v_stride_seq +
                kv_row  * params.v_stride_row;

            const float mask  = ggml_cuda_fattn_nvfp4_mask_value(params, q_row, kv_row, seq);
            const float score = ggml_cuda_fattn_nvfp4_dot_q_k(q_ptr, k_ptr, params.k_head_dim) * params.scale + mask;
            const float kq_max_new = fmaxf(kq_max, score);
            const float scale_old = rowsum == 0.0f ? 0.0f : expf(kq_max - kq_max_new);
            const float scale_new = score - kq_max_new >= SOFTMAX_FTZ_THRESHOLD ? expf(score - kq_max_new) : 0.0f;

            pv = pv * scale_old + scale_new * ggml_cuda_fattn_nvfp4_dequant_row_value(v_ptr, col);
            rowsum = rowsum * scale_old + scale_new;
            kq_max = kq_max_new;
        }

        dst_ptr[col] = rowsum == 0.0f ? 0.0f : pv / rowsum;
    }
#else
    GGML_UNUSED(params);
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}

#if defined(GGML_CUDA_NVFP4_KV_EXEC_LAYOUT) && defined(GGML_CUDA_NVFP4_KV_EXEC_P1_SCALAR)
#if defined(GGML_CUDA_NVFP4_KV_EXEC_P1_NATIVE_KQ)
__global__ void fattn_nvfp4_p1_kx_mma_kernel(
        const fattn_nvfp4_mtp4_params params,
        const ggml_cuda_nvfp4_kx_layout kx_layout) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    using tile_A = ggml_cuda_mma::tile<16, 8, int>;
    using tile_B = ggml_cuda_mma::tile< 8, 8, int>;
    using tile_C = ggml_cuda_mma::tile<16, 8, float>;

    __shared__ int      kq_a_tile[tile_A::I * tile_A::J];
    __shared__ int      kq_b_tile[tile_B::I * tile_B::J];
    __shared__ uint32_t kq_a_scale[tile_A::I];
    __shared__ uint32_t kq_b_scale[tile_B::I];
    __shared__ float    smem_kq_score[GGML_CUDA_NVFP4_KX_KV_ROWS];

    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int col_warp = threadIdx.y;
    const int64_t q_head = (int64_t) blockIdx.x;
    const int64_t col_base = (int64_t) col_warp * WARP_SIZE;
    const int64_t col = col_base + lane;
    const int64_t seq = (int64_t) blockIdx.y;
    const int64_t kv_head = q_head / params.gqa_ratio;
    const int nfrag = (int) (params.k_head_dim / QK_NVFP4);

    if (q_head >= params.ne_q_heads || seq >= params.ne_seqs || kv_head >= params.ne_kv_heads) {
        return;
    }

    const float * q_ptr = params.Q +
        q_head * params.q_stride_head +
        seq    * params.q_stride_seq;
    float * dst_ptr = params.dst +
        q_head * params.dst_stride_head +
        seq    * params.dst_stride_seq;

    float kq_max = -3.402823466e+38F;
    float rowsum = 0.0f;
    float pv = 0.0f;

#pragma unroll 1
    for (int64_t kv_tile = 0; kv_tile < kx_layout.kv_tile_count; ++kv_tile) {
        if (col_warp == 0) {
            float kq_score[GGML_CUDA_NVFP4_KX_KV_ROWS] = {};

#pragma unroll 1
            for (int q_frag = 0; q_frag < nfrag; ++q_frag) {
                const block_nvfp4 q_blk = ggml_cuda_fattn_nvfp4_quantize_q_frag(q_ptr, q_frag);
                const uint32_t * q_qs = reinterpret_cast<const uint32_t *>(q_blk.qs);
                const uint32_t q_scale = ggml_cuda_fattn_nvfp4_block_scale(q_blk);
                const ggml_cuda_nvfp4_kx_tile * kx_tile =
                    ggml_cuda_nvfp4_kx_tile_ptr(kx_layout, seq, kv_head, q_frag, kv_tile);

                for (int i = lane; i < tile_A::I * tile_A::J; i += WARP_SIZE) {
                    kq_a_tile[i] = 0;
                }
                for (int i = lane; i < tile_B::I * tile_B::J; i += WARP_SIZE) {
                    kq_b_tile[i] = kx_tile->qs[i];
                }
                for (int i = lane; i < tile_A::I; i += WARP_SIZE) {
                    kq_a_scale[i] = q_scale;
                }
                for (int i = lane; i < tile_B::I; i += WARP_SIZE) {
                    kq_b_scale[i] = kx_tile->scale[i];
                }
                if (lane < tile_A::J) {
                    kq_a_tile[lane] = (int) q_qs[lane];
                }
                __syncwarp();

                tile_A A;
                tile_B B;
                tile_C C = {};
                ggml_cuda_mma::load_ldmatrix(A, kq_a_tile, tile_A::J);
                ggml_cuda_mma::load_generic(B, kq_b_tile, tile_B::J);

                const int tidx_A = lane / 4 + (lane % 2) * 8;
                const int tidx_B = lane / 4;
                ggml_cuda_mma::mma_block_scaled_fp4<GGML_TYPE_NVFP4>(
                    C, A, B, kq_a_scale[tidx_A], kq_b_scale[tidx_B]);

#pragma unroll
                for (int row = 0; row < GGML_CUDA_NVFP4_KX_KV_ROWS; ++row) {
                    float local_score = 0.0f;
#pragma unroll
                    for (int l = 0; l < tile_C::ne; ++l) {
                        if (tile_C::get_i(l) == 0 && tile_C::get_j(l) == row) {
                            local_score += C.x[l];
                        }
                    }

#pragma unroll
                    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                        local_score += __shfl_down_sync(0xffffffff, local_score, offset);
                    }
                    kq_score[row] += __shfl_sync(0xffffffff, local_score, 0);
                }
                __syncwarp();
            }

#pragma unroll
            for (int row = lane; row < GGML_CUDA_NVFP4_KX_KV_ROWS; row += WARP_SIZE) {
                smem_kq_score[row] = kq_score[row];
            }
        }
        __syncthreads();

#pragma unroll
        for (int row = 0; row < GGML_CUDA_NVFP4_KX_KV_ROWS; ++row) {
            const int64_t kv_row = kv_tile * GGML_CUDA_NVFP4_KX_KV_ROWS + row;
            if (kv_row >= params.ne_kv_rows) {
                continue;
            }

            const float mask = ggml_cuda_fattn_nvfp4_mask_value(params, 0, kv_row, seq);
            const float score = smem_kq_score[row] * params.scale + mask;
            const float kq_max_new = fmaxf(kq_max, score);
            const float scale_old = rowsum == 0.0f ? 0.0f : expf(kq_max - kq_max_new);
            const float scale_new = score - kq_max_new >= SOFTMAX_FTZ_THRESHOLD ? expf(score - kq_max_new) : 0.0f;

            if (col < params.v_head_dim) {
                const block_nvfp4 * v_ptr = params.V +
                    kv_head * params.v_stride_head +
                    seq     * params.v_stride_seq +
                    kv_row  * params.v_stride_row;
                const float v = ggml_cuda_fattn_nvfp4_dequant_row_value(v_ptr, col);
                pv = pv * scale_old + scale_new * v;
            }
            rowsum = rowsum * scale_old + scale_new;
            kq_max = kq_max_new;
        }
        __syncthreads();
    }

    if (col < params.v_head_dim) {
        dst_ptr[col] = rowsum == 0.0f ? 0.0f : pv / rowsum;
    }
#else
    GGML_UNUSED_VARS(params, kx_layout);
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}
#endif // defined(GGML_CUDA_NVFP4_KV_EXEC_P1_NATIVE_KQ)

__global__ void fattn_nvfp4_p1_vx_scalar_kernel(
        const fattn_nvfp4_mtp4_params params,
        const ggml_cuda_nvfp4_vx_layout vx_layout) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int col = threadIdx.x;
    const int64_t q_head = (int64_t) blockIdx.x;
    const int64_t seq = (int64_t) blockIdx.y;
    const int64_t kv_head = q_head / params.gqa_ratio;

    if (col >= params.v_head_dim || q_head >= params.ne_q_heads || seq >= params.ne_seqs || kv_head >= params.ne_kv_heads) {
        return;
    }

    const float * q_ptr = params.Q +
        q_head * params.q_stride_head +
        seq    * params.q_stride_seq;
    float * dst_ptr = params.dst +
        q_head * params.dst_stride_head +
        seq    * params.dst_stride_seq;

    float kq_max = -3.402823466e+38F;
    float rowsum = 0.0f;
    float pv = 0.0f;

#pragma unroll 1
    for (int64_t kv_row = 0; kv_row < params.ne_kv_rows; ++kv_row) {
        const block_nvfp4 * k_ptr = params.K +
            kv_head * params.k_stride_head +
            seq     * params.k_stride_seq +
            kv_row  * params.k_stride_row;

        const float mask = ggml_cuda_fattn_nvfp4_mask_value(params, 0, kv_row, seq);
        const float score = ggml_cuda_fattn_nvfp4_dot_q_k(q_ptr, k_ptr, params.k_head_dim) * params.scale + mask;
        const float kq_max_new = fmaxf(kq_max, score);
        const float scale_old = rowsum == 0.0f ? 0.0f : expf(kq_max - kq_max_new);
        const float scale_new = score - kq_max_new >= SOFTMAX_FTZ_THRESHOLD ? expf(score - kq_max_new) : 0.0f;

        const float v = ggml_cuda_nvfp4_vx_dequant_value(vx_layout, seq, kv_head, kv_row, col);
        pv = pv * scale_old + scale_new * v;
        rowsum = rowsum * scale_old + scale_new;
        kq_max = kq_max_new;
    }

    dst_ptr[col] = rowsum == 0.0f ? 0.0f : pv / rowsum;
#else
    GGML_UNUSED_VARS(params, vx_layout);
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}
#endif // defined(GGML_CUDA_NVFP4_KV_EXEC_LAYOUT) && defined(GGML_CUDA_NVFP4_KV_EXEC_P1_SCALAR)

__global__ void fattn_nvfp4_mtp4_compile_probe(
        const block_nvfp4 * q,
        const block_nvfp4 * k,
        const uint32_t *    v_lut,
        float *             dst) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    const block_nvfp4 q_blk = q[lane % FATTN_NVFP4_MAX_NFRAG];
    const block_nvfp4 k_blk = k[lane % FATTN_NVFP4_MAX_NFRAG];

    const uint32_t * q_qs = reinterpret_cast<const uint32_t *>(q_blk.qs);
    const uint32_t * k_qs = reinterpret_cast<const uint32_t *>(k_blk.qs);

    float kq[FATTN_NVFP4_MTP4_ROWS][4] = {};
    float pv[FATTN_NVFP4_MTP4_ROWS][4] = {};
    uint32_t p_frag[4] = {};
    p_frag[0] = ggml_cuda_fattn_nvfp4_half2_bits(__floats2half2_rn(1.0f, 0.875f));
    p_frag[1] = ggml_cuda_fattn_nvfp4_half2_bits(__floats2half2_rn(0.75f, 0.625f));
    p_frag[2] = ggml_cuda_fattn_nvfp4_half2_bits(__floats2half2_rn(0.5f, 0.375f));
    p_frag[3] = ggml_cuda_fattn_nvfp4_half2_bits(__floats2half2_rn(0.25f, 0.125f));

    const uint32_t q_scale = ggml_cuda_fattn_nvfp4_block_scale(q_blk);
    const uint32_t k_scale = ggml_cuda_fattn_nvfp4_block_scale(k_blk);
    const uint32_t v_scale = (uint32_t) k_blk.d[0] << 8;
    const uint32_t bx0     = v_lut[v_scale | (uint32_t) ((uint8_t *) k_blk.qs)[0]];
    const uint32_t bx1     = v_lut[v_scale | (uint32_t) ((uint8_t *) k_blk.qs)[1]];

#pragma unroll
    for (int row = 0; row < FATTN_NVFP4_MTP4_ROWS; ++row) {
        ggml_cuda_fattn_nvfp4_kq_mma(
            (int) q_qs[(lane + row + 0) & 7],
            (int) q_qs[(lane + row + 1) & 7],
            (int) q_qs[(lane + row + 2) & 7],
            (int) q_qs[(lane + row + 3) & 7],
            (int) k_qs[(lane + 0) & 7],
            (int) k_qs[(lane + 1) & 7],
            q_scale,
            k_scale,
            kq[row][0],
            kq[row][1],
            kq[row][2],
            kq[row][3]);

        ggml_cuda_fattn_nvfp4_pv_mma(
            p_frag[0],
            p_frag[1],
            p_frag[2],
            p_frag[3],
            bx0,
            bx1,
            pv[row][0],
            pv[row][1],
            pv[row][2],
            pv[row][3]);
    }

    if (blockIdx.x == 0 && lane == 0) {
        dst[0] = kq[0][0] + pv[0][0];
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        dst[0] = 0.0f;
    }
    GGML_UNUSED_VARS(q, k, v_lut);
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}

static void ggml_cuda_flash_attn_ext_nvfp4_mtp4_fill_lut(ggml_backend_cuda_context & ctx, uint32_t * lut) {
    cudaStream_t stream = ctx.stream();
    const dim3 block_dim(256, 1, 1);
    const dim3 grid_dim((FATTN_NVFP4_LUT_SIZE + block_dim.x - 1) / block_dim.x, 1, 1);

    fattn_nvfp4_fill_lut<<<grid_dim, block_dim, 0, stream>>>(lut);
    CUDA_CHECK(cudaGetLastError());
}

static bool ggml_cuda_flash_attn_ext_nvfp4_mtp4_stream_capturing(ggml_backend_cuda_context & ctx) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &status));
    return status != cudaStreamCaptureStatusNone;
}

static uint32_t * ggml_cuda_flash_attn_ext_nvfp4_mtp4_get_lut(ggml_backend_cuda_context & ctx) {
    uint32_t *& lut = ctx.nvfp4_fattn_lut[ctx.device];
    cudaEvent_t & ready = ctx.nvfp4_fattn_lut_ready[ctx.device];
    const bool capturing = ggml_cuda_flash_attn_ext_nvfp4_mtp4_stream_capturing(ctx);

    if (lut == nullptr) {
        if (capturing) {
            return nullptr;
        }

        ggml_cuda_set_device(ctx.device);
        CUDA_CHECK(cudaMalloc((void **) &lut, FATTN_NVFP4_LUT_SIZE * sizeof(uint32_t)));
        CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        ggml_cuda_flash_attn_ext_nvfp4_mtp4_fill_lut(ctx, lut);
        CUDA_CHECK(cudaEventRecord(ready, ctx.stream()));
    }

    if (!capturing) {
        CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), ready, 0));
    }

    return lut;
}

static bool ggml_cuda_flash_attn_ext_nvfp4_strides_supported(const ggml_tensor * t) {
    const size_t type_size = ggml_type_size(t->type);

    if (t->nb[0] != type_size) {
        return false;
    }

    for (int dim = 1; dim < GGML_MAX_DIMS; ++dim) {
        if (t->nb[dim] % (int64_t) type_size != 0) {
            return false;
        }
    }

    return true;
}

static bool ggml_cuda_flash_attn_ext_nvfp4_mask_supported(const ggml_tensor * mask, const ggml_tensor * Q, const ggml_tensor * K) {
    if (mask == nullptr || mask->type != GGML_TYPE_F16) {
        return false;
    }

    if (mask->ne[0] != K->ne[1] || mask->ne[1] != Q->ne[1] || mask->ne[2] != 1) {
        return false;
    }

    if (mask->ne[3] != 1 && mask->ne[3] != Q->ne[3]) {
        return false;
    }

    return ggml_cuda_flash_attn_ext_nvfp4_strides_supported(mask);
}

static bool ggml_cuda_flash_attn_ext_nvfp4_mtp4_shape_supported(int device, const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
#else
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    if (Q == nullptr || K == nullptr || V == nullptr) {
        return false;
    }

    const int cc = ggml_cuda_info().devices[device].cc;
    if (!blackwell_mma_available(cc)) {
        return false;
    }

    if (Q->type != GGML_TYPE_F32 || KQV->type != GGML_TYPE_F32) {
        return false;
    }

    if (K->type != GGML_TYPE_NVFP4 || V->type != GGML_TYPE_NVFP4) {
        return false;
    }

    if (!ggml_cuda_flash_attn_ext_nvfp4_strides_supported(Q) ||
        !ggml_cuda_flash_attn_ext_nvfp4_strides_supported(K) ||
        !ggml_cuda_flash_attn_ext_nvfp4_strides_supported(V) ||
        !ggml_cuda_flash_attn_ext_nvfp4_strides_supported(KQV)) {
        return false;
    }

    if (Q->ne[0] != K->ne[0] || V->ne[0] != KQV->ne[0]) {
        return false;
    }

    if ((Q->ne[0] != 256 && Q->ne[0] != 512) || (V->ne[0] != 256 && V->ne[0] != 512)) {
        return false;
    }

    if (K->ne[1] != V->ne[1] || K->ne[2] != V->ne[2] || K->ne[3] != V->ne[3]) {
        return false;
    }

    if (Q->ne[2] % K->ne[2] != 0 || Q->ne[3] != K->ne[3]) {
        return false;
    }

#if defined(GGML_CUDA_NVFP4_FA)
    if (Q->ne[2] == K->ne[2]) {
        return false;
    }
#endif // defined(GGML_CUDA_NVFP4_FA)

    if (!ggml_cuda_flash_attn_ext_nvfp4_mask_supported(mask, Q, K)) {
        return false;
    }

    if (sinks != nullptr) {
        return false;
    }

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,       (const float *) KQV->op_params + 1, sizeof(float));
    memcpy(&logit_softcap,  (const float *) KQV->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }

    // The kernels execute four-row MTP tiles, but internally mask rows beyond
    // the actual query count. This lets normal one-token decode use the native
    // path without claiming a separate single-row optimized schedule.
    if (Q->ne[1] < 1) {
        return false;
    }

    if (K->ne[1] % FATTN_KQ_STRIDE != 0) {
        return false;
    }

    return true;
#endif // FLASH_ATTN_AVAILABLE
}

bool ggml_cuda_flash_attn_ext_nvfp4_mtp4_supported(int device, const ggml_tensor * dst) {
    const bool shape_supported = ggml_cuda_flash_attn_ext_nvfp4_mtp4_shape_supported(device, dst);

#if defined(GGML_CUDA_NVFP4_FA)
    return shape_supported;
#else
    GGML_UNUSED(shape_supported);
    return false;
#endif // GGML_CUDA_NVFP4_FA
}

#if defined(GGML_CUDA_NVFP4_KV_EXEC_LAYOUT) && defined(GGML_CUDA_NVFP4_KV_EXEC_P1_SCALAR)
static bool ggml_cuda_flash_attn_ext_nvfp4_p1_vx_scalar_shape_supported(int device, const ggml_tensor * dst) {
    if (!ggml_cuda_flash_attn_ext_nvfp4_mtp4_shape_supported(device, dst)) {
        return false;
    }

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * V = dst->src[2];

    if (Q->ne[1] != 1) {
        return false;
    }

    if (V->ne[0] > FATTN_NVFP4_MAX_HEAD_DIM) {
        return false;
    }

    return true;
}

#if defined(GGML_CUDA_NVFP4_KV_EXEC_P1_NATIVE_KQ)
bool ggml_cuda_flash_attn_ext_nvfp4_p1_kx_mma(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);

    if (!ggml_cuda_flash_attn_ext_nvfp4_p1_vx_scalar_shape_supported(ctx.device, dst)) {
        return false;
    }

    const ggml_tensor * K = dst->src[1];

    ggml_cuda_nvfp4_kx_layout kx_layout = {};
    if (!ggml_cuda_nvfp4_kx_find(ctx, K, &kx_layout)) {
        return false;
    }

    const fattn_nvfp4_mtp4_params params = ggml_cuda_fattn_nvfp4_mtp4_make_params(dst, nullptr);
    const int64_t col_block_count = (params.v_head_dim + WARP_SIZE - 1) / WARP_SIZE;
    const dim3 block(WARP_SIZE, (uint32_t) col_block_count, 1);
    const dim3 grid((uint32_t) params.ne_q_heads, (uint32_t) params.ne_seqs, 1);
    fattn_nvfp4_p1_kx_mma_kernel<<<grid, block, 0, ctx.stream()>>>(params, kx_layout);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
#endif // defined(GGML_CUDA_NVFP4_KV_EXEC_P1_NATIVE_KQ)

bool ggml_cuda_flash_attn_ext_nvfp4_p1_vx_scalar(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);

    if (!ggml_cuda_flash_attn_ext_nvfp4_p1_vx_scalar_shape_supported(ctx.device, dst)) {
        return false;
    }

    // This p1 execution-layout consumer is still experimental and does not meet
    // the stock FLASH_ATTN_EXT NVFP4 error tolerance for sliding-window masks.
    return false;

    const ggml_tensor * V = dst->src[2];
    ggml_cuda_nvfp4_vx_layout vx_layout = {};
    if (!ggml_cuda_nvfp4_vx_find(ctx, V, &vx_layout)) {
        return false;
    }

    fattn_nvfp4_mtp4_params params = ggml_cuda_fattn_nvfp4_mtp4_make_params(dst, nullptr);
    const dim3 block((uint32_t) params.v_head_dim, 1, 1);
    const dim3 grid((uint32_t) params.ne_q_heads, (uint32_t) params.ne_seqs, 1);
    fattn_nvfp4_p1_vx_scalar_kernel<<<grid, block, 0, ctx.stream()>>>(params, vx_layout);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
#endif // defined(GGML_CUDA_NVFP4_KV_EXEC_LAYOUT) && defined(GGML_CUDA_NVFP4_KV_EXEC_P1_SCALAR)

size_t ggml_cuda_flash_attn_ext_nvfp4_mtp4_get_alloc_size(const ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);
    return ggml_nbytes(dst);
}

void ggml_cuda_flash_attn_ext_nvfp4_mtp4(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    uint32_t * lut = ggml_cuda_flash_attn_ext_nvfp4_mtp4_get_lut(ctx);
    ggml_cuda_pool_alloc<uint32_t> lut_alloc(ctx.pool());

    if (lut == nullptr) {
        lut = lut_alloc.alloc(FATTN_NVFP4_LUT_SIZE);
        ggml_cuda_flash_attn_ext_nvfp4_mtp4_fill_lut(ctx, lut);
    }

    fattn_nvfp4_mtp4_params params = ggml_cuda_fattn_nvfp4_mtp4_make_params(dst, lut);
#if defined(GGML_CUDA_NVFP4_KV_EXEC_LAYOUT)
    (void) ggml_cuda_nvfp4_kx_prepare(ctx, dst->src[1]);
    (void) ggml_cuda_nvfp4_vx_prepare(ctx, dst->src[2]);
#if defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_REQUANT) && defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_VX)
    {
        ggml_cuda_nvfp4_vx_layout native_pv_vx_layout = {};
        if (ggml_cuda_nvfp4_vx_find(ctx, dst->src[2], &native_pv_vx_layout)) {
            params.native_pv_vx_layout = native_pv_vx_layout;
            params.native_pv_use_vx = true;
        }
    }
#endif // defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_REQUANT) && defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_VX)
#if defined(GGML_CUDA_NVFP4_KV_EXEC_VALIDATE)
    if (!ggml_cuda_flash_attn_ext_nvfp4_mtp4_stream_capturing(ctx)) {
        float vx_max_abs_error = 0.0f;
        float vx_mean_abs_error = 0.0f;
        const bool vx_valid = ggml_cuda_nvfp4_vx_validate(ctx, dst->src[2], &vx_max_abs_error, &vx_mean_abs_error);
        GGML_ASSERT(vx_valid);
        GGML_ASSERT(std::isfinite(vx_max_abs_error));
        GGML_ASSERT(std::isfinite(vx_mean_abs_error));

        float kx_max_abs_error = 0.0f;
        float kx_mean_abs_error = 0.0f;
        const bool kx_valid = ggml_cuda_nvfp4_kx_validate(ctx, dst->src[1], &kx_max_abs_error, &kx_mean_abs_error);
        GGML_ASSERT(kx_valid);
        GGML_ASSERT(std::isfinite(kx_max_abs_error));
        GGML_ASSERT(std::isfinite(kx_mean_abs_error));
        GGML_ASSERT(kx_max_abs_error <= 1.0e-6f);
        GGML_ASSERT(kx_mean_abs_error <= 1.0e-6f);
    }
#endif // defined(GGML_CUDA_NVFP4_KV_EXEC_VALIDATE)
#endif // defined(GGML_CUDA_NVFP4_KV_EXEC_LAYOUT)

#if defined(GGML_CUDA_NVFP4_FA_MMA) && defined(GGML_CUDA_NVFP4_FA_MMA_MULTIWARP) && defined(GGML_CUDA_NVFP4_FA_MMA_SPLIT_KV)
    ggml_cuda_pool_alloc<float>  split_partial_alloc(ctx.pool());
#ifdef GGML_CUDA_NVFP4_FA_MMA_DECOUPLED
    ggml_cuda_pool_alloc<float>  split_prob_alloc(ctx.pool());
#if defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_REQUANT)
    ggml_cuda_pool_alloc<int>      native_pv_v_qs_alloc(ctx.pool());
    ggml_cuda_pool_alloc<uint32_t> native_pv_v_scale_alloc(ctx.pool());
#endif // defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_REQUANT)
#endif // GGML_CUDA_NVFP4_FA_MMA_DECOUPLED
    ggml_cuda_pool_alloc<float2> split_meta_alloc(ctx.pool());

    params.kv_split_size = FATTN_NVFP4_SPLIT_KV_ROWS;
    params.kv_split_count = (params.ne_kv_rows + params.kv_split_size - 1) / params.kv_split_size;
    params.kv_split_base = 0;
    params.kv_split_active_count = params.kv_split_count;
#if defined(GGML_CUDA_NVFP4_FA_ACTIVE_DECOUPLED_WINDOW_SPLITS)
    params.kv_split_active_count =
        min((int64_t) GGML_CUDA_NVFP4_FA_ACTIVE_DECOUPLED_WINDOW_SPLITS, params.kv_split_count);
#endif // defined(GGML_CUDA_NVFP4_FA_ACTIVE_DECOUPLED_WINDOW_SPLITS)
    params.split_partial = split_partial_alloc.alloc(
        (size_t) params.kv_split_count * params.ne_seqs * params.ne_q_heads * params.ne_q_rows * params.v_head_dim);
#ifdef GGML_CUDA_NVFP4_FA_MMA_DECOUPLED
    params.split_prob = split_prob_alloc.alloc(
        (size_t) params.kv_split_active_count * params.ne_seqs * params.ne_q_heads * params.ne_q_rows * params.kv_split_size);
#if defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_REQUANT)
    params.native_pv_nkv_tile = (params.kv_split_size + QK_NVFP4 - 1) / QK_NVFP4;
    const int64_t native_pv_ncol_tile = params.v_head_dim / FATTN_NVFP4_PV_COL_TILE;
    const size_t native_pv_tile_count =
        (size_t) params.kv_split_active_count * params.ne_seqs * params.ne_kv_heads *
        native_pv_ncol_tile * params.native_pv_nkv_tile;
#if defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_VX) && defined(GGML_CUDA_NVFP4_KV_EXEC_LAYOUT)
    if (!params.native_pv_use_vx)
#endif // defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_VX) && defined(GGML_CUDA_NVFP4_KV_EXEC_LAYOUT)
    {
        params.native_pv_v_qs = native_pv_v_qs_alloc.alloc(native_pv_tile_count * 64);
        params.native_pv_v_scale = native_pv_v_scale_alloc.alloc(native_pv_tile_count * FATTN_NVFP4_PV_COL_TILE);
    }
#endif // defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_REQUANT)
#endif // GGML_CUDA_NVFP4_FA_MMA_DECOUPLED
    params.split_meta = split_meta_alloc.alloc(
        (size_t) params.kv_split_count * params.ne_seqs * params.ne_q_heads * params.ne_q_rows);
#endif // defined(GGML_CUDA_NVFP4_FA_MMA) && defined(GGML_CUDA_NVFP4_FA_MMA_MULTIWARP) && defined(GGML_CUDA_NVFP4_FA_MMA_SPLIT_KV)

    const dim3 blocks_num = ggml_cuda_fattn_nvfp4_mtp4_blocks(params);

#ifdef GGML_CUDA_NVFP4_FA_MMA
#ifdef GGML_CUDA_NVFP4_FA_MMA_MULTIWARP
#if defined(GGML_CUDA_NVFP4_FA_MMA_SPLIT_KV) && defined(GGML_CUDA_NVFP4_FA_MMA_DECOUPLED)
    const dim3 kq_prob_block(WARP_SIZE, 1, 1);
    const dim3 pv_block(FATTN_NVFP4_TC_PV_WARPS * WARP_SIZE, 1, 1);
#if defined(GGML_CUDA_NVFP4_FA_ACTIVE_DECOUPLED_WINDOW_SPLITS)
    for (int64_t kv_split_base = 0; kv_split_base < params.kv_split_count; kv_split_base += params.kv_split_active_count) {
        params.kv_split_base = kv_split_base;
        params.kv_split_active_count =
            min((int64_t) GGML_CUDA_NVFP4_FA_ACTIVE_DECOUPLED_WINDOW_SPLITS, params.kv_split_count - kv_split_base);
        const dim3 window_blocks_num(blocks_num.x, blocks_num.y, (uint32_t) (params.ne_seqs * params.kv_split_active_count));
        fattn_nvfp4_mtp4_split_kq_prob_kernel<<<window_blocks_num, kq_prob_block, 0, ctx.stream()>>>(params);
#if defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_REQUANT)
#if defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_VX) && defined(GGML_CUDA_NVFP4_KV_EXEC_LAYOUT)
        if (!params.native_pv_use_vx)
#endif // defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_VX) && defined(GGML_CUDA_NVFP4_KV_EXEC_LAYOUT)
        {
            const int64_t native_pv_ncol_tile = params.v_head_dim / FATTN_NVFP4_PV_COL_TILE;
            const int64_t native_pv_tile_count =
                params.kv_split_active_count * params.ne_seqs * params.ne_kv_heads *
                native_pv_ncol_tile * params.native_pv_nkv_tile;
            const dim3 prepack_block(WARP_SIZE, 1, 1);
            const dim3 prepack_grid((uint32_t) native_pv_tile_count, 1, 1);
            fattn_nvfp4_mtp4_prepack_v_kernel<<<prepack_grid, prepack_block, 0, ctx.stream()>>>(params);
        }
#endif // defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_REQUANT)
        fattn_nvfp4_mtp4_split_pv_from_prob_kernel<<<window_blocks_num, pv_block, 0, ctx.stream()>>>(params);
    }
#else
    fattn_nvfp4_mtp4_split_kq_prob_kernel<<<blocks_num, kq_prob_block, 0, ctx.stream()>>>(params);
#if defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_REQUANT)
#if defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_VX) && defined(GGML_CUDA_NVFP4_KV_EXEC_LAYOUT)
    if (!params.native_pv_use_vx)
#endif // defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_VX) && defined(GGML_CUDA_NVFP4_KV_EXEC_LAYOUT)
    {
        const int64_t native_pv_ncol_tile = params.v_head_dim / FATTN_NVFP4_PV_COL_TILE;
        const int64_t native_pv_tile_count =
            params.kv_split_active_count * params.ne_seqs * params.ne_kv_heads *
            native_pv_ncol_tile * params.native_pv_nkv_tile;
        const dim3 prepack_block(WARP_SIZE, 1, 1);
        const dim3 prepack_grid((uint32_t) native_pv_tile_count, 1, 1);
        fattn_nvfp4_mtp4_prepack_v_kernel<<<prepack_grid, prepack_block, 0, ctx.stream()>>>(params);
    }
#endif // defined(GGML_CUDA_NVFP4_FA_NATIVE_PV_REQUANT)
    fattn_nvfp4_mtp4_split_pv_from_prob_kernel<<<blocks_num, pv_block, 0, ctx.stream()>>>(params);
#endif // defined(GGML_CUDA_NVFP4_FA_ACTIVE_DECOUPLED_WINDOW_SPLITS)
#else
    const dim3 block_dim(FATTN_NVFP4_TC_THREADS, 1, 1);
#if defined(GGML_CUDA_NVFP4_FA_MMA_SPLIT_KV) && defined(GGML_CUDA_NVFP4_FA_MMA_TWOPASS)
    fattn_nvfp4_mtp4_multiwarp_twopass_kernel<<<blocks_num, block_dim, 0, ctx.stream()>>>(params);
#else
    fattn_nvfp4_mtp4_multiwarp_kernel<<<blocks_num, block_dim, 0, ctx.stream()>>>(params);
#endif // defined(GGML_CUDA_NVFP4_FA_MMA_SPLIT_KV) && defined(GGML_CUDA_NVFP4_FA_MMA_TWOPASS)
#endif // defined(GGML_CUDA_NVFP4_FA_MMA_SPLIT_KV) && defined(GGML_CUDA_NVFP4_FA_MMA_DECOUPLED)
#ifdef GGML_CUDA_NVFP4_FA_MMA_SPLIT_KV
    const int64_t combine_ne = params.ne_seqs * params.ne_q_heads * params.ne_q_rows * params.v_head_dim;
    const dim3 combine_block(256, 1, 1);
    const dim3 combine_grid((uint32_t) ((combine_ne + combine_block.x - 1) / combine_block.x), 1, 1);
    fattn_nvfp4_mtp4_split_kv_combine_kernel<<<combine_grid, combine_block, 0, ctx.stream()>>>(params);
#endif // GGML_CUDA_NVFP4_FA_MMA_SPLIT_KV
#else
    const dim3 block_dim(WARP_SIZE, 1, 1);
#ifdef GGML_CUDA_NVFP4_FA_MMA_SCALAR_PV
    fattn_nvfp4_mtp4_kernel<<<blocks_num, block_dim, 0, ctx.stream()>>>(params);
#else
    const int ncol_group = ggml_cuda_fattn_nvfp4_ncol_group(params.v_head_dim);
    const dim3 blocks_num_tc(
        blocks_num.x * ncol_group,
        blocks_num.y,
        blocks_num.z);
    fattn_nvfp4_mtp4_kernel<<<blocks_num_tc, block_dim, 0, ctx.stream()>>>(params);
#endif // GGML_CUDA_NVFP4_FA_MMA_SCALAR_PV
#endif // GGML_CUDA_NVFP4_FA_MMA_MULTIWARP
#else
    const dim3 block_dim((uint32_t) params.v_head_dim, 1, 1);
    fattn_nvfp4_mtp4_scalar_correctness_kernel<<<blocks_num, block_dim, 0, ctx.stream()>>>(params);
#endif // GGML_CUDA_NVFP4_FA_MMA
    CUDA_CHECK(cudaGetLastError());
}
