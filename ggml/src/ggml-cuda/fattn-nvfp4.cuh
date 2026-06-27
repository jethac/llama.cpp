#pragma once

#include "common.cuh"

bool ggml_cuda_flash_attn_ext_nvfp4_mtp4_supported(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_nvfp4_mtp4_get_alloc_size(const ggml_tensor * dst);

void ggml_cuda_flash_attn_ext_nvfp4_mtp4(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#if defined(GGML_CUDA_NVFP4_KV_EXEC_LAYOUT) && defined(GGML_CUDA_NVFP4_KV_EXEC_P1_SCALAR)
bool ggml_cuda_flash_attn_ext_nvfp4_p1_vx_scalar(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
#endif
