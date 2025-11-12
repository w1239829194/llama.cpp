#include "cumsum.cuh"

// Get actual warp size at runtime
// RDNA (including RDNA 3.5) uses 32, GCN/CDNA uses 64
static __device__ __forceinline__ int get_warp_size() {
#if defined(GGML_USE_HIP)
    return __builtin_amdgcn_wavefrontsize();
#else
    return 32;
#endif
}

// Warp-level inclusive scan (cumulative sum)
#if defined(GGML_USE_HIP)
using ggml_cuda_warp_mask_t = unsigned long long;
#else
using ggml_cuda_warp_mask_t = unsigned int;
#endif

template<int WARP_SZ>
__device__ inline float warp_cumsum(float val, int lane_id) {
    const ggml_cuda_warp_mask_t full_mask = ~ggml_cuda_warp_mask_t(0);
#pragma unroll
    for (int offset = 1; offset < WARP_SZ; offset *= 2) {
        float n = __shfl_up_sync(full_mask, val, offset, WARP_SZ);
        if (lane_id >= offset) val += n;
    }
    return val;
}

// Kernel for small rows (row_len <= 1024)
// Each block processes one row
template<int BLOCK_SIZE>
__global__ void cumsum_f32_kernel(const float * __restrict__ x, float * __restrict__ dst,
 int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
 int64_t nb0, int64_t nb1, int64_t nb2, int64_t nb3,
 int64_t dst_nb0, int64_t dst_nb1, int64_t dst_nb2, int64_t dst_nb3) {

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    if (i3 >= ne3 || i2 >= ne2 || i1 >= ne1) return;

    const float * src_row = (const float *)((const char *)x + i1*nb1 + i2*nb2 + i3*nb3);
    float * dst_row = (float *)((char *)dst + i1*dst_nb1 + i2*dst_nb2 + i3*dst_nb3);

    const int tid = threadIdx.x;
    const int warp_size = get_warp_size();
    const int lane_id = tid & (warp_size - 1);
    const int warp_id = tid / warp_size;
    const int num_warps = BLOCK_SIZE / warp_size;

    // Shared memory for warp-level sums
    // Use 32 to ensure proper alignment and avoid bank conflicts
    __shared__ float warp_sums[32];

    // Use register for carry instead of shared memory - faster!
    float carry_accum = 0.0f;

    // Process elements in chunks of BLOCK_SIZE
    // Optimized: reduce sync overhead by batching operations
    for (int64_t i = tid; i < ne0; i += BLOCK_SIZE) {
        // Use GGML_CUDA_LDG for read-only data (texture cache on CUDA, regular load on HIP)
        float val = GGML_CUDA_LDG(&src_row[i]);
        
        // Use runtime warp size for better portability
        if (warp_size == 64) {
            val = warp_cumsum<64>(val, lane_id);
        } else {
            val = warp_cumsum<32>(val, lane_id);
        }

        // Store warp-level sum for later
        if (lane_id == warp_size - 1) {
            warp_sums[warp_id] = val;
        }
        __syncthreads();

        // First warp computes prefix sum of warp sums
        if (warp_id == 0) {
            float warp_sum = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
            if (warp_size == 64) {
                warp_sum = warp_cumsum<64>(warp_sum, lane_id);
            } else {
                warp_sum = warp_cumsum<32>(warp_sum, lane_id);
            }
            if (lane_id < num_warps) {
                warp_sums[lane_id] = warp_sum;
            }
        }
        __syncthreads();

        // Add prefix sum of previous warps to current value
        if (warp_id > 0) {
            val += warp_sums[warp_id - 1];
        }
        val += carry_accum;

        dst_row[i] = val;

        // Update carry for next chunk
        if (tid == BLOCK_SIZE - 1) {
            carry_accum = val;
        }
        __syncthreads();
    }
}

// Sequential kernel for very large rows (row_len > 4096)
// Only first thread in block processes the entire row
__global__ void cumsum_f32_sequential_kernel(const float * __restrict__ x, float * __restrict__ dst,
 int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
 int64_t nb0, int64_t nb1, int64_t nb2, int64_t nb3,
 int64_t dst_nb0, int64_t dst_nb1, int64_t dst_nb2, int64_t dst_nb3) {

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    if (i3 >= ne3 || i2 >= ne2 || i1 >= ne1) return;
    if (threadIdx.x != 0) return; // Only first thread in block

    const float * src_row = (const float *)((const char *)x + i1*nb1 + i2*nb2 + i3*nb3);
    float * dst_row = (float *)((char *)dst + i1*dst_nb1 + i2*dst_nb2 + i3*dst_nb3);

    float cumsum = 0.0f;
    for (int64_t i = 0; i < ne0; i++) {
        cumsum += src_row[i];
        dst_row[i] = cumsum;
    }
}

void ggml_cuda_op_cumsum(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;

    cudaStream_t stream = ctx.stream();

    const int64_t ne0 = src0->ne[0];
    const int64_t ne1 = src0->ne[1];
    const int64_t ne2 = src0->ne[2];
    const int64_t ne3 = src0->ne[3];

    const int64_t nb0 = src0->nb[0];
    const int64_t nb1 = src0->nb[1];
    const int64_t nb2 = src0->nb[2];
    const int64_t nb3 = src0->nb[3];

    const int64_t dst_nb0 = dst->nb[0];
    const int64_t dst_nb1 = dst->nb[1];
    const int64_t dst_nb2 = dst->nb[2];
    const int64_t dst_nb3 = dst->nb[3];

    // Launch kernel
    dim3 grid(ne1, ne2, ne3);

    if (ne0 <= 4096) {
        // Use parallel scan for small to medium rows
        const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
        const int cc = ggml_cuda_info().devices[ctx.device].cc;

        int block_size = 512;
        if (ne0 <= 256) {
            block_size = 128;
        } else if (ne0 <= 512) {
            block_size = 256;
        } else if (warp_size == 64 && ne0 >= 2048) {
            // GCN/CDNA: use 256 for large sequences
            block_size = 256;
        }

        switch (block_size) {
            case 128:
                cumsum_f32_kernel<128><<<grid, 128, 0, stream>>>(
                    src0_d, dst_d, ne0, ne1, ne2, ne3,
                    nb0, nb1, nb2, nb3,
                    dst_nb0, dst_nb1, dst_nb2, dst_nb3
                );
                break;
            case 256:
                cumsum_f32_kernel<256><<<grid, 256, 0, stream>>>(
                    src0_d, dst_d, ne0, ne1, ne2, ne3,
                    nb0, nb1, nb2, nb3,
                    dst_nb0, dst_nb1, dst_nb2, dst_nb3
                );
                break;
            default:
                cumsum_f32_kernel<512><<<grid, 512, 0, stream>>>(
                    src0_d, dst_d, ne0, ne1, ne2, ne3,
                    nb0, nb1, nb2, nb3,
                    dst_nb0, dst_nb1, dst_nb2, dst_nb3
                );
                break;
        }
    } else {
        // Use sequential kernel for very large rows
        cumsum_f32_sequential_kernel<<<grid, 1, 0, stream>>>(
            src0_d, dst_d, ne0, ne1, ne2, ne3,
            nb0, nb1, nb2, nb3,
            dst_nb0, dst_nb1, dst_nb2, dst_nb3
        );
    }
}

