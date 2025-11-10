#include "delta-net.cuh"

// Recurrent delta network kernel
// Processes sequences with recurrent state updates (similar to cumsum but with state)
// Optimized with parallel scan for better performance
template<int BLOCK_SIZE>
__global__ void delta_net_parallel_kernel(
    const float * __restrict__ x,
    float * __restrict__ dst,
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
    const int lane_id = tid & 31;
    const int warp_id = tid / 32;
    const int num_warps = BLOCK_SIZE / 32;

    __shared__ float warp_sums[32]; // max 32 warps per block

    float carry_accum = 0.0f;

    // Use parallel scan similar to cumsum
    for (int64_t i = tid; i < ne0; i += BLOCK_SIZE) {
        float val = __ldg(&src_row[i]);
        
        // Warp-level inclusive scan
        #pragma unroll
        for (int offset = 1; offset < 32; offset *= 2) {
            float n = __shfl_up_sync(0xffffffff, val, offset, 32);
            if (lane_id >= offset) val += n;
        }

        // Store warp-level sum
        if (lane_id == 31) {
            warp_sums[warp_id] = val;
        }
        __syncthreads();

        // First warp computes prefix sum of warp sums
        if (warp_id == 0) {
            float warp_sum = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
            #pragma unroll
            for (int offset = 1; offset < 32; offset *= 2) {
                float n = __shfl_up_sync(0xffffffff, warp_sum, offset, 32);
                if (lane_id >= offset) warp_sum += n;
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

// Sequential kernel for very large sequences
__global__ void delta_net_sequential_kernel(
    const float * __restrict__ x,
    float * __restrict__ dst,
    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
    int64_t nb0, int64_t nb1, int64_t nb2, int64_t nb3,
    int64_t dst_nb0, int64_t dst_nb1, int64_t dst_nb2, int64_t dst_nb3) {

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    if (i3 >= ne3 || i2 >= ne2 || i1 >= ne1) return;
    if (threadIdx.x != 0) return;

    const float * src_row = (const float *)((const char *)x + i1*nb1 + i2*nb2 + i3*nb3);
    float * dst_row = (float *)((char *)dst + i1*dst_nb1 + i2*dst_nb2 + i3*dst_nb3);

    float s = 0.0f;
    for (int64_t i = 0; i < ne0; i++) {
        s = s + __ldg(&src_row[i]);
        dst_row[i] = s;
    }
}


void ggml_cuda_op_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
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
        // Use parallel scan for small to medium sequences
        const int block_size = 512;
        delta_net_parallel_kernel<block_size><<<grid, block_size, 0, stream>>>(
            src0_d, dst_d, ne0, ne1, ne2, ne3,
            nb0, nb1, nb2, nb3,
            dst_nb0, dst_nb1, dst_nb2, dst_nb3
        );
    } else {
        // Use sequential kernel for very long sequences
        delta_net_sequential_kernel<<<grid, 1, 0, stream>>>(
            src0_d, dst_d, ne0, ne1, ne2, ne3,
            nb0, nb1, nb2, nb3,
            dst_nb0, dst_nb1, dst_nb2, dst_nb3
        );
    }
}

