#include "delta-net.cuh"

// Recurrent delta network kernel
// Processes sequences with recurrent state updates (similar to cumsum but with state)
template<bool chunked>
__global__ void delta_net_recurrent_kernel(
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

    float s = 0.0f; // Initial state

    if (chunked) {
        // Chunked processing: process in chunks for better memory access
        const int chunk_size = 32;
        for (int64_t t = 0; t < ne0; t += chunk_size) {
            int64_t chunk_end = min(t + chunk_size, ne0);
            for (int64_t i = t; i < chunk_end; i++) {
                s = s + src_row[i]; // Delta update: state = state + delta
                dst_row[i] = s;
            }
        }
    } else {
        // Sequential processing
        for (int64_t i = 0; i < ne0; i++) {
            s = s + src_row[i]; // Delta update: state = state + delta
            dst_row[i] = s;
        }
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

    if (ne0 <= 1024) {
        // Use recurrent kernel for short sequences
        const int block_size = 1; // One thread per row
        delta_net_recurrent_kernel<false><<<grid, block_size, 0, stream>>>(
            src0_d, dst_d, ne0, ne1, ne2, ne3,
            nb0, nb1, nb2, nb3,
            dst_nb0, dst_nb1, dst_nb2, dst_nb3
        );
    } else {
        // Use sequential kernel for very long sequences (similar to cumsum)
        const int block_size = 1;
        delta_net_recurrent_kernel<false><<<grid, block_size, 0, stream>>>(
            src0_d, dst_d, ne0, ne1, ne2, ne3,
            nb0, nb1, nb2, nb3,
            dst_nb0, dst_nb1, dst_nb2, dst_nb3
        );
    }
}

