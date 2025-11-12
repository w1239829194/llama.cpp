#include "solve-tri.cuh"

namespace {

__global__ void solve_tri_f32_kernel(
    const float * __restrict__ a,
    const float * __restrict__ b,
    float * __restrict__ dst,
    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
    int64_t nb0_a, int64_t nb1_a, int64_t nb2_a, int64_t nb3_a,
    int64_t nb0_b, int64_t nb1_b, int64_t nb2_b, int64_t nb3_b,
    int64_t nb0_dst, int64_t nb1_dst, int64_t nb2_dst, int64_t nb3_dst) {

    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ne1) {
        return;
    }

    const int64_t i2 = blockIdx.y;
    const int64_t i3 = blockIdx.z;
    if (i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const char * a_batch = reinterpret_cast<const char *>(a) + i2 * nb2_a + i3 * nb3_a;
    const char * b_batch = reinterpret_cast<const char *>(b) + i2 * nb2_b + i3 * nb3_b;
    char * dst_batch = reinterpret_cast<char *>(dst) + i2 * nb2_dst + i3 * nb3_dst;

    constexpr float eps = 1e-12f;

    for (int64_t row = 0; row < ne0; ++row) {
        const char * a_row = a_batch + row * nb1_a;
        const char * b_row = b_batch + row * nb1_b;
        char * dst_row = dst_batch + row * nb1_dst;

        // Use GGML_CUDA_LDG for read-only data
        const float rhs = GGML_CUDA_LDG(reinterpret_cast<const float *>(b_row + col * nb0_b));

        float accum = 0.0f;
        // Optimize: reduce memory accesses by caching frequently accessed values
        for (int64_t t = 0; t < row; ++t) {
            const float a_val = GGML_CUDA_LDG(reinterpret_cast<const float *>(a_row + t * nb0_a));
            const float x_val = GGML_CUDA_LDG(reinterpret_cast<const float *>(dst_batch + t * nb1_dst + col * nb0_dst));
            accum += a_val * x_val;
        }

        float diag = GGML_CUDA_LDG(reinterpret_cast<const float *>(a_row + row * nb0_a));
        if (fabsf(diag) < eps) {
            diag = (diag >= 0.0f ? eps : -eps);
        }

        const float value = (rhs - accum) / diag;
        *reinterpret_cast<float *>(dst_row + col * nb0_dst) = value;
    }
}

} // namespace

void ggml_cuda_op_solve_tri(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // lower-triangular matrix A
    const ggml_tensor * src1 = dst->src[1]; // right-hand side B

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    const float * a_d = static_cast<const float *>(src0->data);
    const float * b_d = static_cast<const float *>(src1->data);
    float * dst_d = static_cast<float *>(dst->data);

    cudaStream_t stream = ctx.stream();

    const int64_t ne0 = src0->ne[0];
    const int64_t ne1 = src1->ne[1];
    const int64_t ne2 = dst->ne[2];
    const int64_t ne3 = dst->ne[3];

    const int64_t nb0_a = src0->nb[0];
    const int64_t nb1_a = src0->nb[1];
    const int64_t nb2_a = src0->nb[2];
    const int64_t nb3_a = src0->nb[3];

    const int64_t nb0_b = src1->nb[0];
    const int64_t nb1_b = src1->nb[1];
    const int64_t nb2_b = src1->nb[2];
    const int64_t nb3_b = src1->nb[3];

    const int64_t nb0_dst = dst->nb[0];
    const int64_t nb1_dst = dst->nb[1];
    const int64_t nb2_dst = dst->nb[2];
    const int64_t nb3_dst = dst->nb[3];

    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    
    int block_cols = 256;
    if (ne1 <= warp_size) {
        block_cols = warp_size;
    } else if (ne1 <= 128) {
        block_cols = 128;
    } else if (ne1 <= 256) {
        block_cols = 256;
    } else {
        // For RDNA 3.5/4, use 256 for better occupancy
        if (GGML_CUDA_CC_IS_RDNA35(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
            block_cols = 256;
        } else {
            block_cols = 256;
        }
    }

    block_cols = std::max(block_cols, warp_size);
    // RDNA 3.5 can handle up to 512 threads efficiently
    const int max_block_size = (GGML_CUDA_CC_IS_RDNA35(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) ? 512 : 256;
    block_cols = std::min(block_cols, max_block_size);

    const int grid_x = static_cast<int>((ne1 + block_cols - 1) / block_cols);
    dim3 grid(grid_x, ne2, ne3);
    dim3 block(block_cols, 1, 1);

    solve_tri_f32_kernel<<<grid, block, 0, stream>>>(
        a_d, b_d, dst_d,
        ne0, ne1, ne2, ne3,
        nb0_a, nb1_a, nb2_a, nb3_a,
        nb0_b, nb1_b, nb2_b, nb3_b,
        nb0_dst, nb1_dst, nb2_dst, nb3_dst
    );
}


