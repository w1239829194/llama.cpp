#include "tri.cuh"

// Kernel for triangular matrix operations
// Each thread processes one row
__global__ void tri_f32_kernel(const float * __restrict__ x, float * __restrict__ dst,
 int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
 int64_t nb0, int64_t nb1, int64_t nb2, int64_t nb3,
 int64_t dst_nb0, int64_t dst_nb1, int64_t dst_nb2, int64_t dst_nb3,
 int tri_type, float constant, bool keep_org_val) {

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    if (i3 >= ne3 || i2 >= ne2 || i1 >= ne1) return;

    const float * src_row = (const float *)((const char *)x + i1*nb1 + i2*nb2 + i3*nb3);
    float * dst_row = (float *)((char *)dst + i1*dst_nb1 + i2*dst_nb2 + i3*dst_nb3);

    const int r = i1; // current row index

    for (int64_t i = 0; i < ne0; i++) {
        bool cmp = false;
        switch (tri_type) {
            case 0: // GGML_TRI_TYPE_UPPER_DIAG
                cmp = i >= r;
                break;
            case 1: // GGML_TRI_TYPE_UPPER
                cmp = i > r;
                break;
            case 2: // GGML_TRI_TYPE_LOWER_DIAG
                cmp = i <= r;
                break;
            case 3: // GGML_TRI_TYPE_LOWER
            default:
                cmp = i < r;
                break;
        }
        dst_row[i] = cmp ? (keep_org_val ? src_row[i] : constant) : 0.0f;
    }
}

void ggml_cuda_op_tri(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->ne[0] == src0->ne[1]); // must be square matrix

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

    // Get operation parameters
    int tri_type = (int)dst->op_params[0];
    float constant = ggml_get_op_params_f32(dst, 1);
    bool keep_org_val = isnan(constant);

    // Launch kernel
    dim3 grid(ne1, ne2, ne3);
    const int block_size = 1; // One thread per row

    tri_f32_kernel<<<grid, block_size, 0, stream>>>(
        src0_d, dst_d, ne0, ne1, ne2, ne3,
        nb0, nb1, nb2, nb3,
        dst_nb0, dst_nb1, dst_nb2, dst_nb3,
        tri_type, constant, keep_org_val
    );
}

