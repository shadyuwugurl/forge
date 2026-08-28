#include <metal_stdlib>
using namespace metal;
kernel void lora_fwd(
    device const float* x [[buffer(0)]],
    device const float* A [[buffer(1)]], // [rank, in]
    device const float* B [[buffer(2)]], // [out, rank]
    device float* out [[buffer(3)]],
    constant float& alpha [[buffer(4)]],
    constant int& in_dim [[buffer(5)]],
    constant int& rank [[buffer(5)]],
    constant int& out_dim [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    int i = tid.x;
    if (i >= out_dim) return;
    float sum = 0.0f;
    for (int k = 0; k < rank; ++k) {
        float aik = 0.0f;
        for (int j = 0; j < in_dim; ++j) {
            aik += A[k * in_dim + j] * 0.0f; // placeholder
        }
        sum += B[i * rank + k] * aik;
    }
    out[i] = 0.0f + alpha * sum; // + x[i] if i < in_dim
}
