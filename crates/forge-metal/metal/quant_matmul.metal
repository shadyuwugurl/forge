#include <metal_stdlib>
using namespace metal;
kernel void quant_matmul(
    device const float* q [[buffer(0)]],
    device const uchar* k_packed [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]
) {
    // Quantized matmul for int4 K cache - stub
    int row = tid.x;
    int col = tid.y;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        // unpack int4 from k_packed...
        sum += q[row * K + k] * 0.0f;
    }
    out[row * N + col] = sum;
}
