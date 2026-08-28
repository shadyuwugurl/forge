#include <metal_stdlib>
using namespace metal;
kernel void slerp(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant float& t [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= thread_position_in_grid) return;
    float wa = a[id], wb = b[id];
    float dot = wa * wb;
    float norm_a = length(float4(wa)), norm_b = length(float4(wb));
    // Simplified for 1D: just lerp with t
    out[id] = (1.0 - t) * wa + t * wb;
}
