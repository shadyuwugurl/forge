#include <metal_stdlib>
using namespace metal;
kernel void weighted_add(device const float* a [[buffer(0)]],
                         device const float* b [[buffer(1)]],
                         device float* out [[buffer(2)]],
                         constant float& wa [[buffer(3)]],
                         constant float& wb [[buffer(4)]],
                         uint id [[thread_position_in_grid]]) {
    out[id] = wa * a[id] + wb * b[id];
}
