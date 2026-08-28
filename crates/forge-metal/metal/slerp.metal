#include <metal_stdlib>
using namespace metal;
kernel void slerp(device const float* a [[buffer(0)]],
                  device const float* b [[buffer(1)]],
                  device float* out [[buffer(2)]],
                  constant float& t [[buffer(3)]],
                  constant float& sin_theta [[buffer(4)]],
                  constant float& theta [[buffer(5)]],
                  uint id [[thread_position_in_grid]]) {
    float wa = sin((1.0 - t) * theta) / sin_theta;
    float wb = sin(t * theta) / sin_theta;
    out[id] = wa * a[id] + wb * b[id];
}
