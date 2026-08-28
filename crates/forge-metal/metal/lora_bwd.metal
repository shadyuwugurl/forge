#include <metal_stdlib>
using namespace metal;
kernel void lora_bwd(
    device const float* grad_out [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const float* A [[buffer(2)]],
    device const float* B [[buffer(3)]],
    device float* grad_A [[buffer(4)]],
    device float* grad_B [[buffer(5)]],
    constant int& in_dim [[buffer(6)]],
    constant int& rank [[buffer(7)]],
    constant int& out_dim [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    // Gradient computation for LoRA training - stub
}
