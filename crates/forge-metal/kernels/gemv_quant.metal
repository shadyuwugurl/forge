// gemv_quant.metal — quantized GEMV for sub-1-bit MLX weights (M2).
//
// y = W x, one thread per row. W is row-major, LSB-first packed exactly as
// forge-quant::mlx_pack emits it (continuous packing, no per-row padding),
// with one fp32 scale per `group` columns.
//
// Layout contract (must match metal_gemv.rs, the CPU oracle):
//   packed : rows*cols*bits/8 bytes, byte idx = (r*cols + j) / (8/bits)
//   scales : rows*ceil(cols/group) fp32, idx = r*groups_per_row + j/group
//   value  : (code - M) / M * scale, M = (2^bits-1)/2 (symmetric midpoint,
//            the exact forge-quant quantizer convention)
// Constraint: cols*bits divisible by 8 (true for all g128-native shapes).
// Asymmetric variants (dbell/btcllm) need host-side tables — not this kernel.
//
// Build (Xcode toolchain):
//   xcrun -sdk macosx metal -c gemv_quant.metal -o gemv_quant.air
// Swift hosts load the .air via MTLDevice.makeLibrary / makeComputePipelineState;
// no Swift sources ship in-repo (host apps own their Swift).

#include <metal_stdlib>
using namespace metal;

kernel void gemv_quant(
    device const uchar* packed [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device const float* x      [[buffer(2)]],
    device float*       y      [[buffer(3)]],
    constant uint& rows  [[buffer(4)]],
    constant uint& cols  [[buffer(5)]],
    constant uint& bits  [[buffer(6)]],
    constant uint& group [[buffer(7)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= rows) return;
    const uint per_byte = 8u / bits;
    const uint mask = (1u << bits) - 1u;
    const float mid = float(mask) * 0.5f;
    const uint groups_per_row = (cols + group - 1u) / group;
    float acc = 0.0f;
    for (uint j = 0; j < cols; ++j) {
        uint idx = row * cols + j;
        uint byte_idx = idx / per_byte;
        uint shift = (idx % per_byte) * bits;
        float code = float((packed[byte_idx] >> shift) & mask);
        float s = scales[row * groups_per_row + j / group];
        acc += ((code - mid) / mid) * s * x[j];
    }
    y[row] = acc;
}
