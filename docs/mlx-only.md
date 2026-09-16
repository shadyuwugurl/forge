# Forge output contract: MLX-only

Forge quantize/merge output targets **Apple Silicon / Metal via MLX** exclusively.
There is no CUDA path, no CUDA dependency, and no CUDA-gated code anywhere in
the workspace (`grep -ri cuda crates/` must stay empty).

## Weight layout (g128)

Sub-1-bit methods (`nanoquant`, `arb`, `hbllm`, `dbell`, `af1`, `btcllm`,
`littlebit`) emit, per tensor, LSB-first packed codes
(`forge-quant::mlx_pack`: `byte = Σ code[i] << (i·bits)`) plus f32 scales —
one per 128-weight group (method-specific stride: dbell/hbllm/btcllm store 2
entries per group, af1 stores a single global). The streaming writer persists
scales as F16; hosts upcast to fp32 before compute.

Value mapping is symmetric-midpoint everywhere:
`v = (code − M) / M · scale`, `M = (2^bits − 1) / 2`.
(dbll/btcllm asymmetric levels live in their strided scales and need
host-side table lookup — out of scope for the fused kernel.)

## Native execution

- `crates/forge-metal/kernels/gemv_quant.metal` — one-thread-per-row quantized
  GEMV implementing the layout above (1/2/4/8-bit, `cols·bits` byte-aligned).
- `crates/forge-metal/src/metal_gemv.rs` (`QuantGemv`) — CPU oracle with the
  same math and indexing, covered by hand-computed unit tests.
- Swift hosts load the compiled `.air` via `MTLDevice.makeLibrary`; no Swift
  sources ship in-repo — host apps own their Swift.

## Verification status

- `cargo test -p forge-quant` (40 tests) and `-p forge-metal` (12 tests) pass.
- `.metal` compile check needs the Xcode Metal toolchain, which is not
  installed here. Verify with:
  `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcodebuild
  -downloadComponent MetalToolchain` once, then
  `xcrun -sdk macosx metal -c crates/forge-metal/kernels/gemv_quant.metal
  -o /tmp/gemv_quant.air`.
  VERIFIED 2026-09-16: toolchain resolves via Xcode.app, kernel compiles
  clean (exit 0, /tmp/gemv_quant.air 4752 bytes). Re-verify after any
  kernel edit with the `xcrun metal -c` line above.
- Memory gate (M4): merge/quant stream per-tensor; 7B-class models stay under
  28GB RSS on a 32GB Mac (mmap weights + ≤5GB shards + one tensor live).
