//! M5 benchmark harness: baseline timings for hot paths.
//!
//! Run: `cargo run --release -p forge-cli --example m5_bench`
//! Deterministic LCG weights; 3 timed runs each, reports ms + GB/s.

use std::time::Instant;

fn lcg_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn bench(label: &str, bytes: usize, f: impl Fn()) {
    // Warmup.
    f();
    let mut best = u128::MAX;
    for _ in 0..3 {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_micros());
    }
    let ms = best as f64 / 1000.0;
    let gbs = bytes as f64 / (best as f64) / 1000.0;
    println!("{label:34} {ms:10.1} ms   {gbs:8.2} GB/s");
}

fn main() -> anyhow::Result<()> {
    println!("== M5 baseline (release) ==");

    // 1. mlx_pack pack/unpack on 4M codes.
    for bits in [1u8, 2, 4] {
        let levels = 1u32 << bits;
        let n = 4_000_000usize;
        let codes: Vec<u8> = (0..n).map(|i| (i as u32 * 2654435761 % levels) as u8).collect();
        let packed = forge_quant::mlx_pack::pack_uniform(&codes, bits)?;
        bench(&format!("pack_uniform {bits}-bit 4M"), n, || {
            let _ = forge_quant::mlx_pack::pack_uniform(&codes, bits).unwrap();
        });
        bench(&format!("unpack_uniform {bits}-bit 4M"), n, || {
            let _ = forge_quant::mlx_pack::unpack_uniform(&packed, bits, n).unwrap();
        });
    }

    // 2. QuantGemv 1024x4096 2-bit.
    {
        let (rows, cols) = (1024usize, 4096usize);
        let codes: Vec<u8> = (0..rows * cols).map(|i| (i % 4) as u8).collect();
        let packed = forge_quant::mlx_pack::pack_uniform(&codes, 2)?;
        let scales = vec![0.5f32; rows * cols / 128];
        let x = vec![0.01f32; cols];
        let g = forge_metal::metal_gemv::QuantGemv::new(2, 128)?;
        let flops = rows * cols * 2;
        bench("QuantGemv 1024x4096 2-bit", flops * 4, || {
            let _ = g.gemv(&packed, &scales, rows, cols, &x).unwrap();
        });
    }

    // 3. Quantizers on 1M f32.
    {
        let w = lcg_f32(1_000_000, 42);
        let bytes = w.len() * 4;
        let nq = forge_quant::NanoQuantQuantizer::new(1, 50, 128);
        bench("nanoquant 1-bit 1M", bytes, || {
            let _ = nq.quantize(&w).unwrap();
        });
        let af = forge_quant::Af1Quantizer::new(1, 128);
        bench("af1 1-bit 1M", bytes, || {
            let _ = af.quantize(&w).unwrap();
        });
        let lb = forge_quant::LittleBitQuantizer::new(1, true, 128);
        bench("littlebit 1-bit 1M", bytes, || {
            let _ = lb.quantize(&w).unwrap();
        });
    }

    // 4. Chimera compat scoring on 1M vecs.
    {
        let a = lcg_f32(1_000_000, 1);
        let b = lcg_f32(1_000_000, 2);
        bench("chimera score_compat 1M", 2 * a.len() * 4, || {
            let _ = forge_merge::chimera::score_compat(&a, &b);
        });
    }

    // 5. tensor_f32 file round-trip: 8 x 4M-f32 tensors (~128MB).
    {
        let dir = std::env::temp_dir().join("m5_bench_store");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir)?;
        let w = lcg_f32(4_000_000, 7);
        let raw: &[u8] = bytemuck::cast_slice(&w);
        let mut wr = forge_io::StreamingWriter::new(&dir, 5 * 1024 * 1024 * 1024)?;
        for i in 0..8 {
            wr.write_tensor(&format!("t{i}"), raw, "F32", &[4_000_000])?;
        }
        wr.finalize("model")?;
        let store = forge_io::TensorStore::open(&dir.join("model.safetensors"))?;
        let names = store.tensor_names();
        let total = 8 * w.len() * 4;
        bench("tensor_f32 8x4M F32 mmap", total, || {
            for n in &names {
                let _ = store.tensor_f32(n).unwrap();
            }
        });
        let _ = std::fs::remove_dir_all(&dir);
    }

    Ok(())
}
