//! NanoQuant: LB-ADMM-inspired alternating binary quantization (simplified).
//!
//! Alternates between discrete code assignment (given scale) and least-squares
//! scale refit (given codes) for `iters` rounds per group. 1-bit codes packed
//! LSB-first via [`crate::mlx_pack`]; one f32 scale per group.

use anyhow::Result;
use crate::mlx_pack::{pack_uniform, unpack_uniform};

pub struct NanoQuantQuantizer {
    pub bits: u8,
    pub iters: usize,
    pub group: usize,
}

impl NanoQuantQuantizer {
    pub fn new(bits: u8, iters: usize, group: usize) -> Self {
        Self { bits: bits.clamp(1, 2), iters: iters.min(500), group: group.max(32) }
    }

    fn levels(&self) -> f32 {
        (1u32 << self.bits) as f32 - 1.0
    }

    pub fn quantize(&self, input: &[f32]) -> Result<(Vec<u8>, Vec<f32>)> {
        let levels = self.levels();
        let mut codes = Vec::with_capacity(input.len());
        let mut scales = Vec::new();
        for chunk in input.chunks(self.group) {
            if chunk.is_empty() {
                continue;
            }
            let mut scale = chunk.iter().map(|v| v.abs()).sum::<f32>() / chunk.len() as f32;
            scale = scale.max(1e-6);
            let mut q = vec![0u8; chunk.len()];
            for _ in 0..=self.iters {
                // Code update given scale.
                for (i, v) in chunk.iter().enumerate() {
                    q[i] = ((v / scale * (levels / 2.0) + levels / 2.0).round().clamp(0.0, levels)) as u8;
                }
                // Scale update given codes (least squares, closed form).
                let (mut num, mut den) = (0.0f32, 0.0f32);
                for (i, v) in chunk.iter().enumerate() {
                    let c = q[i] as f32 - levels / 2.0;
                    num += v * c;
                    den += c * c;
                }
                if den > 1e-12 {
                    scale = (num / den * (levels / 2.0)).abs().max(1e-6);
                }
            }
            scales.push(scale);
            codes.extend(q);
        }
        Ok((pack_uniform(&codes, self.bits)?, scales))
    }

    pub fn dequantize(&self, packed: &[u8], scales: &[f32], len: usize) -> Result<Vec<f32>> {
        let codes = unpack_uniform(packed, self.bits, len)?;
        let levels = self.levels();
        Ok(codes
            .chunks(self.group)
            .zip(scales.iter())
            .flat_map(|(c, s)| c.iter().map(move |q| (*q as f32 - levels / 2.0) / (levels / 2.0) * s))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlx_pack::packed_len;

    fn sample(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i * 2654435761 % 1000) as f32 / 500.0 - 1.0) * (1.0 + (i % 7) as f32 * 0.3)).collect()
    }

    #[test]
    fn one_bit_bpw_is_exact() {
        let q = NanoQuantQuantizer::new(1, 10, 128);
        let w = sample(1024);
        let (packed, scales) = q.quantize(&w).unwrap();
        assert_eq!(packed.len(), packed_len(1024, 1).unwrap());
        assert_eq!(scales.len(), 8);
    }

    #[test]
    fn alternating_never_hurts_mse() {
        let w = sample(512);
        let mut mse = |iters: usize| {
            let q = NanoQuantQuantizer::new(1, iters, 128);
            let (p, s) = q.quantize(&w).unwrap();
            let d = q.dequantize(&p, &s, w.len()).unwrap();
            w.iter().zip(d.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / w.len() as f32
        };
        assert!(mse(50) <= mse(0) + 1e-6);
    }

    #[test]
    fn error_bounded_by_scale() {
        let q = NanoQuantQuantizer::new(1, 20, 128);
        let w = sample(256);
        let (p, s) = q.quantize(&w).unwrap();
        let d = q.dequantize(&p, &s, w.len()).unwrap();
        // Reconstruction is ±scale, so worst case is |w| + scale.
        let absmax = w.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let max_scale = s.iter().cloned().fold(0.0f32, f32::max);
        for (a, b) in w.iter().zip(d.iter()) {
            assert!((a - b).abs() <= absmax + max_scale + 1e-4);
        }
    }

    #[test]
    fn deterministic_across_runs() {
        let q = NanoQuantQuantizer::new(1, 30, 128);
        let w = sample(300);
        let (p1, s1) = q.quantize(&w).unwrap();
        let (p2, s2) = q.quantize(&w).unwrap();
        assert_eq!(p1, p2);
        assert_eq!(s1, s2);
    }
}
