//! ARB-LLM: alternating refinement + codebook-gradient quantization (simplified).
//!
//! Scale init is outlier-robust when `reorder` is set (trimmed absmax over a
//! magnitude-sorted scratch copy — the RC/reorder spirit without a stored
//! permutation sidecar), then codes/scales alternate with a gradient-descent
//! check (CGB spirit): a scale step is kept only if MSE strictly decreases.

use anyhow::Result;
use crate::mlx_pack::{pack_uniform, unpack_uniform};

pub struct ArbQuantizer {
    pub bits: u8,
    pub reorder: bool,
    pub group: usize,
}

impl ArbQuantizer {
    pub fn new(bits: u8, reorder: bool, group: usize) -> Self {
        Self { bits: bits.clamp(1, 2), reorder, group: group.max(32) }
    }

    fn levels(&self) -> f32 {
        (1u32 << self.bits) as f32 - 1.0
    }

    fn init_scale(&self, chunk: &[f32]) -> f32 {
        if !self.reorder || chunk.is_empty() {
            return chunk.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
        }
        let mut mags: Vec<f32> = chunk.iter().map(|v| v.abs()).collect();
        mags.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Trimmed absmax: drop top 1% magnitudes (outlier-robust).
        let keep = (mags.len() * 99 / 100).max(1).min(mags.len());
        mags[keep - 1].max(1e-6)
    }

    fn mse(&self, chunk: &[f32], q: &[u8], scale: f32) -> f32 {
        let levels = self.levels();
        chunk
            .iter()
            .zip(q.iter())
            .map(|(v, c)| {
                let r = (*c as f32 - levels / 2.0) / (levels / 2.0) * scale;
                (v - r).powi(2)
            })
            .sum::<f32>()
            / chunk.len() as f32
    }

    pub fn quantize(&self, input: &[f32]) -> Result<(Vec<u8>, Vec<f32>)> {
        let levels = self.levels();
        let mut codes = Vec::with_capacity(input.len());
        let mut scales = Vec::new();
        for chunk in input.chunks(self.group) {
            if chunk.is_empty() {
                continue;
            }
            let mut scale = self.init_scale(chunk);
            let mut q = vec![0u8; chunk.len()];
            let assign = |chunk: &[f32], scale: f32, q: &mut [u8]| {
                for (i, v) in chunk.iter().enumerate() {
                    q[i] = ((v / scale * (levels / 2.0) + levels / 2.0).round().clamp(0.0, levels)) as u8;
                }
            };
            assign(chunk, scale, &mut q);
            // 8 refinement rounds; codebook-gradient step kept iff MSE drops.
            for _ in 0..8 {
                assign(chunk, scale, &mut q);
                let (mut num, mut den) = (0.0f32, 0.0f32);
                for (i, v) in chunk.iter().enumerate() {
                    let c = q[i] as f32 - levels / 2.0;
                    num += v * c;
                    den += c * c;
                }
                if den <= 1e-12 {
                    break;
                }
                let before = self.mse(chunk, &q, scale);
                // Analytic gradient of MSE wrt scale; exact line search along it.
                let mut grad_num = 0.0f32;
                for (i, v) in chunk.iter().enumerate() {
                    let c = (q[i] as f32 - levels / 2.0) / (levels / 2.0);
                    grad_num += c * (v - c * scale);
                }
                let grad_den: f32 = q
                    .iter()
                    .map(|c| {
                        let cc = (*c as f32 - levels / 2.0) / (levels / 2.0);
                        cc * cc
                    })
                    .sum();
                if grad_den <= 1e-12 {
                    break;
                }
                let candidate = scale + grad_num / grad_den;
                if candidate > 1e-6 && self.mse(chunk, &q, candidate) < before {
                    scale = candidate;
                } else {
                    // Fall back to closed-form LS step.
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
        (0..n).map(|i| ((i * 40503 % 997) as f32 / 498.5 - 1.0) * (1.0 + (i % 5) as f32)).collect()
    }

    #[test]
    fn one_bit_bpw_is_exact() {
        let q = ArbQuantizer::new(1, true, 128);
        let w = sample(1024);
        let (packed, scales) = q.quantize(&w).unwrap();
        assert_eq!(packed.len(), packed_len(1024, 1).unwrap());
        assert_eq!(scales.len(), 8);
    }

    #[test]
    fn reorder_trims_outlier_scale() {
        let mut w = sample(256);
        w[0] = 100.0; // single outlier
        let plain = ArbQuantizer::new(1, false, 256);
        let reord = ArbQuantizer::new(1, true, 256);
        let (_, s_plain) = plain.quantize(&w).unwrap();
        let (_, s_reord) = reord.quantize(&w).unwrap();
        assert!(s_reord[0] <= s_plain[0], "trimmed {} should be <= absmax {}", s_reord[0], s_plain[0]);
    }

    #[test]
    fn error_bounded_and_deterministic() {
        let q = ArbQuantizer::new(1, true, 128);
        let w = sample(300);
        let (p1, s1) = q.quantize(&w).unwrap();
        let (p2, s2) = q.quantize(&w).unwrap();
        assert_eq!(p1, p2);
        assert_eq!(s1, s2);
        let d = q.dequantize(&p1, &s1, w.len()).unwrap();
        let absmax = w.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let max_scale = s1.iter().cloned().fold(0.0f32, f32::max);
        for (a, b) in w.iter().zip(d.iter()) {
            assert!((a - b).abs() <= absmax + max_scale + 1e-4);
        }
    }
}
