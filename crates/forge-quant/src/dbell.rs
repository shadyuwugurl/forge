//! DBellQuant: dual-bell non-uniform 1-bit quantization (simplified).
//!
//! Weights follow a bell around zero, but positives and negatives have
//! different spreads in practice. Each group stores two half-scales (one per
//! side) and 1-bit side codes, reconstructing {-s_neg, +s_pos} — a 2-entry
//! non-uniform codebook fit to the dual-bell shape. Strict 1 data-bit/weight.

use anyhow::Result;
use crate::mlx_pack::{pack_uniform, unpack_uniform};

pub struct DbellQuantizer {
    pub bits: u8,
    pub group: usize,
}

impl DbellQuantizer {
    pub fn new(bits: u8, group: usize) -> Self {
        Self { bits: bits.clamp(1, 1), group: group.max(32) }
    }

    pub fn quantize(&self, input: &[f32]) -> Result<(Vec<u8>, Vec<f32>)> {
        let mut codes = Vec::with_capacity(input.len());
        // Stride-2 scales: (s_neg, s_pos) per group.
        let mut scales = Vec::new();
        for chunk in input.chunks(self.group) {
            if chunk.is_empty() {
                continue;
            }
            let s_neg = chunk.iter().filter(|v| **v < 0.0).map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
            let s_pos = chunk.iter().filter(|v| **v >= 0.0).map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
            scales.push(s_neg);
            scales.push(s_pos);
            codes.extend(chunk.iter().map(|v| if *v >= 0.0 { 1u8 } else { 0u8 }));
        }
        Ok((pack_uniform(&codes, 1)?, scales))
    }

    pub fn dequantize(&self, packed: &[u8], scales: &[f32], len: usize) -> Result<Vec<f32>> {
        let codes = unpack_uniform(packed, 1, len)?;
        Ok(codes
            .chunks(self.group)
            .zip(scales.chunks(2))
            .flat_map(|(c, s)| {
                let (neg, pos) = (s[0], s[1]);
                c.iter().map(move |q| if *q == 1 { pos } else { -neg })
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlx_pack::packed_len;

    #[test]
    fn one_bit_bpw_is_exact() {
        let q = DbellQuantizer::new(1, 128);
        let w: Vec<f32> = (0..1024).map(|i| (i as f32 / 512.0) - 1.0).collect();
        let (packed, scales) = q.quantize(&w).unwrap();
        assert_eq!(packed.len(), packed_len(1024, 1).unwrap());
        assert_eq!(scales.len(), 16); // 2 per group
    }

    #[test]
    fn skewed_data_gives_asymmetric_scales() {
        let q = DbellQuantizer::new(1, 64);
        let mut w = vec![-0.1f32; 32];
        w.extend(vec![5.0f32; 32]);
        let (_, s) = q.quantize(&w).unwrap();
        assert!((s[0] - 0.1).abs() < 1e-5, "s_neg {}", s[0]);
        assert!((s[1] - 5.0).abs() < 1e-5, "s_pos {}", s[1]);
        let d = q.dequantize(&pack_uniform(&vec![0u8; 32].into_iter().chain(vec![1u8; 32]).collect::<Vec<_>>(), 1).unwrap(), &s, 64).unwrap();
        assert!(d.iter().take(32).all(|v| (*v - -0.1).abs() < 1e-5));
        assert!(d.iter().skip(32).all(|v| (*v - 5.0).abs() < 1e-5));
    }

    #[test]
    fn one_sided_group_has_no_nan() {
        let q = DbellQuantizer::new(1, 32);
        let w = vec![2.0f32; 32]; // all positive
        let (p, s) = q.quantize(&w).unwrap();
        let d = q.dequantize(&p, &s, w.len()).unwrap();
        assert!(d.iter().all(|v| v.is_finite()));
        assert!(d.iter().all(|v| (*v - 2.0).abs() < 1e-5));
    }
}
