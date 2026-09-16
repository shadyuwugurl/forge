//! AF1: strict-1.00bpw quantization (simplified).
//!
//! One sign bit per weight plus a SINGLE global scale for the whole tensor —
//! no per-group scales, no residual sidecar. Total is 1 data-bit/weight plus a
//! negligible 32-bit global (0.00003bpw at 1M params), i.e. strict 1.00bpw.

use anyhow::Result;
use crate::mlx_pack::{pack_uniform, unpack_uniform};

pub struct Af1Quantizer {
    pub bits: u8,
    pub group: usize,
}

impl Af1Quantizer {
    pub fn new(bits: u8, group: usize) -> Self {
        Self { bits: bits.clamp(1, 1), group: group.max(32) }
    }

    pub fn quantize(&self, input: &[f32]) -> Result<(Vec<u8>, Vec<f32>)> {
        let scale = input.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
        let codes: Vec<u8> = input.iter().map(|v| if *v >= 0.0 { 1u8 } else { 0u8 }).collect();
        Ok((pack_uniform(&codes, 1)?, vec![scale]))
    }

    pub fn dequantize(&self, packed: &[u8], scales: &[f32], len: usize) -> Result<Vec<f32>> {
        let scale = *scales.first().unwrap_or(&1.0);
        let codes = unpack_uniform(packed, 1, len)?;
        Ok(codes.iter().map(|c| if *c == 1 { scale } else { -scale }).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlx_pack::packed_len;

    #[test]
    fn single_global_scale_only() {
        let q = Af1Quantizer::new(1, 128);
        let w: Vec<f32> = (0..2048).map(|i| (i as f32 / 1024.0) - 1.0).collect();
        let (packed, scales) = q.quantize(&w).unwrap();
        assert_eq!(packed.len(), packed_len(2048, 1).unwrap());
        assert_eq!(scales.len(), 1, "strict AF1 stores exactly one scale");
        assert!((scales[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn zero_tensor_has_no_nan() {
        let q = Af1Quantizer::new(1, 128);
        let w = vec![0.0f32; 100];
        let (p, s) = q.quantize(&w).unwrap();
        let d = q.dequantize(&p, &s, w.len()).unwrap();
        assert!(d.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn error_bounded_by_global_scale() {
        let q = Af1Quantizer::new(1, 128);
        let w: Vec<f32> = (0..500).map(|i| ((i * 7919 % 613) as f32 / 306.5) - 1.0).collect();
        let (p, s) = q.quantize(&w).unwrap();
        let d = q.dequantize(&p, &s, w.len()).unwrap();
        for (a, b) in w.iter().zip(d.iter()) {
            assert!((a - b).abs() <= s[0] + 1e-4);
        }
    }
}
