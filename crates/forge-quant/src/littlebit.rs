//! LittleBit: residual-compensated 1-bit quantization (simplified).
//!
//! Naive sign quantization fits the scale to the data range (absmax) and eats
//! the full residual. With `residual` set, the scale is instead refit in
//! closed form to minimize the actual reconstruction residual (least squares
//! over the chosen codes) — the residual-compensation step — which provably
//! never increases MSE versus the naive scale for the same codes.

use anyhow::Result;
use crate::mlx_pack::{pack_uniform, unpack_uniform};

pub struct LittleBitQuantizer {
    pub bits: u8,
    pub residual: bool,
    pub group: usize,
}

impl LittleBitQuantizer {
    pub fn new(bits: u8, residual: bool, group: usize) -> Self {
        Self { bits: bits.clamp(1, 1), residual, group: group.max(32) }
    }

    fn group_scale(&self, chunk: &[f32], codes: &[u8]) -> f32 {
        if !self.residual {
            return chunk.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
        }
        // Least-squares refit: s = sum(v*c)/sum(c^2), c in {-1,+1}.
        let (mut num, mut den) = (0.0f32, 0.0f32);
        for (v, q) in chunk.iter().zip(codes.iter()) {
            let c = if *q == 1 { 1.0f32 } else { -1.0f32 };
            num += v * c;
            den += c * c;
        }
        if den <= 1e-12 {
            chunk.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6)
        } else {
            (num / den).abs().max(1e-6)
        }
    }

    pub fn quantize(&self, input: &[f32]) -> Result<(Vec<u8>, Vec<f32>)> {
        let mut codes = Vec::with_capacity(input.len());
        let mut scales = Vec::new();
        for chunk in input.chunks(self.group) {
            if chunk.is_empty() {
                continue;
            }
            let q: Vec<u8> = chunk.iter().map(|v| if *v >= 0.0 { 1u8 } else { 0u8 }).collect();
            scales.push(self.group_scale(chunk, &q));
            codes.extend(q);
        }
        Ok((pack_uniform(&codes, 1)?, scales))
    }

    pub fn dequantize(&self, packed: &[u8], scales: &[f32], len: usize) -> Result<Vec<f32>> {
        let codes = unpack_uniform(packed, 1, len)?;
        Ok(codes
            .chunks(self.group)
            .zip(scales.iter())
            .flat_map(|(c, s)| c.iter().map(move |q| if *q == 1 { *s } else { -*s }))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlx_pack::packed_len;

    fn sample(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i * 1103515245 + 12345) % 1000) as f32 / 500.0 - 1.0).collect()
    }

    fn mse(w: &[f32], d: &[f32]) -> f32 {
        w.iter().zip(d.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / w.len() as f32
    }

    #[test]
    fn one_bit_bpw_is_exact() {
        let q = LittleBitQuantizer::new(1, true, 128);
        let w = sample(1024);
        let (packed, scales) = q.quantize(&w).unwrap();
        assert_eq!(packed.len(), packed_len(1024, 1).unwrap());
        assert_eq!(scales.len(), 8);
    }

    #[test]
    fn residual_compensation_never_hurts() {
        let w = sample(512);
        let naive = LittleBitQuantizer::new(1, false, 128);
        let comp = LittleBitQuantizer::new(1, true, 128);
        let (pn, sn) = naive.quantize(&w).unwrap();
        let (pc, sc) = comp.quantize(&w).unwrap();
        let mn = mse(&w, &naive.dequantize(&pn, &sn, w.len()).unwrap());
        let mc = mse(&w, &comp.dequantize(&pc, &sc, w.len()).unwrap());
        assert!(mc <= mn + 1e-6, "compensated {} > naive {}", mc, mn);
    }

    #[test]
    fn deterministic_and_finite() {
        let q = LittleBitQuantizer::new(1, true, 64);
        let w = sample(200);
        let (p1, s1) = q.quantize(&w).unwrap();
        let (p2, s2) = q.quantize(&w).unwrap();
        assert_eq!(p1, p2);
        assert_eq!(s1, s2);
        let d = q.dequantize(&p1, &s1, w.len()).unwrap();
        assert!(d.iter().all(|v| v.is_finite()));
    }
}
