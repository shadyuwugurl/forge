//! HBLLM: Haar-wavelet-domain binary coding (simplified).
//!
//! Each group is forward Haar-transformed (`levels` stages), coefficients are
//! stored as 1-bit signs plus one shared scale, and dequantization inverts the
//! transform. Energy compaction puts signal in a few large coefficients, so a
//! single global sign bit per coefficient beats time-domain 1-bit on
//! smooth/correlated weights.

use anyhow::Result;
use crate::mlx_pack::{pack_uniform, unpack_uniform};

pub struct HbllmQuantizer {
    pub levels: u8,
    pub group: usize,
}

impl HbllmQuantizer {
    pub fn new(levels: u8, group: usize) -> Self {
        Self { levels: levels.clamp(1, 8), group: group.max(32) }
    }

    /// Working length: next power of two >= group.
    fn work_len(&self) -> usize {
        self.group.next_power_of_two()
    }

    /// Decomposition stages actually applied (clamped to what fits).
    fn stages(&self) -> usize {
        let max = self.work_len().trailing_zeros() as usize;
        (self.levels as usize).clamp(1, max)
    }

    pub fn quantize(&self, input: &[f32]) -> Result<(Vec<u8>, Vec<f32>)> {
        let work = self.work_len();
        let stages = self.stages();
        let avg_len = work >> stages;
        let mut codes = Vec::with_capacity(input.len());
        // Stride-2 scales per group: (approx-band scale, detail-band scale).
        let mut scales = Vec::new();
        for chunk in input.chunks(self.group) {
            if chunk.is_empty() {
                continue;
            }
            let mut buf = vec![0.0f32; work];
            buf[..chunk.len()].copy_from_slice(chunk);
            haar_forward(&mut buf, stages);
            let s_avg = buf[..avg_len].iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
            let s_det = buf[avg_len..].iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
            scales.push(s_avg);
            scales.push(s_det);
            // Store signs for the real coefficients only (padding is zero).
            codes.extend(chunk.iter().enumerate().map(|(i, _)| if buf[i] >= 0.0 { 1u8 } else { 0u8 }));
        }
        Ok((pack_uniform(&codes, 1)?, scales))
    }

    pub fn dequantize(&self, packed: &[u8], scales: &[f32], len: usize) -> Result<Vec<f32>> {
        let codes = unpack_uniform(packed, 1, len)?;
        let work = self.work_len();
        let stages = self.stages();
        let avg_len = work >> stages;
        let mut out = Vec::with_capacity(len);
        for (chunk, s) in codes.chunks(self.group).zip(scales.chunks(2)) {
            let (s_avg, s_det) = (s[0], s[1]);
            let mut buf = vec![0.0f32; work];
            for (i, c) in chunk.iter().enumerate() {
                let band = if i < avg_len { s_avg } else { s_det };
                buf[i] = if *c == 1 { band } else { -band };
            }
            haar_inverse(&mut buf, stages);
            out.extend(buf[..chunk.len()].iter().cloned());
        }
        Ok(out)
    }
}

fn haar_forward(x: &mut [f32], stages: usize) {
    let mut tmp = vec![0.0f32; x.len()];
    let mut len = x.len();
    for _ in 0..stages {
        let half = len / 2;
        for i in 0..half {
            let (a, b) = (x[2 * i], x[2 * i + 1]);
            tmp[i] = (a + b) * 0.5;
            tmp[half + i] = (a - b) * 0.5;
        }
        x[..len].copy_from_slice(&tmp[..len]);
        len = half;
    }
}

fn haar_inverse(x: &mut [f32], stages: usize) {
    let mut tmp = vec![0.0f32; x.len()];
    let mut len = x.len() >> stages;
    for _ in 0..stages {
        for i in 0..len {
            let (a, d) = (x[i], x[len + i]);
            tmp[2 * i] = a + d;
            tmp[2 * i + 1] = a - d;
        }
        x[..2 * len].copy_from_slice(&tmp[..2 * len]);
        len *= 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlx_pack::packed_len;

    #[test]
    fn haar_roundtrip_is_identity() {
        let mut x: Vec<f32> = (0..128).map(|i| (i as f32 * 0.37).sin() * 2.0 - 0.5).collect();
        let orig = x.clone();
        haar_forward(&mut x, 3);
        haar_inverse(&mut x, 3);
        for (a, b) in orig.iter().zip(x.iter()) {
            assert!((a - b).abs() < 1e-4, "{} vs {}", a, b);
        }
    }

    #[test]
    fn one_bit_bpw_is_exact() {
        let q = HbllmQuantizer::new(3, 128);
        let w: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.11).cos()).collect();
        let (packed, scales) = q.quantize(&w).unwrap();
        assert_eq!(packed.len(), packed_len(1024, 1).unwrap());
        assert_eq!(scales.len(), 16); // 2 band scales per group
    }

    #[test]
    fn levels_clamp_and_error_bounded() {
        let q = HbllmQuantizer::new(99, 128); // clamped to 7
        assert_eq!(q.stages(), 7);
        // Flat regions wavelet-compress near-exactly: details vanish, signs
        // reconstruct the approx band at full scale.
        let w = vec![0.75f32; 256];
        let (p, s) = q.quantize(&w).unwrap();
        let d = q.dequantize(&p, &s, w.len()).unwrap();
        let mse_flat: f32 = w.iter().zip(d.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / w.len() as f32;
        assert!(mse_flat < 1e-6, "flat mse {}", mse_flat);
        // High-frequency content pays full 1-bit cost: bounded and finite.
        let w2: Vec<f32> = (0..256).map(|i| (i as f32 * 0.05).sin()).collect();
        let (p2, s2) = q.quantize(&w2).unwrap();
        let d2 = q.dequantize(&p2, &s2, w2.len()).unwrap();
        assert_eq!(d2.len(), w2.len());
        assert!(d2.iter().all(|v| v.is_finite()));
    }
}
