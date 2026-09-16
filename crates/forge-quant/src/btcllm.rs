//! BTC-LLM: vector-codebook 1-bit quantization (simplified).
//!
//! Each group learns a 2^bits-entry codebook via 1-D k-means (5 Lloyd iters,
//! min/max init) and stores code indices (1 bit at bits=1). Centroids are
//! carried in the scales vector (2^bits f32 per group), so the standard
//! (packed, scales) streaming format holds both codes and codebook.

use anyhow::{bail, Result};
use crate::mlx_pack::{pack_uniform, unpack_uniform};

pub struct BtcQuantizer {
    pub bits: u8,
    pub codebook: usize,
    pub group: usize,
}

impl BtcQuantizer {
    pub fn new(bits: u8, codebook: usize, group: usize) -> Self {
        Self { bits: bits.clamp(1, 3), codebook, group: group.max(32) }
    }

    fn entries(&self) -> Result<usize> {
        let want = 1usize << self.bits;
        if self.codebook != want {
            bail!("btc codebook size {} != 2^bits {} — pass --profile <bits>:<2^bits>", self.codebook, want);
        }
        Ok(want)
    }

    fn kmeans_1d(&self, chunk: &[f32], entries: usize) -> Vec<f32> {
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for v in chunk {
            lo = lo.min(*v);
            hi = hi.max(*v);
        }
        if !(hi > lo) {
            return vec![lo; entries];
        }
        let mut centers: Vec<f32> = (0..entries).map(|i| lo + (hi - lo) * i as f32 / (entries - 1) as f32).collect();
        for _ in 0..5 {
            let mut sums = vec![0.0f32; entries];
            let mut counts = vec![0usize; entries];
            for v in chunk {
                let mut best = 0;
                let mut bd = f32::INFINITY;
                for (i, c) in centers.iter().enumerate() {
                    let d = (v - c).abs();
                    if d < bd {
                        bd = d;
                        best = i;
                    }
                }
                sums[best] += *v;
                counts[best] += 1;
            }
            for i in 0..entries {
                if counts[i] > 0 {
                    centers[i] = sums[i] / counts[i] as f32;
                }
            }
        }
        centers
    }

    pub fn quantize(&self, input: &[f32]) -> Result<(Vec<u8>, Vec<f32>)> {
        let entries = self.entries()?;
        let mut codes = Vec::with_capacity(input.len());
        let mut scales = Vec::new();
        for chunk in input.chunks(self.group) {
            if chunk.is_empty() {
                continue;
            }
            let centers = self.kmeans_1d(chunk, entries);
            scales.extend(centers.iter().cloned());
            for v in chunk {
                let mut best = 0u8;
                let mut bd = f32::INFINITY;
                for (i, c) in centers.iter().enumerate() {
                    let d = (v - c).abs();
                    if d < bd {
                        bd = d;
                        best = i as u8;
                    }
                }
                codes.push(best);
            }
        }
        Ok((pack_uniform(&codes, self.bits)?, scales))
    }

    pub fn dequantize(&self, packed: &[u8], scales: &[f32], len: usize) -> Result<Vec<f32>> {
        let entries = 1usize << self.bits;
        let codes = unpack_uniform(packed, self.bits, len)?;
        Ok(codes
            .chunks(self.group)
            .zip(scales.chunks(entries))
            .flat_map(|(c, centers)| c.iter().map(move |q| centers[*q as usize]))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlx_pack::packed_len;

    #[test]
    fn one_bit_bpw_is_exact() {
        let q = BtcQuantizer::new(1, 2, 128);
        let w: Vec<f32> = (0..1024).map(|i| (i as f32 / 512.0) - 1.0).collect();
        let (packed, scales) = q.quantize(&w).unwrap();
        assert_eq!(packed.len(), packed_len(1024, 1).unwrap());
        assert_eq!(scales.len(), 16); // 2 centroids per group
    }

    #[test]
    fn bimodal_centroids_match_modes() {
        let q = BtcQuantizer::new(1, 2, 64);
        let mut w = vec![-3.0f32; 32];
        w.extend(vec![4.0f32; 32]);
        let (_, s) = q.quantize(&w).unwrap();
        let mut c = vec![s[0], s[1]];
        c.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((c[0] - -3.0).abs() < 1e-4, "{:?}", c);
        assert!((c[1] - 4.0).abs() < 1e-4, "{:?}", c);
    }

    #[test]
    fn wrong_codebook_size_bails() {
        let q = BtcQuantizer::new(1, 256, 128);
        assert!(q.quantize(&[0.5, -0.5]).is_err());
    }

    #[test]
    fn dequant_roundtrip_length() {
        let q = BtcQuantizer::new(2, 4, 64);
        let w: Vec<f32> = (0..200).map(|i| ((i * 131 % 89) as f32 / 44.5) - 1.0).collect();
        let (p, s) = q.quantize(&w).unwrap();
        let d = q.dequantize(&p, &s, w.len()).unwrap();
        assert_eq!(d.len(), w.len());
        assert!(d.iter().all(|v| v.is_finite()));
    }
}
