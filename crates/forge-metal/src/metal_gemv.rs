//! Quantized GEMV reference oracle (M2).
//!
//! Computes `y = W x` where `W` is row-major, LSB-first packed exactly as
//! [`forge_quant::mlx_pack`] emits it (continuous packing, no per-row
//! padding), with one f32 scale per group of `group` columns.
//!
//! Layout constraint: `cols * bits` must be divisible by 8 so rows never share
//! a byte (true for all g128-native shapes: 128cols x 1/2/4-bit). This is the
//! CPU oracle for `kernels/gemv_quant.metal` — same math, same indexing.
//!
//! Value mapping is symmetric-midpoint: `v = (code - M) / M * scale` with
//! `M = (2^bits - 1) / 2` — the exact convention the forge-quant quantizers
//! use (`(q - levels/2) / (levels/2)`). This covers nano/arb/hbllm/af1/littlebit. Asymmetric
//! variants (dbell dual scales, btcllm centroids) carry their levels in a
//! strided scales vector and need host-side table lookup (future
//! `gemv_table`); they are out of scope for this kernel.

use anyhow::{bail, Result};

pub struct QuantGemv {
    pub bits: u8,
    pub group: usize,
}

impl QuantGemv {
    pub fn new(bits: u8, group: usize) -> Result<Self> {
        // group must match the quantizer's group (g128 native); any >= 1 works.
        match bits {
            1 | 2 | 4 | 8 => Ok(Self { bits, group: group.max(1) }),
            b => bail!("QuantGemv supports 1, 2, 4 or 8 bits, got {}", b),
        }
    }

    /// `packed`: rows*cols*bits/8 bytes; `scales`: rows*ceil(cols/group) f32.
    pub fn gemv(
        &self,
        packed: &[u8],
        scales: &[f32],
        rows: usize,
        cols: usize,
        x: &[f32],
    ) -> Result<Vec<f32>> {
        if (cols * self.bits as usize) % 8 != 0 {
            bail!("cols*bits must be byte-aligned, got cols={} bits={}", cols, self.bits);
        }
        let per_byte = 8 / self.bits as usize;
        let mask = (1u32 << self.bits) - 1;
        // Quantizer convention: levels = 2^bits - 1, midpoint M = levels/2.
        let mid = mask as f32 / 2.0;
        let row_bytes = cols * self.bits as usize / 8;
        let groups_per_row = cols.div_ceil(self.group);
        if packed.len() < rows * row_bytes {
            bail!("packed too short: {} < {}", packed.len(), rows * row_bytes);
        }
        if scales.len() < rows * groups_per_row {
            bail!("scales too short: {} < {}", scales.len(), rows * groups_per_row);
        }
        if x.len() < cols {
            bail!("x too short: {} < {}", x.len(), cols);
        }
        let mut y = vec![0.0f32; rows];
        for r in 0..rows {
            let mut acc = 0.0f32;
            for j in 0..cols {
                let idx = r * cols + j;
                let byte = packed[idx / per_byte];
                let code = ((byte as u32 >> ((idx % per_byte) * self.bits as usize)) & mask) as f32;
                let s = scales[r * groups_per_row + j / self.group];
                acc += ((code - mid) / mid) * s * x[j];
            }
            y[r] = acc;
        }
        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn four_bit_matches_hand_computation() {
        // 1 row x 4 cols, 4-bit: codes [0, 15, 8, 7], scale 2.0, x all ones.
        // M = 7.5: -2.0, +2.0, +0.1333, -0.1333 -> sums to 0.
        let g = QuantGemv::new(4, 32).unwrap();
        let packed = vec![0xF0u8, 0x78u8]; // lo nibble first: 0,15 then 8,7
        let y = g.gemv(&packed, &[2.0], 1, 4, &[1.0, 1.0, 1.0, 1.0]).unwrap();
        assert!(y[0].abs() < 1e-4, "y={}", y[0]);
        // All-max codes saturate at +scale: 4 cols x (15-7.5)/7.5*2 = 8.
        let y2 = g.gemv(&[0xFF, 0xFF], &[2.0], 1, 4, &[1.0; 4]).unwrap();
        assert!((y2[0] - 8.0).abs() < 1e-4, "y={}", y2[0]);
    }

    #[test]
    fn one_bit_two_rows() {
        // row0 codes: 1,1,0,0,1,0,1,0 -> byte 0b01011001? LSB-first: bit j = code j.
        // bits: j0=1,j1=1,j2=0,j3=0,j4=1,j5=0,j6=1,j7=0 -> 0b01010011 = 0x53.
        // row1 all ones -> 0xFF. scales [1.0, 0.5], x = ones.
        // row0 values: +1,+1,-1,-1,+1,-1,+1,-1 -> sum 0. row1: 8 * 0.5 = 4.
        let g = QuantGemv::new(1, 32).unwrap();
        let y = g.gemv(&[0x53, 0xFF], &[1.0, 0.5], 2, 8, &[1.0; 8]).unwrap();
        assert!((y[0]).abs() < 1e-5, "y0={}", y[0]);
        assert!((y[1] - 4.0).abs() < 1e-5, "y1={}", y[1]);
    }

    #[test]
    fn two_bit_scales_per_group() {
        // cols=8, group=4 -> 2 scales per row. codes all 3 (max), scales [1.0, 2.0].
        // value = (3-1.5)/1.5*s = s. y = 4*1 + 4*2 = 12.
        let g = QuantGemv::new(2, 4).unwrap();
        let y = g.gemv(&[0xFF, 0xFF], &[1.0, 2.0], 1, 8, &[1.0; 8]).unwrap();
        assert!((y[0] - 12.0).abs() < 1e-5, "y={}", y[0]);
    }

    #[test]
    fn unaligned_cols_bail() {
        let g = QuantGemv::new(1, 32).unwrap();
        assert!(g.gemv(&[0xFF; 4], &[1.0], 1, 10, &[1.0; 10]).is_err());
    }

    #[test]
    fn short_buffers_bail() {
        let g = QuantGemv::new(4, 32).unwrap();
        assert!(g.gemv(&[0x00], &[1.0], 1, 4, &[1.0; 4]).is_err());
        assert!(g.gemv(&[0x00, 0x00], &[], 1, 4, &[1.0; 4]).is_err());
        assert!(g.gemv(&[0x00, 0x00], &[1.0], 1, 4, &[1.0; 2]).is_err());
        assert!(QuantGemv::new(3, 32).is_err());
    }
}
