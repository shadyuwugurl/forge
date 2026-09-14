use anyhow::Result;

/// BSQ-AT: block-scaled quantized adaptive training quantizer (Phase B).
/// Default 4-bit block-wise quantization (`--method bsqat`):
/// per-`block` absmax scale stored as F16, weights packed as nibbles.
pub struct BsqatQuantizer {
    pub bits: u8,
    pub block: usize,
}

impl BsqatQuantizer {
    pub fn new(bits: u8, block: usize) -> Self {
        Self { bits: bits.clamp(2, 8), block: block.max(16) }
    }
    /// Quantize one f32 shard -> (packed bytes, scales). Nibble-packed for 4-bit.
    pub fn quantize(&self, input: &[f32]) -> Result<(Vec<u8>, Vec<f32>)> {
        let mut packed = Vec::new();
        let mut scales = Vec::new();
        for chunk in input.chunks(self.block) {
            let max = chunk.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
            scales.push(max);
            let levels = (1u32 << self.bits) as f32 - 1.0;
            let mut nibble_buf: u8 = 0;
            let mut half = false;
            for v in chunk {
                let q = ((v / max * (levels / 2.0) + levels / 2.0).round().clamp(0.0, levels)) as u8;
                if self.bits == 4 {
                    if !half {
                        nibble_buf = q & 0x0F;
                        half = true;
                    } else {
                        packed.push(nibble_buf | (q << 4));
                        half = false;
                    }
                } else {
                    packed.push(q);
                }
            }
            if self.bits == 4 && half {
                packed.push(nibble_buf);
            }
        }
        Ok((packed, scales))
    }
    /// Dequantize for verification.
    pub fn dequantize(&self, packed: &[u8], scales: &[f32], len: usize) -> Vec<f32> {
        let levels = (1u32 << self.bits) as f32 - 1.0;
        let mut qs: Vec<u8> = Vec::with_capacity(len);
        if self.bits == 4 {
            for b in packed {
                qs.push(b & 0x0F);
                if qs.len() >= len {
                    break;
                }
                qs.push(b >> 4);
                if qs.len() >= len {
                    break;
                }
            }
        } else {
            qs.extend(packed.iter().cloned());
        }
        qs.truncate(len);
        qs.chunks(self.block)
            .zip(scales.iter())
            .flat_map(|(c, s)| c.iter().map(move |q| (*q as f32 - levels / 2.0) / (levels / 2.0) * s))
            .collect()
    }
}
