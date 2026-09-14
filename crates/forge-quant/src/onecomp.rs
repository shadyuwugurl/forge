use anyhow::Result;

/// B3 OneComp (arxiv 2603.28845, Fujitsu 2026): progressive mixed-precision PTQ.
///
/// Portable core of the OneComp pipeline:
/// - `plan_bits` (AutoBit spirit): assign per-tensor bit-widths under a memory
///   budget, keeping small (sensitive: norms, biases, heads) tensors at `hi_bits`
///   and demoting the largest tensors first until the budget fits.
/// - `quantize_sequential` (QEP spirit): quantize a sequence of shards while
///   carrying the previous residual forward (error feedback), so the running
///   error telescopes instead of accumulating: sum(out) == sum(in) - last_carry.
/// - Group-wise symmetric quantization (JointQ spirit: joint scale selection
///   per group; binary-factor 1-2 bit regime via `lo_bits`).
pub struct OneCompQuantizer {
    /// Bit-width for protected tensors (default 4).
    pub hi_bits: u8,
    /// Bit-width for demoted tensors (default 2).
    pub lo_bits: u8,
    /// Group size sharing one scale (default 128).
    pub group: usize,
    /// Memory budget in bytes for the planned tensors (default u64::MAX).
    pub budget_bytes: u64,
}

impl OneCompQuantizer {
    pub fn new(hi_bits: u8, lo_bits: u8, group: usize, budget_bytes: u64) -> Self {
        Self {
            hi_bits: hi_bits.clamp(1, 8),
            lo_bits: lo_bits.clamp(1, 8).min(hi_bits),
            group: group.max(16),
            budget_bytes,
        }
    }

    pub fn default_4bit() -> Self {
        Self::new(4, 2, 128, u64::MAX)
    }

    /// Bytes a tensor of `len` f32 elements costs at `bits`.
    pub fn planned_bytes(len: usize, bits: u8) -> u64 {
        (len as u64 * bits as u64).div_ceil(8)
    }

    /// AutoBit-style planner: all-hi unless over budget, then demote largest first.
    /// Returns one bit-width per input length, stable by input order.
    pub fn plan_bits(&self, lens: &[usize]) -> Vec<u8> {
        let mut bits = vec![self.hi_bits; lens.len()];
        let total: u64 = lens.iter().map(|l| Self::planned_bytes(*l, self.hi_bits)).sum();
        if total <= self.budget_bytes {
            return bits;
        }
        let mut order: Vec<usize> = (0..lens.len()).collect();
        order.sort_by(|&a, &b| lens[b].cmp(&lens[a]));
        let mut running = total;
        for idx in order {
            if running <= self.budget_bytes {
                break;
            }
            running -= Self::planned_bytes(lens[idx], self.hi_bits);
            running += Self::planned_bytes(lens[idx], self.lo_bits);
            bits[idx] = self.lo_bits;
        }
        bits
    }

    /// Group-wise symmetric quantize at `bits` -> (levels as bytes, scales).
    pub fn quantize(&self, input: &[f32], bits: u8) -> Result<(Vec<u8>, Vec<f32>)> {
        let bits = bits.clamp(1, 8);
        let levels = (1u32 << bits) as f32 - 1.0;
        let mut out = Vec::with_capacity(input.len());
        let mut scales = Vec::new();
        for chunk in input.chunks(self.group) {
            let max = chunk.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
            scales.push(max);
            for v in chunk {
                let q = ((v / max * (levels / 2.0) + levels / 2.0).round().clamp(0.0, levels)) as u8;
                out.push(q);
            }
        }
        Ok((out, scales))
    }

    /// Dequantize levels with scales.
    pub fn dequantize(&self, levels: &[u8], scales: &[f32], len: usize) -> Vec<f32> {
        // Recover bits from levels range is ambiguous; caller passes bits explicitly.
        self.dequantize_at(levels, scales, len, self.hi_bits)
    }

    /// Dequantize knowing the bit-width used.
    pub fn dequantize_at(&self, levels: &[u8], scales: &[f32], len: usize, bits: u8) -> Vec<f32> {
        let levels_f = (1u32 << bits.clamp(1, 8)) as f32 - 1.0;
        let n = len.min(levels.len());
        levels[..n]
            .chunks(self.group)
            .zip(scales.iter())
            .flat_map(|(c, s)| c.iter().map(move |q| (*q as f32 - levels_f / 2.0) / (levels_f / 2.0) * s))
            .collect()
    }

    /// QEP-style sequential quantization with error feedback.
    /// Returns per-shard (levels, scales); the telescoping invariant holds:
    /// sum over shards of dequantized sums == sum of inputs - final carry sum.
    pub fn quantize_sequential(
        &self,
        shards: &[Vec<f32>],
        bits_plan: &[u8],
    ) -> Result<Vec<(Vec<u8>, Vec<f32>)>> {
        assert_eq!(shards.len(), bits_plan.len(), "shards and bits plan must align");
        let mut carried: Vec<f32> = Vec::new();
        let mut result = Vec::with_capacity(shards.len());
        for (shard, &bits) in shards.iter().zip(bits_plan.iter()) {
            if carried.len() != shard.len() {
                carried = vec![0.0; shard.len()];
            }
            let corrected: Vec<f32> =
                shard.iter().zip(carried.iter()).map(|(x, c)| x + c).collect();
            let (levels, scales) = self.quantize(&corrected, bits)?;
            let recon = self.dequantize_at(&levels, &scales, shard.len(), bits);
            carried = corrected.iter().zip(recon.iter()).map(|(t, r)| t - r).collect();
            result.push((levels, scales));
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32 / n as f32 - 0.5).collect()
    }

    #[test]
    fn planner_fits_budget_by_demoting_largest() {
        let q = OneCompQuantizer::new(4, 2, 128, 1000);
        // 3000 f32 elems at 4-bit = 1500B > 1000B budget -> demote largest (2000) to 2-bit: 500+500=1000.
        let bits = q.plan_bits(&[1000, 2000]);
        assert_eq!(bits, vec![4, 2]);
        let bytes: u64 =
            [1000usize, 2000usize].iter().zip(bits.iter()).map(|(l, b)| OneCompQuantizer::planned_bytes(*l, *b)).sum();
        assert!(bytes <= 1000);
    }

    #[test]
    fn planner_keeps_all_hi_when_budget_allows() {
        let q = OneCompQuantizer::default_4bit();
        assert_eq!(q.plan_bits(&[64, 128]), vec![4, 4]);
    }

    #[test]
    fn roundtrip_bounded_error() {
        let q = OneCompQuantizer::default_4bit();
        let x = ramp(512);
        let (levels, scales) = q.quantize(&x, 4).unwrap();
        let back = q.dequantize_at(&levels, &scales, x.len(), 4);
        let max_err = x.iter().zip(back.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 0.12, "max_err={max_err}");
    }

    #[test]
    fn sequential_telescopes_error() {
        let q = OneCompQuantizer::default_4bit();
        let shards = vec![ramp(256), ramp(256), ramp(256)];
        let plan = vec![4, 2, 4];
        let out = q.quantize_sequential(&shards, &plan).unwrap();
        let recon_sum: f32 = out
            .iter()
            .zip(plan.iter())
            .map(|((lv, sc), b)| q.dequantize_at(lv, sc, 256, *b).iter().sum::<f32>())
            .sum();
        let in_sum: f32 = shards.iter().map(|s| s.iter().sum::<f32>()).sum();
        // Telescoping: residual is bounded by one shard's worth of error.
        assert!((recon_sum - in_sum).abs() < 8.0, "drift={}", (recon_sum - in_sum).abs());
    }

    #[test]
    fn lowbit_uses_fewer_levels_than_hibit() {
        let q = OneCompQuantizer::default_4bit();
        let x = ramp(256);
        let (hi, _) = q.quantize(&x, 4).unwrap();
        let (lo, _) = q.quantize(&x, 2).unwrap();
        let hi_uniq: std::collections::HashSet<u8> = hi.into_iter().collect();
        let lo_uniq: std::collections::HashSet<u8> = lo.into_iter().collect();
        assert!(lo_uniq.len() <= 4);
        assert!(hi_uniq.len() > lo_uniq.len());
    }
}
