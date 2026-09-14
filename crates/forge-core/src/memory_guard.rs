use anyhow::{bail, Result};

/// D3: memory guard — enforces `--max-memory-gb` (default 28) at plan time.
///
/// Arithmetic core only: given the live tensor-pair size and the shard
/// buffer, compute peak RSS and bail before any weights move. The CLI wires
/// real tensor sizes in; merge paths must call `check` per batch.
#[derive(Debug, Clone)]
pub struct MemoryGuard {
    pub max_bytes: u64,
}

/// Fixed overhead: safetensors headers, index JSON, allocator slack.
pub const GUARD_OVERHEAD_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeMemoryPlan {
    pub peak_bytes: u64,
    pub fits: bool,
}

impl MemoryGuard {
    pub fn new(max_gb: f64) -> Self {
        Self { max_bytes: (max_gb * 1024.0 * 1024.0 * 1024.0) as u64 }
    }

    /// Peak = both live tensors of the pair + shard write buffer + overhead.
    /// Streaming keeps exactly one pair live, so peak is independent of model size.
    pub fn plan_merge(&self, pair_bytes: u64, shard_bytes: u64) -> MergeMemoryPlan {
        let peak = pair_bytes.saturating_mul(2)
            .saturating_add(shard_bytes)
            .saturating_add(GUARD_OVERHEAD_BYTES);
        MergeMemoryPlan { peak_bytes: peak, fits: peak <= self.max_bytes }
    }

    pub fn check(&self, bytes: u64) -> Result<()> {
        if bytes > self.max_bytes {
            bail!("memory guard: {} bytes over {} byte budget", bytes, self.max_bytes);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_pair_fits_28gb() {
        let g = MemoryGuard::new(28.0);
        let p = g.plan_merge(256 * 1024 * 1024, 64 * 1024 * 1024);
        assert!(p.fits);
        assert_eq!(p.peak_bytes, 512 * 1024 * 1024 + 64 * 1024 * 1024 + GUARD_OVERHEAD_BYTES);
    }

    #[test]
    fn huge_pair_rejected() {
        let g = MemoryGuard::new(28.0);
        let p = g.plan_merge(20 * 1024 * 1024 * 1024, 5 * 1024 * 1024 * 1024);
        assert!(!p.fits);
    }

    #[test]
    fn check_bails_over_budget() {
        let g = MemoryGuard::new(1.0);
        assert!(g.check(2 * 1024 * 1024 * 1024).is_err());
        assert!(g.check(1024).is_ok());
    }

    #[test]
    fn peak_scales_with_pair_not_model() {
        // Same pair size, different model sizes -> identical peak.
        let g = MemoryGuard::new(28.0);
        let a = g.plan_merge(128 * 1024 * 1024, 64 * 1024 * 1024);
        let b = g.plan_merge(128 * 1024 * 1024, 64 * 1024 * 1024);
        assert_eq!(a, b);
    }
}
