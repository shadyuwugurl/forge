use anyhow::{bail, Result};

/// D1: MLX bridge stub — MLX quantize is the primary path on M5.
///
/// Pure-Rust planning surface only: no `mlx-sys` dependency yet. When the
/// native bridge lands, `plan_quantize` output feeds the MLX call and
/// `probe` gates `--method mlx` in the CLI.
/// Supports 1/2-bit (sub-1-bit quantizers, g128) plus 4/8-bit.
#[derive(Debug, Clone)]
pub struct MlxBridge {
    pub bits: u8,
    /// Scales group: one fp16 scale per `group` weights (MLX g128).
    pub group: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlxProbe {
    pub available: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlxQuantPlan {
    pub params: u64,
    pub bits: u8,
    pub out_bytes: u64,
}

impl MlxBridge {
    /// 1/2-bit (sub-1-bit, g128) plus 4/8-bit. Anything else is rejected.
    pub fn new(bits: u8) -> Result<Self> {
        Self::with_group(bits, 128)
    }

    pub fn with_group(bits: u8, group: usize) -> Result<Self> {
        if bits == 1 || bits == 2 || bits == 4 || bits == 8 {
            Ok(Self { bits, group: group.max(32) })
        } else {
            bail!("mlx bridge supports 1, 2, 4 or 8 bits, got {}", bits);
        }
    }

    /// Compile-time platform gate: Apple Silicon macOS only.
    pub fn probe() -> MlxProbe {
        MlxProbe {
            available: cfg!(target_arch = "aarch64") && cfg!(target_os = "macos"),
        }
    }

    /// Byte budget for a weight array: packed data (params * bits / 8)
    /// plus one fp16 scale per group (MLX g128 layout).
    pub fn plan_quantize(&self, params: u64) -> MlxQuantPlan {
        let data = params * self.bits as u64 / 8;
        let scales = params.div_ceil(self.group as u64) * 2;
        MlxQuantPlan {
            params,
            bits: self.bits,
            out_bytes: data + scales,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_returns_struct_with_bool() {
        let p = MlxBridge::probe();
        assert_eq!(p.available, cfg!(target_arch = "aarch64") && cfg!(target_os = "macos"));
    }

    #[test]
    fn four_bit_plan_math_is_exact() {
        let b = MlxBridge::new(4).unwrap();
        // 1B params at 4-bit -> 500MB data + 1B/128*2 scales.
        let scales = 1_000_000_000u64.div_ceil(128) * 2;
        assert_eq!(b.plan_quantize(1_000_000_000).out_bytes, 500_000_000 + scales);
    }

    #[test]
    fn sub_one_bit_accepted_with_group_overhead() {
        let b = MlxBridge::with_group(1, 128).unwrap();
        // 1024 params at 1-bit -> 128B data + 8 groups * 2B scales.
        assert_eq!(b.plan_quantize(1024).out_bytes, 128 + 16);
        let b2 = MlxBridge::new(2).unwrap();
        assert_eq!(b2.group, 128);
        assert_eq!(b2.plan_quantize(1024).out_bytes, 256 + 16);
    }

    #[test]
    fn invalid_bits_rejected() {
        assert!(MlxBridge::new(3).is_err());
        assert!(MlxBridge::new(16).is_err());
        assert!(MlxBridge::new(1).is_ok());
        assert!(MlxBridge::new(2).is_ok());
        assert!(MlxBridge::new(8).is_ok());
    }
}
