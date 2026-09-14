use anyhow::{bail, Result};

/// D1: MLX bridge stub — 4-bit MLX quantize is the primary path on M5.
///
/// Pure-Rust planning surface only: no `mlx-sys` dependency yet. When the
/// native bridge lands, `plan_quantize` output feeds the MLX call and
/// `probe` gates `--method mlx` in the CLI.
#[derive(Debug, Clone)]
pub struct MlxBridge {
    pub bits: u8,
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
    /// 4-bit is the M5 default; 8-bit allowed. Anything else is rejected.
    pub fn new(bits: u8) -> Result<Self> {
        if bits == 4 || bits == 8 {
            Ok(Self { bits })
        } else {
            bail!("mlx bridge supports 4 or 8 bits, got {}", bits);
        }
    }

    /// Compile-time platform gate: Apple Silicon macOS only.
    pub fn probe() -> MlxProbe {
        MlxProbe {
            available: cfg!(target_arch = "aarch64") && cfg!(target_os = "macos"),
        }
    }

    /// Byte budget for a weight array: params * bits / 8 (group overhead
    /// accounted by the caller once the native bridge exists).
    pub fn plan_quantize(&self, params: u64) -> MlxQuantPlan {
        MlxQuantPlan {
            params,
            bits: self.bits,
            out_bytes: params * self.bits as u64 / 8,
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
        // 1B params at 4-bit -> 500MB.
        assert_eq!(b.plan_quantize(1_000_000_000).out_bytes, 500_000_000);
    }

    #[test]
    fn invalid_bits_rejected() {
        assert!(MlxBridge::new(2).is_err());
        assert!(MlxBridge::new(16).is_err());
        assert!(MlxBridge::new(8).is_ok());
    }
}
