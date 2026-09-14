/// D2: MLX flash-attention stub — capability probe + dispatch plan.
///
/// Pure logic only: decides whether a (head_dim, seq_len) pair may take the
/// flash path once the native kernel exists. No new dependencies.
#[derive(Debug, Clone)]
pub struct MlxFlash {
    pub max_seq: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlashPlan {
    pub use_flash: bool,
    pub head_dim: usize,
    pub reason: &'static str,
}

impl MlxFlash {
    pub fn new(max_seq: usize) -> Self {
        Self { max_seq }
    }

    pub fn plan(&self, head_dim: usize, seq_len: usize) -> FlashPlan {
        if !matches!(head_dim, 64 | 128) {
            return FlashPlan { use_flash: false, head_dim, reason: "unsupported head_dim" };
        }
        if seq_len > self.max_seq {
            return FlashPlan { use_flash: false, head_dim, reason: "seq_len over budget" };
        }
        FlashPlan { use_flash: true, head_dim, reason: "flash path" }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_dims_take_flash_path() {
        let f = MlxFlash::new(8192);
        assert!(f.plan(64, 4096).use_flash);
        assert!(f.plan(128, 8192).use_flash);
    }

    #[test]
    fn odd_head_dim_falls_back() {
        let f = MlxFlash::new(8192);
        let p = f.plan(96, 1024);
        assert!(!p.use_flash);
        assert_eq!(p.reason, "unsupported head_dim");
    }

    #[test]
    fn overlong_seq_falls_back() {
        let f = MlxFlash::new(4096);
        let p = f.plan(64, 8192);
        assert!(!p.use_flash);
        assert_eq!(p.reason, "seq_len over budget");
    }
}
