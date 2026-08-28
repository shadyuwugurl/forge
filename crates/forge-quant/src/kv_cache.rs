use anyhow::Result;

/// KV cache optimizer — organizes & compresses the KV cache for long-context serving.
///
/// Mirrors PMetal's TurboQuant: block paged layout + per-head quantization + eviction.
/// Use: `forge quant --kv-cache --block-size 64 --quant q4` or auto via `forge serve`.

#[derive(Debug, Clone)]
pub struct KvCacheConfig {
    /// Page/block size (tokens per block)
    pub block_size: usize,
    /// Quantization for cached K/V (none, q8, q4)
    pub quant: KvQuant,
    /// Max blocks before eviction (0 = no limit, use unified memory)
    pub max_blocks: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvQuant { None, Q8, Q4 }

impl Default for KvCacheConfig {
    fn default() -> Self { Self { block_size: 64, quant: KvQuant::Q4, max_blocks: 0 } }
}

/// Paged KV cache layout — one paged table per layer/head.
///
/// On Apple Silicon, blocks are Metal buffers; quantization is per-block (scale/bias per head).
/// This organizer just describes the layout; the actual Metal kernels are in `forge-metal`.
pub struct KvCacheOrganizer {
    pub config: KvCacheConfig,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl KvCacheOrganizer {
    pub fn new(num_layers: usize, num_heads: usize, head_dim: usize) -> Self {
        Self { config: KvCacheConfig::default(), num_layers, num_heads, head_dim }
    }

    /// Memory for a full cache of `seq_len` tokens (all layers) at `quant` precision.
    pub fn memory_bytes(&self, seq_len: usize) -> usize {
        let per_token = self.num_layers * self.num_heads * self.head_dim * 2; // K + V
        let bytes_per_elem = match self.config.quant {
            KvQuant::None => 2, // f16
            KvQuant::Q8 => 1,
            KvQuant::Q4 => 1, // packed 4b = 0.5 byte but with scales; approx 0.6
        };
        let raw = seq_len * per_token * bytes_per_elem;
        // Block overhead: one scale/bias per block per head (f16 *2)
        let num_blocks = (seq_len + self.config.block_size - 1) / self.config.block_size;
        let overhead = num_blocks * self.num_layers * self.num_heads * 4;
        raw + overhead
    }

    /// Suggest a config that fits `target_gb` for a given `seq_len` on `available_gb` unified memory.
    pub fn suggest_for_budget(&mut self, seq_len: usize, available_gb: f64, target_gb: f64) {
        let full = self.memory_bytes(seq_len) as f64 / 1e9;
        if full <= target_gb && full <= available_gb - 4.0 { self.config.quant = KvQuant::None; }
        else if full * 0.55 <= target_gb { self.config.quant = KvQuant::Q8; }
        else { self.config.quant = KvQuant::Q4; self.config.block_size = 32; }
    }

    /// Describe layout as JSON for `forge info --kv-cache` and for `forge-metal` to allocate.
    pub fn describe(&self, seq_len: usize) -> serde_json::Value {
        serde_json::json!({
            "layers": self.num_layers, "heads": self.num_heads, "head_dim": self.head_dim,
            "block_size": self.config.block_size, "quant": format!("{:?}", self.config.quant),
            "seq_len": seq_len, "memory_gb": self.memory_bytes(seq_len) as f64 / 1e9,
            "num_blocks": (seq_len + self.config.block_size - 1)/self.config.block_size,
            "layout": "paged_per_head_quantized"
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn memory_fits() {
        let org = KvCacheOrganizer::new(32, 32, 128);
        let bytes = org.memory_bytes(4096);
        assert!(bytes > 0);
        assert!(bytes < 10*1024*1024*1024);
    }
}
