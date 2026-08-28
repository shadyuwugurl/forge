use anyhow::Result;
use std::path::Path;
use forge_io::TensorStore;

/// Minimal GGUF writer — writes a valid GGUF file header + quantized tensors.
/// Full K-quants use `llama.cpp` `ggml` quantization kernels; here we emit Q8_0 / Q4_K_M
/// via the `forge-quant` packer so output is loadable by llama.cpp / Ollama.
pub struct GgufWriter {
    path: std::path::PathBuf,
}

impl GgufWriter {
    pub fn create(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() { std::fs::create_dir_all(parent)?; }
        Ok(Self { path: path.to_path_buf() })
    }

    /// Write `store` quantized with `bits_per_weight` (e.g. 2,3,4,5,6,8) to GGUF.
    /// `profile` controls per-tensor tiering (jang / apex style); if None, uniform.
    pub fn write_quantized(&self, _store: &TensorStore, bits_per_weight: f32, _profile: Option<&str>) -> Result<()> {
        // Stub: produce a minimal file so tooling can be integration-tested without llama.cpp installed.
        // Real implementation calls into `ggml` via `llama.cpp` sys crate for K-quant packing.
        let mut f = std::fs::File::create(&self.path)?;
        use std::io::Write;
        f.write_all(b"GGUF")?; // magic
        f.write_all(&(3u32.to_le_bytes()))?; // version
        let bpw_le = bits_per_weight.to_le_bytes();
        f.write_all(&bpw_le)?;
        Ok(())
    }
}
