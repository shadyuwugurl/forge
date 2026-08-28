// forge-metal — Apple Silicon Metal/ANE acceleration (stubs for v0.4, full dispatch in v0.5)

use anyhow::{Context, Result};

/// Hardware info with Metal/ANE capabilities
#[derive(Debug, Clone)]
pub struct HardwareInfo {
    pub chip_name: String,
    pub gpu_cores: usize,
    pub has_ane: bool,
    pub unified_memory_gb: f64,
    pub metal_supports_fp16: bool,
    pub metal_supports_int4: bool,
    pub metal_supports_int8: bool,
}

impl HardwareInfo {
    pub fn from_sysctl() -> Self {
        let mem: u64 = std::process::Command::new("sysctl").arg("-n").arg("hw.memsize")
            .output().ok().and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.trim().parse().ok()).unwrap_or(0);
        let mem_gb = mem as f64 / 1e9;
        HardwareInfo { chip_name: "Apple Silicon".into(), gpu_cores: 0, has_ane: false, unified_memory_gb: mem_gb, metal_supports_fp16: true, metal_supports_int4: false, metal_supports_int8: false }
    }
}

impl std::fmt::Display for HardwareInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({} GPU cores, {:.1}GB, ANE:{}, Metal:fp16={},int4={},int8={})",
            self.chip_name, self.gpu_cores, self.unified_memory_gb, self.has_ane,
            self.metal_supports_fp16, self.metal_supports_int4, self.metal_supports_int8)
    }
}

/// Metal-backed merge operations (stub for now - real Metal dispatch in v0.5)
pub struct MetalMerge;

impl MetalMerge {
    pub fn new() -> anyhow::Result<Self> { Ok(Self) }

    /// fused weighted add on GPU: out = wa * a + wb * b
    pub fn weighted_add(&self, a: &[f32], b: &[f32], wa: f32, wb: f32) -> anyhow::Result<Vec<f32>> {
        Ok(a.iter().zip(b.iter()).map(|(a, b)| wa * a + wb * b).collect())
    }

    /// SLERP on GPU
    pub fn slerp(&self, a: &[f32], b: &[f32], t: f32) -> anyhow::Result<Vec<f32>> {
        let n = a.len().min(b.len());
        let mut out = vec![0.0f32; n];
        let dot: f32 = a.iter().zip(b.iter()).map(|(a, b)| a * b).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos_theta = (dot / (norm_a * norm_b + 1e-8)).clamp(-1.0, 1.0);
        let theta = cos_theta.acos();
        if theta.abs() < 1e-6 {
            for i in 0..n { out[i] = (1.0 - t) * a[i] + t * b[i]; }
        } else {
            let sin_theta = theta.sin();
            let wa = ((1.0 - t) * theta).sin() / sin_theta;
            let wb = (t * theta).sin() / sin_theta;
            for i in 0..n { out[i] = wa * a[i] + wb * b[i]; }
        }
        Ok(out)
    }
}

/// Metal-backed LoRA training (stub)
pub struct MetalLoRA;
impl MetalLoRA {
    pub fn new() -> anyhow::Result<Self> { Ok(Self) }
    pub fn forward(&self, _x: &[f32], _a: &[f32], _b: &[f32], _alpha: f32) -> anyhow::Result<Vec<f32>> { Ok(vec![]) }
}

/// Metal-backed KV cache
pub struct MetalKVCache;
impl MetalKVCache {
    pub fn new() -> anyhow::Result<Self> { Ok(Self) }
    pub fn quant_matmul(&self, _q: &[f32], _k: &[u8]) -> anyhow::Result<Vec<f32>> { Ok(vec![]) }
}