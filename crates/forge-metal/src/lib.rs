use anyhow::Result;

/// Apple Silicon Metal GPU and ANE acceleration — real hardware detection + kernel stubs.
//
// On macOS we query `sysctl` / IOKit for chip + memory. On other OS we return Unknown.
//
// Metal kernels themselves are MSM-written `.metal` files compiled via `metal` crate at runtime.
// This crate compiles with `cfg(target_os="macos")` gated Metal; CI on Linux gets the stub.

#[derive(Debug, Clone)]
pub struct HardwareInfo {
    pub chip: Chip,
    pub chip_name: String,
    pub gpu_cores: usize,
    pub has_neural_engine: bool,
    pub unified_memory_gb: f64,
    pub memory_bandwidth_gbs: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Chip {
    M1, M1Pro, M1Max, M1Ultra,
    M2, M2Pro, M2Max, M2Ultra,
    M3, M3Pro, M3Max, M3Ultra,
    M4, M4Pro, M4Max,
    M5, M5Pro, M5Max,
    M6,
    Unknown,
}

impl Chip {
    pub fn family(&self) -> &'static str {
        match self {
            Chip::M1 | Chip::M1Pro | Chip::M1Max | Chip::M1Ultra => "M1",
            Chip::M2 | Chip::M2Pro | Chip::M2Max | Chip::M2Ultra => "M2",
            Chip::M3 | Chip::M3Pro | Chip::M3Max | Chip::M3Ultra => "M3",
            Chip::M4 | Chip::M4Pro | Chip::M4Max => "M4",
            Chip::M5 | Chip::M5Pro | Chip::M5Max => "M5",
            Chip::M6 => "M6",
            Chip::Unknown => "Unknown",
        }
    }
}

impl std::fmt::Display for HardwareInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({} GPU cores, {:.1}GB RAM, ANE: {}, bw: {} GB/s)",
            self.chip_name, self.gpu_cores, self.unified_memory_gb, self.has_neural_engine,
            self.memory_bandwidth_gbs.map(|v| format!("{:.0}", v)).unwrap_or_else(|| "?".into()))
    }
}

pub struct MetalBackend;

impl MetalBackend {
    /// Detect host hardware. Never fails — returns Unknown on error.
    pub fn detect_hardware() -> Result<HardwareInfo> {
        #[cfg(target_os = "macos")]
        { detect_macos() }
        #[cfg(not(target_os = "macos"))]
        {
            Ok(HardwareInfo {
                chip: Chip::Unknown,
                chip_name: "Unknown (non-macOS)".into(),
                gpu_cores: 0,
                has_neural_engine: false,
                unified_memory_gb: 0.0,
                memory_bandwidth_gbs: None,
            })
        }
    }

    /// Return max model size that fits in unified memory (reserve ~4GB for OS).
    pub fn max_model_gb(info: &HardwareInfo) -> f64 { (info.unified_memory_gb - 4.0).max(0.0) }
}

#[cfg(target_os = "macos")]
fn detect_macos() -> Result<HardwareInfo> {
    use std::process::Command;

    let brand = Command::new("sysctl").arg("-n").arg("machdep.cpu.brand_string")
        .output().ok().and_then(|o| String::from_utf8(o.stdout).ok()).unwrap_or_default();

    let mem_bytes: u64 = Command::new("sysctl").arg("-n").arg("hw.memsize")
        .output().ok().and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse().ok()).unwrap_or(0);

    let chip = parse_chip(&brand);
    let (gpu_cores, bw) = gpu_info_for_chip(chip);

    Ok(HardwareInfo {
        chip_name: brand.trim().to_string(),
        chip,
        gpu_cores,
        has_neural_engine: chip != Chip::Unknown, // all Apple Silicon has ANE
        unified_memory_gb: mem_bytes as f64 / 1e9,
        memory_bandwidth_gbs: bw,
    })
}

fn parse_chip(brand: &str) -> Chip {
    let b = brand.to_lowercase();
    if b.contains("m6") { return Chip::M6; }
    if b.contains("m5 max") { return Chip::M5Max; }
    if b.contains("m5 pro") { return Chip::M5Pro; }
    if b.contains("m5") { return Chip::M5; }
    if b.contains("m4 max") { return Chip::M4Max; }
    if b.contains("m4 pro") { return Chip::M4Pro; }
    if b.contains("m4") { return Chip::M4; }
    if b.contains("m3 ultra") { return Chip::M3Ultra; }
    if b.contains("m3 max") { return Chip::M3Max; }
    if b.contains("m3 pro") { return Chip::M3Pro; }
    if b.contains("m3") { return Chip::M3; }
    if b.contains("m2 ultra") { return Chip::M2Ultra; }
    if b.contains("m2 max") { return Chip::M2Max; }
    if b.contains("m2 pro") { return Chip::M2Pro; }
    if b.contains("m2") { return Chip::M2; }
    if b.contains("m1 ultra") { return Chip::M1Ultra; }
    if b.contains("m1 max") { return Chip::M1Max; }
    if b.contains("m1 pro") { return Chip::M1Pro; }
    if b.contains("m1") { return Chip::M1; }
    Chip::Unknown
}

fn gpu_info_for_chip(chip: Chip) -> (usize, Option<f64>) {
    match chip {
        Chip::M1 => (8, Some(68.)), Chip::M1Pro => (16, Some(200.)), Chip::M1Max => (32, Some(400.)), Chip::M1Ultra => (64, Some(800.)),
        Chip::M2 => (10, Some(100.)), Chip::M2Pro => (19, Some(200.)), Chip::M2Max => (38, Some(400.)), Chip::M2Ultra => (76, Some(800.)),
        Chip::M3 => (10, Some(100.)), Chip::M3Pro => (18, Some(150.)), Chip::M3Max => (40, Some(400.)), Chip::M3Ultra => (76, Some(800.)),
        Chip::M4 => (10, Some(120.)), Chip::M4Pro => (20, Some(273.)), Chip::M4Max => (40, Some(546.)),
        Chip::M5 => (10, Some(150.)), Chip::M5Pro => (20, Some(300.)), Chip::M5Max => (40, Some(600.)),
        Chip::M6 => (12, Some(200.)),
        Chip::Unknown => (0, None),
    }
}

/// Metal kernels — compiled `.metal` sources (stubbed for now, real MSL in `metal/` dir).
pub mod kernels {
    /// Fused weighted-add: out = Σ w_i * a_i  (merge hot loop)
    pub const WEIGHTED_ADD: &str = include_str!("../metal/weighted_add.metal");
    /// SLERP kernel
    pub const SLERP: &str = include_str!("../metal/slerp.metal");
}
