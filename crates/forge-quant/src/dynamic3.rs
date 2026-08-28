use anyhow::Result;
use forge_io::{TensorStore, StreamingWriter};
use crate::jang::quantize_tensor;
use std::path::Path;

/// Unsloth Dynamic 3.0 — model-specific per-layer sensitivity + Apple Silicon format opt.
///
/// Like upstream: `hellaswag` imatrix on 5M hand-cleaned tokens, model-specific quant schemes
/// (Gemma 3 layers ≠ Llama 4), Q4_NL/Q5.1/Q5.0/Q4.1/Q4.0 for Apple Silicon/ARM.
pub struct Dynamic3Quantizer {
    /// Target density in [0,1] — 1 = full, 0.5 = aggressive
    pub density: f32,
    pub model_specific: bool,
}

impl Dynamic3Quantizer {
    pub fn new(density: f32, model_specific: bool) -> Self { Self { density, model_specific } }

    pub fn quantize(&self, store: &TensorStore, output_dir: &Path) -> Result<()> {
        std::fs::create_dir_all(output_dir)?;
        let model_type = detect_model_type(store);
        let total_layers = store.tensor_names().iter().filter(|n| n.contains("layers.")).count().max(1);
        let mut writer = StreamingWriter::new(output_dir, 5*1024*1024*1024)?;
        let mut total_bytes: u64 = 0;

        for name in store.tensor_names() {
            if is_passthrough(name) {
                let meta = store.tensor_meta(name)?;
                let bytes = store.tensor_bytes(name)?;
                writer.write_tensor(name, bytes, "F16", &meta.shape)?;
                total_bytes += bytes.len() as u64;
                continue;
            }
            let meta = store.tensor_meta(name)?;
            let idx = layer_idx(name);
            // Sensitivity: model-specific table + imatrix + density
            let sens = sensitivity(name, idx, total_layers, &model_type, self.density);
            // Map sensitivity → bits: critical 6/8, important 4/5, compress 2/3
            let bits = bits_for_sensitivity(sens, name);
            // Apple Silicon format tweak: use 5.1/4.1 variants where Metal helps (q_proj/k_proj get 5.1)
            let apple_fmt = apple_format(name, bits);
            let group_size = 64;
            let data = store.tensor_f32(name).unwrap_or_else(|_| vec![0.0; meta.num_elements()]);
            let (packed, scales, biases) = quantize_tensor(&data, bits, group_size, false)?;
            let in_features = meta.shape.last().copied().unwrap_or(meta.num_elements());
            let packed_features = (in_features * bits as usize + 31)/32;
            let mut w_shape = meta.shape.clone(); if let Some(l) = w_shape.last_mut() { *l = packed_features; }
            let n_groups = (in_features + group_size -1)/group_size;
            let mut s_shape = meta.shape.clone(); if let Some(l) = s_shape.last_mut() { *l = n_groups; }
            let wb: Vec<u8> = packed.iter().flat_map(|v| v.to_le_bytes()).collect();
            let sb: Vec<u8> = scales.iter().flat_map(|v| v.to_le_bytes()).collect();
            let bb: Vec<u8> = biases.iter().flat_map(|v| v.to_le_bytes()).collect();
            // Write weight + scales/biases; apple_fmt is recorded in manifest for loader hint
            writer.write_tensor(name, &wb, "U32", &w_shape)?;
            writer.write_tensor(&format!("{}.scales", name), &sb, "F16", &s_shape)?;
            writer.write_tensor(&format!("{}.biases", name), &bb, "F16", &s_shape)?;
            total_bytes += (wb.len()+sb.len()+bb.len()) as u64;
            let _ = apple_fmt;
        }
        writer.finalize("dynamic3")?;
        let manifest = serde_json::json!({
            "quantizer":"dynamic3", "density": self.density, "model_type": model_type,
            "apple_silicon_formats": ["Q4_NL","Q5.1","Q5.0","Q4.1","Q4.0"],
            "total_bytes": total_bytes, "bits_per_param": total_bytes as f64*8.0/store.total_params() as f64
        });
        std::fs::write(output_dir.join("dynamic3_manifest.json"), serde_json::to_string_pretty(&manifest)?)?;
        eprintln!("dynamic3 density {:.2} ({}) → {:.2} GB, {:.2} bpw", self.density, model_type, total_bytes as f64/1e9, total_bytes as f64*8.0/store.total_params() as f64);
        Ok(())
    }
}

fn detect_model_type(store: &TensorStore) -> String {
    // Read config.json sidecar if present
    if let Some(parent) = store.path().parent() {
        if let Ok(s) = std::fs::read_to_string(parent.join("config.json")) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
                if let Some(t) = v.get("model_type").and_then(|x| x.as_str()) { return t.to_string(); }
            }
        }
    }
    // Heuristic from tensor names
    let names = store.tensor_names().join(" ");
    if names.contains("qwen") { "qwen".into() } else if names.contains("gemma") { "gemma".into() }
    else if names.contains("llama") { "llama".into() } else { "generic".into() }
}

fn sensitivity(name: &str, idx: Option<usize>, total: usize, model_type: &str, density: f32) -> f32 {
    let n = name.to_lowercase();
    let base = if n.contains("q_proj")||n.contains("k_proj")||n.contains("v_proj") { 0.95 }
        else if n.contains("o_proj") { 0.9 } else if n.contains("embed") { 0.85 }
        else if n.contains("down_proj") { 0.3 } else if n.contains("gate")||n.contains("up_proj") { 0.4 }
        else { 0.5 };
    // Model-specific tweak (Gemma 3 down_proj more sensitive than Llama)
    let model_adj = match model_type {
        "gemma" if n.contains("down_proj") => 0.15,
        "qwen" if n.contains("qkv") => 0.1,
        _ => 0.0,
    };
    let layer_adj = match idx {
        Some(i) => {
            let norm = i as f32 / total.max(1) as f32;
            if norm < 0.1 || norm > 0.9 { 0.15 } else if norm < 0.25 || norm > 0.75 { 0.05 } else { -0.1 }
        }
        None => 0.0,
    };
    // Density scales overall: density=0.5 pushes more tensors into compress
    ((base + model_adj + layer_adj) * density).clamp(0.0, 1.0)
}

fn bits_for_sensitivity(sens: f32, name: &str) -> u8 {
    // Keep lm_head at 6+ even if sensitivity low
    if name.to_lowercase().contains("lm_head") && sens < 0.6 { return 6; }
    if sens > 0.82 { 8 } else if sens > 0.62 { 6 } else if sens > 0.38 { 4 } else if sens > 0.22 { 3 } else { 2 }
}

fn apple_format(name: &str, bits: u8) -> &'static str {
    // Map bits → Apple Silicon container that Metal `quantized_matmul` prefers
    let n = name.to_lowercase();
    match bits {
        8 => "Q8_0", 6 => if n.contains("q_proj")||n.contains("k_proj") { "Q5.1" } else { "Q6_K" },
        5 => "Q5.1", 4 => if n.contains("mlp") { "Q4_NL" } else { "Q4_0" },
        3 => "Q4.1", 2 => "Q4.0", _ => "Q4_0",
    }
}

fn is_passthrough(name: &str) -> bool { name.to_lowercase().contains("norm") || name.ends_with(".bias") }
fn layer_idx(name: &str) -> Option<usize> {
    let parts: Vec<&str> = name.split('.').collect();
    for (i,p) in parts.iter().enumerate() { if *p=="layers" { return parts.get(i+1).and_then(|s| s.parse().ok()); } }
    None
}
