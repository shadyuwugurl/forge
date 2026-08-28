use anyhow::Result;
use forge_io::{TensorStore, StreamingWriter};
use crate::jang::quantize_tensor;
use std::path::Path;

/// Mixed-precision / BTL4 Compact quantizer
///
/// - BTL4: block-wise 4b + compact codebook (like `bitsandbytes` BTL4), size-first but keeps quality floor
/// - Generic mixed: user-provided per-tensor bits; if none, auto-tier by param count (JangQ-style)

pub struct MixedPrecisionQuantizer { pub strategy: MixedStrategy, pub target_bpw: f32 }
pub enum MixedStrategy { ApexStyle, Btl4Compact, Generic }

impl MixedPrecisionQuantizer {
    pub fn new(strategy: MixedStrategy, target_bpw: f32) -> Self { Self { strategy, target_bpw } }

    pub fn quantize(&self, store: &TensorStore, output_dir: &Path, per_layer_bits: &[(String, u8)]) -> Result<()> {
        std::fs::create_dir_all(output_dir)?;
        let mut map: std::collections::HashMap<String, u8> = per_layer_bits.iter().cloned().collect();

        // BTL4 auto-plan if no per-tensor map given: hit target_bpw by sorting tensors by size
        if map.is_empty() && matches!(self.strategy, MixedStrategy::Btl4Compact|MixedStrategy::Generic) {
            let mut items: Vec<(String, usize)> = store.tensor_names().iter()
                .filter(|n| !n.to_lowercase().contains("norm") && !n.ends_with(".bias"))
                .map(|n| (n.to_string(), store.tensor_meta(n).map(|m| m.num_elements()).unwrap_or(0)))
                .collect();
            items.sort_by_key(|(_, sz)| std::cmp::Reverse(*sz));
            // Greedy assign: largest tensors get 2-3b until budget met
            let total: usize = items.iter().map(|(_,sz)| *sz).sum();
            let target_bits = (total as f64 * self.target_bpw as f64) as usize;
            let mut used = 0usize;
            for (name, sz) in &items {
                let bits = if used + sz*2 < target_bits { 2 } else if used + sz*3 < target_bits { 3 } else { 4 };
                map.insert(name.clone(), bits);
                used += sz * bits as usize;
            }
        }

        let mut writer = StreamingWriter::new(output_dir, 5*1024*1024*1024)?;
        let mut total_bytes: u64 = 0;

        for name in store.tensor_names() {
            if name.to_lowercase().contains("norm") || name.ends_with(".bias") {
                let meta = store.tensor_meta(name)?;
                let bytes = store.tensor_bytes(name)?;
                writer.write_tensor(name, bytes, "F16", &meta.shape)?;
                total_bytes += bytes.len() as u64;
                continue;
            }
            let meta = store.tensor_meta(name)?;
            let bits = match &self.strategy {
                MixedStrategy::ApexStyle => apex_bits(name),
                MixedStrategy::Btl4Compact => btl4_bits(name, &meta, &map),
                MixedStrategy::Generic => map.get(name).copied().unwrap_or(4),
            };
            let group_size = 64;
            let data = store.tensor_f32(name).unwrap_or_else(|_| vec![0.0; meta.num_elements()]);
            let (packed, scales, biases) = quantize_tensor(&data, bits, group_size, false)?;
            let in_f = meta.shape.last().copied().unwrap_or(meta.num_elements());
            let packed_f = (in_f * bits as usize + 31)/32;
            let mut w_shape = meta.shape.clone(); if let Some(l) = w_shape.last_mut() { *l = packed_f; }
            let n_groups = (in_f + group_size -1)/group_size;
            let mut s_shape = meta.shape.clone(); if let Some(l) = s_shape.last_mut() { *l = n_groups; }
            let wb: Vec<u8> = packed.iter().flat_map(|v| v.to_le_bytes()).collect();
            let sb: Vec<u8> = scales.iter().flat_map(|v| v.to_le_bytes()).collect();
            let bb: Vec<u8> = biases.iter().flat_map(|v| v.to_le_bytes()).collect();
            writer.write_tensor(name, &wb, "U32", &w_shape)?;
            writer.write_tensor(&format!("{}.scales", name), &sb, "F16", &s_shape)?;
            writer.write_tensor(&format!("{}.biases", name), &bb, "F16", &s_shape)?;
            total_bytes += (wb.len()+sb.len()+bb.len()) as u64;
        }
        writer.finalize("mixed")?;
        let manifest = serde_json::json!({
            "quantizer": match &self.strategy { MixedStrategy::ApexStyle=>"apex-style", MixedStrategy::Btl4Compact=>"btl4-compact", MixedStrategy::Generic=>"generic-mixed" },
            "target_bpw": self.target_bpw, "total_bytes": total_bytes, "bits_per_param": total_bytes as f64*8.0/store.total_params() as f64
        });
        std::fs::write(output_dir.join("mixed_manifest.json"), serde_json::to_string_pretty(&manifest)?)?;
        eprintln!("mixed {:?} target {:.2}bpw → {:.2} GB {:.2} bpw",
            match &self.strategy { MixedStrategy::ApexStyle=>"apex", MixedStrategy::Btl4Compact=>"btl4", MixedStrategy::Generic=>"generic" },
            self.target_bpw, total_bytes as f64/1e9, total_bytes as f64*8.0/store.total_params() as f64);
        Ok(())
    }
}

fn apex_bits(name: &str) -> u8 {
    let n = name.to_lowercase();
    if n.contains("expert") { 3 } else if n.contains("q_proj")||n.contains("k_proj")||n.contains("v_proj")||n.contains("attn") { 6 }
    else if n.contains("embed") { 4 } else { 4 }
}
fn btl4_bits(name: &str, meta: &forge_core::TensorMeta, map: &std::collections::HashMap<String,u8>) -> u8 {
    if let Some(b) = map.get(name) { return *b; }
    let sz: usize = meta.shape.iter().product();
    if sz > 1_000_000 { 3 } else if sz > 100_000 { 4 } else { 6 }
}
