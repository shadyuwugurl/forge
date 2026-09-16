use std::path::Path;
use anyhow::{Result, Context};
use forge_core::{ModelProfile, TensorMeta, FamilyRegistry, TensorMap, tensor_map::AetherLayout};
use forge_io::TensorStore;

pub fn run(model: &Path, aether: bool) -> Result<()> {
    let (store_path, config_dir) = super::resolve_model(model);
    let store = TensorStore::open(&store_path)?;
    
    // Collect tensor names and shapes for profile detection
    let mut tensor_shapes = Vec::new();
    for name in store.tensor_names() {
        if let Ok(meta) = store.tensor_meta(name) {
            tensor_shapes.push((name.to_string(), meta.shape));
        }
    }
    
    let registry = FamilyRegistry::builtin();
    let profile = ModelProfile::detect(&config_dir, &tensor_shapes)?;
    let tensor_names: Vec<String> = tensor_shapes.iter().map(|(n, _)| n.clone()).collect();
    let tensor_map = TensorMap::build(&profile, &tensor_names, &registry)?;
    
    // Print model profile
    eprintln!("=== Model Profile ===");
    eprintln!("Family: {}", profile.family);
    eprintln!("Layers: {}", profile.num_layers);
    eprintln!("Hidden size: {}", profile.hidden_size);
    eprintln!("Attention heads: {}", profile.num_heads);
    eprintln!("KV heads: {}", profile.kv_heads);
    eprintln!("Head dim: {}", profile.head_dim);
    eprintln!("Intermediate size: {}", profile.intermediate_size);
    eprintln!("Vocab size: {}", profile.vocab_size);
    eprintln!("Tie embeddings: {}", profile.tie_embeddings);
    eprintln!("Attention type: {:?}", profile.attention);
    eprintln!("FFN type: {:?}", profile.ffn);
    eprintln!("Norm type: {:?}", profile.norm);
    eprintln!("Positional type: {:?}", profile.positional);
    eprintln!("Tokenizer type: {:?}", profile.tokenizer);
    eprintln!("MOE experts: {:?}", profile.moe_experts);
    eprintln!("Quirks: {:?}", profile.quirks);
    
    // Print tensor map stats
    eprintln!("\n=== Tensor Map ===");
    eprintln!("Total slots: {}", tensor_map.slots.len());
    eprintln!("Mapped tensors: {}", tensor_map.by_raw.len());
    eprintln!("Common tensors: {}", tensor_map.common.len());
    
    // Print per-slot summary
    for (slot_key, entry) in &tensor_map.slots {
        if !entry.raw_names.is_empty() {
            eprintln!("  {} -> {} tensors{}", slot_key, entry.raw_names.len(), if entry.fused { " (fused)" } else { "" });
        }
    }
    
    // Print unmapped tensors
    let mapped: std::collections::HashSet<_> = tensor_map.by_raw.keys().collect();
    let unmapped: Vec<_> = tensor_names.iter().filter(|n| !mapped.contains(*n)).collect();
    if !unmapped.is_empty() {
        eprintln!("\nUnmapped tensors ({}):", unmapped.len());
        for n in unmapped.iter().take(20) {
            eprintln!("  {}", n);
        }
        if unmapped.len() > 20 {
            eprintln!("  ... and {} more", unmapped.len() - 20);
        }
    }

    if aether {
        eprintln!("\n=== Aether Latin-square layout ===");
        if profile.family != "aether" {
            eprintln!("  Note: detected family is '{}', not aether — showing reference 7x7 map.", profile.family);
        }
        let layout = AetherLayout::aether49();
        for (layer, t) in layout.slot_map() {
            eprintln!("  layer {:>2} -> {}", layer, forge_core::tensor_map::AETHER_ATTN_TYPES[t]);
        }
    }

    Ok(())
}