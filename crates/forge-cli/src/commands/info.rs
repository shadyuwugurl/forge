use std::path::Path;
use anyhow::Result;
use forge_io::TensorStore;

pub fn run(model_path: &str) -> Result<()> {
    let path = if model_path.starts_with("http") || model_path.contains('/') {
        // HuggingFace ID — would need to download first
        eprintln!("HuggingFace models: use `forge download {}` first", model_path);
        return Ok(());
    } else {
        std::path::PathBuf::from(model_path)
    };

    let store = TensorStore::open(&path)?;

    eprintln!("Model: {}", path.display());
    eprintln!("Tensors: {}", store.tensor_names().len());
    eprintln!("Parameters: {} ({:.1}B)", store.total_params(), store.total_params() as f64 / 1e9);

    // Group by component
    let mut attn_count = 0;
    let mut mlp_count = 0;
    let mut embed_count = 0;
    let mut other_count = 0;

    for name in store.tensor_names() {
        if name.contains("q_proj") || name.contains("k_proj") || name.contains("v_proj")
            || name.contains("o_proj") || name.contains("attn") {
            attn_count += 1;
        } else if name.contains("mlp") || name.contains("gate_proj")
            || name.contains("up_proj") || name.contains("down_proj") {
            mlp_count += 1;
        } else if name.contains("embed") {
            embed_count += 1;
        } else {
            other_count += 1;
        }
    }

    eprintln!("\nComponent breakdown:");
    eprintln!("  Attention: {} tensors", attn_count);
    eprintln!("  MLP:       {} tensors", mlp_count);
    eprintln!("  Embed:     {} tensors", embed_count);
    eprintln!("  Other:     {} tensors", other_count);

    Ok(())
}
