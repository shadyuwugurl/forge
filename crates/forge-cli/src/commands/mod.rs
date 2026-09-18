pub mod info;
pub mod search;
pub mod download;
pub mod merge;
pub mod quantize;
pub mod eval;
pub mod fuse;
pub mod extract;
pub mod train;
pub mod imatrix;
pub mod inspect;
pub mod surgery;

/// Resolve a user-supplied model path to `(store_file, config_dir)`.
///
/// Accepts:
/// - a model directory with `model.safetensors` (single file), or
/// - a sharded dir with `model.safetensors.index.json` + `model-*.safetensors`, or
/// - a direct `.safetensors` file (uses the parent dir for `config.json`).
pub fn resolve_model(path: &std::path::Path) -> (std::path::PathBuf, std::path::PathBuf) {
    if path.is_dir() {
        // Single-file layout first
        let single = path.join("model.safetensors");
        if single.exists() {
            return (single, path.to_path_buf());
        }
        // Sharded layout: pick first shard for TensorStore open (orchestrator
        // currently streams one file; full sharded streaming is TODO).
        if let Some(shard) = resolve_model_shard(path) {
            return (shard, path.to_path_buf());
        }
        // Default to single path so error messages point at the expected file
        (single, path.to_path_buf())
    } else {
        let dir = path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::path::PathBuf::from("."));
        (path.to_path_buf(), dir)
    }
}

/// Find the first `model-*.safetensors` shard in a directory, sorted.
pub fn resolve_model_shard(dir: &std::path::Path) -> Option<std::path::PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut shards: Vec<std::path::PathBuf> = entries
        .filter_map(|e| e.ok().map(|x| x.path()))
        .filter(|p| {
            p.is_file()
                && p.extension().map(|x| x == "safetensors").unwrap_or(false)
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("model-") || n.starts_with("pytorch_model-"))
                    .unwrap_or(false)
        })
        .collect();
    shards.sort();
    shards.into_iter().next()
}

/// List all safetensors shards in a model dir (single + sharded).
pub fn list_model_shards(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    if dir.join("model.safetensors").exists() {
        return vec![dir.join("model.safetensors")];
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return vec![],
    };
    let mut shards: Vec<std::path::PathBuf> = entries
        .filter_map(|e| e.ok().map(|x| x.path()))
        .filter(|p| {
            p.is_file() && p.extension().map(|x| x == "safetensors").unwrap_or(false)
        })
        .collect();
    shards.sort();
    shards
}
