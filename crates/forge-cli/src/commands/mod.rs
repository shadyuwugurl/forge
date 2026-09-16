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
/// Accepts a model directory (uses `model.safetensors` + dir-level
/// `config.json`) or a direct `.safetensors` file (uses the parent dir for
/// `config.json`). Pass-through for anything else.
pub fn resolve_model(path: &std::path::Path) -> (std::path::PathBuf, std::path::PathBuf) {
    if path.is_dir() {
        (path.join("model.safetensors"), path.to_path_buf())
    } else {
        let dir = path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::path::PathBuf::from("."));
        (path.to_path_buf(), dir)
    }
}
