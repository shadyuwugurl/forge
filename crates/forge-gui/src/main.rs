//! forge-gui — Tauri + Vue desktop app
//!
//! Build the full Tauri app:
//!   cd crates/forge-gui/frontend && npm install && npm run build
//!   cd crates/forge-gui && cargo build --features tauri --release
//!
//! Without the `tauri` feature this binary is a headless stub that prints help
//! and is used for `cargo check` / CI where webview deps are unavailable.

#[cfg(feature = "tauri")]
mod tauri_app {
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize)]
    pub struct MergeRequest {
        pub models: Vec<String>,
        pub method: String,
        pub output: String,
    }

    #[tauri::command]
    fn merge_models(req: MergeRequest) -> Result<String, String> {
        // Dispatch to forge-merge via internal API (stub — wire in Phase 2)
        Ok(format!("merge {:?} via {} -> {}", req.models, req.method, req.output))
    }

    #[tauri::command]
    fn model_info(path: String) -> Result<String, String> {
        Ok(format!("info for {}", path))
    }

    pub fn run() {
        tauri::Builder::default()
            .invoke_handler(tauri::generate_handler![merge_models, model_info])
            .run(tauri::generate_context!())
            .expect("error while running tauri application");
    }
}

fn main() -> anyhow::Result<()> {
    #[cfg(feature = "tauri")]
    {
        tauri_app::run();
        return Ok(());
    }

    #[cfg(not(feature = "tauri"))]
    {
        eprintln!("forge-gui: Tauri GUI not built.");
        eprintln!("  To build the desktop app:");
        eprintln!("    cd crates/forge-gui/frontend && npm install && npm run build");
        eprintln!("    cargo run -p forge-gui --features tauri");
        eprintln!("");
        eprintln!("Available CLI alternatives:");
        eprintln!("  forge tui        # terminal UI (ratatui)");
        eprintln!("  forge --help     # CLI");
        Ok(())
    }
}
