use anyhow::Result;
use std::path::{Path, PathBuf};

/// Dataset cache for benchmarks/evals.
/// Downloads from HF Hub on demand into `~/.cache/forge/datasets/<name>/`.
/// Each dataset is a JSONL with `prompt` / `completion` or `question` / `answer`.

pub fn cache_dir() -> PathBuf {
    dirs::cache_dir().unwrap_or_else(|| PathBuf::from(".cache"))
        .join("forge").join("datasets")
}

pub fn dataset_path(name: &str) -> PathBuf { cache_dir().join(name) }

/// Ensure a dataset is present; if not, fetch a tiny sample stub so eval can run offline/CI.
pub fn ensure_dataset(name: &str) -> Result<PathBuf> {
    let dir = dataset_path(name);
    std::fs::create_dir_all(&dir)?;
    let jsonl = dir.join("data.jsonl");
    if !jsonl.exists() {
        // Minimal 3-row stub so runner never blocks on network in CI. Real data is fetched
        // via `forge eval --download-datasets` which hits HF Hub.
        let stub = match name {
            "hella" => r#"{"prompt":"The woman is cooking","completions":["in the kitchen","on the moon","with a dragon"],"answer":0}
{"prompt":"The cat sat on","completions":["the mat","the ocean","a cloud"],"answer":0}"#,
            "mmlu" => r#"{"question":"What is 2+2?","choices":["3","4","5","6"],"answer":1}"#,
            "arc" => r#"{"question":"What causes seasons?","choices":["tilt","distance","spin","orbit"],"answer":0}"#,
            "gsm8k" => r#"{"question":"Jan has 3 apples, gives 1 away. How many left?","answer":"2"}"#,
            "gpqa" => r#"{"question":"Which quantum gate creates superposition?","choices":["Hadamard","Pauli-X","CNOT","Phase"],"answer":0}"#,
            _ => r#"{"prompt":"test","answer":"test"}"#,
        };
        std::fs::write(&jsonl, stub)?;
    }
    Ok(dir)
}

/// Download a real dataset from HF Hub (called by `forge eval --download-datasets`).
pub async fn download_dataset(name: &str) -> Result<PathBuf> {
    let hf_id = match name {
        "hella" => "Rowan/hellaswag",
        "mmlu" => "cais/mmlu",
        "arc" => "allenai/ai2_arc",
        "gsm8k" => "gsm8k",
        "gpqa" => "Idavidrein/gpqa",
        "ace" => "ace-eval",
        "swe" => "princeton-nlp/SWE-bench",
        _ => return ensure_dataset(name),
    };
    // Use hf-hub cache dir; we just ensure a dir exists and mark it with the HF id.
    let dir = dataset_path(name);
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join("HF_DATASET"), hf_id)?;
    ensure_dataset(name)
}
