use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[cfg(feature = "tauri")]
mod tauri_app {
    use crate::commands::{MergeRequest, QuantizeRequest, EvalRequest, ModelInfoRequest, ExtractRequest, TrainRequest};
    use crate::forge_core::config::{MergeConfig, QuantMethod, EvalConfig, TrainConfig};
    use crate::forge_merge::{execute_merge, MergeOptions};
    use crate::forge_quant::{JangQuantizer, Dynamic3Quantizer, ApexQuantizer, MixedPrecisionQuantizer, MixedStrategy};
    use crate::forge_eval::{EvalRunner, ComparisonTable};
    use crate::forge_train::{Trainer, TrainConfig, TrainMethod};
    use crate::forge_io::TensorStore;
    use std::path::PathBuf;

    #[derive(Serialize, Deserialize)]
    pub struct MergeRequest {
        pub models: Vec<String>,
        pub method: String,
        pub output: String,
        pub config: Option<String>,
    }

    #[derive(Serialize, Deserialize)]
    pub struct QuantizeRequest {
        pub model: String,
        pub method: String,
        pub profile: Option<String>,
        pub output: String,
        pub density: Option<f32>,
    }

    #[derive(Serialize, Deserialize)]
    pub struct EvalRequest {
        pub model: String,
        pub benchmarks: Option<String>,
        pub evals: Option<String>,
        pub original: Option<String>,
    }

    #[derive(Serialize, Deserialize)]
    pub struct ModelInfoRequest {
        pub model: String,
    }

    #[derive(Serialize, Deserialize)]
    pub struct ExtractRequest {
        pub model: String,
        pub base: String,
        pub output: String,
        pub method: String,
        pub rank: usize,
    }

    #[derive(Serialize, Deserialize)]
    pub struct TrainRequest {
        pub model: String,
        pub dataset: String,
        pub output: String,
        pub method: String,
        pub rank: usize,
        pub alpha: f32,
        pub lr: f32,
        pub epochs: usize,
        pub batch_size: usize,
    }

    #[tauri::command]
    async fn merge_models(req: MergeRequest) -> Result<String, String> {
        let output_path = std::path::PathBuf::from(&req.output);
        std::fs::create_dir_all(&output_path).map_err(|e| e.to_string())?;

        let writer = crate::forge_io::StreamingWriter::new(&output_path, 5 * 1024 * 1024 * 1024)
            .map_err(|e| e.to_string())?;

        let stores: Vec<crate::forge_io::TensorStore> = req.models.iter()
            .map(|m| crate::forge_io::TensorStore::open(std::path::Path::new(m)))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;

        let store_refs: Vec<&crate::forge_io::TensorStore> = stores.iter().collect();

        let options = MergeOptions {
            output_dtype: crate::forge_core::DType::BF16,
            base_model_dir: None,
            quiet: false,
            verbose: true,
        };

        // For now, use simple linear merge
        let merge_op = crate::forge_merge::LinearMerge::new(
            req.models.iter().zip(stores.iter()).map(|(m, s)| {
                let weight = if m.contains("base") { 0.5 } else { 0.5 };
                (s, weight)
            }).collect(),
            true,
        );

        crate::forge_merge::orchestrator::execute_merge(&merge_op, &mut writer, &options)
            .map_err(|e| e.to_string())?;

        writer.finalize("merged").map_err(|e| e.to_string())?;

        Ok(format!("Merged {} models via {} -> {}", req.models.len(), req.method, req.output))
    }

    #[tauri::command]
    async fn quantize_model(req: QuantizeRequest) -> Result<String, String> {
        let store = crate::forge_io::TensorStore::open(std::path::Path::new(&req.model))
            .map_err(|e| e.to_string())?;
        std::fs::create_dir_all(&req.output).map_err(|e| e.to_string())?;

        match req.method.as_str() {
            "jang" => {
                let profile = req.profile.unwrap_or_else(|| "JANG_2L".to_string());
                let q = crate::forge_quant::JangQuantizer::new(&profile, crate::forge_quant::jang::JangFormat::Mlx);
                q.quantize(&store, std::path::Path::new(&req.output)).map_err(|e| e.to_string())?;
            }
            "dynamic3" | "dynamic" => {
                let d = req.density.unwrap_or(0.5);
                let q = crate::forge_quant::Dynamic3Quantizer::new(d, true);
                q.quantize(&store, std::path::Path::new(&req.output)).map_err(|e| e.to_string())?;
            }
            "apex" => {
                let tier = req.profile.unwrap_or_else(|| "balanced".to_string());
                crate::forge_quant::ApexQuantizer::new(&tier).quantize(&store, std::path::Path::new(&req.output)).map_err(|e| e.to_string())?;
            }
            "gguf" => {
                let qtype = req.profile.and_then(|p| crate::forge_quant::GGUFQuantType::from_str(&p))
                    .unwrap_or(crate::forge_quant::GGUFQuantType::Q4_K_M);
                let mut writer = crate::forge_quant::GgufWriter::create(std::path::Path::new(&req.output)).map_err(|e| e.to_string())?;
                writer.set_metadata("general.architecture", serde_json::Value::String("generic".into()));
                writer.set_metadata("general.name", serde_json::Value::String("forge-quantized".into()));
                writer.write_quantized(&store, qtype).map_err(|e| e.to_string())?;
            }
            _ => return Err(format!("Unknown quant method: {}", req.method)),
        }
        Ok(format!("Quantized {} with {} -> {}", req.model, req.method, req.output))
    }

    #[tauri::command]
    async fn eval_model(req: EvalRequest) -> Result<String, String> {
        let runner = crate::forge_eval::EvalRunner::new(std::path::Path::new(&req.model));
        
        let bench_names: Vec<String> = req.benchmarks
            .map(|b| b.split(',').map(|s| s.trim().to_string()).collect())
            .unwrap_or_else(|| vec!["hella".into(),"mmlu".into(),"arc".into(),"gsm8k".into(),"gpqa".into()]);

        let eval_names: Vec<String> = req.evals
            .map(|e| e.split(',').map(|s| s.trim().to_string()).collect())
            .unwrap_or_else(|| vec!["ace".into(),"swe".into(),"terminal".into(),"gaia".into(),"hle".into()]);

        let mut results = runner.run_benchmarks(&bench_names).map_err(|e| e.to_string())?;
        results.extend(runner.run_evals(&eval_names).map_err(|e| e.to_string())?);

        let output = if let Some(orig) = req.original {
            let table = runner.compare(std::path::Path::new(&orig), std::path::Path::new(&req.model), &bench_names, &eval_names)
                .map_err(|e| e.to_string())?;
            format!("{:?}", table)
        } else {
            format!("{:?}", results)
        };
        Ok(output)
    }

    #[tauri::command]
    async fn model_info(req: ModelInfoRequest) -> Result<String, String> {
        let store = crate::forge_io::TensorStore::open(std::path::Path::new(&req.model))
            .map_err(|e| e.to_string())?;
        
        let mut info = format!("Model: {}\n", req.model);
        info.push_str(&format!("Tensors: {}\n", store.tensor_names().len()));
        info.push_str(&format!("Parameters: {} ({:.1}B)\n", store.total_params(), store.total_params() as f64 / 1e9));
        Ok(info)
    }

    #[tauri::command]
    async fn extract_adapter(req: ExtractRequest) -> Result<String, String> {
        let base_store = crate::forge_io::TensorStore::open(std::path::Path::new(&req.base))
            .map_err(|e| e.to_string())?;
        let model_store = crate::forge_io::TensorStore::open(std::path::Path::new(&req.model))
            .map_err(|e| e.to_string())?;

        let adapter = crate::forge_train::LoraExtractor::extract(&base_store, &model_store, req.rank)
            .map_err(|e| e.to_string())?;

        std::fs::create_dir_all(&req.output).map_err(|e| e.to_string())?;
        adapter.save_peft(std::path::Path::new(&req.output)).map_err(|e| e.to_string())?;

        Ok(format!("Extracted LoRA (rank={}) -> {}", req.rank, req.output))
    }

    #[tauri::command]
    async fn train_model(req: TrainRequest) -> Result<String, String> {
        let method = match req.method.as_str() {
            "lora" => crate::forge_train::TrainMethod::LoRA,
            "qlora" => crate::forge_train::TrainMethod::QLoRA,
            "dora" => crate::forge_train::TrainMethod::DoRA,
            "grpo" => crate::forge_train::TrainMethod::GRPO,
            "dapo" => crate::forge_train::TrainMethod::DAPO,
            _ => return Err(format!("Unknown method: {}", req.method)),
        };

        let config = TrainConfig {
            model_path: req.model,
            dataset: req.dataset,
            output: req.output.clone(),
            rank: req.rank,
            alpha: req.alpha,
            learning_rate: req.lr,
            epochs: req.epochs,
            batch_size: req.batch_size,
            method,
            quant: None,
        };

        let mut trainer = crate::forge_train::Trainer::new(config).map_err(|e| e.to_string())?;
        trainer.train().map_err(|e| e.to_string())?;

        Ok(format!("Training complete -> {}", req.output))
    }

    pub fn run() {
        tauri::Builder::default()
            .invoke_handler(tauri::generate_handler![
                merge_models,
                quantize_model,
                eval_model,
                model_info,
                extract_adapter,
                train_model,
            ])
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