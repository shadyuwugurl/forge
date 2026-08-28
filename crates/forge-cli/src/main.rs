use std::path::PathBuf;
use clap::{Parser, Subcommand};

mod commands;

#[derive(Parser)]
#[command(
    name = "forge",
    version,
    about = "Unified model merging, quantization, and evaluation tool for Apple Silicon"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose logging
    #[arg(long, short)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Show model information (tensors, params, dtype, arch)
    Info {
        /// Model path (local or HuggingFace ID)
        model: String,
    },

    /// Search for models on HuggingFace Hub
    Search {
        /// Search query
        query: String,
        /// Maximum results
        #[arg(long, default_value = "10")]
        limit: usize,
    },

    /// Download a model from HuggingFace Hub
    Download {
        /// Model ID (e.g., meta-llama/Llama-3.1-8B-Instruct)
        model: String,
        /// Output directory
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Merge models using a config file or CLI args
    Merge {
        /// Path to merge config YAML
        #[arg(short, long)]
        config: Option<PathBuf>,
        /// Model paths to merge
        #[arg(short, long, num_args = 2..)]
        models: Option<Vec<PathBuf>>,
        /// Merge method (linear, slerp, ties, dare, della, passthrough, darwin, frankenmerge)
        #[arg(short, long)]
        method: Option<String>,
        /// Output directory
        #[arg(short, long)]
        output: PathBuf,
        /// Interpolation factor for SLERP
        #[arg(long)]
        t: Option<f32>,
        /// Darwin generations (if using darwin method)
        #[arg(long, default_value = "30")]
        generations: usize,
        /// Darwin population size
        #[arg(long, default_value = "40")]
        population: usize,
    },

    /// Quantize a model
    #[command(visible_alias = "quant")]
    Quantize {
        /// Model path
        model: String,
        /// Quantization method (jang, dynamic3, apex, btl4, mixed)
        #[arg(short, long)]
        method: String,
        /// Profile/tier for the method
        #[arg(short, long)]
        profile: Option<String>,
        /// Output directory
        #[arg(short, long)]
        output: PathBuf,
        /// Target density (for dynamic3)
        #[arg(long)]
        density: Option<f32>,
    },

    /// Evaluate a model on benchmarks and evals
    Eval {
        /// Model path
        model: String,
        /// Benchmarks to run (comma-separated: hella,mmlu,arc,gsm8k,gpqa)
        #[arg(short, long)]
        benchmarks: Option<String>,
        /// Evals to run (comma-separated: ace,swe,terminal,gaia,hle)
        #[arg(short, long)]
        evals: Option<String>,
        /// Original model for comparison (optional)
        #[arg(long)]
        original: Option<PathBuf>,
    },

    /// Fuse adapters into a base model
    Fuse {
        /// Base model path
        base: PathBuf,
        /// Directory of adapters/LoRAs
        #[arg(short, long)]
        adapters: PathBuf,
        /// Output directory
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Extract adapter / training data from a fine-tuned model
    Extract {
        /// Fine-tuned model path
        model: PathBuf,
        /// Base model path
        base: PathBuf,
        /// Output path for extracted data
        #[arg(short, long)]
        output: PathBuf,
        /// Extraction method (lora, weight-diff, activation-probe, distill)
        #[arg(short, long, default_value = "lora")]
        method: String,
        /// LoRA rank (for lora method)
        #[arg(long, default_value = "16")]
        rank: usize,
        /// Calibration data file (JSONL) for activation-probe/distill
        #[arg(long)]
        calib: Option<String>,
        /// Teacher model path (for distill method)
        #[arg(long)]
        teacher: Option<String>,
    },

    /// Train a model with LoRA/QLoRA/DoRA/GRPO
    Train {
        /// Base model path
        model: String,
        /// Training dataset (JSONL)
        dataset: String,
        /// Output directory
        #[arg(short, long)]
        output: PathBuf,
        /// Training method (lora, qlora, dora, grpo, dapo)
        #[arg(short, long, default_value = "lora")]
        method: String,
        /// LoRA rank
        #[arg(long, default_value = "16")]
        rank: usize,
        /// LoRA alpha
        #[arg(long, default_value = "32.0")]
        alpha: f32,
        /// Learning rate
        #[arg(long, default_value = "2e-4")]
        lr: f32,
        /// Number of epochs
        #[arg(long, default_value = "3")]
        epochs: usize,
        /// Batch size
        #[arg(long, default_value = "4")]
        batch_size: usize,
    },

    /// Launch terminal UI
    Tui,

    /// Launch desktop GUI
    Gui,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    if cli.verbose {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();
    } else {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    }

    match cli.command {
        Commands::Info { model } => commands::info::run(&model),
        Commands::Search { query, limit } => {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(commands::search::run(&query, limit))
        }
        Commands::Download { model, output } => {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(commands::download::run(&model, output.as_deref()))
        }
        Commands::Merge { config, models, method, output, t, generations, population } => {
            commands::merge::run(config.as_deref(), models.as_deref(), method.as_deref(), &output, t, generations, population)
        }
        Commands::Quantize { model, method, profile, output, density } => {
            commands::quantize::run(&model, &method, profile.as_deref(), &output, density)
        }
        Commands::Eval { model, benchmarks, evals, original } => {
            commands::eval::run(&model, benchmarks.as_deref(), evals.as_deref(), original.as_deref())
        }
        Commands::Fuse { base, adapters, output } => {
            commands::fuse::run(&base, &adapters, &output)
        }
        Commands::Extract { model, base, output, method, rank, calib, teacher } => {
            commands::extract::run(&model, &base, &output, rank, Some(method), calib, teacher)
        }
        Commands::Train { model, dataset, output, method, rank, alpha, lr, epochs, batch_size } => {
            commands::train::run(&model, &dataset, &output, &method, rank, alpha, lr, epochs, batch_size)
        }
        Commands::Tui => {
            eprintln!("TUI not yet implemented — use `forge-tui` binary");
            Ok(())
        }
        Commands::Gui => {
            eprintln!("GUI not yet implemented — use `forge-gui` binary");
            Ok(())
        }
    }
}
