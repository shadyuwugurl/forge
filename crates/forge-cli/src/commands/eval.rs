use std::path::Path;
use anyhow::Result;
use forge_eval::{EvalRunner, ComparisonTable};

pub fn run(model: &str, benchmarks: Option<&str>, evals: Option<&str>, original: Option<&Path>) -> Result<()> {
    let runner = EvalRunner::new(std::path::Path::new(model));

    let bench_names: Vec<String> = benchmarks
        .map(|b| b.split(',').map(String::from).collect())
        .unwrap_or_default();

    let eval_names: Vec<String> = evals
        .map(|e| e.split(',').map(String::from).collect())
        .unwrap_or_default();

    eprintln!("Evaluating model: {}", model);

    let mut results = runner.run_benchmarks(&bench_names)?;
    results.extend(runner.run_evals(&eval_names)?);

    // If original model provided, compare
    if let Some(orig_path) = original {
        let orig_runner = EvalRunner::new(orig_path);
        let orig_results = orig_runner.run_benchmarks(&bench_names)?;
        // TODO: Merge orig_results with eval results into comparison table
    }

    for r in &results {
        eprintln!("  {}: {:.2}%", r.name, r.score * 100.0);
    }

    Ok(())
}
