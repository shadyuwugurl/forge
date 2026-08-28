use std::path::Path;
use anyhow::Result;
use forge_eval::{EvalRunner, ComparisonTable};

pub fn run(model: &str, benchmarks: Option<&str>, evals: Option<&str>, original: Option<&Path>) -> Result<()> {
    let bench_names: Vec<String> = benchmarks.map(|b| b.split(',').map(|s| s.trim().to_string()).collect()).unwrap_or_else(|| vec!["hella".into(),"mmlu".into(),"arc".into(),"gsm8k".into(),"gpqa".into()]);
    let eval_names: Vec<String> = evals.map(|e| e.split(',').map(|s| s.trim().to_string()).collect()).unwrap_or_else(|| vec!["ace".into(),"swe".into(),"terminal".into(),"gaia".into(),"hle".into()]);

    if let Some(orig) = original {
        let runner = EvalRunner::new(std::path::Path::new(model));
        let table = runner.compare(orig, std::path::Path::new(model), &bench_names, &eval_names)?;
        table.print();
    } else {
        let runner = EvalRunner::new(std::path::Path::new(model));
        let mut results = runner.run_benchmarks(&bench_names)?;
        results.extend(runner.run_evals(&eval_names)?);
        for r in &results { eprintln!("  {}: {:.2}%", r.name, r.score * 100.0); }
    }
    Ok(())
}
