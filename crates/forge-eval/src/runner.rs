use anyhow::Result;
use std::path::Path;
use crate::benchmarks::{Benchmark, self};
use crate::evals::{Eval, self};

/// Result of a single benchmark/eval run
#[derive(Debug, Clone)]
pub struct EvalResult {
    pub name: String,
    pub score: f64,
    pub details: Option<String>,
}

/// Runs benchmarks and evals against a model
pub struct EvalRunner {
    pub model_path: Box<Path>,
    pub llama_cpp_path: Option<Box<Path>>,
}

impl EvalRunner {
    pub fn new(model_path: &Path) -> Self {
        Self {
            model_path: model_path.into(),
            llama_cpp_path: None,
        }
    }

    /// Run specified benchmarks
    pub fn run_benchmarks(&self, benchmark_names: &[String]) -> Result<Vec<EvalResult>> {
        let mut results = Vec::new();

        for name in benchmark_names {
            if let Some(bench) = benchmarks::get_benchmark(name) {
                eprintln!("Running benchmark: {} ({})", bench.display_name, bench.description);

                // TODO: Actual benchmark execution via llama.cpp
                let score = self.run_single_benchmark(&bench)?;
                results.push(EvalResult {
                    name: bench.display_name,
                    score,
                    details: None,
                });
            }
        }

        Ok(results)
    }

    /// Run specified evals
    pub fn run_evals(&self, eval_names: &[String]) -> Result<Vec<EvalResult>> {
        let mut results = Vec::new();

        for name in eval_names {
            if let Some(eval) = evals::get_eval(name) {
                eprintln!("Running eval: {} ({})", eval.display_name, eval.description);

                // TODO: Actual eval execution
                let score = self.run_single_eval(&eval)?;
                results.push(EvalResult {
                    name: eval.display_name,
                    score,
                    details: None,
                });
            }
        }

        Ok(results)
    }

    fn run_single_benchmark(&self, bench: &Benchmark) -> Result<f64> {
        // Stub: in production, invoke llama.cpp with the model
        // and run the actual benchmark dataset
        Ok(0.0)
    }

    fn run_single_eval(&self, eval: &Eval) -> Result<f64> {
        // Stub: in production, invoke the eval framework
        Ok(0.0)
    }
}
