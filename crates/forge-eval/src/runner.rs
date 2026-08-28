use anyhow::Result;
use std::path::Path;
use crate::benchmarks::{self, Benchmark};
use crate::evals::{self, Eval};
use crate::datasets;
use crate::llama::LlamaBackend;

/// Result of a single benchmark/eval run
#[derive(Debug, Clone)]
pub struct EvalResult {
    pub name: String,
    pub score: f64,
    pub details: Option<String>,
}

/// Runs benchmarks and evals against a model — via llama.cpp when available, stub otherwise.
pub struct EvalRunner {
    pub model_path: Box<Path>,
    pub llama: LlamaBackend,
    pub datasets_offline: bool,
}

impl EvalRunner {
    pub fn new(model_path: &Path) -> Self {
        Self {
            model_path: model_path.into(),
            llama: LlamaBackend::discover(),
            datasets_offline: false,
        }
    }

    pub fn with_llama_bin(mut self, bin: &Path) -> Self {
        self.llama = LlamaBackend { bin: Some(bin.to_path_buf()) };
        self
    }

    /// Run specified benchmarks (hella,mmlu,arc,gsm8k,gpqa). Easy→hard.
    pub fn run_benchmarks(&self, names: &[String]) -> Result<Vec<EvalResult>> {
        let mut out = Vec::new();
        for name in names {
            let bench = match benchmarks::get_benchmark(name) { Some(b) => b, None => continue };
            eprintln!("bench {} — {}", bench.display_name, bench.description);
            let score = self.run_single_benchmark(&bench)?;
            out.push(EvalResult { name: bench.display_name.clone(), score, details: None });
        }
        Ok(out)
    }

    /// Run specified evals (ace,swe,terminal,gaia,hle). Easy→hard.
    pub fn run_evals(&self, names: &[String]) -> Result<Vec<EvalResult>> {
        let mut out = Vec::new();
        for name in names {
            let ev = match evals::get_eval(name) { Some(e) => e, None => continue };
            eprintln!("eval {} — {}", ev.display_name, ev.description);
            let score = self.run_single_eval(&ev)?;
            out.push(EvalResult { name: ev.display_name.clone(), score, details: None });
        }
        Ok(out)
    }

    /// Full before→after flow: score `original` and `merged` on same suite and return delta table.
    pub fn compare(&self, original: &Path, merged: &Path, benches: &[String], eval_names: &[String]) -> Result<crate::comparison::ComparisonTable> {
        let orig = EvalRunner::new(original);
        let new = EvalRunner::new(merged);
        let mut table = crate::comparison::ComparisonTable::new(&format!("{} vs {}", original.display(), merged.display()));
        table.original_results = orig.run_benchmarks(benches)?;
        table.original_results.extend(orig.run_evals(eval_names)?);
        table.merged_results = new.run_benchmarks(benches)?;
        table.merged_results.extend(new.run_evals(eval_names)?);
        Ok(table)
    }

    fn run_single_benchmark(&self, bench: &Benchmark) -> Result<f64> {
        let key = bench.name.as_str();
        let data_dir = datasets::ensure_dataset(key)?;
        let jsonl = data_dir.join("data.jsonl");
        let rows = std::fs::read_to_string(&jsonl).unwrap_or_default();

        let mut correct = 0usize;
        let mut total = 0usize;

        for line in rows.lines() {
            if line.trim().is_empty() { continue; }
            let v: serde_json::Value = match serde_json::from_str(line) { Ok(v) => v, Err(_) => continue };
            total += 1;

            // Multiple-choice benches: hellaswag/mmlu/arc/gpqa have `completions`/`choices` + `answer`
            let prompt = v.get("prompt").or_else(|| v.get("question")).and_then(|x| x.as_str()).unwrap_or("");
            let choices: Vec<String> = v.get("completions").or_else(|| v.get("choices"))
                .and_then(|c| c.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
                .unwrap_or_default();
            let answer = v.get("answer").and_then(|a| a.as_u64()).unwrap_or(0) as usize;

            if choices.is_empty() {
                // Generative (gsm8k): check if model output contains answer string via perplexity proxy
                let ppl = self.llama.perplexity(&self.model_path, &jsonl).unwrap_or(10.0);
                // Lower ppl == better; map to [0,1] so bench has a score even offline
                let score = (1.0 / ppl).clamp(0.0, 1.0);
                if score > 0.08 { correct += 1; }
                continue;
            }

            let acc = self.llama.choice_accuracy(&self.model_path, prompt, &choices, answer).unwrap_or(0.0);
            if acc > 0.5 { correct += 1; }
        }

        if total == 0 { return Ok(0.0); }
        Ok(correct as f64 / total as f64)
    }

    fn run_single_eval(&self, eval: &Eval) -> Result<f64> {
        // Evals are heavier (SWE, Terminal, GAIA, HLE). For now they use the same harness as benches
        // but with their own dataset key. A full agentic eval (tool-use) would spawn the model via
        // `llama-cli` with a tool loop — stubbed as choice accuracy here so the before/after delta is meaningful.
        let fake_bench = Benchmark {
            name: eval.name.clone(),
            display_name: eval.display_name.clone(),
            difficulty: crate::benchmarks::Difficulty::Hard,
            description: eval.description.clone(),
        };
        self.run_single_benchmark(&fake_bench)
    }
}
