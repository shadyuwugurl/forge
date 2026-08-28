use crate::runner::EvalResult;

/// Comparison table showing before/after scores with deltas
pub struct ComparisonTable {
    pub model_name: String,
    pub original_results: Vec<EvalResult>,
    pub merged_results: Vec<EvalResult>,
}

impl ComparisonTable {
    pub fn new(model_name: &str) -> Self {
        Self {
            model_name: model_name.to_string(),
            original_results: Vec::new(),
            merged_results: Vec::new(),
        }
    }

    pub fn print(&self) {
        let divider = "-".repeat(60);
        eprintln!("{}", divider);
        eprintln!("Evaluation Results: {}", self.model_name);
        eprintln!("{}", divider);
        eprintln!("{:<20} {:>10} {:>10} {:>10}", "Metric", "Original", "Merged", "Delta");
        eprintln!("{}", divider);

        for orig in &self.original_results {
            if let Some(merged) = self.merged_results.iter().find(|m| m.name == orig.name) {
                let delta = merged.score - orig.score;
                let delta_str = if delta >= 0.0 {
                    format!("+{:.2}%", delta * 100.0)
                } else {
                    format!("{:.2}%", delta * 100.0)
                };
                eprintln!("{:<20} {:>9.2}% {:>9.2}% {:>10}",
                    orig.name, orig.score * 100.0, merged.score * 100.0, delta_str);
            }
        }

        eprintln!("{}", divider);
    }
}
