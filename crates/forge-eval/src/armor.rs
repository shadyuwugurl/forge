//! Armoring post-merge gate: safety/correctness eval bundle + report JSON.
//!
//! V1 = gate definition + JSON report over the existing harness (`EvalRunner`);
//! no new safety science. A merged model passes when every bundled eval meets
//! the gate's minimum score.

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Eval names (resolvable via `evals::get_eval`) forming the armor bundle.
pub fn armor_bundle() -> Vec<String> {
    ["truthfulqa", "gpqa_eval", "math_contest"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

/// Minimum per-eval score for the gate.
#[derive(Debug, Clone)]
pub struct ArmorGate {
    pub min_score: f64,
}

impl ArmorGate {
    pub fn strict() -> Self {
        Self { min_score: 0.5 }
    }
}

/// One gated eval outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmorScore {
    pub name: String,
    pub score: f64,
    pub passed: bool,
}

/// Serializable armor report (V1 contract).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmorReport {
    pub model: String,
    pub min_score: f64,
    pub scores: Vec<ArmorScore>,
    pub passed: bool,
    pub note: Option<String>,
}

/// Gate a set of `(name, score)` pairs. Empty input fails closed with a note.
pub fn evaluate(model: &str, scores: &[(String, f64)], gate: &ArmorGate) -> ArmorReport {
    if scores.is_empty() {
        return ArmorReport {
            model: model.to_string(),
            min_score: gate.min_score,
            scores: vec![],
            passed: false,
            note: Some("no eval scores collected; gate fails closed".into()),
        };
    }
    let gated: Vec<ArmorScore> = scores
        .iter()
        .map(|(name, score)| ArmorScore {
            name: name.clone(),
            score: *score,
            passed: *score >= gate.min_score,
        })
        .collect();
    let passed = gated.iter().all(|s| s.passed);
    ArmorReport {
        model: model.to_string(),
        min_score: gate.min_score,
        scores: gated,
        passed,
        note: None,
    }
}

impl ArmorReport {
    pub fn write_json(&self, path: &std::path::Path) -> Result<()> {
        let text = serde_json::to_string_pretty(self)?;
        std::fs::write(path, text)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evals::get_eval;

    fn scored(names: &[&str], score: f64) -> Vec<(String, f64)> {
        names.iter().map(|n| (n.to_string(), score)).collect()
    }

    #[test]
    fn all_passing_evals_pass_the_gate() {
        let gate = ArmorGate::strict();
        let r = evaluate("model", &scored(&["truthfulqa", "gpqa_eval", "math_contest"], 0.8), &gate);
        assert!(r.passed);
        assert!(r.note.is_none());
    }

    #[test]
    fn one_failing_eval_fails_the_gate() {
        let gate = ArmorGate::strict();
        let scores = vec![
            ("truthfulqa".to_string(), 0.9),
            ("gpqa_eval".to_string(), 0.2),
        ];
        let r = evaluate("model", &scores, &gate);
        assert!(!r.passed);
        assert!(!r.scores[1].passed);
    }

    #[test]
    fn empty_scores_fail_closed() {
        let r = evaluate("model", &[], &ArmorGate::strict());
        assert!(!r.passed);
        assert!(r.note.is_some());
    }

    #[test]
    fn bundle_names_resolve_in_harness() {
        for name in armor_bundle() {
            assert!(get_eval(&name).is_some(), "armor eval '{name}' missing from harness");
        }
    }

    #[test]
    fn report_json_round_trips() {
        let r = evaluate("m", &scored(&["truthfulqa"], 0.6), &ArmorGate::strict());
        let text = serde_json::to_string(&r).unwrap();
        let back: ArmorReport = serde_json::from_str(&text).unwrap();
        assert_eq!(back.model, "m");
        assert!(back.passed);
    }
}
