/// Evaluation definitions (scaling easy → hard)
#[derive(Debug, Clone)]
pub struct Eval {
    pub name: String,
    pub display_name: String,
    pub difficulty: Difficulty,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Difficulty {
    Easy,
    Medium,
    MediumHard,
    Hard,
    VeryHard,
}

pub fn all_evals() -> Vec<Eval> {
    vec![
        Eval {
            name: "ace".to_string(),
            display_name: "ACE (Agentic Coding Eval)".to_string(),
            difficulty: Difficulty::Easy,
            description: "Basic code generation and following instructions".to_string(),
        },
        Eval {
            name: "swe".to_string(),
            display_name: "SWE-bench".to_string(),
            difficulty: Difficulty::Medium,
            description: "Real GitHub issue resolution".to_string(),
        },
        Eval {
            name: "terminal".to_string(),
            display_name: "TerminalBench".to_string(),
            difficulty: Difficulty::MediumHard,
            description: "CLI/system operations".to_string(),
        },
        Eval {
            name: "gaia".to_string(),
            display_name: "GAIA".to_string(),
            difficulty: Difficulty::Hard,
            description: "General AI assistants, multi-step real-world tasks".to_string(),
        },
        Eval {
            name: "hle".to_string(),
            display_name: "HLE (Humanity's Last Exam)".to_string(),
            difficulty: Difficulty::VeryHard,
            description: "Expert-level cross-domain reasoning".to_string(),
        },
    ]
}

pub fn get_eval(name: &str) -> Option<Eval> {
    all_evals().into_iter().find(|e| e.name == name)
}
