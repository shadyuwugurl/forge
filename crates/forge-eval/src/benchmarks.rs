/// Benchmark definitions (scaling easy → hard)
#[derive(Debug, Clone)]
pub struct Benchmark {
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

pub fn all_benchmarks() -> Vec<Benchmark> {
    vec![
        Benchmark {
            name: "hella".to_string(),
            display_name: "HellaSwag".to_string(),
            difficulty: Difficulty::Easy,
            description: "Commonsense reasoning, sentence completion".to_string(),
        },
        Benchmark {
            name: "mmlu".to_string(),
            display_name: "MMLU".to_string(),
            difficulty: Difficulty::Medium,
            description: "57 subjects, knowledge recall".to_string(),
        },
        Benchmark {
            name: "arc".to_string(),
            display_name: "ARC-Challenge".to_string(),
            difficulty: Difficulty::MediumHard,
            description: "Science reasoning, grade-school to college".to_string(),
        },
        Benchmark {
            name: "gsm8k".to_string(),
            display_name: "GSM8K".to_string(),
            difficulty: Difficulty::Hard,
            description: "Grade-school math, multi-step reasoning".to_string(),
        },
        Benchmark {
            name: "gpqa".to_string(),
            display_name: "GPQA Diamond".to_string(),
            difficulty: Difficulty::VeryHard,
            description: "Graduate-level science, expert questions".to_string(),
        },
    ]
}

pub fn get_benchmark(name: &str) -> Option<Benchmark> {
    all_benchmarks().into_iter().find(|b| b.name == name)
}
