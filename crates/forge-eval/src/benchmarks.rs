/// Benchmark definitions (scaling easy → hard)
#[derive(Debug, Clone)]
pub struct Benchmark {
    pub name: String,
    pub display_name: String,
    pub difficulty: Difficulty,
    pub description: String,
    pub category: BenchmarkCategory,
    pub dataset_name: String,
    pub metric: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
pub enum Difficulty {
    Easy,
    Medium,
    MediumHard,
    Hard,
    VeryHard,
    Expert,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum BenchmarkCategory {
    Reasoning,
    Knowledge,
    Math,
    Coding,
    Language,
    Science,
    Conversation,
    Instruction,
    ReasoningEfficiency,
    GeneralQA,
    ReadingComprehension,
    GradeSchool,
    University,
    PhDLevel,
}

impl BenchmarkCategory {
    pub fn all() -> Vec<BenchmarkCategory> {
        vec![
            BenchmarkCategory::Reasoning,
            BenchmarkCategory::Knowledge,
            BenchmarkCategory::Math,
            BenchmarkCategory::Coding,
            BenchmarkCategory::Language,
            BenchmarkCategory::Science,
            BenchmarkCategory::Conversation,
            BenchmarkCategory::Instruction,
            BenchmarkCategory::ReasoningEfficiency,
            BenchmarkCategory::GeneralQA,
            BenchmarkCategory::ReadingComprehension,
            BenchmarkCategory::GradeSchool,
            BenchmarkCategory::University,
            BenchmarkCategory::PhDLevel,
        ]
    }
}

pub fn all_benchmarks() -> Vec<Benchmark> {
    vec![
        // === Reasoning Benchmarks ===
        Benchmark {
            name: "hella".to_string(),
            display_name: "HellaSwag".to_string(),
            difficulty: Difficulty::Easy,
            description: "Commonsense reasoning, sentence completion".to_string(),
            category: BenchmarkCategory::Reasoning,
            dataset_name: "hellaswag".to_string(),
            metric: "accuracy".to_string(),
        },
        Benchmark {
            name: "arc_easy".to_string(),
            display_name: "ARC-Easy".to_string(),
            difficulty: Difficulty::Easy,
            description: "Grade-school science questions (easy set)".to_string(),
            category: BenchmarkCategory::Reasoning,
            dataset_name: "ai2_arc".to_string(),
            metric: "accuracy".to_string(),
        },
        Benchmark {
            name: "arc".to_string(),
            display_name: "ARC-Challenge".to_string(),
            difficulty: Difficulty::MediumHard,
            description: "Grade-school to college science questions (challenge set)".to_string(),
            category: BenchmarkCategory::Reasoning,
            dataset_name: "ai2_arc".to_string(),
            metric: "accuracy".to_string(),
        },
        Benchmark {
            name: "arc_agi_3".to_string(),
            display_name: "ARC-AGI-3".to_string(),
            difficulty: Difficulty::VeryHard,
            description: "Abstract reasoning corpus - visual analogy tasks (ARC-AGI-3)".to_string(),
            category: BenchmarkCategory::Reasoning,
            dataset_name: "arc_agi_3".to_string(),
            metric: "accuracy".to_string(),
        },
        Benchmark {
            name: "logiq".to_string(),
            display_name: "LogiQA".to_string(),
            difficulty: Difficulty::Hard,
            description: "Logical reasoning questions from Chinese civil service exams".to_string(),
            category: BenchmarkCategory::Reasoning,
            dataset_name: "logiqa".to_string(),
            metric: "accuracy".to_string(),
        },
        Benchmark {
            name: "reclor".to_string(),
            display_name: "ReClor".to_string(),
            difficulty: Difficulty::Hard,
            description: "Logical reasoning from LSAT/GMAT style questions".to_string(),
            category: BenchmarkCategory::Reasoning,
            dataset_name: "reclor".to_string(),
            metric: "accuracy".to_string(),
        },

        // === Knowledge Benchmarks ===
        Benchmark {
            name: "mmlu".to_string(),
            display_name: "MMLU".to_string(),
            difficulty: Difficulty::Medium,
            description: "57 subjects, knowledge recall across STEM, humanities, social sciences".to_string(),
            category: BenchmarkCategory::Knowledge,
            dataset_name: "mmlu".to_string(),
            metric: "accuracy".to_string(),
        },
        Benchmark {
            name: "mmlu_pro".to_string(),
            display_name: "MMLU-Pro".to_string(),
            difficulty: Difficulty::Hard,
            description: "Enhanced MMLU with more challenging questions and fewer ambiguous options".to_string(),
            category: BenchmarkCategory::Knowledge,
            dataset_name: "mmlu_pro".to_string(),
            metric: "accuracy".to_string(),
        },
        Benchmark {
            name: "gpqa".to_string(),
            display_name: "GPQA Diamond".to_string(),
            difficulty: Difficulty::VeryHard,
            description: "Graduate-level biology, physics, chemistry questions".to_string(),
            category: BenchmarkCategory::Knowledge,
            dataset_name: "gpqa".to_string(),
            metric: "accuracy".to_string(),
        },
        Benchmark {
            name: "gpqa_extended".to_string(),
            display_name: "GPQA Extended".to_string(),
            difficulty: Difficulty::Expert,
            description: "Extended GPQA with more domains and harder questions".to_string(),
            category: BenchmarkCategory::Knowledge,
            dataset_name: "gpqa_extended".to_string(),
            metric: "accuracy".to_string(),
        },
        Benchmark {
            name: "triviaqa".to_string(),
            display_name: "TriviaQA".to_string(),
            difficulty: Difficulty::Medium,
            description: "Reading comprehension with trivia questions".to_string(),
            category: BenchmarkCategory::Knowledge,
            dataset_name: "triviaqa".to_string(),
            metric: "f1".to_string(),
        },
        Benchmark {
            name: "natural_questions".to_string(),
            display_name: "Natural Questions".to_string(),
            difficulty: Difficulty::Medium,
            description: "Real user questions from Google search".to_string(),
            category: BenchmarkCategory::Knowledge,
            dataset_name: "natural_questions".to_string(),
            metric: "f1".to_string(),
        },
        Benchmark {
            name: "hotpotqa".to_string(),
            display_name: "HotpotQA".to_string(),
            difficulty: Difficulty::Hard,
            description: "Multi-hop reasoning over Wikipedia articles".to_string(),
            category: BenchmarkCategory::Knowledge,
            dataset_name: "hotpotqa".to_string(),
            metric: "f1".to_string(),
        },

        // === Math Benchmarks ===
        Benchmark {
            name: "gsm8k".to_string(),
            display_name: "GSM8K".to_string(),
            difficulty: Difficulty::Hard,
            description: "Grade-school math, multi-step reasoning".to_string(),
            category: BenchmarkCategory::Math,
            dataset_name: "gsm8k".to_string(),
            metric: "accuracy".to_string(),
        },
        Benchmark {
            name: "gsm8k_cot".to_string(),
            display_name: "GSM8K-CoT".to_string(),
            difficulty: Difficulty::Hard,
            description: "GSM8K with chain-of-thought prompting".to_string(),
            category: BenchmarkCategory::Math,
            dataset_name: "gsm8k".to_string(),
            metric: "accuracy".to_string(),
        },
        Benchmark {
            name: "math".to_string(),
            display_name: "MATH".to_string(),
            difficulty: Difficulty::VeryHard,
            description: "High-school math competition problems (algebra, geometry, calculus)".to_string(),
            category: BenchmarkCategory::Math,
            dataset_name: "math".to_string(),
            metric: "accuracy".to_string(),
        },
        Benchmark {
            name: "math_500".to_string(),
            display_name: "MATH-500".to_string(),
            difficulty: Difficulty::VeryHard,
            description: "500 hardest problems from MATH dataset".to_string(),
            category: BenchmarkCategory::Math,
            dataset_name: "math_500".to_string(),
            metric: "accuracy".to_string(),
        },
        Benchmark {
            name: "minerva".to_string(),
            display_name: "Minerva Math".to_string(),
            difficulty: Difficulty::Expert,
            description: "STEM math problems at university/PhD level".to_string(),
            category: BenchmarkCategory::Math,
            dataset_name: "minerva".to_string(),
            metric: "accuracy".to_string(),
        },
        Benchmark {
            name: "svamp".to_string(),
            display_name: "SVAMP".to_string(),
            difficulty: Difficulty::Medium,
            description: "Simple arithmetic word problems with variations".to_string(),
            category: BenchmarkCategory::Math,
            dataset_name: "svamp".to_string(),
            metric: "accuracy".to_string(),
        },
        Benchmark {
            name: "asdiv".to_string(),
            display_name: "ASDiv-A".to_string(),
            difficulty: Difficulty::Medium,
            description: "Diverse arithmetic word problems".to_string(),
            category: BenchmarkCategory::Math,
            dataset_name: "asdiv".to_string(),
            metric: "accuracy".to_string(),
        },

        // === Coding Benchmarks ===
        Benchmark {
            name: "humaneval".to_string(),
            display_name: "HumanEval".to_string(),
            difficulty: Difficulty::Hard,
            description: "Function-level code generation from docstrings".to_string(),
            category: BenchmarkCategory::Coding,
            dataset_name: "humaneval".to_string(),
            metric: "pass@1".to_string(),
        },
        Benchmark {
            name: "humaneval_plus".to_string(),
            display_name: "HumanEval+".to_string(),
            difficulty: Difficulty::Hard,
            description: "HumanEval with more comprehensive test suites".to_string(),
            category: BenchmarkCategory::Coding,
            dataset_name: "humaneval_plus".to_string(),
            metric: "pass@1".to_string(),
        },
        Benchmark {
            name: "mbpp".to_string(),
            display_name: "MBPP".to_string(),
            difficulty: Difficulty::Medium,
            description: "Basic programming problems from Google".to_string(),
            category: BenchmarkCategory::Coding,
            dataset_name: "mbpp".to_string(),
            metric: "pass@1".to_string(),
        },
        Benchmark {
            name: "mbpp_plus".to_string(),
            display_name: "MBPP+".to_string(),
            difficulty: Difficulty::Medium,
            description: "MBPP with enhanced test cases".to_string(),
            category: BenchmarkCategory::Coding,
            dataset_name: "mbpp_plus".to_string(),
            metric: "pass@1".to_string(),
        },
        Benchmark {
            name: "codex_eval".to_string(),
            display_name: "CodexEval".to_string(),
            difficulty: Difficulty::Hard,
            description: "Code generation from natural language with complex requirements".to_string(),
            category: BenchmarkCategory::Coding,
            dataset_name: "codex_eval".to_string(),
            metric: "pass@1".to_string(),
        },
        Benchmark {
            name: "ds1000".to_string(),
            display_name: "DS-1000".to_string(),
            difficulty: Difficulty::VeryHard,
            description: "Data science code generation with libraries".to_string(),
            category: BenchmarkCategory::Coding,
            dataset_name: "ds1000".to_string(),
            metric: "pass@1".to_string(),
        },

        // === Language & Instruction Following ===
        Benchmark {
            name: "ifeval".to_string(),
            display_name: "IFEval".to_string(),
            difficulty: Difficulty::Medium,
            description: "Instruction following evaluation with verifiable constraints".to_string(),
            category: BenchmarkCategory::Instruction,
            dataset_name: "ifeval".to_string(),
            metric: "accuracy".to_string(),
        },
        Benchmark {
            name: "alpaca_eval".to_string(),
            display_name: "AlpacaEval 2.0".to_string(),
            difficulty: Difficulty::Medium,
            description: "Instruction following with GPT-4 as judge".to_string(),
            category: BenchmarkCategory::Instruction,
            dataset_name: "alpaca_eval".to_string(),
            metric: "win_rate".to_string(),
        },
        Benchmark {
            name: "mt_bench".to_string(),
            display_name: "MT-Bench".to_string(),
            difficulty: Difficulty::Hard,
            description: "Multi-turn conversation benchmark with LLM judge".to_string(),
            category: BenchmarkCategory::Conversation,
            dataset_name: "mt_bench".to_string(),
            metric: "score".to_string(),
        },
        Benchmark {
            name: "wildbench".to_string(),
            display_name: "WildBench".to_string(),
            difficulty: Difficulty::Hard,
            description: "Real-world user instructions with diverse tasks".to_string(),
            category: BenchmarkCategory::Conversation,
            dataset_name: "wildbench".to_string(),
            metric: "score".to_string(),
        },

        // === Reading Comprehension ===
        Benchmark {
            name: "squad".to_string(),
            display_name: "SQuAD 2.0".to_string(),
            difficulty: Difficulty::Medium,
            description: "Reading comprehension with unanswerable questions".to_string(),
            category: BenchmarkCategory::ReadingComprehension,
            dataset_name: "squad_v2".to_string(),
            metric: "f1".to_string(),
        },
        Benchmark {
            name: "squad_1".to_string(),
            display_name: "SQuAD 1.1".to_string(),
            difficulty: Difficulty::Medium,
            description: "Reading comprehension (original)".to_string(),
            category: BenchmarkCategory::ReadingComprehension,
            dataset_name: "squad".to_string(),
            metric: "f1".to_string(),
        },
        Benchmark {
            name: "drop".to_string(),
            display_name: "DROP".to_string(),
            difficulty: Difficulty::Hard,
            description: "Discrete reasoning over paragraphs".to_string(),
            category: BenchmarkCategory::ReadingComprehension,
            dataset_name: "drop".to_string(),
            metric: "f1".to_string(),
        },
        Benchmark {
            name: "race".to_string(),
            display_name: "RACE".to_string(),
            difficulty: Difficulty::Hard,
            description: "Reading comprehension from English exams".to_string(),
            category: BenchmarkCategory::ReadingComprehension,
            dataset_name: "race".to_string(),
            metric: "accuracy".to_string(),
        },
        Benchmark {
            name: "quoref".to_string(),
            display_name: "Quoref".to_string(),
            difficulty: Difficulty::Hard,
            description: "Coreferential reasoning in reading comprehension".to_string(),
            category: BenchmarkCategory::ReadingComprehension,
            dataset_name: "quoref".to_string(),
            metric: "f1".to_string(),
        },

        // === Science & Specialized ===
        Benchmark {
            name: "pubmedqa".to_string(),
            display_name: "PubMedQA".to_string(),
            difficulty: Difficulty::Hard,
            description: "Biomedical research question answering".to_string(),
            category: BenchmarkCategory::Science,
            dataset_name: "pubmedqa".to_string(),
            metric: "accuracy".to_string(),
        },
        Benchmark {
            name: "bioasq".to_string(),
            display_name: "BioASQ".to_string(),
            difficulty: Difficulty::Expert,
            description: "Biomedical semantic indexing and QA".to_string(),
            category: BenchmarkCategory::Science,
            dataset_name: "bioasq".to_string(),
            metric: "f1".to_string(),
        },
        Benchmark {
            name: "scitail".to_string(),
            display_name: "SciTail".to_string(),
            difficulty: Difficulty::Medium,
            description: "Science entailment from textbooks".to_string(),
            category: BenchmarkCategory::Science,
            dataset_name: "scitail".to_string(),
            metric: "accuracy".to_string(),
        },

        // === Grade School / University / PhD Level ===
        Benchmark {
            name: "grade_school_math".to_string(),
            display_name: "Grade School Math".to_string(),
            difficulty: Difficulty::Easy,
            description: "Elementary arithmetic word problems".to_string(),
            category: BenchmarkCategory::GradeSchool,
            dataset_name: "grade_school_math".to_string(),
            metric: "accuracy".to_string(),
        },
        Benchmark {
            name: "university_math".to_string(),
            display_name: "University Math".to_string(),
            difficulty: Difficulty::Hard,
            description: "Undergraduate math problems".to_string(),
            category: BenchmarkCategory::University,
            dataset_name: "university_math".to_string(),
            metric: "accuracy".to_string(),
        },
        Benchmark {
            name: "phd_level_reasoning".to_string(),
            display_name: "PhD-Level Reasoning".to_string(),
            difficulty: Difficulty::Expert,
            description: "PhD-level reasoning across domains".to_string(),
            category: BenchmarkCategory::PhDLevel,
            dataset_name: "phd_reasoning".to_string(),
            metric: "accuracy".to_string(),
        },

        // === Reasoning Efficiency ===
        Benchmark {
            name: "reasoning_efficiency".to_string(),
            display_name: "Reasoning Efficiency".to_string(),
            difficulty: Difficulty::Hard,
            description: "Token efficiency in multi-step reasoning".to_string(),
            category: BenchmarkCategory::ReasoningEfficiency,
            dataset_name: "reasoning_efficiency".to_string(),
            metric: "tokens_per_correct".to_string(),
        },
    ]
}

pub fn get_benchmark(name: &str) -> Option<Benchmark> {
    all_benchmarks().into_iter().find(|b| b.name == name)
}

pub fn benchmarks_by_category(category: BenchmarkCategory) -> Vec<Benchmark> {
    all_benchmarks().into_iter().filter(|b| b.category == category).collect()
}

pub fn benchmarks_by_difficulty(difficulty: Difficulty) -> Vec<Benchmark> {
    all_benchmarks().into_iter().filter(|b| b.difficulty == difficulty).collect()
}

pub fn all_categories() -> Vec<BenchmarkCategory> {
    BenchmarkCategory::all()
}