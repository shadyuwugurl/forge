/// Evaluation definitions (scaling easy → hard)
#[derive(Debug, Clone)]
pub struct Eval {
    pub name: String,
    pub display_name: String,
    pub difficulty: Difficulty,
    pub description: String,
    pub category: EvalCategory,
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
pub enum EvalCategory {
    GeneralQA,
    Coding,
    Reasoning,
    Conversation,
    Instruction,
    ReasoningEfficiency,
    Science,
    Math,
    Language,
    InstructionFollowing,
    Creative,
    Roleplay,
    Safety,
    Multilingual,
    LongContext,
    ToolUse,
    RAG,
}

impl EvalCategory {
    pub fn all() -> Vec<EvalCategory> {
        vec![
            EvalCategory::GeneralQA,
            EvalCategory::Coding,
            EvalCategory::Reasoning,
            EvalCategory::Conversation,
            EvalCategory::Instruction,
            EvalCategory::ReasoningEfficiency,
            EvalCategory::Science,
            EvalCategory::Math,
            EvalCategory::Language,
            EvalCategory::InstructionFollowing,
            EvalCategory::Creative,
            EvalCategory::Roleplay,
            EvalCategory::Safety,
            EvalCategory::Multilingual,
            EvalCategory::LongContext,
            EvalCategory::ToolUse,
            EvalCategory::RAG,
        ]
    }
}

pub fn all_evals() -> Vec<Eval> {
    vec![
        // === General QA ===
        Eval {
            name: "ace".to_string(),
            display_name: "ACE (Agentic Coding Eval)".to_string(),
            difficulty: Difficulty::Easy,
            description: "Basic code generation and following instructions".to_string(),
            category: EvalCategory::Coding,
            dataset_name: "ace".to_string(),
            metric: "pass@1".to_string(),
        },
        Eval {
            name: "truthfulqa".to_string(),
            display_name: "TruthfulQA".to_string(),
            difficulty: Difficulty::Medium,
            description: "Measuring model truthfulness vs imitation of falsehoods".to_string(),
            category: EvalCategory::GeneralQA,
            dataset_name: "truthfulqa".to_string(),
            metric: "truth_score".to_string(),
        },
        Eval {
            name: "halueval".to_string(),
            display_name: "Halueval".to_string(),
            difficulty: Difficulty::Medium,
            description: "Hallucination evaluation benchmark".to_string(),
            category: EvalCategory::GeneralQA,
            dataset_name: "halueval".to_string(),
            metric: "f1".to_string(),
        },

        // === Coding Evals ===
        Eval {
            name: "swe".to_string(),
            display_name: "SWE-bench".to_string(),
            difficulty: Difficulty::Medium,
            description: "Real GitHub issue resolution in Python repos".to_string(),
            category: EvalCategory::Coding,
            dataset_name: "swe_bench".to_string(),
            metric: "resolved_rate".to_string(),
        },
        Eval {
            name: "swe_bench_lite".to_string(),
            display_name: "SWE-bench Lite".to_string(),
            difficulty: Difficulty::Medium,
            description: "Lightweight subset of SWE-bench".to_string(),
            category: EvalCategory::Coding,
            dataset_name: "swe_bench_lite".to_string(),
            metric: "resolved_rate".to_string(),
        },
        Eval {
            name: "swe_bench_verified".to_string(),
            display_name: "SWE-bench Verified".to_string(),
            difficulty: Difficulty::Medium,
            description: "Human-verified subset of SWE-bench".to_string(),
            category: EvalCategory::Coding,
            dataset_name: "swe_bench_verified".to_string(),
            metric: "resolved_rate".to_string(),
        },
        Eval {
            name: "code_contests".to_string(),
            display_name: "Code Contests".to_string(),
            difficulty: Difficulty::VeryHard,
            description: "Competitive programming problems".to_string(),
            category: EvalCategory::Coding,
            dataset_name: "code_contests".to_string(),
            metric: "pass@1".to_string(),
        },
        Eval {
            name: "apps".to_string(),
            display_name: "APPS".to_string(),
            difficulty: Difficulty::VeryHard,
            description: "Automated Python Programming Problems".to_string(),
            category: EvalCategory::Coding,
            dataset_name: "apps".to_string(),
            metric: "pass@1".to_string(),
        },

        // === Terminal/CLI ===
        Eval {
            name: "terminal".to_string(),
            display_name: "TerminalBench".to_string(),
            difficulty: Difficulty::MediumHard,
            description: "CLI/system operations and shell command generation".to_string(),
            category: EvalCategory::ToolUse,
            dataset_name: "terminal_bench".to_string(),
            metric: "success_rate".to_string(),
        },
        Eval {
            name: "shell_qa".to_string(),
            display_name: "Shell QA".to_string(),
            difficulty: Difficulty::Medium,
            description: "Shell command question answering".to_string(),
            category: EvalCategory::ToolUse,
            dataset_name: "shell_qa".to_string(),
            metric: "accuracy".to_string(),
        },

        // === General AI Assistant ===
        Eval {
            name: "gaia".to_string(),
            display_name: "GAIA".to_string(),
            difficulty: Difficulty::Hard,
            description: "General AI assistants, multi-step real-world tasks".to_string(),
            category: EvalCategory::GeneralQA,
            dataset_name: "gaia".to_string(),
            metric: "level_1_2_3_accuracy".to_string(),
        },
        Eval {
            name: "webshop".to_string(),
            display_name: "WebShop".to_string(),
            difficulty: Difficulty::Hard,
            description: "E-commerce navigation and purchase tasks".to_string(),
            category: EvalCategory::ToolUse,
            dataset_name: "webshop".to_string(),
            metric: "success_rate".to_string(),
        },
        Eval {
            name: "tool_bench".to_string(),
            display_name: "ToolBench".to_string(),
            difficulty: Difficulty::Hard,
            description: "Tool use with 16,000+ APIs".to_string(),
            category: EvalCategory::ToolUse,
            dataset_name: "toolbench".to_string(),
            metric: "success_rate".to_string(),
        },

        // === Reasoning / Multi-step ===
        Eval {
            name: "hle".to_string(),
            display_name: "HLE (Humanity's Last Exam)".to_string(),
            difficulty: Difficulty::VeryHard,
            description: "Expert-level cross-domain reasoning".to_string(),
            category: EvalCategory::Reasoning,
            dataset_name: "hle".to_string(),
            metric: "accuracy".to_string(),
        },
        Eval {
            name: "hle_hard".to_string(),
            display_name: "HLE Hard Subset".to_string(),
            difficulty: Difficulty::Expert,
            description: "Hardest subset of HLE".to_string(),
            category: EvalCategory::Reasoning,
            dataset_name: "hle_hard".to_string(),
            metric: "accuracy".to_string(),
        },
        Eval {
            name: "omnimath".to_string(),
            display_name: "OmniMath".to_string(),
            difficulty: Difficulty::Expert,
            description: "Olympiad-level math with formal proofs".to_string(),
            category: EvalCategory::Math,
            dataset_name: "omnimath".to_string(),
            metric: "accuracy".to_string(),
        },
        Eval {
            name: "proofnet".to_string(),
            display_name: "ProofNet".to_string(),
            difficulty: Difficulty::Expert,
            description: "Formal theorem proving in Lean".to_string(),
            category: EvalCategory::Reasoning,
            dataset_name: "proofnet".to_string(),
            metric: "proof_success_rate".to_string(),
        },

        // === Conversational / Chat ===
        Eval {
            name: "mt_bench".to_string(),
            display_name: "MT-Bench".to_string(),
            difficulty: Difficulty::Hard,
            description: "Multi-turn conversation with LLM judge".to_string(),
            category: EvalCategory::Conversation,
            dataset_name: "mt_bench".to_string(),
            metric: "score_1_to_10".to_string(),
        },
        Eval {
            name: "alpaca_eval".to_string(),
            display_name: "AlpacaEval 2.0".to_string(),
            difficulty: Difficulty::Medium,
            description: "Instruction following with GPT-4 as judge".to_string(),
            category: EvalCategory::Conversation,
            dataset_name: "alpaca_eval_v2".to_string(),
            metric: "win_rate_vs_gpt4".to_string(),
        },
        Eval {
            name: "wildchat".to_string(),
            display_name: "WildChat".to_string(),
            difficulty: Difficulty::Hard,
            description: "Real user conversations with diverse topics".to_string(),
            category: EvalCategory::Conversation,
            dataset_name: "wildchat".to_string(),
            metric: "win_rate".to_string(),
        },
        Eval {
            name: "lmsys_chatbot_arena".to_string(),
            display_name: "LMSYS Chatbot Arena".to_string(),
            difficulty: Difficulty::Hard,
            description: "Crowdsourced pairwise model comparisons".to_string(),
            category: EvalCategory::Conversation,
            dataset_name: "lmsys_chatbot_arena".to_string(),
            metric: "elo_rating".to_string(),
        },

        // === Instruction Following ===
        Eval {
            name: "ifeval".to_string(),
            display_name: "IFEval".to_string(),
            difficulty: Difficulty::Medium,
            description: "Instruction following with verifiable constraints".to_string(),
            category: EvalCategory::InstructionFollowing,
            dataset_name: "ifeval".to_string(),
            metric: "strict_accuracy".to_string(),
        },
        Eval {
            name: "promptbench".to_string(),
            display_name: "PromptBench".to_string(),
            difficulty: Difficulty::Medium,
            description: "Robustness to prompt variations".to_string(),
            category: EvalCategory::InstructionFollowing,
            dataset_name: "promptbench".to_string(),
            metric: "robustness_score".to_string(),
        },

        // === Safety / Alignment ===
        Eval {
            name: "harmbench".to_string(),
            display_name: "HarmBench".to_string(),
            difficulty: Difficulty::Medium,
            description: "Standardized harmful behavior evaluation".to_string(),
            category: EvalCategory::Safety,
            dataset_name: "harmbench".to_string(),
            metric: "refusal_rate".to_string(),
        },
        Eval {
            name: "xstest".to_string(),
            display_name: "XSTest".to_string(),
            difficulty: Difficulty::Medium,
            description: "Exaggerated safety test for over-refusal".to_string(),
            category: EvalCategory::Safety,
            dataset_name: "xstest".to_string(),
            metric: "over_refusal_rate".to_string(),
        },
        Eval {
            name: "safe_rlhf".to_string(),
            display_name: "Safe-RLHF".to_string(),
            difficulty: Difficulty::Medium,
            description: "Human preference data for safety".to_string(),
            category: EvalCategory::Safety,
            dataset_name: "safe_rlhf".to_string(),
            metric: "preference_accuracy".to_string(),
        },

        // === Multilingual ===
        Eval {
            name: "xlsum".to_string(),
            display_name: "XLSum".to_string(),
            difficulty: Difficulty::Medium,
            description: "Cross-lingual summarization in 44 languages".to_string(),
            category: EvalCategory::Multilingual,
            dataset_name: "xlsum".to_string(),
            metric: "rouge".to_string(),
        },
        Eval {
            name: "flores".to_string(),
            display_name: "FLORES-200".to_string(),
            difficulty: Difficulty::Medium,
            description: "Translation benchmark for 200 languages".to_string(),
            category: EvalCategory::Multilingual,
            dataset_name: "flores_200".to_string(),
            metric: "spbleu".to_string(),
        },
        Eval {
            name: "xnli".to_string(),
            display_name: "XNLI".to_string(),
            difficulty: Difficulty::Medium,
            description: "Cross-lingual natural language inference".to_string(),
            category: EvalCategory::Multilingual,
            dataset_name: "xnli".to_string(),
            metric: "accuracy".to_string(),
        },

        // === Long Context ===
        Eval {
            name: "longbench".to_string(),
            display_name: "LongBench".to_string(),
            difficulty: Difficulty::Hard,
            description: "Long context understanding (up to 100k tokens)".to_string(),
            category: EvalCategory::LongContext,
            dataset_name: "longbench".to_string(),
            metric: "f1_rouge".to_string(),
        },
        Eval {
            name: "needle_in_haystack".to_string(),
            display_name: "Needle in Haystack".to_string(),
            difficulty: Difficulty::Hard,
            description: "Retrieve specific info from long context".to_string(),
            category: EvalCategory::LongContext,
            dataset_name: "needle_in_haystack".to_string(),
            metric: "retrieval_accuracy".to_string(),
        },
        Eval {
            name: "ruler".to_string(),
            display_name: "RULER".to_string(),
            difficulty: Difficulty::VeryHard,
            description: "Long context retrieval and reasoning".to_string(),
            category: EvalCategory::LongContext,
            dataset_name: "ruler".to_string(),
            metric: "accuracy".to_string(),
        },

        // === RAG / Retrieval ===
        Eval {
            name: "ragbench".to_string(),
            display_name: "RAGBench".to_string(),
            difficulty: Difficulty::Medium,
            description: "Retrieval-augmented generation benchmark".to_string(),
            category: EvalCategory::RAG,
            dataset_name: "ragbench".to_string(),
            metric: "f1_answer".to_string(),
        },
        Eval {
            name: "kilt".to_string(),
            display_name: "KILT".to_string(),
            difficulty: Difficulty::Medium,
            description: "Knowledge-intensive language tasks".to_string(),
            category: EvalCategory::RAG,
            dataset_name: "kilt".to_string(),
            metric: "r_precision".to_string(),
        },

        // === Creative / Roleplay ===
        Eval {
            name: "creative_writing".to_string(),
            display_name: "Creative Writing".to_string(),
            difficulty: Difficulty::Medium,
            description: "Story generation, poetry, creative prompts".to_string(),
            category: EvalCategory::Creative,
            dataset_name: "creative_writing".to_string(),
            metric: "human_preference".to_string(),
        },
        Eval {
            name: "roleplay_bench".to_string(),
            display_name: "Roleplay Bench".to_string(),
            difficulty: Difficulty::Medium,
            description: "Character consistency in roleplay".to_string(),
            category: EvalCategory::Roleplay,
            dataset_name: "roleplay_bench".to_string(),
            metric: "consistency_score".to_string(),
        },

        // === Reasoning Efficiency ===
        Eval {
            name: "reasoning_efficiency".to_string(),
            display_name: "Reasoning Efficiency".to_string(),
            difficulty: Difficulty::Hard,
            description: "Tokens per correct answer in multi-step reasoning".to_string(),
            category: EvalCategory::ReasoningEfficiency,
            dataset_name: "reasoning_efficiency".to_string(),
            metric: "tokens_per_correct".to_string(),
        },
        Eval {
            name: "cot_efficiency".to_string(),
            display_name: "CoT Efficiency".to_string(),
            difficulty: Difficulty::Hard,
            description: "Chain-of-thought token efficiency".to_string(),
            category: EvalCategory::ReasoningEfficiency,
            dataset_name: "cot_efficiency".to_string(),
            metric: "tokens_per_correct".to_string(),
        },

        // === Science & Specialized ===
        Eval {
            name: "gpqa_eval".to_string(),
            display_name: "GPQA Eval".to_string(),
            difficulty: Difficulty::VeryHard,
            description: "Graduate-level physics/chemistry/biology QA".to_string(),
            category: EvalCategory::Science,
            dataset_name: "gpqa".to_string(),
            metric: "accuracy".to_string(),
        },
        Eval {
            name: "math_contest".to_string(),
            display_name: "Math Contest".to_string(),
            difficulty: Difficulty::Expert,
            description: "IMO/AMC/AIME level problems".to_string(),
            category: EvalCategory::Math,
            dataset_name: "math_contest".to_string(),
            metric: "accuracy".to_string(),
        },

        // === HLE (Humanity's Last Exam) ===
        Eval {
            name: "hle".to_string(),
            display_name: "HLE (Humanity's Last Exam)".to_string(),
            difficulty: Difficulty::VeryHard,
            description: "Expert-level cross-domain reasoning across 100+ domains".to_string(),
            category: EvalCategory::Reasoning,
            dataset_name: "hle".to_string(),
            metric: "accuracy".to_string(),
        },
    ]
}

pub fn get_eval(name: &str) -> Option<Eval> {
    all_evals().into_iter().find(|e| e.name == name)
}

pub fn evals_by_category(category: EvalCategory) -> Vec<Eval> {
    all_evals().into_iter().filter(|e| e.category == category).collect()
}

pub fn evals_by_difficulty(difficulty: Difficulty) -> Vec<Eval> {
    all_evals().into_iter().filter(|e| e.difficulty == difficulty).collect()
}

pub fn all_categories() -> Vec<EvalCategory> {
    EvalCategory::all()
}