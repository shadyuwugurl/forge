use serde::{Deserialize, Serialize};

/// The 14-dimensional Darwin merge genome.
/// Different genome values → different child model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DarwinGenome {
    /// Global merge ratio
    pub gamma: f32,
    /// Attention component ratio
    pub alpha_attn: f32,
    /// FFN component ratio
    pub alpha_ffn: f32,
    /// Embedding component ratio
    pub alpha_emb: f32,
    /// Parent A density (what fraction of A's weights to keep)
    pub rho_a: f32,
    /// Parent B density
    pub rho_b: f32,
    /// Block-level merge ratios (6 independent layer-block ratios)
    pub r: [f32; 6],
    /// MRI-Trust parameter (balances diagnostic vs genome ratio)
    pub tau: f32,
    /// Regularization parameter
    pub lambda: f32,
}

impl DarwinGenome {
    /// Create a random genome for initial population
    pub fn random(seed: u64) -> Self {
        let mut rng_state = seed;
        let mut next_f32 = |min: f32, max: f32| -> f32 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let t = (rng_state >> 33) as f32 / (1u32 << 31) as f32;
            min + t * (max - min)
        };

        Self {
            gamma: next_f32(0.0, 1.0),
            alpha_attn: next_f32(0.0, 1.0),
            alpha_ffn: next_f32(0.0, 1.0),
            alpha_emb: next_f32(0.0, 1.0),
            rho_a: next_f32(0.3, 0.8),
            rho_b: next_f32(0.3, 0.8),
            r: [
                next_f32(0.0, 1.0),
                next_f32(0.0, 1.0),
                next_f32(0.0, 1.0),
                next_f32(0.0, 1.0),
                next_f32(0.0, 1.0),
                next_f32(0.0, 1.0),
            ],
            tau: next_f32(0.3, 0.6),
            lambda: next_f32(0.01, 0.2),
        }
    }

    /// Get the merge ratio for a specific tensor based on its component type and layer
    pub fn tensor_ratio(&self, tensor_name: &str, layer_idx: Option<usize>, total_layers: usize) -> f32 {
        let base_ratio = self.gamma;

        // Adjust for component type
        let component_ratio = if tensor_name.contains("q_proj") || tensor_name.contains("k_proj")
            || tensor_name.contains("v_proj") || tensor_name.contains("o_proj")
            || tensor_name.contains("attn")
        {
            self.alpha_attn
        } else if tensor_name.contains("mlp") || tensor_name.contains("ffn")
            || tensor_name.contains("gate_proj") || tensor_name.contains("up_proj")
            || tensor_name.contains("down_proj")
        {
            self.alpha_ffn
        } else if tensor_name.contains("embed") || tensor_name.contains("word_embeddings")
        {
            self.alpha_emb
        } else {
            0.5
        };

        // Adjust for layer position
        let layer_ratio = if let Some(idx) = layer_idx {
            let block_size = (total_layers / 6).max(1);
            let block_idx = (idx / block_size).min(5);
            self.r[block_idx]
        } else {
            0.5
        };

        // Combine: weighted average of base, component, and layer ratios
        (base_ratio * 0.4 + component_ratio * 0.3 + layer_ratio * 0.3).clamp(0.0, 1.0)
    }

    /// Crossover two genomes (uniform crossover)
    pub fn crossover(a: &DarwinGenome, b: &DarwinGenome, rng_state: &mut u64) -> DarwinGenome {
        let mut next_bool = || -> bool {
            *rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (*rng_state >> 33) & 1 == 0
        };

        DarwinGenome {
            gamma: if next_bool() { a.gamma } else { b.gamma },
            alpha_attn: if next_bool() { a.alpha_attn } else { b.alpha_attn },
            alpha_ffn: if next_bool() { a.alpha_ffn } else { b.alpha_ffn },
            alpha_emb: if next_bool() { a.alpha_emb } else { b.alpha_emb },
            rho_a: if next_bool() { a.rho_a } else { b.rho_a },
            rho_b: if next_bool() { a.rho_b } else { b.rho_b },
            r: std::array::from_fn(|i| if next_bool() { a.r[i] } else { b.r[i] }),
            tau: if next_bool() { a.tau } else { b.tau },
            lambda: if next_bool() { a.lambda } else { b.lambda },
        }
    }

    /// Mutate genome with given rate
    pub fn mutate(&self, rate: f32, rng_state: &mut u64) -> DarwinGenome {
        let mut next_mutation = |val: f32, min: f32, max: f32| -> f32 {
            *rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let roll = (*rng_state >> 33) as f32 / (1u32 << 31) as f32;
            if roll < rate {
                *rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let delta = ((*rng_state >> 33) as f32 / (1u32 << 31) as f32 - 0.5) * 0.2;
                (val + delta).clamp(min, max)
            } else {
                val
            }
        };

        DarwinGenome {
            gamma: next_mutation(self.gamma, 0.0, 1.0),
            alpha_attn: next_mutation(self.alpha_attn, 0.0, 1.0),
            alpha_ffn: next_mutation(self.alpha_ffn, 0.0, 1.0),
            alpha_emb: next_mutation(self.alpha_emb, 0.0, 1.0),
            rho_a: next_mutation(self.rho_a, 0.1, 0.9),
            rho_b: next_mutation(self.rho_b, 0.1, 0.9),
            r: std::array::from_fn(|i| next_mutation(self.r[i], 0.0, 1.0)),
            tau: next_mutation(self.tau, 0.1, 0.9),
            lambda: next_mutation(self.lambda, 0.01, 0.5),
        }
    }
}
