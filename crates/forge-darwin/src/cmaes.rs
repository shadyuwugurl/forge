use crate::genome::DarwinGenome;

/// CMA-ES (Covariance Matrix Adaptation Evolution Strategy) state.
/// Used to evolve the 14-dimensional genome toward optimal merge parameters.
pub struct CmaEsState {
    /// Population size
    pub population_size: usize,
    /// Mean genome (center of search distribution)
    pub mean: DarwinGenome,
    /// Step size
    pub sigma: f32,
    /// Learning rates
    pub cc: f32,
    pub cs: f32,
    pub c1: f32,
    pub cmu: f32,
    /// Evolution path
    pub pc: Vec<f32>,
    pub ps: Vec<f32>,
    /// Covariance matrix (diagonal approximation for 14 dims)
    pub cov_diag: Vec<f32>,
    /// Generation counter
    pub generation: usize,
}

impl CmaEsState {
    /// Initialize CMA-ES from a starting genome
    pub fn new(start: &DarwinGenome, population_size: usize, sigma: f32) -> Self {
        let dim = 14;
        let n = population_size as f32;

        Self {
            population_size,
            mean: start.clone(),
            sigma,
            cc: (4.0 + dim as f32 / n) / (dim as f32 + 4.0 + 2.0 / n),
            cs: (n + 2.0) / (dim as f32 + n + 5.0),
            c1: 2.0 / ((dim as f32 + 1.3).powi(3) + n),
            cmu: (n - 2.0 + 1.0 / n) / ((dim as f32 + 2.0).powi(2) + n),
            pc: vec![0.0; dim],
            ps: vec![0.0; dim],
            cov_diag: vec![1.0; dim],
            generation: 0,
        }
    }

    /// Sample N candidate genomes from the current distribution
    pub fn sample_population(&self, rng_state: &mut u64) -> Vec<DarwinGenome> {
        let mut population = Vec::with_capacity(self.population_size);

        for _ in 0..self.population_size {
            let mut genome = self.mean.clone();

            // Sample from diagonal Gaussian
            let mut genome_vec = genome_to_vec(&genome);
            for i in 0..14 {
                let z = sample_normal(rng_state);
                genome_vec[i] += self.sigma * self.cov_diag[i].sqrt() * z;
            }
            genome = vec_to_genome(&genome_vec);
            clamp_genome(&mut genome);

            population.push(genome);
        }

        population
    }

    /// Update the distribution based on fitness scores
    pub fn update(&mut self, ranked_population: &[(DarwinGenome, f32)]) {
        // ranked_population is sorted by fitness (best first)
        let mu = self.population_size / 2; // top half

        // Update mean toward best candidates
        let mut new_mean_vec = vec![0.0f32; 14];
        for i in 0..mu {
            let genome_vec = genome_to_vec(&ranked_population[i].0);
            let weight = (mu as f32 - i as f32) / mu as f32;
            for j in 0..14 {
                new_mean_vec[j] += genome_vec[j] * weight;
            }
        }
        let total_weight: f32 = (0..mu).map(|i| (mu as f32 - i as f32) / mu as f32).sum();
        for val in new_mean_vec.iter_mut() {
            *val /= total_weight;
        }

        self.mean = vec_to_genome(&new_mean_vec);
        clamp_genome(&mut self.mean);

        // Update step size with CSA
        let step_diff: Vec<f32> = genome_to_vec(&self.mean).iter()
            .zip(new_mean_vec.iter())
            .map(|(old, new)| (new - old) / self.sigma)
            .collect();

        let norm_diff: f32 = step_diff.iter().map(|x| x * x).sum::<f32>().sqrt();
        let chi_n = (14.0f32).sqrt(); // expected norm of N(0,I)

        // Simplified CSA update
        self.sigma *= (self.cs / self.cs * (norm_diff / chi_n - 1.0) * 0.1).exp();
        self.sigma = self.sigma.clamp(0.001, 1.0);

        self.generation += 1;
    }
}

/// Convert genome to 14-element vector for CMA-ES operations
fn genome_to_vec(g: &DarwinGenome) -> Vec<f32> {
    vec![
        g.gamma, g.alpha_attn, g.alpha_ffn, g.alpha_emb,
        g.rho_a, g.rho_b,
        g.r[0], g.r[1], g.r[2], g.r[3], g.r[4], g.r[5],
        g.tau, g.lambda,
    ]
}

/// Convert 14-element vector back to genome
fn vec_to_genome(v: &[f32]) -> DarwinGenome {
    DarwinGenome {
        gamma: v[0],
        alpha_attn: v[1],
        alpha_ffn: v[2],
        alpha_emb: v[3],
        rho_a: v[4],
        rho_b: v[5],
        r: [v[6], v[7], v[8], v[9], v[10], v[11]],
        tau: v[12],
        lambda: v[13],
    }
}

/// Clamp genome values to valid ranges
fn clamp_genome(g: &mut DarwinGenome) {
    g.gamma = g.gamma.clamp(0.0, 1.0);
    g.alpha_attn = g.alpha_attn.clamp(0.0, 1.0);
    g.alpha_ffn = g.alpha_ffn.clamp(0.0, 1.0);
    g.alpha_emb = g.alpha_emb.clamp(0.0, 1.0);
    g.rho_a = g.rho_a.clamp(0.1, 0.9);
    g.rho_b = g.rho_b.clamp(0.1, 0.9);
    for r in g.r.iter_mut() {
        *r = r.clamp(0.0, 1.0);
    }
    g.tau = g.tau.clamp(0.1, 0.9);
    g.lambda = g.lambda.clamp(0.01, 0.5);
}

/// Sample from standard normal distribution using Box-Muller
fn sample_normal(rng_state: &mut u64) -> f32 {
    *rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
    let u1 = (*rng_state >> 33) as f32 / (1u32 << 31) as f32;
    *rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
    let u2 = (*rng_state >> 33) as f32 / (1u32 << 31) as f32;

    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}
