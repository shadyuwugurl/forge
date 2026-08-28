use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use crate::genome::DarwinGenome;
use crate::mri_trust::MriTrustFusion;
use crate::cmaes::CmaEsState;
use forge_io::TensorStore;
use forge_core::TensorMeta;

/// Darwin V6 evolutionary model merger.
///
/// The flagship Darwin-27B-Opus achieved 86.9% on GPQA Diamond,
/// ranking #6 among 1,252 evaluated models.
pub struct DarwinEvolver {
    pub generations: usize,
    pub population_size: usize,
    pub seed: u64,
    pub verbose: bool,
}

impl DarwinEvolver {
    pub fn new(generations: usize, population_size: usize) -> Self {
        Self {
            generations,
            population_size,
            seed: 42,
            verbose: false,
        }
    }

    /// Run the full Darwin V6 evolution loop.
    /// Returns the best genome found.
    pub fn evolve(
        &self,
        store_a: &TensorStore,
        store_b: &TensorStore,
        mri: &MriTrustFusion,
        fitness_fn: &dyn Fn(&DarwinGenome) -> f32,
    ) -> Result<DarwinGenome> {
        let mut rng_state = self.seed;

        // Initialize population with random genomes
        let initial = DarwinGenome::random(self.seed);
        let mut cmaes = CmaEsState::new(&initial, self.population_size, 0.3);

        let pb = if self.verbose {
            let pb = ProgressBar::new(self.generations as u64);
            pb.set_style(ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] generation {pos}/{len} | σ={msg}")
                .unwrap());
            Some(pb)
        } else {
            None
        };

        for gen in 0..self.generations {
            // Sample population
            let population = cmaes.sample_population(&mut rng_state);

            // Evaluate fitness for each candidate
            let mut scored: Vec<(DarwinGenome, f32)> = population.into_iter()
                .map(|g| {
                    let fitness = fitness_fn(&g);
                    (g, fitness)
                })
                .collect();

            // Sort by fitness (descending)
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            if let Some(ref pb) = pb {
                pb.set_message(format!("{:.4}", cmaes.sigma));
                pb.inc(1);
            }

            if self.verbose {
                eprintln!("gen {} | best fitness: {:.4} | mean: {:.4}",
                    gen, scored[0].1,
                    scored.iter().map(|(_, f)| f).sum::<f32>() / scored.len() as f32);
            }

            // Update CMA-ES distribution
            cmaes.update(&scored);
        }

        // Final evaluation: sample from final distribution, pick best
        let final_pop = cmaes.sample_population(&mut rng_state);
        let best = final_pop.into_iter()
            .map(|g| {
                let fitness = fitness_fn(&g);
                (g, fitness)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(g, _)| g)
            .unwrap_or(initial);

        Ok(best)
    }

    /// Apply a genome to merge two models' tensors
    pub fn apply_genome(
        genome: &DarwinGenome,
        tensor_a: &[f32],
        tensor_b: &[f32],
        tensor_name: &str,
        layer_idx: Option<usize>,
        total_layers: usize,
        mri: &MriTrustFusion,
    ) -> Vec<f32> {
        let genome_ratio = genome.tensor_ratio(tensor_name, layer_idx, total_layers);

        // Compute MRI scores for this tensor
        let mri_a = mri.mri_score(tensor_name);
        let mri_b = mri.mri_score(tensor_name); // In practice, different per parent

        let final_ratio = mri.final_ratio(tensor_name, mri_a, mri_b, genome_ratio, genome.tau);

        // Pure convex combination
        tensor_a.iter().zip(tensor_b.iter())
            .map(|(a, b)| (1.0 - final_ratio) * a + final_ratio * b)
            .collect()
    }
}
