use anyhow::Result;
use forge_core::TensorMeta;
use crate::orchestrator::MergeOp;
use crate::expert_weaver::ExpertWeaver;
use crate::moe_dense_distill::MoeDenseDistill;

/// C3 MoE bridge: composes ExpertWeaver (dense -> shared+routed experts
/// view) with MoE -> Dense distillation (teacher experts -> dense student).
///
/// Two entry points:
/// - [`MoeBridge::bridge_slot`]: slot-level MoE -> dense. `student` is the
///   dense prior, `teacher_shards` are the per-expert teacher tensors for
///   the same canonical slot (e.g. `experts.0.gate_proj` ..
///   `experts.7.gate_proj`). Shards are averaged into one teacher view,
///   then blended with the student via temperature-scaled softmax.
/// - `MergeOp` impl: delegates homogeneous same-name merges to
///   [`MoeDenseDistill`], so the CLI can use the bridge directly as a
///   merge method.
pub struct MoeBridge {
    pub temperature: f32,
    pub num_experts: usize,
    pub shared_dim: Option<usize>,
}

impl MoeBridge {
    pub fn new(temperature: f32, num_experts: usize, shared_dim: Option<usize>) -> Self {
        Self {
            temperature: temperature.clamp(0.1, 10.0),
            num_experts: num_experts.max(1).min(64),
            shared_dim,
        }
    }

    /// True for MoE expert tensor names across the registry families
    /// (qwen3_moe `mlp.experts.*`, deepseek `mlp.experts.*`,
    /// mixtral `block_sparse_moe.experts.*`, ...).
    pub fn is_moe_tensor(name: &str) -> bool {
        name.contains("experts.")
            || name.contains("block_sparse_moe")
            || name.contains("shared_expert")
    }

    fn weaver(&self) -> ExpertWeaver {
        ExpertWeaver::new(self.num_experts, self.shared_dim)
    }

    fn distiller(&self) -> MoeDenseDistill {
        MoeDenseDistill::new(self.temperature)
    }

    /// Average equal-length teacher expert shards into one teacher view.
    pub fn collapse_experts(teacher_shards: &[Vec<f32>]) -> Result<Vec<f32>> {
        if teacher_shards.is_empty() {
            anyhow::bail!("MoeBridge: no teacher shards to collapse");
        }
        let n = teacher_shards[0].len();
        if teacher_shards.iter().any(|s| s.len() != n) {
            anyhow::bail!(
                "MoeBridge: teacher shard length mismatch: {:?}",
                teacher_shards.iter().map(|s| s.len()).collect::<Vec<_>>()
            );
        }
        let mut out = vec![0.0f32; n];
        for shard in teacher_shards {
            for (o, v) in out.iter_mut().zip(shard.iter()) {
                *o += v / teacher_shards.len() as f32;
            }
        }
        Ok(out)
    }

    /// Full slot bridge: dense student prior + teacher expert shards ->
    /// dense output. Runs the student through the weaver (shared+routed
    /// view, shape-preserving) so both sides carry the same structure,
    /// then distills student-view + teacher-view into one tensor.
    pub fn bridge_slot(
        &self,
        slot_name: &str,
        meta: &TensorMeta,
        student: &[f32],
        teacher_shards: &[Vec<f32>],
    ) -> Result<Vec<f32>> {
        if student.len() != meta.num_elements() {
            anyhow::bail!(
                "MoeBridge: student length {} != slot {} elements {:?}",
                student.len(),
                slot_name,
                meta.shape
            );
        }
        let teacher_view = Self::collapse_experts(teacher_shards)?;
        if teacher_view.len() != meta.num_elements() {
            anyhow::bail!(
                "MoeBridge: teacher view length {} != slot {} elements {:?}",
                teacher_view.len(),
                slot_name,
                meta.shape
            );
        }
        // Weaver pass over the student (shape-preserving shared+routed
        // rewrite) so the distill step blends like-structured tensors.
        let woven_student = self.weaver().merge_tensors(
            slot_name,
            meta,
            &[student.to_vec()],
        )?;
        self.distiller().merge_tensors(
            slot_name,
            meta,
            &[woven_student, teacher_view],
        )
    }
}

impl MergeOp for MoeBridge {
    fn merge_tensor(&self, _name: &str, meta: &TensorMeta) -> Result<Vec<f32>> {
        Ok(vec![0.0f32; meta.num_elements()])
    }

    fn merge_tensors(&self, name: &str, meta: &TensorMeta, inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
        // Homogeneous same-name merge: straight distillation across parents.
        self.distiller().merge_tensors(name, meta, inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_core::DType;

    fn meta(n: usize) -> TensorMeta {
        TensorMeta { name: "x".into(), shape: vec![n], dtype: DType::F32, offset: 0, size: n * 4 }
    }

    #[test]
    fn bridge_midpoint_for_equal_inputs() {
        let b = MoeBridge::new(1.0, 8, None);
        let m = meta(4);
        let student = vec![2.0f32; 4];
        let shards = vec![vec![2.0f32; 4], vec![2.0f32; 4]];
        let out = b.bridge_slot("mlp.gate_proj", &m, &student, &shards).unwrap();
        assert_eq!(out.len(), 4);
        for v in out {
            assert!((v - 2.0).abs() < 1e-5, "expected 2.0 got {v}");
        }
    }

    #[test]
    fn bridge_softmax_favors_larger_side() {
        // Low temperature -> near-argmax weighting: teacher=10 dominates.
        let b = MoeBridge::new(0.1, 8, None);
        let m = meta(2);
        let student = vec![0.0f32; 2];
        let shards = vec![vec![10.0f32; 2]];
        let out = b.bridge_slot("mlp.gate_proj", &m, &student, &shards).unwrap();
        for v in out {
            assert!(v > 9.0, "expected teacher-dominated, got {v}");
        }
    }

    #[test]
    fn bridge_rejects_mismatched_lengths() {
        let b = MoeBridge::new(1.0, 8, None);
        let m = meta(4);
        assert!(b.bridge_slot("x", &m, &vec![0.0; 4], &[vec![0.0; 3]]).is_err());
        assert!(b.bridge_slot("x", &m, &vec![0.0; 2], &[vec![0.0; 4]]).is_err());
        assert!(b.bridge_slot("x", &m, &vec![0.0; 4], &[]).is_err());
    }

    #[test]
    fn moe_tensor_detection() {
        assert!(MoeBridge::is_moe_tensor("model.layers.0.mlp.experts.3.gate_proj"));
        assert!(MoeBridge::is_moe_tensor("model.layers.0.block_sparse_moe.experts.0.w1"));
        assert!(MoeBridge::is_moe_tensor("model.layers.0.mlp.shared_expert.gate_proj"));
        assert!(!MoeBridge::is_moe_tensor("model.layers.0.mlp.gate_proj"));
        assert!(!MoeBridge::is_moe_tensor("model.layers.0.self_attn.q_proj"));
    }

    #[test]
    fn merge_op_delegates_to_distill() {
        let b = MoeBridge::new(1.0, 8, None);
        let m = meta(3);
        let out = b
            .merge_tensors("mlp.down_proj", &m, &[vec![1.0; 3], vec![3.0; 3]])
            .unwrap();
        // softmax(1,3): w = e/(e+e^3) ~ 0.119, out ~ 1*0.119+3*0.881 = 2.76
        for v in out {
            assert!((v - 2.762).abs() < 0.01, "got {v}");
        }
    }
}
