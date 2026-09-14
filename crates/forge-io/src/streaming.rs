use forge_core::tensor_map::TensorMap;

/// D4: streaming aligner — groups canonical slots into memory-bounded batches.
///
/// Walks a [`TensorMap`] in sorted slot-key order and greedily packs slots so
/// each batch's summed raw-tensor bytes stay under `max_bytes`. Tensor sizes
/// come from the caller (index JSON / TensorStore metadata), keeping this
/// crate free of weight I/O. An oversize single slot gets its own batch
/// rather than stalling the packer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamingPlan {
    pub batches: Vec<Vec<String>>,
}

impl StreamingPlan {
    pub fn build(map: &TensorMap, max_bytes: u64, size_of: &dyn Fn(&str) -> u64) -> Self {
        let mut keys: Vec<&String> = map.slots.keys().collect();
        keys.sort();
        let mut batches: Vec<Vec<String>> = Vec::new();
        let mut cur: Vec<String> = Vec::new();
        let mut cur_bytes: u64 = 0;
        for key in keys {
            let entry = &map.slots[key];
            let slot_bytes: u64 = entry.raw_names.iter().map(|r| size_of(r)).sum();
            if !cur.is_empty() && cur_bytes + slot_bytes > max_bytes {
                batches.push(std::mem::take(&mut cur));
                cur_bytes = 0;
            }
            cur.push((*key).clone());
            cur_bytes += slot_bytes;
        }
        if !cur.is_empty() {
            batches.push(cur);
        }
        Self { batches }
    }

    pub fn batch_count(&self) -> usize {
        self.batches.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
use forge_core::tensor_map::{CanonicalSlot, SlotEntry, TensorMap};

    fn sizes<'a>(pairs: &'a [(&'a str, u64)]) -> impl Fn(&str) -> u64 + 'a {
        move |name: &str| {
            pairs.iter().find(|(n, _)| *n == name).map(|(_, s)| *s).unwrap_or(0)
        }
    }

    fn map_with_slots(slots: &[(&str, &[&str])]) -> TensorMap {
        // Minimal map: bypass build(), insert slots directly.
        let mut m = TensorMap { slots: Default::default(), by_raw: Default::default(), common: Default::default() };
        for (key, raws) in slots {
            let entry = SlotEntry {
                slot: Some(CanonicalSlot::named(None, key.to_string())),
                raw_names: raws.iter().map(|s| s.to_string()).collect(),
                fused: false,
            };
            m.slots.insert(key.to_string(), entry);
        }
        m
    }

    #[test]
    fn empty_map_yields_no_batches() {
        let m = map_with_slots(&[]);
        let plan = StreamingPlan::build(&m, 1024, &sizes(&[]));
        assert_eq!(plan.batch_count(), 0);
    }

    #[test]
    fn fitting_slots_pack_into_one_sorted_batch() {
        let m = map_with_slots(&[("b", &["rb"][..]), ("a", &["ra"][..])]);
        let plan = StreamingPlan::build(&m, 1024, &sizes(&[("ra", 100), ("rb", 100)]));
        assert_eq!(plan.batches, vec![vec!["a".to_string(), "b".to_string()]]);
    }

    #[test]
    fn over_budget_slots_split_batches() {
        let m = map_with_slots(&[("a", &["ra"][..]), ("b", &["rb"][..])]);
        let plan = StreamingPlan::build(&m, 1000, &sizes(&[("ra", 600), ("rb", 600)]));
        assert_eq!(plan.batches, vec![vec!["a".to_string()], vec!["b".to_string()]]);
    }

    #[test]
    fn oversize_slot_gets_own_batch_without_stall() {
        let m = map_with_slots(&[("big", &["rbig"][..]), ("small", &["rsmall"][..])]);
        let plan = StreamingPlan::build(&m, 1000, &sizes(&[("rbig", 5000), ("rsmall", 100)]));
        assert_eq!(plan.batch_count(), 2);
        assert_eq!(plan.batches[0], vec!["big".to_string()]);
        assert_eq!(plan.batches[1], vec!["small".to_string()]);
    }
}
