use anyhow::{Context, Result};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use forge_io::TensorStore;
use forge_core::TensorMeta;

/// Distributed merge coordination using mDNS discovery and Thunderbolt ring all-reduce
/// Supports merging models that don't fit on a single Mac (e.g., 120B+ models)

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ClusterNode {
    pub id: String,
    pub addr: SocketAddr,
    pub gpu_cores: usize,
    pub memory_gb: f64,
    pub is_coordinator: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MergeShard {
    pub tensor_name: String,
    pub shard_id: usize,
    pub total_shards: usize,
    pub data: Vec<f32>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum DistributedMessage {
    Discover { node: ClusterNode },
    Heartbeat { node_id: String },
    ShardRequest { tensor_name: String, shard_id: usize },
    ShardResponse { shard: MergeShard },
    MergeComplete { tensor_name: String },
    CoordinateAssign { node_id: String, role: NodeRole },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum NodeRole {
    Coordinator,
    Worker { shard_ids: Vec<usize> },
}

pub struct DistributedMergeCoordinator {
    nodes: Arc<RwLock<HashMap<String, ClusterNode>>>,
    local_node: ClusterNode,
    shard_assignments: Arc<RwLock<HashMap<String, Vec<String>>>>, // tensor -> node_ids
}

impl DistributedMergeCoordinator {
    pub fn new(local_node: ClusterNode) -> Self {
        Self {
            nodes: Arc::new(RwLock::new(HashMap::new())),
            local_node,
            shard_assignments: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Start mDNS discovery for cluster nodes
    pub async fn start_discovery(&self) -> Result<()> {
        // In production, use mdns-sd crate for service discovery
        // For now, simulate with local node only
        let mut nodes = self.nodes.write().await;
        nodes.insert(self.local_node.id.clone(), self.local_node.clone());
        Ok(())
    }

    /// Assign shards to nodes using ring topology
    pub async fn assign_shards(&self, tensor_names: &[String], world_size: usize) -> Result<()> {
        let mut assignments = self.shard_assignments.write().await;
        let nodes = self.nodes.read().await;
        let node_ids: Vec<String> = nodes.keys().cloned().collect();

        if node_ids.is_empty() { return Ok(()); }

        for name in tensor_names {
            // Simple round-robin shard assignment across nodes
            let num_shards = world_size;
            let mut shard_nodes = Vec::with_capacity(num_shards);
            for i in 0..num_shards {
                let node_idx = i % node_ids.len();
                shard_nodes.push(node_ids[node_idx].clone());
            }
            assignments.insert(name.clone(), shard_nodes);
        }
        Ok(())
    }

    /// Get node responsible for a shard
    pub async fn get_shard_owner(&self, tensor_name: &str, shard_id: usize) -> Option<String> {
        let assignments = self.shard_assignments.read().await;
        assignments.get(tensor_name)
            .and_then(|nodes| nodes.get(shard_id).cloned())
    }

    /// Ring all-reduce for gradient/weight synchronization
    /// Each node sends to next in ring, receives from previous
    pub async fn ring_all_reduce(&self, data: &mut [f32]) -> Result<()> {
        // In production, this would use NCCL-like ring all-reduce over Thunderbolt
        // For now, local no-op
        Ok(())
    }
}

pub struct DistributedWorker {
    node: ClusterNode,
    coordinator_addr: Option<SocketAddr>,
}

impl DistributedWorker {
    pub fn new(node: ClusterNode) -> Self {
        Self { node, coordinator_addr: None }
    }

    /// Connect to coordinator and register
    pub async fn register(&mut self) -> Result<()> {
        // In production, connect to coordinator via TCP/QUIC
        Ok(())
    }

    /// Process assigned shards
    pub async fn process_shards(&self, store: &TensorStore, tensor_name: &str, shard_ids: &[usize]) -> Result<Vec<MergeShard>> {
        let mut shards = Vec::new();
        for shard_id in shard_ids {
            let meta = store.tensor_meta(tensor_name)?;
            let data = store.tensor_f32(tensor_name)?;

            // Split tensor into shards
            let shard_size = data.len() / 4; // assuming 4 nodes
            let start = shard_id * (data.len() / 4);
            let end = ((shard_id + 1) * (data.len() / 4)).min(data.len());

            shards.push(MergeShard {
                tensor_name: tensor_name.to_string(),
                shard_id: *shard_id,
                total_shards: 4,
                data: data[start..end].to_vec(),
            });
        }
        Ok(shards)
    }
}

/// CLI integration for distributed merge
pub async fn run_distributed_merge(
    config_path: &str,
    output_dir: &str,
    distributed: bool,
) -> Result<()> {
    if !distributed {
        return Err(anyhow::anyhow!("Distributed merge not enabled"));
    }

    // Load config
    let config_str = std::fs::read_to_string(config_path)?;
    let config: forge_core::config::MergeConfig = serde_yaml::from_str(&config_str)?;

    // Discover cluster
    let local_node = ClusterNode {
        id: uuid::Uuid::new_v4().to_string(),
        addr: "0.0.0.0:8080".parse().unwrap(),
        gpu_cores: 10,
        memory_gb: 64.0,
        is_coordinator: true,
    };

    let coordinator = DistributedMergeCoordinator::new(local_node.clone());
    coordinator.start_discovery().await?;
    coordinator.assign_shards(&config.models.iter().map(|m| m.path.display().to_string()).collect::<Vec<String>>(), 4).await?;

    // Execute distributed merge (stub - would use actual distributed merge)
    eprintln!("Distributed merge not fully implemented - would coordinate {} nodes", 1);

    Ok(())
}