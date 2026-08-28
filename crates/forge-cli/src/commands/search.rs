use anyhow::Result;
use forge_hub::HubClient;

pub async fn run(query: &str, limit: usize) -> Result<()> {
    let client = HubClient::new().await?;
    let models = client.search(query, limit).await?;

    eprintln!("Search results for '{}':\n", query);
    eprintln!("{:<50} {:>10} {:>10}", "Model", "Downloads", "Likes");
    eprintln!("{}", "-".repeat(72));

    for m in &models {
        eprintln!("{:<50} {:>10} {:>10}", m.id, m.downloads, m.likes);
    }

    Ok(())
}
