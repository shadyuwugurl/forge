use std::path::Path;
use anyhow::Result;
use forge_hub::HubClient;

pub async fn run(model_id: &str, output: Option<&Path>) -> Result<()> {
    let client = HubClient::new().await?;

    let output_dir = output.unwrap_or(std::path::Path::new("."));
    std::fs::create_dir_all(output_dir)?;

    eprintln!("Downloading {}...", model_id);
    let path = client.download(model_id, &output_dir.to_path_buf()).await?;
    eprintln!("Downloaded to: {}", path.display());

    Ok(())
}
