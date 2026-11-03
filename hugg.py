from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="openai/privacy-filter",
    local_dir=r"D:\Models\privacy-filter",
)
print("Model saved to:", path)