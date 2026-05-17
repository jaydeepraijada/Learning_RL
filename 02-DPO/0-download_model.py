import os
import ssl

# Disable SSL verification globally
ssl._create_default_https_context = ssl._create_unverified_context

os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["PYTHONHTTPSVERIFY"] = "0"

from huggingface_hub import snapshot_download

LOCAL_MODEL_DIR = "./Qwen2.5-0.5B-Instruct"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def download_model():

    if os.path.exists(LOCAL_MODEL_DIR):
        print(f"Model already exists at {LOCAL_MODEL_DIR}")
        return

    print(f"Downloading {MODEL_ID} ...")

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=LOCAL_MODEL_DIR,
        local_dir_use_symlinks=False,
    )

    print("Download complete.")


if __name__ == "__main__":
    download_model()