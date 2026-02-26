#!/usr/bin/env python3
"""
Download MNIST in CSV format from Kaggle and save to the data folder.

Dataset: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

Requires:
  - pip install kaggle
  - Kaggle API credentials (script will prompt if not configured)
"""

import json
import os
import shutil
import stat
from pathlib import Path


def setup_kaggle_credentials() -> None:
    """
    Ensure Kaggle credentials exist. If not, prompt user and either save to
    ~/.kaggle/kaggle.json or use env vars for this run only.
    """
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    # Already configured
    if kaggle_json.exists():
        return
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return

    print("Kaggle API credentials not found.")
    print("Create an API token at: https://www.kaggle.com/settings -> Create New API Token")
    print()
    username = input("Kaggle username: ").strip()
    key = input("Kaggle API key: ").strip()

    if not username or not key:
        raise SystemExit("Username and API key are required.")

    save = input("Save credentials to ~/.kaggle/kaggle.json for future runs? [Y/n]: ").strip().lower()
    if save != "n":
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        kaggle_dir.chmod(stat.S_IRWXU)  # 700: only owner
        kaggle_json.write_text(json.dumps({"username": username, "key": key}))
        kaggle_json.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 600
        print("Credentials saved.")
    else:
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key
        print("Using credentials for this run only.")


def main():
    setup_kaggle_credentials()

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise SystemExit(
            "Kaggle API is required. Install with: pip install kaggle"
        )

    # Paths relative to project root
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    dataset = "oddrationale/mnist-in-csv"
    print(f"Downloading {dataset}...")

    # Download and unzip to data directory
    api.dataset_download_files(dataset, path=str(data_dir), unzip=True)

    # Kaggle extracts to a subfolder (e.g. data/mnist-in-csv/)
    # Move CSV files to data/ if they're in a subfolder
    subfolders = [d for d in data_dir.iterdir() if d.is_dir()]
    for subfolder in subfolders:
        for csv_file in subfolder.glob("*.csv"):
            dest = data_dir / csv_file.name
            if dest != csv_file:
                shutil.move(str(csv_file), str(dest))
                print(f"  Moved {csv_file.name} -> {dest}")
        # Remove empty subfolder
        try:
            subfolder.rmdir()
        except OSError:
            pass

    # Verify expected files exist and print stats
    train_path = data_dir / "mnist_train.csv"
    test_path = data_dir / "mnist_test.csv"
    if train_path.exists() and test_path.exists():
        print(f"\nDone. Saved to {data_dir}:")
        print(f"  - {train_path.name}")
        print(f"  - {test_path.name}")
        import sys
        import pandas as pd
        sys.path.insert(0, str(Path(__file__).parent))
        from dataset_stats import print_dataset_stats
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        print_dataset_stats(train_df, test_df, label_col="label", dim=784)
    else:
        print("\nDownload complete. Check data/ for mnist_train.csv and mnist_test.csv")


if __name__ == "__main__":
    main()
