#!/usr/bin/env python3
"""
Apply dimensionality reduction to MNIST CSV datasets as described in DR.md:

1. Crop 3 pixels from all edges: 28x28 → 22x22 (484-D)
2. Average pooling 2x2 with stride 2: 22x22 → 11x11 (121-D)

Result: 121/784 ≈ 15.4% of original dimensionality.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def reduce_mnist_image(pixels_1d: np.ndarray) -> np.ndarray:
    """
    Apply dimensionality reduction to a single MNIST image (784 pixels).
    
    Steps:
    1. Reshape to 28x28
    2. Crop 3 pixels from all edges → 22x22
    3. Average pooling 2x2 with stride 2 → 11x11
    """
    # Reshape to 28x28 (row-major: first 28 are row 0, etc.)
    img = pixels_1d.reshape(28, 28)
    
    # Crop 3 pixels from all edges: keep [3:25, 3:25] → 22x22
    cropped = img[3:25, 3:25]
    
    # Average pooling: 2x2 kernel, stride 2
    # Result: 11x11
    pooled = np.zeros((11, 11))
    for i in range(11):
        for j in range(11):
            block = cropped[i*2:(i+1)*2, j*2:(j+1)*2]
            pooled[i, j] = np.mean(block)
    
    return pooled.flatten()


def process_mnist_csv(input_path: str, output_path: str) -> None:
    """
    Read MNIST CSV, apply dimensionality reduction, write reduced CSV.
    """
    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)
    
    # Separate label and pixel columns
    label_col = "label"
    pixel_cols = [c for c in df.columns if c != label_col]
    
    # Verify we have 784 pixel columns
    assert len(pixel_cols) == 784, f"Expected 784 pixel columns, got {len(pixel_cols)}"
    
    # Extract pixel values in correct order (1x1, 1x2, ..., 28x28)
    pixels = df[pixel_cols].values.astype(np.float64)
    
    # Apply reduction to each row
    print("Applying dimensionality reduction...")
    reduced = np.array([reduce_mnist_image(row) for row in pixels])
    
    # Build output dataframe (spec: class + 1x1..11x11)
    new_cols = [f"{i}x{j}" for i in range(1, 12) for j in range(1, 12)]
    out_df = pd.DataFrame(
        np.column_stack([df[label_col].values, reduced]),
        columns=["class"] + new_cols
    )
    
    # Ensure class is integer
    out_df["class"] = out_df["class"].astype(int)
    
    out_df.to_csv(output_path, index=False)
    print(f"Saved {len(out_df)} rows to {output_path} (121 features)")


def main():
    base = Path(__file__).parent
    
    process_mnist_csv(
        str(base / "mnist_train.csv"),
        str(base / "mnist_train_reduced.csv")
    )
    
    process_mnist_csv(
        str(base / "mnist_test.csv"),
        str(base / "mnist_test_reduced.csv")
    )
    
    print("Done. Created mnist_train_reduced.csv and mnist_test_reduced.csv")


if __name__ == "__main__":
    main()
