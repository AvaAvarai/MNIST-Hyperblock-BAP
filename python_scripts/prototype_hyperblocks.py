#!/usr/bin/env python3
"""
Prototype Hyperblock (PHyper): K-means centroids as single-point hyperblocks.
Achieves 95%+ accuracy on MNIST (121-D reduced) with ≤2000 hyperblocks.
See design_documentation/prototype_hyperblock.md.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List

# Import shared types and utilities from hyperblock_algorithms
from hyperblock_algorithms import Hyperblock, learn_k, accuracy


def prototype_hyperblocks(X: np.ndarray, y: np.ndarray, max_blocks: int,
                         random_state: int = 42) -> List[Hyperblock]:
    """
    Prototype selection: K-means centroids as single-point HBs.
    Achieves 95%+ with ≤2000 HBs.
    """
    from sklearn.cluster import MiniBatchKMeans
    classes = np.unique(y)
    n_total = len(X)
    hyperblocks = []
    for c in classes:
        X_c = X[y == c]
        n_c = len(X_c)
        k = min(max(1, int(max_blocks * n_c / n_total)), n_c)
        if k < 1:
            continue
        km = MiniBatchKMeans(n_clusters=k, random_state=random_state, batch_size=1000, n_init=10)
        km.fit(X_c)
        for centroid in km.cluster_centers_:
            p = centroid.astype(np.float64)
            hyperblocks.append(Hyperblock(p, p, int(c)))
    return hyperblocks


def main():
    parser = argparse.ArgumentParser(description="Prototype hyperblocks via K-means centroids")
    parser.add_argument("--max-hyperblocks", type=int, default=2000,
                        help="Max hyperblocks (default: 2000)")
    parser.add_argument("--max-points", type=int, default=None,
                        help="Max training points (default: all)")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    data_dir = base / "data"

    train_path = data_dir / "mnist_train_dr.csv"
    test_path = data_dir / "mnist_test_dr.csv"
    if not train_path.exists() or not test_path.exists():
        raise SystemExit(
            f"Data not found in {data_dir}. Run apply_dimensionality_reduction.py first."
        )

    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    label_col = "class" if "class" in train_df.columns else "label"
    feat_cols = [c for c in train_df.columns if c != label_col]
    X_train = train_df[feat_cols].values.astype(np.float64)
    y_train = train_df[label_col].values.astype(int)
    X_test = test_df[feat_cols].values.astype(np.float64)
    y_test = test_df[label_col].values.astype(int)

    n_train = len(X_train)
    n_val = n_train // 5
    n_fit = n_train - n_val
    X_fit = X_train[:n_fit]
    y_fit = y_train[:n_fit]
    X_val = X_train[n_fit:]
    y_val = y_train[n_fit:]

    if args.max_points is not None and len(X_fit) > args.max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_fit), args.max_points, replace=False)
        X_fit = X_fit[idx]
        y_fit = y_fit[idx]
        print(f"Subsampled to {args.max_points} training points")

    print(f"Training on {len(X_fit)} points, validating on {len(X_val)}")
    print(f"Selecting {args.max_hyperblocks} prototypes (K-means centroids)...")

    hyperblocks = prototype_hyperblocks(X_fit, y_fit, args.max_hyperblocks)
    print(f"Created {len(hyperblocks)} hyperblocks")

    print("Learning k on validation set...")
    k_best = learn_k(X_val, y_val, hyperblocks, k_max=min(50, len(hyperblocks)))
    val_acc = accuracy(X_val, y_val, hyperblocks, k_best)
    test_acc = accuracy(X_test, y_test, hyperblocks, k_best)

    print(f"Best k = {k_best}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    # Save hyperblocks
    out_path = data_dir / "hyperblocks_prototype.csv"
    dim_names = [f"{i}x{j}" for i in range(1, 12) for j in range(1, 12)]
    rows = []
    for hb in hyperblocks:
        row = {"class": hb.label}
        for j, name in enumerate(dim_names):
            row[name] = hb.lower[j]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved {len(hyperblocks)} hyperblocks to {out_path}")

    pd.DataFrame([{
        "k": k_best,
        "n_hyperblocks": len(hyperblocks),
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
    }]).to_csv(data_dir / "hyperblock_metadata_prototype.csv", index=False)


if __name__ == "__main__":
    main()
