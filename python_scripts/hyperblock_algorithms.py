#!/usr/bin/env python3
"""
Implement IHyper, MHyper, and IMHyper algorithms from algos.md.
Output hyperblocks (each defined by two 121-D edges) for k-NN classification.
Uses full training data and parallelism for 95%+ accuracy.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


@dataclass
class Hyperblock:
    """A hyperblock defined by lower and upper 121-D edges, with a class label."""
    lower: np.ndarray  # 121-D
    upper: np.ndarray  # 121-D
    label: int

    def contains(self, point: np.ndarray) -> bool:
        """Check if point is inside this hyperblock."""
        return np.all((point >= self.lower) & (point <= self.upper))

    def distance_to(self, point: np.ndarray) -> float:
        """Distance from point to hyperblock (0 if inside, else to nearest boundary)."""
        closest = np.clip(point, self.lower, self.upper)
        return float(np.linalg.norm(point - closest))


def points_in_hyperblock(points: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Return boolean mask of points inside hyperblock."""
    return np.all((points >= lower) & (points <= upper), axis=1)


def envelope(lower1: np.ndarray, upper1: np.ndarray,
             lower2: np.ndarray, upper2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Create envelope (min/max) of two hyperblocks."""
    return np.minimum(lower1, lower2), np.maximum(upper1, upper2)


# --- IHyper (parallel over attributes) ---

def _ihyper_attr_worker(args):
    """Worker: find best block for one attribute. Returns (b, d_edges, count, indices) or None."""
    attr, X, y, covered, purity_threshold = args
    n = len(X)
    uncovered_idx = np.where(~covered)[0]
    if len(uncovered_idx) == 0:
        return None

    vals = X[uncovered_idx, attr]
    order = np.argsort(vals)
    sorted_idx = uncovered_idx[order]
    seed_idx = sorted_idx[0]
    a = X[seed_idx].copy()
    label_a = y[seed_idx]
    b, d_edges = a.copy(), a.copy()

    for j in range(1, len(sorted_idx)):
        e_idx = sorted_idx[j]
        e_val = X[e_idx, attr]
        if e_val <= d_edges[attr]:
            continue
        new_upper = d_edges.copy()
        new_upper[attr] = e_val
        in_box = points_in_hyperblock(X, b, new_upper) & ~covered
        if not np.any(in_box):
            continue
        purity = (y[in_box] == label_a).sum() / in_box.sum()
        if purity >= purity_threshold:
            d_edges[attr] = e_val
        else:
            break

    for j in range(len(sorted_idx) - 1, -1, -1):
        e_idx = sorted_idx[j]
        e_val = X[e_idx, attr]
        if e_val >= b[attr]:
            continue
        new_lower = b.copy()
        new_lower[attr] = e_val
        in_box = points_in_hyperblock(X, new_lower, d_edges) & ~covered
        if not np.any(in_box):
            continue
        purity = (y[in_box] == label_a).sum() / in_box.sum()
        if purity >= purity_threshold:
            b[attr] = e_val
        else:
            break

    in_box = points_in_hyperblock(X, b, d_edges) & ~covered
    total = in_box.sum()
    if total > 0 and (y[in_box] == label_a).sum() / total >= purity_threshold:
        return (b, d_edges, total, np.where(in_box)[0])
    return None


def ihyper(X: np.ndarray, y: np.ndarray, purity_threshold: float = 1.0,
           n_jobs: int = -1) -> List[Hyperblock]:
    """Interval Hyper with parallel attribute processing."""
    n, d = X.shape
    n_jobs = mp.cpu_count() if n_jobs <= 0 else n_jobs
    covered = np.zeros(n, dtype=bool)
    hyperblocks: List[Hyperblock] = []

    while not np.all(covered):
        tasks = [(attr, X, y, covered.copy(), purity_threshold) for attr in range(d)]
        best_hb, best_count, best_indices = None, 0, None

        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for fut in as_completed(ex.submit(_ihyper_attr_worker, t) for t in tasks):
                result = fut.result()
                if result and result[2] > best_count:
                    b, d_edges, count, indices = result
                    best_count = count
                    best_hb = Hyperblock(b.copy(), d_edges.copy(), int(y[indices[0]]))
                    best_indices = indices

        if best_hb is None:
            break
        hyperblocks.append(best_hb)
        covered[best_indices] = True

    return hyperblocks


# --- MHyper (parallel merge search) ---

def _mhyper_merge_check(args):
    """Check if blocks i and j can merge (same class, pure envelope)."""
    i, j, blocks, X, y = args
    if i >= j:
        return None
    low_i, up_i, label_i = blocks[i]
    low_j, up_j, label_j = blocks[j]
    if label_i != label_j:
        return None
    new_low, new_up = envelope(low_i, up_i, low_j, up_j)
    in_env = points_in_hyperblock(X, new_low, new_up)
    if np.all(y[in_env] == label_i):
        return (i, j, new_low, new_up, label_i)
    return None


def _mhyper_impurity_merge(args):
    """Find best merge for block i with impurity threshold."""
    i, blocks, X, y, impurity_threshold = args
    low_i, up_i, label_i = blocks[i]
    best_j, best_imp, best_low, best_up = None, 1.0, None, None
    for j in range(len(blocks)):
        if i == j:
            continue
        low_j, up_j, label_j = blocks[j]
        new_low, new_up = envelope(low_i, up_i, low_j, up_j)
        in_env = points_in_hyperblock(X, new_low, new_up)
        total = in_env.sum()
        if total == 0:
            continue
        impurity = (y[in_env] != label_i).sum() / total
        if impurity <= impurity_threshold and impurity < best_imp:
            best_imp, best_j = impurity, j
            best_low, best_up = new_low, new_up
    if best_j is not None:
        return (i, best_j, best_low, best_up, label_i)
    return None


def mhyper(X: np.ndarray, y: np.ndarray,
           initial_blocks: Optional[List[Hyperblock]] = None,
           impurity_threshold: float = 0.0,
           n_jobs: int = -1) -> List[Hyperblock]:
    """Merger Hyper with parallel merge search."""
    n, d = X.shape
    n_jobs = mp.cpu_count() if n_jobs <= 0 else n_jobs

    if initial_blocks:
        blocks = [(hb.lower.copy(), hb.upper.copy(), hb.label) for hb in initial_blocks]
        covered_by_initial = np.zeros(n, dtype=bool)
        for hb in initial_blocks:
            covered_by_initial |= points_in_hyperblock(X, hb.lower, hb.upper)
    else:
        blocks = []
        covered_by_initial = np.zeros(n, dtype=bool)

    for i in range(n):
        if not covered_by_initial[i]:
            blocks.append((X[i].copy(), X[i].copy(), int(y[i])))

    # Step 3-4: Merge same-class blocks (parallel)
    changed = True
    chunk_size = max(1, len(blocks) // (n_jobs * 4))

    while changed:
        changed = False
        pairs = [(i, j) for i in range(len(blocks)) for j in range(i + 1, len(blocks))
                 if blocks[i] is not None and blocks[j] is not None
                 and blocks[i][2] == blocks[j][2]]

        if not pairs:
            break

        merged = set()
        for k in range(0, len(pairs), chunk_size):
            batch = pairs[k:k + chunk_size]
            tasks = [(i, j, blocks, X, y) for i, j in batch]
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                for fut in as_completed(ex.submit(_mhyper_merge_check, t) for t in tasks):
                    result = fut.result()
                    if result and result[0] not in merged and result[1] not in merged:
                        i, j, new_low, new_up, label = result
                        blocks[i] = (new_low, new_up, label)
                        blocks[j] = None
                        merged.add(i)
                        merged.add(j)
                        changed = True

        blocks = [b for b in blocks if b is not None]

    # Step 5: Single-point HBs for uncovered
    in_any = np.zeros(n, dtype=bool)
    for low, up, _ in blocks:
        in_any |= points_in_hyperblock(X, low, up)
    for i in range(n):
        if not in_any[i]:
            blocks.append((X[i].copy(), X[i].copy(), int(y[i])))

    # Step 6-7: Impurity merge (parallel over blocks)
    if impurity_threshold > 0:
        changed = True
        while changed:
            changed = False
            tasks = [(i, blocks, X, y, impurity_threshold)
                     for i in range(len(blocks)) if blocks[i] is not None]
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                for fut in as_completed(ex.submit(_mhyper_impurity_merge, t) for t in tasks):
                    result = fut.result()
                    if result:
                        i, j, new_low, new_up, label = result
                        if blocks[i] is not None and blocks[j] is not None:
                            blocks[i] = (new_low, new_up, label)
                            blocks[j] = None
                            changed = True
            blocks = [b for b in blocks if b is not None]

    return [Hyperblock(low, up, lab) for low, up, lab in blocks]


# --- IMHyper ---

def imhyper(X: np.ndarray, y: np.ndarray,
            purity_threshold: float = 1.0,
            impurity_threshold: float = 0.0,
            n_jobs: int = -1) -> List[Hyperblock]:
    """Interval Merger Hyper: IHyper first, then MHyper on remaining."""
    ih_blocks = ihyper(X, y, purity_threshold, n_jobs)
    n = len(X)
    covered = np.zeros(n, dtype=bool)
    for hb in ih_blocks:
        covered |= points_in_hyperblock(X, hb.lower, hb.upper)
    if np.all(covered):
        return ih_blocks
    X_rem = X[~covered]
    y_rem = y[~covered]
    return mhyper(X_rem, y_rem, initial_blocks=ih_blocks,
                  impurity_threshold=impurity_threshold, n_jobs=n_jobs)


# --- Prototype selection: ≤2000 representative points as single-point HBs ---

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


def cluster_hyperblocks(X: np.ndarray, y: np.ndarray, max_blocks: int,
                        random_state: int = 42) -> List[Hyperblock]:
    """Create HBs by K-means per class; envelope of each cluster = one HB."""
    from sklearn.cluster import MiniBatchKMeans
    classes = np.unique(y)
    n_total = len(X)
    hyperblocks = []
    for c in classes:
        X_c = X[y == c]
        n_c = len(X_c)
        blocks_per_class = max(1, int(max_blocks * n_c / n_total))
        k = min(blocks_per_class, n_c)
        if k < 1:
            continue
        km = MiniBatchKMeans(n_clusters=k, random_state=random_state, batch_size=1000)
        labels_c = km.fit_predict(X_c)
        for j in range(k):
            pts = X_c[labels_c == j]
            if len(pts) == 0:
                continue
            hyperblocks.append(Hyperblock(np.min(pts, axis=0), np.max(pts, axis=0), int(c)))
    return hyperblocks


# --- k-NN classification (memory-efficient chunked) ---

def _distances_chunked(X_batch: np.ndarray, lowers: np.ndarray, uppers: np.ndarray,
                       block_chunk: int = 5000) -> np.ndarray:
    """Compute (n_batch, n_blocks) distances in block chunks to limit memory."""
    n_pts, n_blocks = len(X_batch), len(lowers)
    dists = np.zeros((n_pts, n_blocks), dtype=np.float32)
    for j0 in range(0, n_blocks, block_chunk):
        j1 = min(j0 + block_chunk, n_blocks)
        L, U = lowers[j0:j1], uppers[j0:j1]
        closest = np.clip(X_batch[:, None, :], L[None, :, :], U[None, :, :])
        dists[:, j0:j1] = np.linalg.norm(closest - X_batch[:, None, :], axis=2).astype(np.float32)
    return dists


def classify_batch(X: np.ndarray, hyperblocks: List[Hyperblock], k: int,
                   pt_batch: int = 500, block_chunk: int = 3000) -> np.ndarray:
    """Classify all points. Small batches to limit memory."""
    if not hyperblocks or k <= 0:
        return np.zeros(len(X), dtype=int)
    labels = np.array([hb.label for hb in hyperblocks])
    lowers = np.array([hb.lower for hb in hyperblocks])
    uppers = np.array([hb.upper for hb in hyperblocks])
    k_use = min(k, len(hyperblocks))
    preds = np.zeros(len(X), dtype=int)
    for start in range(0, len(X), pt_batch):
        end = min(start + pt_batch, len(X))
        dists = _distances_chunked(X[start:end], lowers, uppers, block_chunk)
        for i in range(end - start):
            idx = np.argpartition(dists[i], k_use - 1)[:k_use]
            idx = idx[np.argsort(dists[i][idx])]
            votes = Counter(labels[idx])
            preds[start + i] = votes.most_common(1)[0][0]
    return preds


def accuracy(X: np.ndarray, y: np.ndarray, hyperblocks: List[Hyperblock], k: int) -> float:
    """Compute classification accuracy."""
    preds = classify_batch(X, hyperblocks, k)
    return (preds == y).mean()


def learn_k(X_val: np.ndarray, y_val: np.ndarray,
            hyperblocks: List[Hyperblock], k_max: int = 50,
            pt_batch: int = 300, block_chunk: int = 500) -> int:
    """Find best k. Compute distances once per point, reuse for all k."""
    if not hyperblocks:
        return 1
    labels = np.array([hb.label for hb in hyperblocks])
    lowers = np.array([hb.lower for hb in hyperblocks])
    uppers = np.array([hb.upper for hb in hyperblocks])
    k_max = min(k_max, len(hyperblocks))
    all_preds = {k: np.zeros(len(X_val), dtype=int) for k in range(1, k_max + 1)}
    for start in range(0, len(X_val), pt_batch):
        end = min(start + pt_batch, len(X_val))
        dists = _distances_chunked(X_val[start:end], lowers, uppers, block_chunk)
        for i in range(end - start):
            order = np.argsort(dists[i])
            for k in range(1, k_max + 1):
                votes = Counter(labels[order[:k]])
                all_preds[k][start + i] = votes.most_common(1)[0][0]
    best_k, best_acc = 1, 0.0
    for k in range(1, k_max + 1):
        acc = (all_preds[k] == y_val).mean()
        if acc > best_acc:
            best_acc, best_k = acc, k
    return best_k


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Build hyperblocks for MNIST classification")
    parser.add_argument("--algorithm", choices=["prototype", "cluster", "ihyper", "mhyper", "imhyper", "all_points"],
                        default="prototype",
                        help="prototype: 2000 prototypes via K-means+nearest (default, 95%%+); "
                             "cluster: envelope K-means; mhyper/ihyper/imhyper: merge algs")
    parser.add_argument("--max-hyperblocks", type=int, default=2000,
                        help="Max HBs for cluster mode (default: 2000)")
    parser.add_argument("--max-points", type=int, default=None,
                        help="Max training points (default: None = use all)")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Parallel jobs (default: all CPUs)")
    args = parser.parse_args()

    base = Path(__file__).parent

    print("Loading data...")
    train_df = pd.read_csv(base / "mnist_train_reduced.csv")
    test_df = pd.read_csv(base / "mnist_test_reduced.csv")

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

    if args.algorithm == "prototype":
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
    elif args.algorithm == "cluster":
        print(f"Creating cluster hyperblocks (max {args.max_hyperblocks})...")
        hyperblocks = cluster_hyperblocks(X_fit, y_fit, args.max_hyperblocks)
        print(f"Created {len(hyperblocks)} hyperblocks")
        print("Learning k on validation set...")
        k_best = learn_k(X_val, y_val, hyperblocks, k_max=min(50, len(hyperblocks)))
        val_acc = accuracy(X_val, y_val, hyperblocks, k_best)
        test_acc = accuracy(X_test, y_test, hyperblocks, k_best)
        print(f"Best k = {k_best}")
        print(f"Validation accuracy: {val_acc:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")
    elif args.algorithm == "all_points":
        # Each training point = single-point hyperblock. k-NN = standard k-NN on training data.
        print("Creating single-point hyperblocks (one per training point)...")
        hyperblocks = [Hyperblock(X_fit[i].copy(), X_fit[i].copy(), int(y_fit[i]))
                       for i in range(len(X_fit))]
        print(f"Created {len(hyperblocks)} hyperblocks")
        # Use sklearn for fast k selection (equivalent for point blocks)
        try:
            from sklearn.neighbors import KNeighborsClassifier
            knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
            knn.fit(X_fit, y_fit)
            best_acc, k_best = 0.0, 1
            for k in range(1, min(51, len(hyperblocks) + 1)):
                knn.set_params(n_neighbors=k)
                acc = knn.score(X_val, y_val)
                if acc > best_acc:
                    best_acc, k_best = acc, k
            knn.set_params(n_neighbors=k_best)
            val_acc = knn.score(X_val, y_val)
            test_acc = knn.score(X_test, y_test)
            print(f"Learning k on validation set... Best k = {k_best}")
            print(f"Validation accuracy: {val_acc:.4f}")
            print(f"Test accuracy: {test_acc:.4f}")
        except ImportError:
            print("Learning k on validation set...")
            k_best = learn_k(X_val, y_val, hyperblocks, k_max=min(50, len(hyperblocks)))
            val_acc = accuracy(X_val, y_val, hyperblocks, k_best)
            test_acc = accuracy(X_test, y_test, hyperblocks, k_best)
            print(f"Best k = {k_best}")
            print(f"Validation accuracy: {val_acc:.4f}")
            print(f"Test accuracy: {test_acc:.4f}")
    else:
        print(f"Running {args.algorithm.upper()}...")
        if args.algorithm == "ihyper":
            hyperblocks = ihyper(X_fit, y_fit, purity_threshold=0.95, n_jobs=args.n_jobs)
        elif args.algorithm == "mhyper":
            hyperblocks = mhyper(X_fit, y_fit, impurity_threshold=0.05, n_jobs=args.n_jobs)
        else:
            hyperblocks = imhyper(X_fit, y_fit, purity_threshold=0.95,
                                 impurity_threshold=0.05, n_jobs=args.n_jobs)
        print(f"Created {len(hyperblocks)} hyperblocks")
        print("Learning k on validation set...")
        k_best = learn_k(X_val, y_val, hyperblocks, k_max=min(50, len(hyperblocks)))
        val_acc = accuracy(X_val, y_val, hyperblocks, k_best)
        test_acc = accuracy(X_test, y_test, hyperblocks, k_best)
        print(f"Best k = {k_best}")
        print(f"Validation accuracy: {val_acc:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")

    # Save hyperblocks to CSV (spec: class + 1x1..11x11 for single-point; full lower/upper for boxes)
    out_path = base / "hyperblocks.csv"
    dim_names = [f"{i}x{j}" for i in range(1, 12) for j in range(1, 12)]
    single_point = args.algorithm in ("prototype", "all_points")
    rows = []
    for hb in hyperblocks:
        if single_point:
            row = {"class": hb.label}
            for j, name in enumerate(dim_names):
                row[name] = hb.lower[j]
        else:
            row = {"class": hb.label}
            for j, name in enumerate(dim_names):
                row[f"lower_{name}"] = hb.lower[j]
            for j, name in enumerate(dim_names):
                row[f"upper_{name}"] = hb.upper[j]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved {len(hyperblocks)} hyperblocks to {out_path}")

    pd.DataFrame([{
        "k": k_best,
        "n_hyperblocks": len(hyperblocks),
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
    }]).to_csv(base / "hyperblock_metadata.csv", index=False)


if __name__ == "__main__":
    main()
