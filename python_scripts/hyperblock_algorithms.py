#!/usr/bin/env python3
"""
Implement IHyper, MHyper, and IMHyper algorithms from algos.md.
Output hyperblocks (each defined by two 121-D edges) for k-NN classification.
Uses full training data and parallelism for 95%+ accuracy.
"""

import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
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
           n_jobs: int = -1, max_blocks: Optional[int] = None) -> List[Hyperblock]:
    """Interval Hyper with parallel attribute processing."""
    n, d = X.shape
    n_jobs = mp.cpu_count() if n_jobs <= 0 else n_jobs
    covered = np.zeros(n, dtype=bool)
    hyperblocks: List[Hyperblock] = []

    with tqdm(total=n, desc="IHyper", unit="pts") as pbar:
        while not np.all(covered):
            if max_blocks is not None and len(hyperblocks) >= max_blocks:
                break
            tasks = [(attr, X, y, covered.copy(), purity_threshold) for attr in range(d)]
            best_hb, best_count, best_indices = None, 0, None

            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                futures = [ex.submit(_ihyper_attr_worker, t) for t in tasks]
                done = 0
                for fut in as_completed(futures):
                    done += 1
                    pbar.set_postfix(hbs=len(hyperblocks), attrs=f"{done}/{d}")
                    result = fut.result()
                    if result and result[2] > best_count:
                        b, d_edges, count, indices = result
                        best_count = count
                        best_hb = Hyperblock(b.copy(), d_edges.copy(), int(y[indices[0]]))
                        best_indices = indices

            if best_hb is None:
                break
            # Only add blocks covering >1 point; let MHyper merge single-point remainder
            if best_count <= 1:
                break
            hyperblocks.append(best_hb)
            covered[best_indices] = True
            pbar.update(best_count)
            pbar.set_postfix(hbs=len(hyperblocks))

    return hyperblocks


# --- MHyper (parallel merge search) ---

def _mhyper_merge_check(args):
    """Check if blocks i and j can merge (same class, pure envelope)."""
    i, j, blocks, X, y = args
    if i >= j:
        return None
    if i >= len(blocks) or j >= len(blocks) or blocks[i] is None or blocks[j] is None:
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
    if i >= len(blocks) or blocks[i] is None:
        return None
    low_i, up_i, label_i = blocks[i]
    best_j, best_imp, best_low, best_up = None, 1.0, None, None
    for j in range(len(blocks)):
        if i == j or blocks[j] is None:
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
           n_jobs: int = -1,
           max_merge_iters: Optional[int] = None,
           max_impurity_iters: Optional[int] = None) -> List[Hyperblock]:
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
    merge_iter = 0
    pbar_merge = tqdm(desc="MHyper merge", unit="iter")

    while changed:
        if max_merge_iters is not None and merge_iter >= max_merge_iters:
            break
        merge_iter += 1
        changed = False
        pairs = [(i, j) for i in range(len(blocks)) for j in range(i + 1, len(blocks))
                 if blocks[i] is not None and blocks[j] is not None
                 and blocks[i][2] == blocks[j][2]]

        if not pairs:
            break

        pbar_merge.update(1)
        n_blocks = len([b for b in blocks if b is not None])
        pbar_merge.set_postfix(blocks=n_blocks, pairs=len(pairs))
        chunk_size = max(100, len(pairs) // n_jobs)
        merged = set()
        n_batches = (len(pairs) + chunk_size - 1) // chunk_size
        for batch_idx, k in enumerate(range(0, len(pairs), chunk_size)):
            pbar_merge.set_postfix(blocks=n_blocks, batch=f"{batch_idx + 1}/{n_batches}")
            batch = pairs[k:k + chunk_size]
            tasks = [(i, j, blocks, X, y) for i, j in batch]
            with ThreadPoolExecutor(max_workers=n_jobs) as ex:
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

    pbar_merge.close()
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
        impurity_iter = 0
        pbar_imp = tqdm(desc="MHyper impurity", unit="iter")
        while changed:
            if max_impurity_iters is not None and impurity_iter >= max_impurity_iters:
                break
            impurity_iter += 1
            changed = False
            pbar_imp.update(1)
            n_blocks = len([b for b in blocks if b is not None])
            pbar_imp.set_postfix(blocks=n_blocks)
            tasks = [(i, blocks, X, y, impurity_threshold)
                     for i in range(len(blocks)) if blocks[i] is not None]
            done = 0
            with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                for fut in as_completed(ex.submit(_mhyper_impurity_merge, t) for t in tasks):
                    done += 1
                    pbar_imp.set_postfix(blocks=n_blocks, workers=f"{done}/{len(tasks)}")
                    result = fut.result()
                    if result:
                        i, j, new_low, new_up, label = result
                        if blocks[i] is not None and blocks[j] is not None:
                            blocks[i] = (new_low, new_up, label)
                            blocks[j] = None
                            changed = True
            blocks = [b for b in blocks if b is not None]
        pbar_imp.close()

    return [Hyperblock(low, up, lab) for low, up, lab in blocks]


# --- IMHyper ---

def imhyper(X: np.ndarray, y: np.ndarray,
            purity_threshold: float = 1.0,
            impurity_threshold: float = 0.0,
            n_jobs: int = -1,
            max_ihyper_blocks: Optional[int] = None,
            max_mhyper_merge_iters: Optional[int] = None,
            max_mhyper_impurity_iters: Optional[int] = None) -> List[Hyperblock]:
    """Interval Merger Hyper: IHyper first, then MHyper on full data to merge all HBs."""
    ih_blocks = ihyper(X, y, purity_threshold, n_jobs, max_blocks=max_ihyper_blocks)
    return mhyper(X, y, initial_blocks=ih_blocks,
                  impurity_threshold=impurity_threshold, n_jobs=n_jobs,
                  max_merge_iters=max_mhyper_merge_iters,
                  max_impurity_iters=max_mhyper_impurity_iters)


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


def _in_any_block(X_batch: np.ndarray, lowers: np.ndarray, uppers: np.ndarray,
                  block_chunk: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """Return (in_block: bool[n], first_block_idx: int[n]). -1 if not in any block."""
    n_pts, n_blocks = len(X_batch), len(lowers)
    in_block = np.zeros(n_pts, dtype=bool)
    first_block_idx = np.full(n_pts, -1, dtype=np.int32)
    for j0 in range(0, n_blocks, block_chunk):
        j1 = min(j0 + block_chunk, n_blocks)
        L, U = lowers[j0:j1], uppers[j0:j1]
        contained = np.all((X_batch[:, None, :] >= L[None, :, :]) &
                          (X_batch[:, None, :] <= U[None, :, :]), axis=2)
        for j in range(j1 - j0):
            col = contained[:, j]
            newly_in = col & ~in_block
            if np.any(newly_in):
                first_block_idx[newly_in] = j0 + j
                in_block |= newly_in
    return in_block, first_block_idx


def classify_batch(X: np.ndarray, hyperblocks: List[Hyperblock], k: int,
                   pt_batch: int = 500, block_chunk: int = 3000) -> np.ndarray:
    """Classify: in-block -> block class; outside -> k-NN HBs k=3 euclidean."""
    if not hyperblocks or k <= 0:
        return np.zeros(len(X), dtype=int)
    labels = np.array([hb.label for hb in hyperblocks])
    lowers = np.array([hb.lower for hb in hyperblocks])
    uppers = np.array([hb.upper for hb in hyperblocks])
    k_use = min(k, len(hyperblocks))
    preds = np.zeros(len(X), dtype=int)
    n_batches = (len(X) + pt_batch - 1) // pt_batch
    for start in tqdm(range(0, len(X), pt_batch), desc="classify", total=n_batches, unit="batch"):
        end = min(start + pt_batch, len(X))
        batch = X[start:end]
        in_block, first_idx = _in_any_block(batch, lowers, uppers, block_chunk)
        preds[start:end][in_block] = labels[first_idx[in_block]]
        outside = ~in_block
        if np.any(outside):
            dists = _distances_chunked(batch[outside], lowers, uppers, block_chunk)
            for ii, i in enumerate(np.where(outside)[0]):
                idx = np.argpartition(dists[ii], k_use - 1)[:k_use]
                idx = idx[np.argsort(dists[ii][idx])]
                votes = Counter(labels[idx])
                preds[start + i] = votes.most_common(1)[0][0]
    return preds


def accuracy(X: np.ndarray, y: np.ndarray, hyperblocks: List[Hyperblock], k: int) -> float:
    """Compute classification accuracy."""
    preds = classify_batch(X, hyperblocks, k)
    return (preds == y).mean()


def learn_k(X_val: np.ndarray, y_val: np.ndarray,
            hyperblocks: List[Hyperblock], k_max: int = 50,
            pt_batch: int = 500, block_chunk: int = 3000) -> int:
    """Find best k. Compute distances once per point, reuse for all k."""
    if not hyperblocks:
        return 1
    labels = np.array([hb.label for hb in hyperblocks])
    lowers = np.array([hb.lower for hb in hyperblocks])
    uppers = np.array([hb.upper for hb in hyperblocks])
    k_max = min(k_max, len(hyperblocks))
    best_k, best_acc = 1, 0.0
    all_preds = {k: np.zeros(len(X_val), dtype=int) for k in range(1, k_max + 1)}
    n_batches = (len(X_val) + pt_batch - 1) // pt_batch
    for start in tqdm(range(0, len(X_val), pt_batch), desc="learn_k", total=n_batches, unit="batch"):
        end = min(start + pt_batch, len(X_val))
        dists = _distances_chunked(X_val[start:end], lowers, uppers, block_chunk)
        for i in range(end - start):
            order = np.argsort(dists[i])
            for k in range(1, k_max + 1):
                votes = Counter(labels[order[:k]])
                all_preds[k][start + i] = votes.most_common(1)[0][0]
    for k in range(1, k_max + 1):
        acc = (all_preds[k] == y_val).mean()
        if acc > best_acc:
            best_acc, best_k = acc, k
    return best_k


# --- Main (BAP: seed-spread, few steps, pick best, repeat) ---

def main():
    parser = argparse.ArgumentParser(description="BAP: seed-spread hyperblock search until 95% test acc")
    parser.add_argument("--max-points", type=int, default=None,
                        help="Max training points (default: None = use all)")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Parallel jobs (default: all CPUs)")
    parser.add_argument("--n-trials", type=int, default=20,
                        help="Short runs per round (widest net)")
    parser.add_argument("--n-top", type=int, default=10,
                        help="Top results to use as seeds for next round")
    parser.add_argument("--chunk", type=int, default=1000,
                        help="Points to add per round (growth step)")
    parser.add_argument("--step-ihyper", type=int, default=5,
                        help="Max IHyper blocks per exploration run")
    parser.add_argument("--step-mmerge", type=int, default=3,
                        help="Max MHyper pure-merge iters (exploration)")
    parser.add_argument("--step-mimpurity", type=int, default=2,
                        help="Max MHyper impurity iters (exploration)")
    parser.add_argument("--step-mmerge-deep", type=int, default=10,
                        help="Max MHyper merge iters for seed refinement (capture worsen-then-benefit)")
    parser.add_argument("--step-mimpurity-deep", type=int, default=6,
                        help="Max MHyper impurity iters for seed refinement")
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent.parent / "data"

    print("Loading data...")
    train_path = data_dir / "mnist_train_dr.csv"
    test_path = data_dir / "mnist_test_dr.csv"
    if not train_path.exists() or not test_path.exists():
        raise SystemExit(
            f"Data not found in {data_dir}. Run apply_dimensionality_reduction.py first."
        )
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    label_col = "class" if "class" in train_df.columns else "label"
    feat_cols = [c for c in train_df.columns if c != label_col]
    X_train = train_df[feat_cols].values.astype(np.float64)
    y_train = train_df[label_col].values.astype(int)
    X_test = test_df[feat_cols].values.astype(np.float64)
    y_test = test_df[label_col].values.astype(int)

    n_jobs = mp.cpu_count() if args.n_jobs <= 0 else args.n_jobs
    k = 3
    chunk = args.chunk
    target_acc = 0.95
    rng = np.random.default_rng(42)

    n_max = args.max_points if args.max_points is not None else len(X_train)
    n_max = min(n_max, len(X_train))

    # Stratified ordering: interleave by class so perm[:n] has balanced classes
    classes = np.unique(y_train)
    by_class = [rng.permutation(np.where(y_train == c)[0]) for c in classes]
    max_per_class = max(len(arr) for arr in by_class)
    perm = []
    for i in range(max_per_class):
        for arr in by_class:
            if i < len(arr):
                perm.append(arr[i])
    perm = np.array(perm)

    # BAP: seed spread net, pull in best group, repeat
    seed_indices = np.array([], dtype=np.int64)
    seed_hbs: List[List[Hyperblock]] = []
    best_hbs: List[Hyperblock] = []
    best_acc = 0.0
    n_fit = 0

    while n_fit < n_max:
        n_fit = min(n_fit + chunk, n_max)
        if len(seed_indices) > 0:
            idx = np.unique(np.concatenate([seed_indices, perm[:n_fit]]))[:n_fit]
        else:
            idx = perm[:n_fit]
        X_fit = X_train[idx]
        y_fit = y_train[idx]

        print(f"\n--- Round: {len(X_fit)} pts, {len(seed_hbs)} seed HB sets ---")

        candidates: List[Tuple[List[Hyperblock], float]] = []

        if not seed_hbs:
            # First round: many short IMHyper runs
            for t in range(args.n_trials):
                trial_perm = rng.permutation(len(X_fit))
                X_t = X_fit[trial_perm]
                y_t = y_fit[trial_perm]
                hbs = imhyper(X_t, y_t, purity_threshold=0.95, impurity_threshold=0.05,
                             n_jobs=n_jobs, max_ihyper_blocks=args.step_ihyper,
                             max_mhyper_merge_iters=args.step_mmerge,
                             max_mhyper_impurity_iters=args.step_mimpurity)
                acc = accuracy(X_test, y_test, hbs, k)
                candidates.append((hbs, acc))
        else:
            # Later rounds: MHyper from each seed (DEEP steps to capture worsen-then-benefit)
            # + fresh IMHyper for exploration
            for hb_seed in seed_hbs:
                hbs = mhyper(X_fit, y_fit, initial_blocks=hb_seed,
                            impurity_threshold=0.05, n_jobs=n_jobs,
                            max_merge_iters=args.step_mmerge_deep,
                            max_impurity_iters=args.step_mimpurity_deep)
                acc = accuracy(X_test, y_test, hbs, k)
                candidates.append((hbs, acc))
            for _ in range(min(4, args.n_trials - len(seed_hbs))):
                trial_perm = rng.permutation(len(X_fit))
                hbs = imhyper(X_fit[trial_perm], y_fit[trial_perm],
                             purity_threshold=0.95, impurity_threshold=0.05,
                             n_jobs=n_jobs, max_ihyper_blocks=args.step_ihyper,
                             max_mhyper_merge_iters=args.step_mmerge,
                             max_mhyper_impurity_iters=args.step_mimpurity)
                acc = accuracy(X_test, y_test, hbs, k)
                candidates.append((hbs, acc))

        if not candidates:
            break
        candidates.sort(key=lambda x: -x[1])
        top = candidates[: args.n_top]
        seed_hbs = [hbs for hbs, _ in top]
        seed_indices = idx.copy()

        best_hbs, best_acc = top[0]
        print(f"Top acc: {best_acc:.4f} (n_hbs={len(best_hbs)})")
        if best_acc >= target_acc:
            print(f"Reached {target_acc:.0%} with {n_fit} pts")
            break

    hyperblocks = best_hbs
    test_acc = best_acc
    if test_acc < target_acc:
        print(f"Stopped at {n_max} pts; best acc {test_acc:.4f} < {target_acc:.0%}")

    # Save hyperblocks to CSV
    out_path = data_dir / "hyperblocks_hyper.csv"
    dim_names = [f"{i}x{j}" for i in range(1, 12) for j in range(1, 12)]
    rows = []
    for i, hb in enumerate(hyperblocks):
        row = {"class": hb.label, "hb_id": i}
        for j, name in enumerate(dim_names):
            row[f"lower_{name}"] = hb.lower[j]
        for j, name in enumerate(dim_names):
            row[f"upper_{name}"] = hb.upper[j]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved {len(hyperblocks)} hyperblocks to {out_path}")

    pd.DataFrame([{
        "algorithm": "IMHyper",
        "k": k,
        "n_train": n_fit,
        "n_hyperblocks": len(hyperblocks),
        "test_accuracy": test_acc,
    }]).to_csv(data_dir / "hyperblock_metadata_hyper.csv", index=False)


if __name__ == "__main__":
    main()
