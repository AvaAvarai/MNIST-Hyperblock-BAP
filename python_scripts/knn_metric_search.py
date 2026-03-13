#!/usr/bin/env python3
"""
Standalone k-NN grid search over k and distance metrics using 10-fold CV.
Uses the same splits for every (k, metric) test. Parallelized; reports best k and metric.
"""
import argparse
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm
import os
import time

# Optional scipy for extra metrics
try:
    from scipy.spatial.distance import canberra, braycurtis, sqeuclidean
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

    def sqeuclidean(u, v):
        return np.sum((np.asarray(u) - np.asarray(v)) ** 2)


def _levenshtein_vec(u, v):
    """Levenshtein-style distance for two 1D numeric arrays (same length).
    Uses substitution cost = 1 if elements differ (by tolerance), else 0.
    Insert/delete cost = 1. O(n^2) DP."""
    n = len(u)
    assert len(v) == n
    tol = 1e-9
    # dp[i][j] = edit distance between u[:i] and v[:j]
    dp = np.zeros((n + 1, n + 1), dtype=float)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(n + 1)
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            sub = 0.0 if np.abs(u[i - 1] - v[j - 1]) <= tol else 1.0
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + sub,
            )
    return dp[n, n]


def _angular_distance(u, v):
    """Angular distance based on cosine similarity: arccos(cos_sim).

    This is a monotone transform of cosine distance and uses L2 normalization
    internally, so it is effectively a normalized angular metric."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0.0 or nv == 0.0:
        return np.pi  # maximal angle if one vector is zero
    cos_sim = float(np.dot(u, v) / (nu * nv))
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return float(np.arccos(cos_sim))


def load_and_prepare(data_path: str, test_path: str = None):
    """Load CSV(s), extract X/y, normalize, return arrays and metadata."""
    if test_path:
        train_df = pd.read_csv(data_path)
        test_df = pd.read_csv(test_path)
        class_col = [c for c in train_df.columns if c.lower() == "class"][0]
        X = train_df.drop(columns=[class_col]).values
        y_raw = train_df[class_col].astype(str).values
        X_test = test_df.drop(columns=[class_col]).values
        y_test_raw = test_df[class_col].astype(str).values
        # Use train for CV; optionally concatenate for "all data" CV
        le = LabelEncoder()
        le.fit(np.unique(np.concatenate([y_raw, y_test_raw])))
        y = le.transform(y_raw)
        y_test = le.transform(y_test_raw)
    else:
        df = pd.read_csv(data_path)
        class_col = [c for c in df.columns if c.lower() == "class"][0]
        X = df.drop(columns=[class_col]).values
        y_raw = df[class_col].astype(str).values
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        X_test = y_test = None

    # Handle NaN
    col_mean = np.nanmean(X, axis=0)
    if np.isnan(X).any():
        X = np.where(np.isnan(X), col_mean, X)
    # Min-max normalize
    mn, mx = X.min(axis=0), X.max(axis=0)
    rng = mx - mn
    rng[rng == 0] = 1.0
    X = (X - mn) / rng
    if X_test is not None:
        if np.isnan(X_test).any():
            X_test = np.where(np.isnan(X_test), col_mean, X_test)
        X_test = (X_test - mn) / rng

    n_features = X.shape[1]
    n_samples = X.shape[0]
    return X, y, X_test, y_test, n_features, n_samples, le


def get_metric_config(metric_name: str):
    """Return (metric, metric_params, algorithm) for KNeighborsClassifier.
    metric can be string or callable."""
    cfg = {
        "euclidean": ("euclidean", {}, "auto"),
        "manhattan": ("manhattan", {}, "auto"),
        "hamming": ("hamming", {}, "auto"),
        "cosine": ("cosine", {}, "auto"),
        "correlation": ("correlation", {}, "auto"),
        "angular": (_angular_distance, {}, "brute"),
        "chebyshev": ("chebyshev", {}, "auto"),
        "squared_euclidean": (sqeuclidean, {}, "brute"),
        "canberra": ("canberra", {}, "auto"),
        "braycurtis": ("braycurtis", {}, "auto"),
        "levenshtein": (_levenshtein_vec, {}, "brute"),
        "mahalanobis": ("mahalanobis", {}, "brute"),  # VI set per fold
    }
    return cfg.get(metric_name.lower(), (metric_name, {}, "brute"))


METRIC_NAMES = [
    "euclidean",
    "manhattan",
    "hamming",
    "cosine",
    "correlation",
    "angular",
    "chebyshev",
    "squared_euclidean",
    "canberra",
    "braycurtis",
    "levenshtein",
    "mahalanobis",
]


def _build_class_distance_keys(labels, distances):
    """Precompute sorted distance lists and padded lexicographic keys per class for one sample.

    Returns:
        class_to_sorted: dict[class_label] -> 1D np.ndarray of sorted distances
        class_to_key:    dict[class_label] -> 1D np.ndarray of fixed length len(distances),
                         where missing positions are filled with +inf (for lexicographic compare).
    """
    class_to_dist = {}
    for d, c in zip(distances, labels):
        class_to_dist.setdefault(c, []).append(d)

    k = len(distances)
    class_to_sorted = {}
    class_to_key = {}
    for c, dists in class_to_dist.items():
        arr = np.sort(np.asarray(dists, dtype=float))
        class_to_sorted[c] = arr
        key = np.full(k, np.inf, dtype=float)
        key[: len(arr)] = arr
        class_to_key[c] = key
    return class_to_sorted, class_to_key


def _lex_is_better(key_a, key_b):
    """Return -1 if key_a is lexicographically better (smaller) than key_b,
    1 if worse, 0 if identical."""
    if np.array_equal(key_a, key_b):
        return 0
    # Compare element-wise
    for a, b in zip(key_a, key_b):
        if a < b:
            return -1
        if a > b:
            return 1
    return 0


def _predict_with_tiebreak(labels, distances):
    """Custom majority vote with distance-based deterministic tie-breaking for one sample.

    Args:
        labels: 1D array of neighbor labels (length k)
        distances: 1D array of neighbor distances (length k)

    Returns:
        Chosen class label.
    """
    labels = np.asarray(labels)
    distances = np.asarray(distances, dtype=float)

    # Majority vote
    unique, counts = np.unique(labels, return_counts=True)
    max_count = counts.max()
    tied_classes = unique[counts == max_count]

    # No tie: straight majority
    if len(tied_classes) == 1:
        return tied_classes[0]

    # Precompute per-class sorted distances and fixed-length keys (once per sample)
    class_to_sorted, class_to_key = _build_class_distance_keys(labels, distances)

    # Restrict to tied classes
    tied = list(tied_classes)

    # Copeland-style tournament over tied classes using lexicographic distance keys
    scores = {c: 0 for c in tied}
    for i in range(len(tied)):
        for j in range(i + 1, len(tied)):
            ci, cj = tied[i], tied[j]
            ki, kj = class_to_key[ci], class_to_key[cj]
            cmp = _lex_is_better(ki, kj)
            if cmp < 0:
                scores[ci] += 1
                scores[cj] -= 1
            elif cmp > 0:
                scores[ci] -= 1
                scores[cj] += 1
            # cmp == 0 -> no score change

    max_score = max(scores.values())
    best_classes = [c for c, s in scores.items() if s == max_score]

    if len(best_classes) == 1:
        return best_classes[0]

    # Still tied after Copeland: choose best lexicographically
    # using the precomputed keys; if still identical, smallest label wins.
    best_c = None
    best_key = None
    for c in best_classes:
        key = class_to_key[c]
        if best_c is None:
            best_c, best_key = c, key
        else:
            cmp = _lex_is_better(key, best_key)
            if cmp < 0:
                best_c, best_key = c, key
            elif cmp == 0 and c < best_c:
                best_c = c
                best_key = key
    return best_c


def run_one_k_metric(args):
    """Worker: (k, metric_name, X_train, y_train, X_test, y_test).
    Train on the full training set and evaluate on the held-out test set.
    Returns (k, metric_name, test_accuracy, elapsed_seconds)."""
    k, metric_name, X_train, y_train, X_test, y_test = args
    start = time.time()

    metric, metric_params, algorithm = get_metric_config(metric_name)
    # Mahalanobis needs VI computed from the training data
    if metric_name.lower() == "mahalanobis":
        try:
            cov = np.cov(X_train.T)
            cov += 1e-6 * np.eye(cov.shape[0])
            VI = np.linalg.inv(cov)
            metric_params = {"VI": VI}
        except np.linalg.LinAlgError:
            return (k, metric_name, 0.0, np.inf)

    try:
        clf = KNeighborsClassifier(
            n_neighbors=k,
            metric=metric,
            metric_params=metric_params if metric_params else None,
            algorithm=algorithm,
            n_jobs=1,
        )
        clf.fit(X_train, y_train)
        # Use custom prediction with deterministic distance-based tie-breaking
        dists, inds = clf.kneighbors(X_test, n_neighbors=k, return_distance=True)
        preds = []
        for dist_row, ind_row in zip(dists, inds):
            neighbor_labels = y_train[ind_row]
            preds.append(_predict_with_tiebreak(neighbor_labels, dist_row))
        preds = np.asarray(preds)
        acc = accuracy_score(y_test, preds)
    except Exception:
        acc = np.nan

    elapsed = time.time() - start
    return (k, metric_name, acc, elapsed)


def main():
    parser = argparse.ArgumentParser(description="k-NN k and distance metric search with 10-fold CV")
    parser.add_argument("--data", required=True, help="Training data CSV (or single dataset)")
    parser.add_argument("--test-data", default=None, help="Optional test CSV (CV is on train only)")
    parser.add_argument("--n-jobs", type=int, default=None, help="Parallel jobs (default: CPU count - 1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for folds")
    parser.add_argument(
        "--k-values",
        help="Comma-separated list of k values to test (overrides automatic sqrt/full ranges)",
    )
    parser.add_argument(
        "--metrics",
        help="Comma-separated list of distance metrics to test "
             f"(subset of: {', '.join(METRIC_NAMES)})",
    )
    parser.add_argument(
        "--log-each-test",
        action="store_true",
        help="Print per-(k, metric) accuracy and timing while still showing tqdm",
    )
    parser.add_argument(
        "--slow-threshold",
        type=float,
        default=60.0,
        help="Seconds above which a single (k, metric) run is considered slow and triggers a warning",
    )
    args = parser.parse_args()

    print("Loading data...")
    X, y, X_test, y_test, n_features, n_samples, le = load_and_prepare(args.data, args.test_data)
    if X_test is None or y_test is None:
        raise SystemExit("This script now requires --test-data so we can train on the training set and evaluate on a held-out test set.")
    print(f"Samples: {n_samples}, features: {n_features}, test samples: {X_test.shape[0]}")

    k_sqrt = max(1, int(sqrt(n_features)))
    # k cannot exceed number of training samples
    train_size = n_samples
    k_max_full = min(n_features, train_size)

    # Determine which metrics to use
    if args.metrics:
        requested = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
        selected_metrics = []
        for m in METRIC_NAMES:
            if m.lower() in requested:
                selected_metrics.append(m)
        if not selected_metrics:
            raise ValueError(f"No valid metrics selected from {requested}; available: {METRIC_NAMES}")
        metric_names = selected_metrics
    else:
        metric_names = METRIC_NAMES

    # Build tasks: (k, metric) from lowest k to highest; for each k all metrics
    if args.k_values:
        k_values = sorted({
            int(x) for x in args.k_values.split(",") if x.strip()
        })
        # Filter invalid ks (must be >=1 and <= train_size)
        k_values = [k for k in k_values if 1 <= k <= train_size]
        if not k_values:
            raise ValueError("No valid k values after filtering; ensure 1 <= k <= number of training samples")
        tasks_all = [(k, m) for k in k_values for m in metric_names]
        phases = [("All k", tasks_all)]
    else:
        # Default two-phase behaviour based on dimensionality
        # Phase 1: k = 1 .. sqrt(dim)
        k_phase1 = list(range(1, k_sqrt + 1))
        # Phase 2: k = sqrt(dim)+1 .. full dimensionality (cap by train size)
        k_phase2 = list(range(k_sqrt + 1, k_max_full + 1))
        tasks_phase1 = [(k, m) for k in k_phase1 for m in metric_names]
        tasks_phase2 = [(k, m) for k in k_phase2 for m in metric_names]
        phases = [("Phase 1", tasks_phase1), ("Phase 2", tasks_phase2)]

    def run_tasks(tasks, phase_name):
        task_args = [
            (k, metric_name, X, y, X_test, y_test)
            for k, metric_name in tasks
        ]
        n_jobs = args.n_jobs or max(1, multiprocessing.cpu_count() - 1)
        results = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(run_one_k_metric, a): a for a in task_args}
            for future in tqdm(as_completed(futures), total=len(futures), desc=phase_name):
                try:
                    k, metric_name, mean_acc, elapsed = future.result()
                except Exception:
                    a = futures[future]
                    k, metric_name, mean_acc, elapsed = a[0], a[1], np.nan, np.nan
                results.append((k, metric_name, mean_acc, elapsed))
                # Optional per-test logging and slowdown alerts; use tqdm.write so bar stays clean
                if args.log_each_test:
                    tqdm.write(
                        f"[{phase_name}] k={k}, metric={metric_name}, "
                        f"accuracy={mean_acc:.4f}, time={elapsed:.2f}s"
                    )
                if elapsed is not None and not np.isnan(elapsed) and elapsed > args.slow_threshold:
                    tqdm.write(
                        f"WARNING: [{phase_name}] k={k}, metric={metric_name} took "
                        f"{elapsed:.2f}s (> {args.slow_threshold:.1f}s); "
                        f"search may be slowing toward infeasible runtimes."
                    )
        return results

    all_results = []
    for phase_name, tasks in phases:
        if phase_name == "Phase 1":
            print(f"\nPhase 1: k from 1 to sqrt(dim) = {k_sqrt}")
        elif phase_name == "Phase 2":
            print(f"\nPhase 2: k from {k_sqrt + 1} to {k_max_full}")
        else:
            print(f"\n{phase_name}")
        all_results.extend(run_tasks(tasks, phase_name))

    # Output: k, metric, accuracy (low to high k)
    all_results.sort(key=lambda r: (r[0], r[1]))

    print("\n--- Results (k, metric, test-set accuracy) ---")
    for k, metric, acc, elapsed in all_results:
        print(f"k={k:3d}  metric={metric:22s}  accuracy={acc:.4f}  time={elapsed:.2f}s")

    valid = [(k, m, a, t) for k, m, a, t in all_results if not np.isnan(a)]
    if not valid:
        print("No valid results.")
        return
    best = max(valid, key=lambda x: x[2])
    print("\n--- Best (train on train, evaluate on test) ---")
    print(f"Best k: {best[0]}, Best metric: {best[1]}, test accuracy: {best[2]:.4f}")

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "knn_metric_search_results.csv")
    with open(out_path, "w") as f:
        f.write("k,metric,accuracy_test,time_seconds\n")
        for k, metric, acc, elapsed in all_results:
            f.write(f"{k},{metric},{acc:.6f},{elapsed:.4f}\n")
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
