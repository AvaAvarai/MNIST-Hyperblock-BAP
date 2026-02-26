"""
Print dataset statistics: total cases, dimensionality, classes, class distribution.
"""

import pandas as pd


def print_dataset_stats(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
    dim: int,
    train_name: str = "train",
    test_name: str = "test",
) -> None:
    """
    Print statistics for train, test, and combined datasets.
    """
    n_train = len(train_df)
    n_test = len(test_df)
    n_total = n_train + n_test

    classes = sorted(train_df[label_col].unique().tolist())
    n_classes = len(classes)

    def _distribution(df: pd.DataFrame) -> dict:
        return df[label_col].value_counts().sort_index().to_dict()

    dist_train = _distribution(train_df)
    dist_test = _distribution(test_df)
    combined = pd.concat([train_df, test_df], ignore_index=True)
    dist_combined = _distribution(combined)

    def _format_dist(d: dict, total: int) -> str:
        return ", ".join(
            f"{c}: {n:,} ({100 * n / total:.1f}%)" if total > 0 else f"{c}: {n:,}"
            for c, n in d.items()
        )

    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"\nTrain ({train_name}):")
    print(f"  Total cases:  {n_train:,}")
    print(f"  Dimensionality: {dim}")
    print(f"  Classes: {classes}")
    print(f"  Distribution: {_format_dist(dist_train, n_train)}")

    print(f"\nTest ({test_name}):")
    print(f"  Total cases:  {n_test:,}")
    print(f"  Dimensionality: {dim}")
    print(f"  Classes: {classes}")
    print(f"  Distribution: {_format_dist(dist_test, n_test)}")

    print(f"\nCombined (train + test):")
    print(f"  Total cases:  {n_total:,}")
    print(f"  Dimensionality: {dim}")
    print(f"  Classes: {classes}")
    print(f"  Distribution: {_format_dist(dist_combined, n_total)}")
    print("=" * 60)
