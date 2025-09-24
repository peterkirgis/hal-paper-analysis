"""
Pareto frontier computation utilities for cost-accuracy analysis.

This module provides functions to compute Pareto-efficient points for cost vs accuracy
analysis, where lower cost is better and higher accuracy is better.
"""

import numpy as np
import pandas as pd


def _pareto_indices(resources: np.ndarray, accs: np.ndarray) -> np.ndarray:
    """
    Return index array of points on the Pareto frontier.

    Args:
        resources: Array of resource values (cost, tokens, etc.)
        accs: Array of accuracy values

    Returns:
        Array of indices of Pareto-efficient points
    """
    if len(resources) != len(accs):
        raise ValueError("resources and accs must have the same length")

    if len(resources) == 0:
        return np.array([], dtype=int)

    # Get Pareto frontier points including origin
    frontier = compute_pareto_frontier_with_origin(resources, accs)

    # Find original indices by matching coordinates (exclude origin at [0,0])
    pareto_indices = []
    for x, y in frontier[1:]:  # Skip origin
        for i, (res, acc) in enumerate(zip(resources, accs)):
            if abs(res - x) < 1e-10 and abs(acc - y) < 1e-10:
                pareto_indices.append(i)
                break

    return np.array(pareto_indices, dtype=int)


def compute_pareto_frontier_with_origin(resources: np.ndarray, accs: np.ndarray) -> np.ndarray:
    """
    Compute Pareto frontier points, always starting from (0,0).

    Args:
        resources: Array of resource values (cost, tokens, etc.)
        accs: Array of accuracy values

    Returns:
        Array of [resource, accuracy] points including (0,0) as first point
    """
    if len(resources) != len(accs):
        raise ValueError("resources and accs must have the same length")

    # Create points array and add origin
    points = np.column_stack([resources, accs])
    points = np.vstack([[0, 0], points])

    # Sort by resource first, then accuracy
    points = points[np.lexsort((points[:, 1], points[:, 0]))]

    # Compute convex hull
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    hull = []
    for p in points:
        while len(hull) >= 2 and cross(hull[-2], hull[-1], p) >= 0:
            hull.pop()
        hull.append(tuple(p))

    # Keep only points with non-decreasing accuracy
    hull = [(x, y) for i, (x, y) in enumerate(hull)
            if i == 0 or y >= hull[i-1][1]]

    return np.array(hull)


def compute_pareto_frontier_flexible(df: pd.DataFrame, resource_col: str = 'total_cost', accuracy_col: str = 'accuracy') -> pd.DataFrame:
    """
    Compute Pareto frontier for any resource column vs accuracy.

    Args:
        df: DataFrame containing the data
        resource_col: Name of the resource column (e.g., 'total_cost', 'total_tokens')
        accuracy_col: Name of the accuracy column

    Returns:
        DataFrame with only the Pareto-efficient points
    """
    if resource_col not in df.columns:
        raise ValueError(f"Resource column '{resource_col}' not found in DataFrame")
    if accuracy_col not in df.columns:
        raise ValueError(f"Accuracy column '{accuracy_col}' not found in DataFrame")

    # Drop rows with missing values for the specified columns
    clean_df = df.dropna(subset=[resource_col, accuracy_col]).copy()

    if len(clean_df) == 0:
        return pd.DataFrame()

    resources = clean_df[resource_col].to_numpy()
    accs = clean_df[accuracy_col].to_numpy()

    # Get Pareto indices
    p_idx = _pareto_indices(resources, accs)

    # Return the Pareto-efficient rows
    return clean_df.iloc[p_idx].copy()


def compute_pareto_statistics(df: pd.DataFrame, resource_col: str = 'total_cost') -> pd.DataFrame:
    """
    Compute the number and fraction of models on the Pareto frontier for each benchmark.
    Now accounts for the origin (0,0) as the baseline point and works with any resource column.

    Args:
        df: DataFrame containing benchmark data with columns:
            - benchmark_name: Name of benchmark
            - model: Model name
            - {resource_col}: Resource values (cost, tokens, etc.)
            - accuracy: Accuracy values

    Returns:
        DataFrame with Pareto statistics per benchmark
    """
    required_cols = ["benchmark_name", "model", resource_col, "accuracy"]

    # Aggregate to model × benchmark means
    plot_df = (
        df.dropna(subset=required_cols)
          .groupby(["benchmark_name","model"], as_index=False)
          .agg({resource_col: "mean", "accuracy": "mean"})
    )

    pareto_stats = []

    for bench in sorted(plot_df['benchmark_name'].unique()):
        sub = plot_df[plot_df['benchmark_name'] == bench].copy()

        if sub.empty:
            continue

        total_models = len(sub)
        resources = sub[resource_col].to_numpy()
        accs = sub['accuracy'].to_numpy()

        # Add origin point (0,0) to the comparison set
        resources_with_origin = np.concatenate([[0], resources])
        accs_with_origin = np.concatenate([[0], accs])

        # Compute Pareto frontier including origin
        p_idx = _pareto_indices(resources_with_origin, accs_with_origin)

        # Count only actual models (exclude origin at index 0)
        pareto_models = len([idx for idx in p_idx if idx > 0])
        pareto_fraction = pareto_models / total_models

        pareto_stats.append({
            'benchmark_name': bench,
            'total_models': total_models,
            'pareto_models': pareto_models,
            'pareto_fraction': pareto_fraction,
            'resource_column': resource_col
        })

    return pd.DataFrame(pareto_stats)


def compute_model_pareto_rankings(df: pd.DataFrame, resource_col: str = 'total_cost') -> pd.DataFrame:
    """
    Rank models by how frequently they appear on Pareto frontiers across benchmarks.
    Now accounts for the origin (0,0) as the baseline point and works with any resource column.

    Args:
        df: DataFrame containing benchmark data with columns:
            - benchmark_name: Name of benchmark
            - model: Model name
            - {resource_col}: Resource values (cost, tokens, etc.)
            - accuracy: Accuracy values

    Returns:
        DataFrame with model rankings by Pareto efficiency
    """
    required_cols = ["benchmark_name", "model", resource_col, "accuracy"]

    # Aggregate to model × benchmark means
    plot_df = (
        df.dropna(subset=required_cols)
          .groupby(["benchmark_name","model"], as_index=False)
          .agg({resource_col: "mean", "accuracy": "mean"})
    )

    model_pareto_counts = {}
    model_benchmark_counts = {}

    # Count how many times each model appears on Pareto frontiers
    for bench in sorted(plot_df['benchmark_name'].unique()):
        sub = plot_df[plot_df['benchmark_name'] == bench].copy()

        if sub.empty:
            continue

        resources = sub[resource_col].to_numpy()
        accs = sub['accuracy'].to_numpy()

        # Add origin point (0,0) to the comparison set
        resources_with_origin = np.concatenate([[0], resources])
        accs_with_origin = np.concatenate([[0], accs])

        # Compute Pareto frontier including origin
        p_idx = _pareto_indices(resources_with_origin, accs_with_origin)

        # Filter out the origin (index 0) to get only actual model indices
        actual_model_indices = [idx - 1 for idx in p_idx if idx > 0]
        pareto_models = sub.iloc[actual_model_indices]['model'].tolist()

        # Count appearances for each model
        for model in sub['model']:
            model_benchmark_counts[model] = model_benchmark_counts.get(model, 0) + 1
            if model in pareto_models:
                model_pareto_counts[model] = model_pareto_counts.get(model, 0) + 1

    # Create ranking DataFrame
    rankings = []
    for model in model_benchmark_counts:
        pareto_count = model_pareto_counts.get(model, 0)
        benchmark_count = model_benchmark_counts[model]
        pareto_rate = pareto_count / benchmark_count

        rankings.append({
            'model': model,
            'benchmarks_tested': benchmark_count,
            'pareto_appearances': pareto_count,
            'pareto_rate': pareto_rate,
            'resource_column': resource_col
        })

    # Sort by Pareto rate (descending), then by absolute count
    rankings_df = pd.DataFrame(rankings)
    rankings_df = rankings_df.sort_values(['pareto_rate', 'pareto_appearances'], ascending=[False, False])
    rankings_df['rank'] = range(1, len(rankings_df) + 1)

    return rankings_df