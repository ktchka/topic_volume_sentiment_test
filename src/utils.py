"""
Common utilities for topic volume and sentiment validation.
"""

import hashlib
import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_json_data(file_path: str) -> dict[str, Any]:
    """Load JSON data file."""
    with open(file_path) as f:
        return json.load(f)


def save_json_data(data: dict[str, Any], file_path: str) -> None:
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Convert numpy types to Python native types
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    converted_data = convert_numpy_types(data)

    with open(file_path, 'w') as f:
        json.dump(converted_data, f, indent=2)


def create_text_hash(text: str) -> str:
    """Create a hash for text to use as cache key."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def load_cache(cache_file: str) -> dict[str, Any]:
    """Load cache from file."""
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def save_cache(cache: dict[str, Any], cache_file: str) -> None:
    """Save cache to file."""
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)


def validate_data_integrity(tickets: list[dict[str, Any]]) -> None:
    """Validate basic data integrity."""
    if not tickets:
        raise ValueError("No tickets found in data")

    required_fields = ['original_message', 'sentiment__filter']
    for i, ticket in enumerate(tickets):
        for field in required_fields:
            if field not in ticket:
                raise ValueError(f"Ticket {i} missing required field: {field}")

        if not ticket.get('original_message', '').strip():
            raise ValueError(f"Ticket {i} has empty original_message")


def stratified_sample(df: pd.DataFrame, column: str, n_per_group: int,
                     random_state: int = 42) -> pd.DataFrame:
    """Create stratified sample from DataFrame."""
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in DataFrame")

    # Group by the specified column
    groups = df.groupby(column)

    sampled_dfs = []
    for _group_name, group_df in groups:
        if len(group_df) >= n_per_group:
            # Sample n_per_group from this group
            sampled = group_df.sample(n=n_per_group, random_state=random_state)
        else:
            # Take all available samples
            sampled = group_df

        sampled_dfs.append(sampled)

    # Combine all sampled groups
    if sampled_dfs:
        return pd.concat(sampled_dfs, ignore_index=True)
    else:
        return pd.DataFrame()


def calculate_confidence_interval(data: list[float], confidence: float = 0.95) -> tuple[float, float]:
    """Calculate confidence interval for a list of values."""
    if not data:
        return 0.0, 0.0

    data_array = np.array(data)
    n = len(data_array)
    mean = np.mean(data_array)
    std_err = stats.sem(data_array)

    # Calculate confidence interval
    h = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)

    return mean - h, mean + h


def bootstrap_confidence_interval(data: list[float], n_bootstrap: int = 1000,
                                confidence: float = 0.95) -> tuple[float, float]:
    """Calculate bootstrap confidence interval."""
    if not data:
        return 0.0, 0.0

    data_array = np.array(data)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        # Sample with replacement
        bootstrap_sample = np.random.choice(data_array, size=len(data_array), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))

    # Calculate confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)

    return lower_bound, upper_bound


def print_validation_results(results: dict[str, Any], title: str) -> None:
    """Print validation results in a formatted way."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

    if 'overall_pass' in results:
        status = "✅ PASS" if results['overall_pass'] else "❌ FAIL"
        print(f"Overall Status: {status}")

    if 'metrics' in results:
        print("\nMetrics:")
        for metric, value in results['metrics'].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

    if 'validation_results' in results:
        print("\nThreshold Validation:")
        for check, passed in results['validation_results'].items():
            status = "✅" if passed else "❌"
            print(f"  {check}: {status}")

    print(f"{'='*60}\n")
