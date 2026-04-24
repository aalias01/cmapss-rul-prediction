"""
Shared utilities for CMAPSS project.
"""

import matplotlib.pyplot as plt
import pandas as pd


def plot_degradation_trajectories(df: pd.DataFrame, sensor: str, n_units: int = 10, ax=None):
    """Plot sensor readings over cycle for a sample of engines to visualize degradation."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))
    sample_units = df["unit"].unique()[:n_units]
    for unit in sample_units:
        unit_df = df[df["unit"] == unit]
        ax.plot(unit_df["cycle"], unit_df[sensor], alpha=0.6, linewidth=0.8)
    ax.set_xlabel("Cycle")
    ax.set_ylabel(sensor)
    ax.set_title(f"Degradation trajectory — {sensor} ({n_units} engines)")
    return ax


def plot_rul_distribution(df: pd.DataFrame, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["rul"], bins=40, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Remaining Useful Life (cycles)")
    ax.set_ylabel("Count")
    ax.set_title("RUL distribution (clipped at 125 cycles)")
    return ax
