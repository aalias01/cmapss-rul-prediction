"""
Plotting utilities for CMAPSS turbofan RUL prediction.

These helpers are used in the EDA and feature-engineering notebooks to produce
the visualisations that drive modelling decisions: degradation trajectories
(justify the rolling window), the RUL distribution (justify the 125-cycle cap),
and sensor-level variance or correlation plots.

All functions accept an optional `ax` argument so they can be embedded in
multi-panel figures alongside other plots.
"""

import matplotlib.pyplot as plt
import pandas as pd


def plot_degradation_trajectories(
    df: pd.DataFrame,
    sensor: str,
    n_units: int = 10,
    ax=None,
) -> plt.Axes:
    """
    Plot a sensor's reading over cycle for a sample of engines.

    Each line represents one engine unit. The resulting overlay reveals
    whether a sensor has a consistent directional trend as engines approach
    failure — the key visual criterion for deciding which sensors are
    informative predictors of RUL.

    Parameters
    ----------
    df : pd.DataFrame
        Training DataFrame with 'unit', 'cycle', and the requested sensor column.
    sensor : str
        Name of the sensor column to plot (e.g. 'sensor_2').
    n_units : int, optional
        Number of engines to overlay. Defaults to 10. Uses the first n_units
        unique engine IDs in the DataFrame.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, a new figure and axes are created.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object, for further customisation or tight_layout() calls.
    """
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


def plot_rul_distribution(df: pd.DataFrame, ax=None) -> plt.Axes:
    """
    Plot the distribution of clipped RUL labels across all training rows.

    The spike at 125 cycles represents the early-life plateau introduced by
    the piecewise-linear RUL cap. The remainder of the distribution reflects
    the accumulation of late-stage degradation cycles. Use this plot to verify
    that the cap is in effect and that the distribution has enough late-life
    examples for the model to learn from.

    Parameters
    ----------
    df : pd.DataFrame
        Training DataFrame with a 'rul' column produced by src/features.add_rul().
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, a new figure and axes are created.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object, for further customisation or tight_layout() calls.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["rul"], bins=40, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Remaining Useful Life (cycles)")
    ax.set_ylabel("Count")
    ax.set_title("RUL distribution (clipped at 125 cycles)")
    return ax
