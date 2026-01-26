"""
Plotting Functions for Baseline Model Analysis

Visualization functions specific to baseline model evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List, Tuple


def plot_baseline_forecasts_timeseries(
    actual: pd.Series,
    forecasts: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot actual RV with all baseline forecasts overlaid.

    Parameters
    ----------
    actual : pd.Series
        Actual realized volatility
    forecasts : pd.DataFrame
        DataFrame with baseline forecast columns
    start_date : str, optional
        Start date for plotting window
    end_date : str, optional
        End date for plotting window
    figsize : tuple, default=(16, 6)
        Figure size
    save_path : Path, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Filter date range if specified
    if start_date or end_date:
        mask = pd.Series(True, index=actual.index)
        if start_date:
            mask &= (actual.index >= start_date)
        if end_date:
            mask &= (actual.index <= end_date)
        actual_plot = actual[mask]
        forecasts_plot = forecasts[mask]
    else:
        actual_plot = actual
        forecasts_plot = forecasts

    # Plot actual
    ax.plot(actual_plot.index, actual_plot, 
            color='black', linewidth=1.5, alpha=0.7, label='Actual RV', zorder=3)

    # Plot forecasts
    colors = {'persistence': '#3498db', 'rolling_avg': '#e74c3c', 'ewma': '#2ecc71'}
    linestyles = {'persistence': '-', 'rolling_avg': '--', 'ewma': '-.'}

    for col in forecasts_plot.columns:
        color = colors.get(col, None)
        linestyle = linestyles.get(col, '-')
        ax.plot(forecasts_plot.index, forecasts_plot[col],
                color=color, linestyle=linestyle, linewidth=1.2, 
                alpha=0.8, label=col.replace('_', ' ').title(), zorder=2)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Realized Volatility', fontsize=12)
    ax.set_title('Baseline Forecasts vs Actual Realized Volatility', 
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_metrics_comparison(
    metrics_df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot bar charts comparing metrics across baseline models.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame with models as rows and metrics as columns
    figsize : tuple, default=(14, 5)
        Figure size
    save_path : Path, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Determine number of metrics
    n_metrics = len(metrics_df.columns)

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    colors = {'persistence': '#3498db', 'rolling_avg': '#e74c3c', 'ewma': '#2ecc71'}

    for idx, metric in enumerate(metrics_df.columns):
        ax = axes[idx]
        
        # Get colors for each model
        bar_colors = [colors.get(model, '#95a5a6') for model in metrics_df.index]
        
        # Plot bars
        bars = ax.bar(range(len(metrics_df)), metrics_df[metric], 
                    color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1)
        
        # Add value labels
        for i, (model, value) in enumerate(zip(metrics_df.index, metrics_df[metric])):
            ax.text(i, value, f'{value:.6f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xticks(range(len(metrics_df)))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_df.index], 
                        rotation=0, ha='center')
        ax.set_ylabel(metric.upper(), fontsize=11)
        ax.set_title(f'{metric.upper()} Comparison', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_regime_conditional_metrics(
    regime_metrics_dict: Dict[str, pd.DataFrame],
    metric: str = 'rmse',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot regime-conditional performance for all baseline models.

    Parameters
    ----------
    regime_metrics_dict : dict
        Dictionary mapping model names to regime metrics DataFrames
    metric : str, default='rmse'
        Metric to plot ('rmse', 'mae', or 'r2')
    figsize : tuple, default=(12, 6)
        Figure size
    save_path : Path, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Extract regimes
    first_model = list(regime_metrics_dict.keys())[0]
    regimes = regime_metrics_dict[first_model].index.tolist()

    # Set up bar positions
    n_regimes = len(regimes)
    n_models = len(regime_metrics_dict)
    x = np.arange(n_regimes)
    width = 0.25

    colors = {'persistence': '#3498db', 'rolling_avg': '#e74c3c', 'ewma': '#2ecc71'}

    # Plot bars for each model
    for idx, (model_name, metrics_df) in enumerate(regime_metrics_dict.items()):
        offset = width * (idx - n_models/2 + 0.5)
        values = [metrics_df.loc[r, metric] for r in regimes]
        
        ax.bar(x + offset, values, width, 
            label=model_name.replace('_', ' ').title(),
            color=colors.get(model_name, '#95a5a6'),
            alpha=0.7, edgecolor='black', linewidth=0.8)

    ax.set_xlabel('Volatility Regime', fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(f'{metric.upper()} by Regime Across Baseline Models', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_forecast_error_distribution(
    actual: pd.Series,
    forecasts: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot forecast error distributions for all baseline models.

    Parameters
    ----------
    actual : pd.Series
        Actual realized volatility
    forecasts : pd.DataFrame
        DataFrame with baseline forecast columns
    figsize : tuple, default=(14, 5)
        Figure size
    save_path : Path, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    n_models = len(forecasts.columns)
    fig, axes = plt.subplots(1, n_models, figsize=figsize, sharey=True)

    if n_models == 1:
        axes = [axes]

    colors = {'persistence': '#3498db', 'rolling_avg': '#e74c3c', 'ewma': '#2ecc71'}

    for idx, model in enumerate(forecasts.columns):
        ax = axes[idx]
        
        # Compute errors
        errors = actual - forecasts[model]
        errors = errors.dropna()
        
        # Plot histogram
        ax.hist(errors, bins=50, color=colors.get(model, '#95a5a6'),
            alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add vertical line at zero
        ax.axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Add statistics
        mean_error = errors.mean()
        std_error = errors.std()
        ax.text(0.05, 0.95, 
            f'Mean: {mean_error:.6f}\nStd: {std_error:.6f}',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Forecast Error', fontsize=11)
        if idx == 0:
            ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(model.replace('_', ' ').title(), 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Forecast Error Distributions', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig