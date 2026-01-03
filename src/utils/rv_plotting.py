"""
Plotting Utilities for Realized Volatility Analysis

Functions for creating standard plots for volatility forecasting notebooks.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from typing import Optional, Tuple, List
from pathlib import Path


def plot_return_diagnostics(
    returns: pd.Series,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create histogram and QQ-plot for return distribution.
    
    Parameters
    ----------
    returns : pd.Series
        Log returns series
    figsize : tuple, default=(14, 5)
        Figure size
    save_path : Path, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram with normal overlay
    ax = axes[0]
    n, bins, patches = ax.hist(
        returns, 
        bins=50, 
        density=True, 
        alpha=0.7,
        color='steelblue', 
        edgecolor='black', 
        label='Empirical'
    )
    
    # Fit normal distribution
    mu, std = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    normal_density = stats.norm.pdf(x, mu, std)
    ax.plot(
        x, 
        normal_density, 
        'r-', 
        linewidth=2, 
        label=f'Normal(μ={mu:.4f}, σ={std:.4f})'
    )
    
    ax.set_xlabel('Log Return', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Histogram of Daily Log Returns', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # QQ plot
    ax = axes[1]
    stats.probplot(returns, dist="norm", plot=ax)
    ax.set_xlabel('Theoretical Quantiles', fontsize=11)
    ax.set_ylabel('Sample Quantiles', fontsize=11)
    ax.set_title('QQ Plot: Log Returns vs Normal', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_rv_distribution(
    rv: pd.Series,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot realized volatility distribution (linear and log scale).
    
    Parameters
    ----------
    rv : pd.Series
        Realized volatility series
    figsize : tuple, default=(14, 5)
        Figure size
    save_path : Path, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Linear scale histogram
    ax = axes[0]
    ax.hist(rv, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Realized Volatility (RV)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Distribution of Realized Volatility', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Log scale histogram
    ax = axes[1]
    rv_nonzero = rv[rv > 0]
    ax.hist(np.log10(rv_nonzero), bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax.set_xlabel('log₁₀(RV)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Realized Volatility (Log Scale)', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_time_series_comparison(
    data: pd.DataFrame,
    cols: List[str],
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot multiple time series for comparison.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with time series data
    cols : list of str
        Column names to plot
    labels : list of str, optional
        Labels for legend
    colors : list of str, optional
        Colors for each series
    figsize : tuple, default=(14, 5)
        Figure size
    save_path : Path, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is None:
        labels = cols
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(cols)))
    
    for col, label, color in zip(cols, labels, colors):
        ax.plot(data.index, data[col], label=label, color=color, linewidth=1.5)
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_acf_comparison(
    data: pd.DataFrame,
    cols: List[str],
    titles: Optional[List[str]] = None,
    lags: int = 60,
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot autocorrelation functions side-by-side.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with time series data
    cols : list of str
        Column names to compute ACF for
    titles : list of str, optional
        Titles for each subplot
    lags : int, default=60
        Number of lags
    figsize : tuple, default=(12, 4)
        Figure size
    save_path : Path, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    n_plots = len(cols)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    if n_plots == 1:
        axes = [axes]
    
    if titles is None:
        titles = [f'ACF of {col}' for col in cols]
    
    for ax, col, title in zip(axes, cols, titles):
        plot_acf(data[col].dropna(), lags=lags, ax=ax)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Lag', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_rolling_statistics(
    series: pd.Series,
    windows: List[int],
    stat: str = 'mean',
    labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot rolling statistics (mean, std, etc.) at multiple windows.
    
    Parameters
    ----------
    series : pd.Series
        Time series data
    windows : list of int
        Window sizes for rolling calculation
    stat : str, default='mean'
        Statistic to compute ('mean', 'std', 'median')
    labels : list of str, optional
        Labels for each window
    figsize : tuple, default=(14, 6)
        Figure size
    save_path : Path, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is None:
        labels = [f'{w}d rolling {stat}' for w in windows]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(windows)))
    
    for window, label, color in zip(windows, labels, colors):
        if stat == 'mean':
            rolling = series.rolling(window).mean()
        elif stat == 'std':
            rolling = series.rolling(window).std()
        elif stat == 'median':
            rolling = series.rolling(window).median()
        else:
            raise ValueError(f"Unknown stat: {stat}")
        
        ax.plot(series.index, rolling, label=label, color=color, linewidth=1.5)
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel(f'Rolling {stat.capitalize()}', fontsize=11)
    ax.set_title(
        f'Rolling {stat.capitalize()} of {series.name}', 
        fontsize=12, 
        fontweight='bold'
    )
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_rv_squared_returns_comparison(
    rv: pd.Series,
    squared_returns: pd.Series,
    sample_size: int = 500,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Compare realized volatility with squared returns proxy.
    
    Parameters
    ----------
    rv : pd.Series
        Realized volatility
    squared_returns : pd.Series
        Squared returns
    sample_size : int, default=500
        Number of recent observations to plot
    figsize : tuple, default=(14, 5)
        Figure size
    save_path : Path, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Take recent sample
    rv_sample = rv.iloc[-sample_size:]
    sr_sample = squared_returns.iloc[-sample_size:]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Time series comparison
    ax = axes[0]
    ax.plot(rv_sample.index, rv_sample, label='RV (high-frequency)', alpha=0.8, linewidth=1.5)
    ax.plot(sr_sample.index, sr_sample, label='r² (daily proxy)', alpha=0.6, linewidth=1)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Volatility Proxy', fontsize=11)
    ax.set_title('RV vs Squared Returns (Recent Period)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Scatter plot
    ax = axes[1]
    ax.scatter(rv_sample, sr_sample, alpha=0.5, s=10)
    
    # Add diagonal line
    max_val = max(rv_sample.max(), sr_sample.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1, label='Perfect agreement')
    
    ax.set_xlabel('RV (high-frequency)', fontsize=11)
    ax.set_ylabel('r² (daily proxy)', fontsize=11)
    ax.set_title('RV vs Squared Returns Correlation', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
