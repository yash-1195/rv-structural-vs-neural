"""
Plotting Functions for Regime Analysis

This module provides visualization functions specific to regime definition
and validation analysis.

Author: Victor
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List, Tuple


def plot_regime_distributions(
    distributions: Dict[str, pd.Series],
    figsize: Tuple[int, int] = (14, 10),
    bins: int = 50,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot separate histograms of RV for each regime in horizontal subplots.
    
    Parameters
    ----------
    distributions : dict
        Dictionary mapping regime name to RV series
    figsize : tuple, default=(14, 10)
        Figure size
    bins : int, default=50
        Number of histogram bins
    save_path : Path or None
        If provided, save figure to this path
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    # Define colors for regimes
    colors = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}
    
    # Create figure with 3 horizontal subplots
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    regime_order = ['Low', 'Medium', 'High']
    
    for idx, regime_name in enumerate(regime_order):
        if regime_name in distributions:
            data = distributions[regime_name]
            
            # Plot histogram
            axes[idx].hist(
                data,
                bins=bins,
                color=colors[regime_name],
                alpha=0.7,
                edgecolor='black',
                linewidth=0.5
            )
            
            # Add regime label and sample size
            axes[idx].set_ylabel('Frequency', fontsize=11)
            axes[idx].text(
                0.98, 0.95, 
                f'{regime_name} Regime (n={len(data)})',
                transform=axes[idx].transAxes,
                fontsize=12,
                fontweight='bold',
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
            axes[idx].grid(True, alpha=0.3, axis='y')
    
    # Set common xlabel only on bottom subplot
    axes[2].set_xlabel('Realized Volatility (RV)', fontsize=12)
    
    # Add overall title
    fig.suptitle('Distribution of Realized Volatility by Regime', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_regime_boxplots(
    rv: pd.Series,
    regimes: pd.Series,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot boxplots of RV by regime.
    
    Parameters
    ----------
    rv : pd.Series
        Realized volatility series
    regimes : pd.Series
        Regime labels
    figsize : tuple, default=(10, 6)
        Figure size
    save_path : Path or None
        If provided, save figure to this path
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create dataframe
    df = pd.DataFrame({'rv': rv, 'regime': regimes}).dropna()
    
    # Create boxplot
    colors = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}
    
    box_parts = ax.boxplot(
        [df[df['regime'] == r]['rv'] for r in ['Low', 'Medium', 'High']],
        labels=['Low', 'Medium', 'High'],
        patch_artist=True,
        widths=0.6
    )
    
    # Color boxes
    for patch, regime in zip(box_parts['boxes'], ['Low', 'Medium', 'High']):
        patch.set_facecolor(colors[regime])
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Realized Volatility (RV)', fontsize=12)
    ax.set_xlabel('Regime', fontsize=12)
    ax.set_title('Realized Volatility Distribution by Regime', fontsize=14, pad=15)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_regime_timeseries(
    rv: pd.Series,
    regimes: pd.Series,
    trailing_stat: Optional[pd.Series] = None,
    figsize: Tuple[int, int] = (16, 8),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot realized volatility with regime overlays.
    
    Parameters
    ----------
    rv : pd.Series
        Realized volatility series
    regimes : pd.Series
        Regime labels
    trailing_stat : pd.Series or None
        Optional trailing statistic to overlay
    figsize : tuple, default=(16, 8)
        Figure size
    save_path : Path or None
        If provided, save figure to this path
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    # Create figure
    if trailing_stat is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
    
    # Create dataframe
    df = pd.DataFrame({
        'rv': rv,
        'regime': regimes
    }).dropna()
    
    # Define regime colors
    regime_colors = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}
    
    # Plot RV with regime background shading
    ax1.plot(df.index, df['rv'], color='black', linewidth=0.8, label='Realized Volatility')
    
    # Add regime background shading
    for regime_name, color in regime_colors.items():
        regime_mask = df['regime'] == regime_name
        if regime_mask.any():
            # Find contiguous regions
            regime_changes = regime_mask.astype(int).diff().fillna(0)
            starts = df.index[regime_changes == 1]
            ends = df.index[regime_changes == -1]
            
            # Handle edge cases
            if regime_mask.iloc[0]:
                starts = df.index[0:1].append(starts)
            if regime_mask.iloc[-1]:
                ends = ends.append(df.index[-1:])
            
            # Shade regions
            for start, end in zip(starts, ends):
                ax1.axvspan(start, end, alpha=0.2, color=color)
    
    # Create custom legend for regimes
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=regime_colors['Low'], alpha=0.2, label='Low Regime'),
        Patch(facecolor=regime_colors['Medium'], alpha=0.2, label='Medium Regime'),
        Patch(facecolor=regime_colors['High'], alpha=0.2, label='High Regime')
    ]
    
    ax1.set_ylabel('Realized Volatility', fontsize=12)
    ax1.set_title('Realized Volatility and Regime Classification Over Time', fontsize=14, pad=15)
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot trailing statistic if provided
    if trailing_stat is not None:
        df_trail = pd.DataFrame({
            'trailing_stat': trailing_stat,
            'regime': regimes
        }).dropna()
        
        ax2.plot(df_trail.index, df_trail['trailing_stat'], 
                color='steelblue', linewidth=1.2, label='Trailing Statistic')
        
        # Add regime shading
        for regime_name, color in regime_colors.items():
            regime_mask = df_trail['regime'] == regime_name
            if regime_mask.any():
                regime_changes = regime_mask.astype(int).diff().fillna(0)
                starts = df_trail.index[regime_changes == 1]
                ends = df_trail.index[regime_changes == -1]
                
                if regime_mask.iloc[0]:
                    starts = df_trail.index[0:1].append(starts)
                if regime_mask.iloc[-1]:
                    ends = ends.append(df_trail.index[-1:])
                
                for start, end in zip(starts, ends):
                    ax2.axvspan(start, end, alpha=0.2, color=color)
        
        ax2.set_ylabel('Trailing Mean RV', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.legend(loc='upper left', fontsize=10)
        ax2.grid(True, alpha=0.3)
    else:
        ax1.set_xlabel('Date', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_regime_persistence(
    persistence_stats: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot regime duration statistics.
    
    Parameters
    ----------
    persistence_stats : pd.DataFrame
        DataFrame from compute_regime_persistence()
    figsize : tuple, default=(10, 6)
        Figure size
    save_path : Path or None
        If provided, save figure to this path
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define colors
    colors = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}
    
    # Plot bars
    x_pos = np.arange(len(persistence_stats))
    bars = ax.bar(
        x_pos,
        persistence_stats['mean_duration'],
        color=[colors[r] for r in persistence_stats['regime_label']],
        alpha=0.7,
        edgecolor='black',
        linewidth=1
    )
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(persistence_stats.iterrows()):
        ax.text(
            i,
            row['mean_duration'] + 1,
            f"{row['mean_duration']:.1f} days",
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(persistence_stats['regime_label'])
    ax.set_ylabel('Average Duration (days)', fontsize=12)
    ax.set_xlabel('Regime', fontsize=12)
    ax.set_title('Average Regime Duration', fontsize=14, pad=15)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_transition_matrix(
    transition_counts: pd.DataFrame,
    normalize: bool = True,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot regime transition matrix as heatmap.
    
    Parameters
    ----------
    transition_counts : pd.DataFrame
        Transition matrix from compute_regime_transitions()
    normalize : bool, default=True
        If True, show transition probabilities instead of counts
    figsize : tuple, default=(8, 6)
        Figure size
    save_path : Path or None
        If provided, save figure to this path
        
    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize if requested
    if normalize:
        # Normalize rows to get transition probabilities
        transition_probs = transition_counts.div(
            transition_counts.sum(axis=1), axis=0
        )
        data = transition_probs
        fmt = '.2%'
        vmax = 1.0
        cbar_label = 'Transition Probability'
    else:
        data = transition_counts
        fmt = 'd'
        vmax = None
        cbar_label = 'Transition Count'
    
    # Create heatmap
    sns.heatmap(
        data,
        annot=True,
        fmt=fmt,
        cmap='YlOrRd',
        vmin=0,
        vmax=vmax,
        cbar_kws={'label': cbar_label},
        linewidths=1,
        linecolor='gray',
        ax=ax
    )
    
    ax.set_xlabel('To Regime', fontsize=12)
    ax.set_ylabel('From Regime', fontsize=12)
    ax.set_title('Regime Transition Matrix', fontsize=14, pad=15)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
