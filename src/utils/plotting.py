"""
Plotting Utilities

Standard matplotlib configuration and helper functions.
"""

import matplotlib.pyplot as plt
from pathlib import Path


def set_mpl_defaults():
    """Set standard matplotlib configuration for all plots."""
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.grid'] = False  # Enable per-plot
    

def savefig(fig, filepath: Path, **kwargs):
    """
    Save figure with standard settings.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    filepath : Path
        Output file path
    **kwargs : dict
        Additional arguments to pass to savefig
    """
    default_kwargs = {'dpi': 300, 'bbox_inches': 'tight'}
    default_kwargs.update(kwargs)
    fig.savefig(filepath, **default_kwargs)
