"""
Seed Utilities

Functions for setting random seeds for reproducibility.
"""

import numpy as np
import random


def set_global_seed(seed: int = 42):
    """
    Set random seeds for numpy and Python's random module.
    
    Parameters
    ----------
    seed : int, default=42
        Random seed value
    """
    np.random.seed(seed)
    random.seed(seed)
