"""
Volatility Regime Definition and Validation

This module provides functions for defining volatility regimes using trailing
information (no look-ahead bias) and validating regime properties.

Author: Victor
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
import warnings


def compute_trailing_statistic(
    rv: pd.Series,
    window: int,
    stat: str = 'mean'
) -> pd.Series:
    """
    Compute a trailing statistic using only past observations.
    
    This function ensures no look-ahead bias by using only information
    available up to and including each time point.
    
    Parameters
    ----------
    rv : pd.Series
        Realized volatility series
    window : int
        Number of periods to include in trailing window
    stat : str, default='mean'
        Statistic to compute ('mean', 'median', 'std')
        
    Returns
    -------
    pd.Series
        Trailing statistic series
        
    Notes
    -----
    The first (window-1) values will be NaN since we need at least
    window observations to compute the statistic.
    """
    if stat == 'mean':
        return rv.rolling(window=window, min_periods=window).mean()
    elif stat == 'median':
        return rv.rolling(window=window, min_periods=window).median()
    elif stat == 'std':
        return rv.rolling(window=window, min_periods=window).std()
    else:
        raise ValueError(f"Unsupported statistic: {stat}")


def compute_regime_thresholds(
    trailing_stat: pd.Series,
    quantiles: Tuple[float, float] = (0.33, 0.67),
    method: str = 'quantile'
) -> Tuple[float, float]:
    """
    Compute thresholds for regime classification.
    
    Parameters
    ----------
    trailing_stat : pd.Series
        Trailing volatility statistic
    quantiles : tuple of float, default=(0.33, 0.67)
        Quantiles defining regime boundaries
    method : str, default='quantile'
        Method for threshold selection ('quantile' or 'fixed')
        
    Returns
    -------
    tuple of float
        (lower_threshold, upper_threshold)
        
    Notes
    -----
    Using quantiles ensures roughly balanced regime populations.
    The thresholds are computed on the entire sample but applied
    point-by-point using only trailing information.
    """
    if method == 'quantile':
        # Drop NaN values before computing quantiles
        valid_stat = trailing_stat.dropna()
        
        if len(valid_stat) == 0:
            raise ValueError("Trailing statistic contains only NaN values")
        
        lower_threshold = valid_stat.quantile(quantiles[0])
        upper_threshold = valid_stat.quantile(quantiles[1])
        
        return lower_threshold, upper_threshold
    
    elif method == 'fixed':
        # Could implement fixed thresholds if needed
        raise NotImplementedError("Fixed threshold method not yet implemented")
    
    else:
        raise ValueError(f"Unsupported threshold method: {method}")


def assign_regime_labels(
    trailing_stat: pd.Series,
    lower_threshold: float,
    upper_threshold: float,
    regime_names: Tuple[str, str, str] = ('Low', 'Medium', 'High')
) -> pd.Series:
    """
    Assign regime labels based on trailing statistic and thresholds.
    
    Parameters
    ----------
    trailing_stat : pd.Series
        Trailing volatility statistic
    lower_threshold : float
        Boundary between Low and Medium regimes
    upper_threshold : float
        Boundary between Medium and High regimes
    regime_names : tuple of str, default=('Low', 'Medium', 'High')
        Names for the three regimes
        
    Returns
    -------
    pd.Series
        Regime labels (categorical)
        
    Notes
    -----
    Classification rule:
    - Low: trailing_stat < lower_threshold
    - Medium: lower_threshold <= trailing_stat < upper_threshold
    - High: trailing_stat >= upper_threshold
    
    Periods with NaN trailing statistics are labeled as NaN.
    """
    # Initialize with NaN
    regimes = pd.Series(index=trailing_stat.index, dtype='object')
    
    # Assign labels based on thresholds
    regimes[trailing_stat < lower_threshold] = regime_names[0]
    regimes[(trailing_stat >= lower_threshold) & 
            (trailing_stat < upper_threshold)] = regime_names[1]
    regimes[trailing_stat >= upper_threshold] = regime_names[2]
    
    # Convert to categorical for ordered display
    regimes = pd.Categorical(
        regimes,
        categories=regime_names,
        ordered=True
    )
    
    return pd.Series(regimes, index=trailing_stat.index)


def compute_regime_persistence(
    regimes: pd.Series
) -> pd.DataFrame:
    """
    Compute regime duration statistics.
    
    Parameters
    ----------
    regimes : pd.Series
        Regime labels
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - regime: regime name
        - mean_duration: average duration in days
        - median_duration: median duration in days
        - min_duration: minimum duration in days
        - max_duration: maximum duration in days
        - count: number of regime episodes
        
    Notes
    -----
    A regime episode is a consecutive sequence of days in the same regime.
    """
    # Drop NaN values
    valid_regimes = regimes.dropna()
    
    # Identify regime changes
    regime_change = valid_regimes != valid_regimes.shift(1)
    regime_ids = regime_change.cumsum()
    
    # Compute duration of each regime episode
    durations = valid_regimes.groupby(regime_ids).size()
    regime_labels = valid_regimes.groupby(regime_ids).first()
    
    # Create dataframe of durations by regime
    duration_df = pd.DataFrame({
        'regime_label': regime_labels,
        'duration': durations
    })
    
    # Compute statistics by regime
    persistence_stats = duration_df.groupby('regime_label')['duration'].agg([
        ('mean_duration', 'mean'),
        ('median_duration', 'median'),
        ('min_duration', 'min'),
        ('max_duration', 'max'),
        ('count', 'count')
    ]).reset_index()
    
    return persistence_stats


def compute_regime_transitions(
    regimes: pd.Series
) -> pd.DataFrame:
    """
    Compute regime transition frequency matrix.
    
    Parameters
    ----------
    regimes : pd.Series
        Regime labels
        
    Returns
    -------
    pd.DataFrame
        Transition matrix where entry (i,j) is the count of
        transitions from regime i to regime j
        
    Notes
    -----
    Diagonal elements represent regime persistence (no transition).
    Off-diagonal elements represent actual transitions.
    """
    # Drop NaN values
    valid_regimes = regimes.dropna()
    
    # Get unique regime names
    regime_names = valid_regimes.cat.categories.tolist()
    
    # Create transition pairs
    from_regime = valid_regimes[:-1].values
    to_regime = valid_regimes[1:].values
    
    # Count transitions
    transition_counts = pd.DataFrame(
        index=regime_names,
        columns=regime_names,
        data=0
    )
    
    for from_r, to_r in zip(from_regime, to_regime):
        transition_counts.loc[from_r, to_r] += 1
    
    return transition_counts


def compute_regime_distributions(
    rv: pd.Series,
    regimes: pd.Series
) -> Dict[str, pd.Series]:
    """
    Compute realized volatility distributions within each regime.
    
    Parameters
    ----------
    rv : pd.Series
        Realized volatility series
    regimes : pd.Series
        Regime labels
        
    Returns
    -------
    dict
        Dictionary mapping regime name to RV series within that regime
        
    Notes
    -----
    This is used to verify that regimes meaningfully separate
    volatility levels.
    """
    # Create dataframe
    df = pd.DataFrame({
        'rv': rv,
        'regime': regimes
    }).dropna()
    
    # Extract RV by regime
    distributions = {}
    for regime_name in df['regime'].cat.categories:
        distributions[regime_name] = df[df['regime'] == regime_name]['rv']
    
    return distributions


def validate_regime_definition(
    rv: pd.Series,
    regimes: pd.Series,
    trailing_stat: pd.Series
) -> Dict[str, any]:
    """
    Run comprehensive validation checks on regime definition.
    
    Parameters
    ----------
    rv : pd.Series
        Realized volatility series
    regimes : pd.Series
        Regime labels
    trailing_stat : pd.Series
        Trailing statistic used for regime definition
        
    Returns
    -------
    dict
        Validation results including:
        - regime_counts: number of observations per regime
        - regime_fractions: fraction of observations per regime
        - rv_means: mean RV in each regime
        - rv_stds: std of RV in each regime
        - trailing_stat_means: mean trailing stat in each regime
        - separation_ratio: ratio of max to min regime mean
        
    Notes
    -----
    This function provides summary statistics to verify that:
    1. Regimes are reasonably balanced
    2. Regimes meaningfully separate volatility levels
    3. No regime is trivially empty or dominant
    """
    # Create dataframe
    df = pd.DataFrame({
        'rv': rv,
        'regime': regimes,
        'trailing_stat': trailing_stat
    }).dropna()
    
    # Count observations per regime
    regime_counts = df['regime'].value_counts().sort_index()
    regime_fractions = regime_counts / len(df)
    
    # Compute RV statistics by regime
    rv_stats = df.groupby('regime')['rv'].agg(['mean', 'std'])
    
    # Compute trailing stat statistics by regime
    trailing_stats = df.groupby('regime')['trailing_stat'].mean()
    
    # Compute separation ratio
    separation_ratio = rv_stats['mean'].max() / rv_stats['mean'].min()
    
    return {
        'regime_counts': regime_counts,
        'regime_fractions': regime_fractions,
        'rv_means': rv_stats['mean'],
        'rv_stds': rv_stats['std'],
        'trailing_stat_means': trailing_stats,
        'separation_ratio': separation_ratio
    }


def label_regimes(
    rv: pd.Series,
    window: int = 63,
    quantiles: Tuple[float, float] = (0.33, 0.67),
    stat: str = 'mean',
    regime_names: Tuple[str, str, str] = ('Low', 'Medium', 'High')
) -> Tuple[pd.Series, pd.Series, Tuple[float, float]]:
    """
    Complete regime labeling pipeline.
    
    This is a convenience function that combines all steps:
    1. Compute trailing statistic
    2. Compute thresholds
    3. Assign regime labels
    
    Parameters
    ----------
    rv : pd.Series
        Realized volatility series
    window : int, default=63
        Trailing window size (in days)
    quantiles : tuple of float, default=(0.33, 0.67)
        Quantiles for threshold definition
    stat : str, default='mean'
        Statistic to compute on trailing window
    regime_names : tuple of str, default=('Low', 'Medium', 'High')
        Names for the three regimes
        
    Returns
    -------
    tuple
        (regimes, trailing_stat, (lower_threshold, upper_threshold))
        
    Examples
    --------
    >>> regimes, trailing_rv, thresholds = label_regimes(
    ...     rv=df['rv'],
    ...     window=63,
    ...     quantiles=(0.33, 0.67)
    ... )
    """
    # Step 1: Compute trailing statistic
    trailing_stat = compute_trailing_statistic(rv, window=window, stat=stat)
    
    # Step 2: Compute thresholds
    lower_threshold, upper_threshold = compute_regime_thresholds(
        trailing_stat,
        quantiles=quantiles
    )
    
    # Step 3: Assign labels
    regimes = assign_regime_labels(
        trailing_stat,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
        regime_names=regime_names
    )
    
    return regimes, trailing_stat, (lower_threshold, upper_threshold)
