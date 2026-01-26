"""
Baseline Models for Volatility Forecasting

Simple baseline models that serve as lower bounds for comparison
with more sophisticated volatility models.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
import warnings


def forecast_persistence(
    rv: pd.Series,
    name: str = 'persistence'
) -> pd.Series:
    """
    Generate persistence (random walk) forecasts.
    
    Forecast rule: RV_hat(t) = RV(t-1)
    
    Parameters
    ----------
    rv : pd.Series
        Realized volatility series with DatetimeIndex
    name : str, default='persistence'
        Name for the forecast series
        
    Returns
    -------
    pd.Series
        One-step-ahead forecasts aligned with rv index
        
    Notes
    -----
    First forecast is NaN since it requires RV(t-1).
    This is the simplest possible volatility forecast and often
    performs surprisingly well due to strong volatility persistence.
    """
    forecasts = rv.shift(1)
    forecasts.name = name
    return forecasts


def forecast_rolling_average(
    rv: pd.Series,
    window: int = 5,
    name: str = 'rolling_avg'
) -> pd.Series:
    """
    Generate rolling average forecasts.
    
    Forecast rule: RV_hat(t) = mean(RV(t-1), ..., RV(t-window))
    
    Parameters
    ----------
    rv : pd.Series
        Realized volatility series
    window : int, default=5
        Number of periods for rolling average
    name : str, default='rolling_avg'
        Name for the forecast series
        
    Returns
    -------
    pd.Series
        One-step-ahead forecasts
        
    Notes
    -----
    First 'window' forecasts are NaN due to insufficient history.
    """
    # Compute rolling average, then shift to align as forecast
    rolling_avg = rv.rolling(window=window, min_periods=window).mean()
    forecasts = rolling_avg.shift(1)
    forecasts.name = name
    return forecasts


def forecast_ewma(
    rv: pd.Series,
    lambda_param: float = 0.94,
    name: str = 'ewma'
) -> pd.Series:
    """
    Generate EWMA (Exponentially Weighted Moving Average) forecasts.
    
    Recursive formula:
        RV_hat(t) = lambda * RV_hat(t-1) + (1 - lambda) * RV(t-1)
    
    Parameters
    ----------
    rv : pd.Series
        Realized volatility series
    lambda_param : float, default=0.94
        Decay parameter (0 < lambda < 1)
        Higher values = more weight on past forecasts
        0.94 corresponds to ~16 day effective window
    name : str, default='ewma'
        Name for the forecast series
        
    Returns
    -------
    pd.Series
        One-step-ahead forecasts
        
    Notes
    -----
    Initialized with first observed RV value.
    This is equivalent to an exponentially weighted moving average
    of past observations.
    
    The RiskMetrics methodology uses lambda = 0.94 for daily data.
    """
    if not 0 < lambda_param < 1:
        raise ValueError(f"lambda_param must be in (0,1), got {lambda_param}")
    
    # Initialize forecasts series
    forecasts = pd.Series(index=rv.index, dtype=float)
    
    # Initialize with first RV value
    forecasts.iloc[0] = rv.iloc[0]
    
    # Generate recursive forecasts
    for t in range(1, len(rv)):
        forecasts.iloc[t] = (
            lambda_param * forecasts.iloc[t-1] + 
            (1 - lambda_param) * rv.iloc[t-1]
        )
    
    forecasts.name = name
    
    return forecasts


def generate_all_baseline_forecasts(
    rv: pd.Series,
    rolling_window: int = 5,
    ewma_lambda: float = 0.94
) -> pd.DataFrame:
    """
    Generate all baseline forecasts in one call.
    
    Parameters
    ----------
    rv : pd.Series
        Realized volatility series
    rolling_window : int, default=5
        Window for rolling average
    ewma_lambda : float, default=0.94
        Lambda parameter for EWMA
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: persistence, rolling_avg, ewma
        
    Examples
    --------
    >>> forecasts = generate_all_baseline_forecasts(df['rv'])
    >>> forecasts.head()
    """
    forecasts = pd.DataFrame(index=rv.index)
    
    forecasts['persistence'] = forecast_persistence(rv)
    forecasts['rolling_avg'] = forecast_rolling_average(rv, window=rolling_window)
    forecasts['ewma'] = forecast_ewma(rv, lambda_param=ewma_lambda)
    
    return forecasts


def align_forecasts_with_actuals(
    forecasts: pd.DataFrame,
    actuals: pd.Series,
    drop_na: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Align forecasts with actual values for evaluation.
    
    Parameters
    ----------
    forecasts : pd.DataFrame
        Forecast DataFrame
    actuals : pd.Series
        Actual realized volatility
    drop_na : bool, default=True
        Whether to drop rows with any NaN values
        
    Returns
    -------
    tuple of (pd.DataFrame, pd.Series)
        Aligned forecasts and actuals
        
    Notes
    -----
    This ensures forecasts and actuals have:
    - Same index
    - No NaN values (if drop_na=True)
    - Proper alignment for evaluation
    """
    # Combine into single dataframe for alignment
    combined = forecasts.copy()
    combined['actual'] = actuals
    
    if drop_na:
        combined = combined.dropna()
    
    # Split back
    aligned_forecasts = combined.drop('actual', axis=1)
    aligned_actuals = combined['actual']
    
    return aligned_forecasts, aligned_actuals