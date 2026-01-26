"""
Evaluation Metrics for Volatility Forecasting

Standard metrics for evaluating forecast accuracy.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
import warnings


def rmse(actual: Union[pd.Series, np.ndarray], 
         forecast: Union[pd.Series, np.ndarray]) -> float:
    """
    Compute Root Mean Squared Error.
    
    Parameters
    ----------
    actual : pd.Series or np.ndarray
        Actual values
    forecast : pd.Series or np.ndarray
        Forecast values
        
    Returns
    -------
    float
        RMSE value
        
    Notes
    -----
    RMSE penalizes large errors heavily due to squaring.
    Lower values indicate better forecasts.
    """
    errors = np.array(actual) - np.array(forecast)
    return np.sqrt(np.mean(errors ** 2))


def mae(actual: Union[pd.Series, np.ndarray], 
        forecast: Union[pd.Series, np.ndarray]) -> float:
    """
    Compute Mean Absolute Error.
    
    Parameters
    ----------
    actual : pd.Series or np.ndarray
        Actual values
    forecast : pd.Series or np.ndarray
        Forecast values
        
    Returns
    -------
    float
        MAE value
        
    Notes
    -----
    MAE is more robust to outliers than RMSE.
    Interpretable in original units.
    """
    errors = np.array(actual) - np.array(forecast)
    return np.mean(np.abs(errors))


def r_squared(actual: Union[pd.Series, np.ndarray], 
              forecast: Union[pd.Series, np.ndarray]) -> float:
    """
    Compute R² (coefficient of determination).
    
    Parameters
    ----------
    actual : pd.Series or np.ndarray
        Actual values
    forecast : pd.Series or np.ndarray
        Forecast values
        
    Returns
    -------
    float
        R² value (can be negative for poor forecasts)
        
    Notes
    -----
    R² measures proportion of variance explained relative to
    predicting the unconditional mean.
    
    R² = 1 - (SS_residual / SS_total)
    
    For volatility forecasting, typical R² values are 0.30-0.50.
    Values below 0 indicate forecasts worse than the mean.
    """
    actual_arr = np.array(actual)
    forecast_arr = np.array(forecast)
    
    ss_residual = np.sum((actual_arr - forecast_arr) ** 2)
    ss_total = np.sum((actual_arr - np.mean(actual_arr)) ** 2)
    
    if ss_total == 0:
        warnings.warn("Total sum of squares is zero, R² undefined")
        return np.nan
    
    return 1 - (ss_residual / ss_total)


def compute_metrics(
    actual: Union[pd.Series, np.ndarray],
    forecast: Union[pd.Series, np.ndarray],
    metrics: Optional[list] = None
) -> Dict[str, float]:
    """
    Compute multiple metrics at once.
    
    Parameters
    ----------
    actual : pd.Series or np.ndarray
        Actual values
    forecast : pd.Series or np.ndarray
        Forecast values
    metrics : list, optional
        List of metric names to compute
        Default: ['rmse', 'mae', 'r2']
        
    Returns
    -------
    dict
        Dictionary mapping metric names to values
        
    Examples
    --------
    >>> results = compute_metrics(actuals, forecasts)
    >>> print(f"RMSE: {results['rmse']:.6f}")
    """
    if metrics is None:
        metrics = ['rmse', 'mae', 'r2']
    
    results = {}
    
    for metric in metrics:
        if metric == 'rmse':
            results['rmse'] = rmse(actual, forecast)
        elif metric == 'mae':
            results['mae'] = mae(actual, forecast)
        elif metric in ['r2', 'r_squared']:
            results['r2'] = r_squared(actual, forecast)
        else:
            warnings.warn(f"Unknown metric: {metric}")
    
    return results


def compute_metrics_by_regime(
    actual: pd.Series,
    forecast: pd.Series,
    regimes: pd.Series,
    metrics: Optional[list] = None
) -> pd.DataFrame:
    """
    Compute metrics separately for each regime.
    
    Parameters
    ----------
    actual : pd.Series
        Actual values with DatetimeIndex
    forecast : pd.Series
        Forecast values with DatetimeIndex
    regimes : pd.Series
        Regime labels with DatetimeIndex
    metrics : list, optional
        List of metric names to compute
        
    Returns
    -------
    pd.DataFrame
        DataFrame with regimes as rows and metrics as columns
        
    Examples
    --------
    >>> regime_metrics = compute_metrics_by_regime(
    ...     actual=df['rv'],
    ...     forecast=forecasts['persistence'],
    ...     regimes=df['regime']
    ... )
    """
    if metrics is None:
        metrics = ['rmse', 'mae', 'r2']
    
    # Align all series
    df = pd.DataFrame({
        'actual': actual,
        'forecast': forecast,
        'regime': regimes
    }).dropna()
    
    # Get unique regimes
    regime_names = df['regime'].cat.categories if hasattr(df['regime'], 'cat') else df['regime'].unique()
    
    # Compute metrics for each regime
    results = []
    
    for regime_name in regime_names:
        regime_mask = df['regime'] == regime_name
        regime_data = df[regime_mask]
        
        if len(regime_data) == 0:
            continue
        
        regime_metrics = compute_metrics(
            actual=regime_data['actual'],
            forecast=regime_data['forecast'],
            metrics=metrics
        )
        regime_metrics['regime'] = regime_name
        regime_metrics['n_obs'] = len(regime_data)
        results.append(regime_metrics)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.set_index('regime')
    
    return results_df


def format_metrics_table(
    metrics_dict: Dict[str, Dict[str, float]],
    transpose: bool = False
) -> pd.DataFrame:
    """
    Format metrics dictionary as a clean DataFrame for display.
    
    Parameters
    ----------
    metrics_dict : dict
        Dictionary mapping model names to metric dictionaries
    transpose : bool, default=False
        If True, models as rows and metrics as columns
        
    Returns
    -------
    pd.DataFrame
        Formatted metrics table
        
    Examples
    --------
    >>> all_metrics = {
    ...     'persistence': compute_metrics(actual, fcst_pers),
    ...     'rolling_avg': compute_metrics(actual, fcst_roll),
    ...     'ewma': compute_metrics(actual, fcst_ewma)
    ... }
    >>> table = format_metrics_table(all_metrics, transpose=True)
    """
    df = pd.DataFrame(metrics_dict)
    
    if transpose:
        df = df.T
    
    return df