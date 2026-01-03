"""
Realized Volatility Data Utilities

Functions for loading, preprocessing, and validating Oxford-Man Institute
realized volatility data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def load_oxford_man_rv(
    filepath: Path,
    symbol: str = ".SPX",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load Oxford-Man Institute realized volatility data.
    
    Parameters
    ----------
    filepath : Path
        Path to the Oxford-Man CSV file
    symbol : str, default=".SPX"
        Market symbol to filter for
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format
        
    Returns
    -------
    pd.DataFrame
        DataFrame with date index and RV columns
    """
    logger.info(f"Loading Oxford-Man RV data from {filepath}")
    
    # Load data
    df = pd.read_csv(filepath)
    
    # Parse dates
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df['date'] = df['date'].dt.tz_localize(None)  # Remove timezone for simplicity
    
    # Filter by symbol
    df = df[df['Symbol'] == symbol].copy()
    
    if len(df) == 0:
        raise ValueError(f"No data found for symbol {symbol}")
    
    # Set date as index
    df = df.set_index('date').sort_index()
    
    # Filter by date range if specified
    if start_date is not None:
        df = df[df.index >= start_date]
    if end_date is not None:
        df = df[df.index <= end_date]
    
    logger.info(f"Loaded {len(df)} observations from {df.index.min()} to {df.index.max()}")
    
    return df


def extract_rv_series(
    df: pd.DataFrame,
    rv_col: str = 'rv5',
    price_col: str = 'close_price',
) -> pd.DataFrame:
    """
    Extract realized volatility and price series.
    
    Parameters
    ----------
    df : pd.DataFrame
        Oxford-Man dataframe with date index
    rv_col : str, default='rv5'
        Column name for realized volatility
    price_col : str, default='close_price'
        Column name for prices
        
    Returns
    -------
    pd.DataFrame
        DataFrame with rv and price columns
    """
    # Select relevant columns
    cols = [price_col, rv_col]
    
    # Add alternative RV estimators if available
    for col in ['rv10', 'medrv']:
        if col in df.columns:
            cols.append(col)
    
    result = df[cols].copy()
    
    # Rename for clarity
    result = result.rename(columns={
        price_col: 'price',
        rv_col: 'rv'
    })
    
    return result


def compute_returns_from_prices(
    prices: pd.Series,
    eps: float = 1e-12,
) -> pd.Series:
    """
    Compute log returns from price series.
    
    Parameters
    ----------
    prices : pd.Series
        Price series with date index
    eps : float, default=1e-12
        Small constant to avoid log(0)
        
    Returns
    -------
    pd.Series
        Log returns
    """
    prices = prices + eps  # Avoid log(0)
    returns = np.log(prices / prices.shift(1))
    return returns


def validate_rv_data(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate realized volatility data quality.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with RV data
        
    Returns
    -------
    dict
        Validation report with checks and warnings
    """
    report = {
        'n_rows': len(df),
        'start_date': df.index.min(),
        'end_date': df.index.max(),
        'missing_values': {},
        'negative_values': {},
        'extreme_values': {},
        'warnings': []
    }
    
    # Check for missing values
    for col in df.columns:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            report['missing_values'][col] = n_missing
            report['warnings'].append(f"{col}: {n_missing} missing values")
    
    # Check for negative RV values (should not happen)
    for col in ['rv', 'rv5', 'rv10', 'medrv']:
        if col in df.columns:
            n_negative = (df[col] < 0).sum()
            if n_negative > 0:
                report['negative_values'][col] = n_negative
                report['warnings'].append(f"{col}: {n_negative} negative values (data error)")
    
    # Check for extreme values (potential outliers)
    for col in ['rv', 'rv5', 'rv10', 'medrv']:
        if col in df.columns:
            q99 = df[col].quantile(0.99)
            n_extreme = (df[col] > 10 * q99).sum()
            if n_extreme > 0:
                report['extreme_values'][col] = n_extreme
                report['warnings'].append(
                    f"{col}: {n_extreme} values > 10x 99th percentile (potential outliers)"
                )
    
    # Check date continuity
    date_diffs = df.index.to_series().diff()
    expected_freq = pd.Timedelta(days=1)
    gaps = date_diffs[date_diffs > 3 * expected_freq]  # Allow for weekends
    
    if len(gaps) > 0:
        report['date_gaps'] = len(gaps)
        report['warnings'].append(f"{len(gaps)} date gaps > 3 days detected")
    
    return report


def print_validation_report(report: Dict[str, any]) -> None:
    """Print validation report in readable format."""
    print("\n" + "="*70)
    print("REALIZED VOLATILITY DATA VALIDATION REPORT")
    print("="*70)
    print(f"Rows: {report['n_rows']:,}")
    print(f"Date range: {report['start_date']} to {report['end_date']}")
    
    if report['warnings']:
        print("\nWARNINGS:")
        for warning in report['warnings']:
            print(f"  - {warning}")
    else:
        print("\nNo data quality issues detected.")
    
    print("="*70 + "\n")
