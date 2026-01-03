"""
I/O Utilities

Functions for directory management and file operations.
"""

from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> Path:
    """
    Ensure directory exists, create if needed.
    
    Parameters
    ----------
    path : Path
        Directory path to create
        
    Returns
    -------
    Path
        The created/existing directory path
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dirs(paths: list[Path]) -> list[Path]:
    """
    Ensure multiple directories exist.
    
    Parameters
    ----------
    paths : list of Path
        Directory paths to create
        
    Returns
    -------
    list of Path
        The created/existing directory paths
    """
    return [ensure_dir(p) for p in paths]


def save_dataframe(df, filepath: Path, index: bool = True) -> None:
    """
    Save DataFrame to CSV with logging.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    filepath : Path
        Output file path
    index : bool, default=True
        Whether to save index
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    df.to_csv(filepath, index=index)
    logger.info(f"Saved DataFrame to {filepath}")
