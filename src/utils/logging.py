"""
Logging Utilities

Configure logging for notebooks and modules.
"""

import logging
import sys


def get_logger(name: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Get configured logger instance.
    
    Parameters
    ----------
    name : str, optional
        Logger name (defaults to root logger)
    level : int, default=logging.INFO
        Logging level
        
    Returns
    -------
    logging.Logger
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Only add handler if logger doesn't have one
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        formatter = logging.Formatter(
            '[%(levelname)s] %(message)s'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        logger.setLevel(level)
    
    return logger
