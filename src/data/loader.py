"""
Data loading utilities for NYC Taxi Trip Data.
"""

import pandas as pd
from typing import Union, List, Optional
from pathlib import Path

def load_parquet(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a parquet file into a pandas DataFrame.
    
    Args:
        file_path: Path to the parquet file
        
    Returns:
        DataFrame containing the data
    """
    return pd.read_parquet(file_path)

def load_multiple_parquet(file_paths: List[Union[str, Path]]) -> pd.DataFrame:
    """
    Load multiple parquet files and concatenate them into a single DataFrame.
    
    Args:
        file_paths: List of paths to parquet files
        
    Returns:
        Concatenated DataFrame
    """
    dfs = [load_parquet(file) for file in file_paths]
    return pd.concat(dfs, ignore_index=True)

def load_taxi_data(taxi_type: str, months: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load taxi data for specified type and months.
    
    Args:
        taxi_type: Type of taxi data ('yellow', 'green', 'fhv', or 'fhvhv')
        months: List of months to load (e.g., ['2025-01', '2025-02'])
                If None, loads all available months
    
    Returns:
        DataFrame containing the taxi data
    """
    from src.config import DATA_DIR
    
    if months is None:
        months = ['2025-01', '2025-02']  # Default to all available months
    
    file_paths = []
    for month in months:
        file_name = f"{taxi_type}_tripdata_{month}.parquet"
        file_path = DATA_DIR / file_name
        if file_path.exists():
            file_paths.append(file_path)
    
    if not file_paths:
        raise FileNotFoundError(f"No {taxi_type} taxi data found for months {months}")
    
    return load_multiple_parquet(file_paths)
