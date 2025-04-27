"""
Helper utilities for NYC Taxi Trip Data Analysis.
"""

import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Optional, Union
import time
from datetime import datetime

def log_message(message: str, log_file: Optional[str] = None):
    """
    Log a message with timestamp.
    
    Args:
        message: Message to log
        log_file: Path to log file (if None, prints to console)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    
    if log_file:
        with open(log_file, 'a') as f:
            f.write(log_entry + '\n')
    else:
        print(log_entry)

def save_metrics(metrics: Dict, 
                output_path: str):
    """
    Save metrics dictionary to a JSON file.
    
    Args:
        metrics: Dictionary of metrics
        output_path: Path to save the metrics
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert numpy types to Python native types for JSON serialization
    clean_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (np.int64, np.int32, np.float64, np.float32)):
            clean_metrics[key] = value.item()
        else:
            clean_metrics[key] = value
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(clean_metrics, f, indent=4)

def load_metrics(input_path: str) -> Dict:
    """
    Load metrics dictionary from a JSON file.
    
    Args:
        input_path: Path to the metrics file
        
    Returns:
        Dictionary of metrics
    """
    with open(input_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics

def timer_decorator(func):
    """
    Decorator to time function execution.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function that prints execution time
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def sample_dataframe(df: pd.DataFrame, 
                    n: int = 100000, 
                    random_state: int = 42) -> pd.DataFrame:
    """
    Sample a large DataFrame to a smaller size for faster processing.
    
    Args:
        df: DataFrame to sample
        n: Number of rows to sample
        random_state: Random seed for reproducibility
        
    Returns:
        Sampled DataFrame
    """
    if len(df) > n:
        return df.sample(n=n, random_state=random_state)
    else:
        return df

def memory_usage(df: pd.DataFrame) -> str:
    """
    Get memory usage of a DataFrame in a human-readable format.
    
    Args:
        df: DataFrame to check
        
    Returns:
        String with memory usage information
    """
    memory_bytes = df.memory_usage(deep=True).sum()
    
    # Convert to appropriate unit
    if memory_bytes < 1024:
        return f"{memory_bytes} bytes"
    elif memory_bytes < 1024**2:
        return f"{memory_bytes/1024:.2f} KB"
    elif memory_bytes < 1024**3:
        return f"{memory_bytes/1024**2:.2f} MB"
    else:
        return f"{memory_bytes/1024**3:.2f} GB"

def print_dataframe_info(df: pd.DataFrame, name: str = "DataFrame"):
    """
    Print comprehensive information about a DataFrame.
    
    Args:
        df: DataFrame to analyze
        name: Name to use for the DataFrame in the output
    """
    print(f"\n{'='*50}")
    print(f"Information for {name}")
    print(f"{'='*50}")
    
    print(f"\nShape: {df.shape}")
    print(f"Memory usage: {memory_usage(df)}")
    
    print("\nColumn Data Types:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")
    
    print("\nMissing Values:")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_info = pd.DataFrame({
        'Missing Values': missing,
        'Percentage': missing_percent
    })
    print(missing_info[missing_info['Missing Values'] > 0])
    
    print("\nSample Data:")
    print(df.head(5))
    
    print(f"\n{'='*50}\n")
