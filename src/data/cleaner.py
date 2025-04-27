"""
Data cleaning utilities for NYC Taxi Trip Data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

def calculate_trip_duration(df: pd.DataFrame, 
                           pickup_col: str, 
                           dropoff_col: str) -> pd.DataFrame:
    """
    Calculate trip duration in minutes and add it as a new column.
    
    Args:
        df: DataFrame containing taxi trip data
        pickup_col: Name of the pickup datetime column
        dropoff_col: Name of the dropoff datetime column
        
    Returns:
        DataFrame with added trip_duration column
    """
    df = df.copy()
    df['trip_duration'] = (df[dropoff_col] - df[pickup_col]).dt.total_seconds() / 60
    return df

def remove_outliers(df: pd.DataFrame, 
                   columns: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """
    Remove outliers from specified columns based on provided thresholds.
    
    Args:
        df: DataFrame containing taxi trip data
        columns: Dictionary mapping column names to (min, max) threshold tuples
        
    Returns:
        DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    for col, (min_val, max_val) in columns.items():
        if col in df.columns:
            mask = (df[col] >= min_val) & (df[col] <= max_val)
            df_clean = df_clean[mask]
    
    return df_clean

def handle_missing_values(df: pd.DataFrame, 
                         strategy: Dict[str, str] = None) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.
    
    Args:
        df: DataFrame containing taxi trip data
        strategy: Dictionary mapping column names to strategies
                 ('drop', 'mean', 'median', 'mode', 'zero', or a constant value)
        
    Returns:
        DataFrame with missing values handled
    """
    if strategy is None:
        strategy = {}
    
    df_clean = df.copy()
    
    # Default strategy for all columns not specified
    default_strategy = strategy.get('default', 'drop')
    
    # Apply strategies to specific columns
    for col in df.columns:
        col_strategy = strategy.get(col, default_strategy)
        
        if col_strategy == 'drop':
            df_clean = df_clean.dropna(subset=[col])
        elif col_strategy == 'mean':
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        elif col_strategy == 'median':
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        elif col_strategy == 'mode':
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        elif col_strategy == 'zero':
            df_clean[col] = df_clean[col].fillna(0)
        elif isinstance(col_strategy, (int, float, str)):
            df_clean[col] = df_clean[col].fillna(col_strategy)
    
    return df_clean

def clean_yellow_taxi_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean yellow taxi data with specific rules.
    
    Args:
        df: DataFrame containing yellow taxi trip data
        
    Returns:
        Cleaned DataFrame
    """
    # Calculate trip duration
    df = calculate_trip_duration(df, 'tpep_pickup_datetime', 'tpep_dropoff_datetime')
    
    # Remove outliers
    outlier_thresholds = {
        'trip_duration': (0, 180),  # 0 to 3 hours in minutes
        'trip_distance': (0, 100),  # 0 to 100 miles
        'fare_amount': (0, 1000),   # $0 to $1000
        'passenger_count': (1, 8)   # 1 to 8 passengers
    }
    df = remove_outliers(df, outlier_thresholds)
    
    # Handle missing values
    missing_strategies = {
        'passenger_count': 'median',
        'trip_distance': 'median',
        'fare_amount': 'median',
        'default': 'drop'
    }
    df = handle_missing_values(df, missing_strategies)
    
    # Remove trips with zero distance but non-zero duration
    df = df[~((df['trip_distance'] == 0) & (df['trip_duration'] > 0))]
    
    # Remove trips with negative fare amounts
    df = df[df['fare_amount'] >= 0]
    
    return df

def clean_green_taxi_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean green taxi data with specific rules.
    
    Args:
        df: DataFrame containing green taxi trip data
        
    Returns:
        Cleaned DataFrame
    """
    # Calculate trip duration
    df = calculate_trip_duration(df, 'lpep_pickup_datetime', 'lpep_dropoff_datetime')
    
    # Remove outliers
    outlier_thresholds = {
        'trip_duration': (0, 180),  # 0 to 3 hours in minutes
        'trip_distance': (0, 100),  # 0 to 100 miles
        'fare_amount': (0, 1000),   # $0 to $1000
        'passenger_count': (1, 8)   # 1 to 8 passengers
    }
    df = remove_outliers(df, outlier_thresholds)
    
    # Handle missing values
    missing_strategies = {
        'passenger_count': 'median',
        'trip_distance': 'median',
        'fare_amount': 'median',
        'default': 'drop'
    }
    df = handle_missing_values(df, missing_strategies)
    
    # Remove trips with zero distance but non-zero duration
    df = df[~((df['trip_distance'] == 0) & (df['trip_duration'] > 0))]
    
    # Remove trips with negative fare amounts
    df = df[df['fare_amount'] >= 0]
    
    return df

def clean_fhv_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean FHV (For-Hire Vehicle) data with specific rules.
    
    Args:
        df: DataFrame containing FHV trip data
        
    Returns:
        Cleaned DataFrame
    """
    # Calculate trip duration
    df = calculate_trip_duration(df, 'pickup_datetime', 'dropOff_datetime')
    
    # Handle missing values for location IDs
    df = df.dropna(subset=['PUlocationID', 'DOlocationID'])
    
    return df

def clean_fhvhv_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean FHVHV (High-Volume For-Hire Vehicle) data with specific rules.
    
    Args:
        df: DataFrame containing FHVHV trip data
        
    Returns:
        Cleaned DataFrame
    """
    # Calculate trip duration
    df = calculate_trip_duration(df, 'pickup_datetime', 'dropoff_datetime')
    
    # Remove outliers
    outlier_thresholds = {
        'trip_duration': (0, 180),  # 0 to 3 hours in minutes
        'trip_miles': (0, 100),     # 0 to 100 miles
        'base_passenger_fare': (0, 1000)  # $0 to $1000
    }
    df = remove_outliers(df, outlier_thresholds)
    
    # Handle missing values
    df = df.dropna(subset=['PULocationID', 'DOLocationID'])
    
    return df
