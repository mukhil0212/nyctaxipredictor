"""
Feature engineering utilities for NYC Taxi Trip Data.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Union

def extract_datetime_features(df: pd.DataFrame, 
                             datetime_col: str) -> pd.DataFrame:
    """
    Extract datetime features from a datetime column.
    
    Args:
        df: DataFrame containing taxi trip data
        datetime_col: Name of the datetime column
        
    Returns:
        DataFrame with added datetime features
    """
    df = df.copy()
    
    # Extract basic datetime components
    df[f'{datetime_col}_hour'] = df[datetime_col].dt.hour
    df[f'{datetime_col}_day'] = df[datetime_col].dt.day
    df[f'{datetime_col}_dayofweek'] = df[datetime_col].dt.dayofweek
    df[f'{datetime_col}_month'] = df[datetime_col].dt.month
    
    # Create period of day feature
    bins = [0, 6, 12, 18, 24]
    labels = ['Night', 'Morning', 'Afternoon', 'Evening']
    df[f'{datetime_col}_period'] = pd.cut(df[datetime_col].dt.hour, 
                                         bins=bins, 
                                         labels=labels, 
                                         include_lowest=True)
    
    # Create weekend indicator
    df[f'{datetime_col}_is_weekend'] = df[datetime_col].dt.dayofweek >= 5
    
    # Create rush hour indicator (7-10 AM and 4-7 PM on weekdays)
    morning_rush = (df[f'{datetime_col}_hour'].between(7, 9)) & (~df[f'{datetime_col}_is_weekend'])
    evening_rush = (df[f'{datetime_col}_hour'].between(16, 18)) & (~df[f'{datetime_col}_is_weekend'])
    df[f'{datetime_col}_is_rush_hour'] = morning_rush | evening_rush
    
    return df

def calculate_distance_features(df: pd.DataFrame, 
                               distance_col: str = 'trip_distance',
                               duration_col: str = 'trip_duration') -> pd.DataFrame:
    """
    Calculate distance-related features.
    
    Args:
        df: DataFrame containing taxi trip data
        distance_col: Name of the distance column
        duration_col: Name of the duration column
        
    Returns:
        DataFrame with added distance features
    """
    df = df.copy()
    
    # Calculate speed (miles per minute)
    df['speed'] = df[distance_col] / df[duration_col].clip(lower=0.1)
    
    # Convert to mph for better interpretability
    df['speed_mph'] = df['speed'] * 60
    
    # Create distance bins
    bins = [0, 1, 3, 5, 10, float('inf')]
    labels = ['Very Short', 'Short', 'Medium', 'Long', 'Very Long']
    df['distance_category'] = pd.cut(df[distance_col], 
                                    bins=bins, 
                                    labels=labels, 
                                    include_lowest=True)
    
    return df

def create_location_features(df: pd.DataFrame,
                            pickup_loc_col: str,
                            dropoff_loc_col: str) -> pd.DataFrame:
    """
    Create features based on pickup and dropoff locations.
    
    Args:
        df: DataFrame containing taxi trip data
        pickup_loc_col: Name of the pickup location column
        dropoff_loc_col: Name of the dropoff location column
        
    Returns:
        DataFrame with added location features
    """
    df = df.copy()
    
    # Create same zone indicator
    df['same_zone'] = df[pickup_loc_col] == df[dropoff_loc_col]
    
    # Create pickup and dropoff location frequency features
    pickup_counts = df[pickup_loc_col].value_counts().to_dict()
    dropoff_counts = df[dropoff_loc_col].value_counts().to_dict()
    
    df['pickup_loc_frequency'] = df[pickup_loc_col].map(pickup_counts)
    df['dropoff_loc_frequency'] = df[dropoff_loc_col].map(dropoff_counts)
    
    return df

def create_payment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features based on payment information.
    
    Args:
        df: DataFrame containing taxi trip data
        
    Returns:
        DataFrame with added payment features
    """
    df = df.copy()
    
    # Check if payment-related columns exist
    if 'payment_type' in df.columns:
        # Map payment types (specific to yellow/green taxis)
        payment_map = {
            1: 'Credit Card',
            2: 'Cash',
            3: 'No Charge',
            4: 'Dispute',
            5: 'Unknown',
            6: 'Voided Trip'
        }
        df['payment_method'] = df['payment_type'].map(payment_map)
    
    # Calculate tip percentage if tip and fare columns exist
    if 'tip_amount' in df.columns and 'fare_amount' in df.columns:
        df['tip_percentage'] = (df['tip_amount'] / df['fare_amount'].clip(lower=0.01)) * 100
        
        # Cap extreme values
        df['tip_percentage'] = df['tip_percentage'].clip(upper=100)
        
        # Create tip category
        bins = [0, 5, 10, 15, 20, float('inf')]
        labels = ['No Tip', 'Low Tip', 'Standard Tip', 'Good Tip', 'Excellent Tip']
        df['tip_category'] = pd.cut(df['tip_percentage'], 
                                   bins=bins, 
                                   labels=labels, 
                                   include_lowest=True)
    
    return df

def engineer_features(df: pd.DataFrame, taxi_type: str) -> pd.DataFrame:
    """
    Apply feature engineering based on taxi type.
    
    Args:
        df: DataFrame containing taxi trip data
        taxi_type: Type of taxi data ('yellow', 'green', 'fhv', or 'fhvhv')
        
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    if taxi_type == 'yellow':
        # Extract datetime features
        df = extract_datetime_features(df, 'tpep_pickup_datetime')
        
        # Calculate distance features
        df = calculate_distance_features(df)
        
        # Create location features
        df = create_location_features(df, 'PULocationID', 'DOLocationID')
        
        # Create payment features
        df = create_payment_features(df)
        
    elif taxi_type == 'green':
        # Extract datetime features
        df = extract_datetime_features(df, 'lpep_pickup_datetime')
        
        # Calculate distance features
        df = calculate_distance_features(df)
        
        # Create location features
        df = create_location_features(df, 'PULocationID', 'DOLocationID')
        
        # Create payment features
        df = create_payment_features(df)
        
    elif taxi_type == 'fhv':
        # Extract datetime features
        df = extract_datetime_features(df, 'pickup_datetime')
        
        # Create location features if location columns exist
        if 'PUlocationID' in df.columns and 'DOlocationID' in df.columns:
            df = create_location_features(df, 'PUlocationID', 'DOlocationID')
            
    elif taxi_type == 'fhvhv':
        # Extract datetime features
        df = extract_datetime_features(df, 'pickup_datetime')
        
        # Calculate distance features using trip_miles instead of trip_distance
        if 'trip_miles' in df.columns:
            df = calculate_distance_features(df, 'trip_miles', 'trip_duration')
        
        # Create location features
        df = create_location_features(df, 'PULocationID', 'DOLocationID')
        
        # Create payment-like features if applicable columns exist
        if 'tips' in df.columns and 'base_passenger_fare' in df.columns:
            df['tip_percentage'] = (df['tips'] / df['base_passenger_fare'].clip(lower=0.01)) * 100
            df['tip_percentage'] = df['tip_percentage'].clip(upper=100)
    
    return df
