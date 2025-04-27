"""
Feature selection utilities for NYC Taxi Trip Data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from sklearn.feature_selection import (
    SelectKBest, mutual_info_regression, f_regression,
    RFE, SelectFromModel, VarianceThreshold
)
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def select_features_mutual_info(X: pd.DataFrame, 
                              y: pd.Series, 
                              k: int = 15) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select top k features using mutual information.
    
    Args:
        X: Feature matrix
        y: Target vector
        k: Number of features to select
        
    Returns:
        Tuple containing (selected feature matrix, list of selected feature names)
    """
    # Initialize the selector
    selector = SelectKBest(mutual_info_regression, k=k)
    
    # Fit the selector
    X_selected = selector.fit_transform(X, y)
    
    # Get the selected feature names
    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask].tolist()
    
    # Create a DataFrame with selected features
    X_selected_df = pd.DataFrame(X_selected, columns=selected_features)
    
    return X_selected_df, selected_features

def select_features_f_regression(X: pd.DataFrame, 
                               y: pd.Series, 
                               k: int = 15) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select top k features using F-statistic.
    
    Args:
        X: Feature matrix
        y: Target vector
        k: Number of features to select
        
    Returns:
        Tuple containing (selected feature matrix, list of selected feature names)
    """
    # Initialize the selector
    selector = SelectKBest(f_regression, k=k)
    
    # Fit the selector
    X_selected = selector.fit_transform(X, y)
    
    # Get the selected feature names
    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask].tolist()
    
    # Create a DataFrame with selected features
    X_selected_df = pd.DataFrame(X_selected, columns=selected_features)
    
    return X_selected_df, selected_features

def select_features_rfe(X: pd.DataFrame, 
                      y: pd.Series, 
                      k: int = 15,
                      estimator=None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select top k features using Recursive Feature Elimination.
    
    Args:
        X: Feature matrix
        y: Target vector
        k: Number of features to select
        estimator: Estimator to use for feature importance (default: RandomForestRegressor)
        
    Returns:
        Tuple containing (selected feature matrix, list of selected feature names)
    """
    # Set default estimator if none provided
    if estimator is None:
        estimator = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Initialize the selector
    selector = RFE(estimator, n_features_to_select=k, step=1)
    
    # Fit the selector
    X_selected = selector.fit_transform(X, y)
    
    # Get the selected feature names
    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask].tolist()
    
    # Create a DataFrame with selected features
    X_selected_df = pd.DataFrame(X_selected, columns=selected_features)
    
    return X_selected_df, selected_features

def select_features_lasso(X: pd.DataFrame, 
                        y: pd.Series, 
                        alpha: float = 0.01) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select features using Lasso regularization.
    
    Args:
        X: Feature matrix
        y: Target vector
        alpha: Regularization strength
        
    Returns:
        Tuple containing (selected feature matrix, list of selected feature names)
    """
    # Scale features for better Lasso performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize and fit Lasso
    lasso = Lasso(alpha=alpha, random_state=42)
    lasso.fit(X_scaled, y)
    
    # Get feature importances
    importances = np.abs(lasso.coef_)
    
    # Select features with non-zero coefficients
    selected_mask = importances > 0
    selected_features = X.columns[selected_mask].tolist()
    
    # Create a DataFrame with selected features
    X_selected = X.iloc[:, selected_mask]
    
    return X_selected, selected_features

def select_features_variance_threshold(X: pd.DataFrame, 
                                     threshold: float = 0.01) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove features with low variance.
    
    Args:
        X: Feature matrix
        threshold: Variance threshold
        
    Returns:
        Tuple containing (selected feature matrix, list of selected feature names)
    """
    # Initialize the selector
    selector = VarianceThreshold(threshold=threshold)
    
    # Fit the selector
    X_selected = selector.fit_transform(X)
    
    # Get the selected feature names
    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask].tolist()
    
    # Create a DataFrame with selected features
    X_selected_df = pd.DataFrame(X_selected, columns=selected_features)
    
    return X_selected_df, selected_features

def select_features_random_forest(X: pd.DataFrame, 
                                y: pd.Series, 
                                threshold: str = 'mean') -> Tuple[pd.DataFrame, List[str]]:
    """
    Select features using Random Forest feature importance.
    
    Args:
        X: Feature matrix
        y: Target vector
        threshold: Threshold strategy ('mean', 'median', or a float)
        
    Returns:
        Tuple containing (selected feature matrix, list of selected feature names)
    """
    # Initialize and fit Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Initialize the selector
    selector = SelectFromModel(rf, threshold=threshold)
    
    # Fit the selector
    X_selected = selector.fit_transform(X, y)
    
    # Get the selected feature names
    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask].tolist()
    
    # Create a DataFrame with selected features
    X_selected_df = pd.DataFrame(X_selected, columns=selected_features)
    
    return X_selected_df, selected_features

def compare_feature_selection_methods(X: pd.DataFrame, 
                                    y: pd.Series, 
                                    k: int = 15) -> Dict[str, List[str]]:
    """
    Compare different feature selection methods.
    
    Args:
        X: Feature matrix
        y: Target vector
        k: Number of features to select
        
    Returns:
        Dictionary mapping method names to lists of selected features
    """
    results = {}
    
    # Mutual Information
    _, mi_features = select_features_mutual_info(X, y, k)
    results['mutual_info'] = mi_features
    
    # F-regression
    _, f_features = select_features_f_regression(X, y, k)
    results['f_regression'] = f_features
    
    # RFE
    _, rfe_features = select_features_rfe(X, y, k)
    results['rfe'] = rfe_features
    
    # Lasso
    _, lasso_features = select_features_lasso(X, y)
    results['lasso'] = lasso_features
    
    # Random Forest
    _, rf_features = select_features_random_forest(X, y)
    results['random_forest'] = rf_features
    
    return results

def get_common_features(feature_lists: Dict[str, List[str]], 
                       min_methods: int = 2) -> List[str]:
    """
    Get features that are selected by multiple methods.
    
    Args:
        feature_lists: Dictionary mapping method names to lists of selected features
        min_methods: Minimum number of methods that must select a feature
        
    Returns:
        List of common features
    """
    # Count how many methods selected each feature
    feature_counts = {}
    for method, features in feature_lists.items():
        for feature in features:
            if feature in feature_counts:
                feature_counts[feature] += 1
            else:
                feature_counts[feature] = 1
    
    # Get features that appear in at least min_methods methods
    common_features = [feature for feature, count in feature_counts.items() 
                      if count >= min_methods]
    
    return common_features
