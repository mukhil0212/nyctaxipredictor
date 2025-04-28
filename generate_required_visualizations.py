"""
Script to generate required visualizations for NYC Taxi Trip Data Analysis.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import folium
from folium.plugins import HeatMap
import math

# Set up plotting
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Create results directory if it doesn't exist
results_dir = 'results/required_visualizations'
os.makedirs(results_dir, exist_ok=True)

print("Loading data...")
from src.data.loader import load_taxi_data
from src.data.cleaner import clean_yellow_taxi_data
from src.data.feature_engineering import engineer_features
from src.data.feature_selection import (
    select_features_mutual_info, select_features_lasso,
    compare_feature_selection_methods, get_common_features
)
from src.models.trainer import (
    prepare_data_for_modeling, train_baseline_model, train_linear_regression,
    train_ridge_regression, train_random_forest, train_gradient_boosting,
    train_knn_regressor, evaluate_model, get_feature_importance
)

# Load yellow taxi data for January 2025
taxi_type = 'yellow'
months = ['2025-01']
raw_df = load_taxi_data(taxi_type, months)

# Sample for faster processing
sample_size = 10000
if len(raw_df) > sample_size:
    print(f"Sampling {sample_size} rows from {len(raw_df)} total rows")
    raw_df = raw_df.sample(n=sample_size, random_state=42)

# Clean the data
print("Cleaning data...")
clean_df = clean_yellow_taxi_data(raw_df)

# Engineer features
print("Engineering features...")
df = engineer_features(clean_df, taxi_type)

# Prepare data for modeling
print("Preparing data for modeling...")
target_col = 'trip_duration'
X_train, X_test, y_train, y_test, cat_cols, num_cols = prepare_data_for_modeling(df, target_col)
print(f"Data split into train ({len(X_train)} rows) and test ({len(X_test)} rows) sets")

# 1. FEATURE SELECTION VISUALIZATION
print("Generating feature selection visualizations...")

# Handle categorical columns for feature selection
# We need to exclude categorical columns for mutual information calculation
X_train_numeric = X_train.select_dtypes(include=['number'])
print(f"Using {X_train_numeric.shape[1]} numeric features for feature selection")

# Compare different feature selection methods
max_features = min(20, X_train_numeric.shape[1])
feature_selection_results = {}

# Use mutual information for feature selection
_, mi_features = select_features_mutual_info(X_train_numeric, y_train, k=max_features)
feature_selection_results['mutual_info'] = mi_features

# Use Lasso for feature selection
_, lasso_features = select_features_lasso(X_train_numeric, y_train)
feature_selection_results['lasso'] = lasso_features

# Count how many methods selected each feature
feature_counts = {}
for method, features in feature_selection_results.items():
    for feature in features:
        if feature in feature_counts:
            feature_counts[feature] += 1
        else:
            feature_counts[feature] = 1

# Create a DataFrame for visualization
feature_selection_df = pd.DataFrame({
    'feature': list(feature_counts.keys()),
    'count': list(feature_counts.values())
}).sort_values('count', ascending=False)

# Visualize feature selection frequency
plt.figure(figsize=(14, 10))
sns.barplot(data=feature_selection_df.head(20), x='count', y='feature')
plt.title('Features Selected by Multiple Methods', fontsize=16)
plt.xlabel('Number of Methods That Selected the Feature', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.grid(axis='x')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'feature_selection_frequency.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Created feature selection frequency plot")

# Get mutual information scores for numeric features
mi_scores = pd.DataFrame({
    'feature': X_train_numeric.columns,
    'mutual_info_score': [0] * len(X_train_numeric.columns)
})
for i, feature in enumerate(mi_features):
    mi_scores.loc[mi_scores['feature'] == feature, 'mutual_info_score'] = len(mi_features) - i

# Visualize mutual information scores
plt.figure(figsize=(14, 10))
sns.barplot(data=mi_scores.sort_values('mutual_info_score', ascending=False).head(20),
            x='mutual_info_score', y='feature')
plt.title('Feature Importance by Mutual Information', fontsize=16)
plt.xlabel('Mutual Information Score (Rank)', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.grid(axis='x')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'mutual_info_feature_importance.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Created mutual information feature importance plot")

# Since we only have two methods, just use the mutual information features
selected_features = mi_features
print(f"Selected {len(selected_features)} features for modeling")

# Filter data to include only selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]
cat_cols_selected = [col for col in cat_cols if col in selected_features]
num_cols_selected = [col for col in num_cols if col in selected_features]

# 2. MODEL TRAINING AND COMPARISON
print("Training models and generating comparison visualizations...")

# Train baseline model
baseline_model, baseline_metrics = train_baseline_model(X_train_selected, y_train,
                                                       cat_cols_selected, num_cols_selected)
baseline_test_metrics = evaluate_model(baseline_model, X_test_selected, y_test)
baseline_metrics.update(baseline_test_metrics)

# Train linear regression model
linear_model, linear_metrics = train_linear_regression(X_train_selected, y_train,
                                                      cat_cols_selected, num_cols_selected)
linear_test_metrics = evaluate_model(linear_model, X_test_selected, y_test)
linear_metrics.update(linear_test_metrics)

# Train ridge regression model
ridge_model, ridge_metrics = train_ridge_regression(X_train_selected, y_train,
                                                   cat_cols_selected, num_cols_selected)
ridge_test_metrics = evaluate_model(ridge_model, X_test_selected, y_test)
ridge_metrics.update(ridge_test_metrics)

# Train random forest model
rf_model, rf_metrics = train_random_forest(X_train_selected, y_train,
                                          cat_cols_selected, num_cols_selected)
rf_test_metrics = evaluate_model(rf_model, X_test_selected, y_test)
rf_metrics.update(rf_test_metrics)

# Train KNN model
knn_model, knn_metrics = train_knn_regressor(X_train_selected, y_train,
                                            cat_cols_selected, num_cols_selected)
knn_test_metrics = evaluate_model(knn_model, X_test_selected, y_test)
knn_metrics.update(knn_test_metrics)

# Collect all model metrics
model_metrics = {
    'Baseline': baseline_metrics,
    'Linear Regression': linear_metrics,
    'Ridge Regression': ridge_metrics,
    'Random Forest': rf_metrics,
    'K-Nearest Neighbors': knn_metrics
}

# Create a DataFrame for model comparison
metrics_df = pd.DataFrame({
    'Model': list(model_metrics.keys()),
    'Train RMSE': [metrics['train_rmse'] for metrics in model_metrics.values()],
    'Test RMSE': [metrics['test_rmse'] for metrics in model_metrics.values()],
    'Train R²': [metrics['train_r2'] for metrics in model_metrics.values()],
    'Test R²': [metrics['test_r2'] for metrics in model_metrics.values()]
})

# Sort by test RMSE (ascending)
metrics_df = metrics_df.sort_values('Test RMSE')

# Visualize model comparison (RMSE)
plt.figure(figsize=(12, 8))
sns.barplot(data=metrics_df, x='Test RMSE', y='Model')
plt.title('Model Comparison: Test RMSE (Lower is Better)', fontsize=16)
plt.xlabel('Root Mean Squared Error (RMSE)', fontsize=14)
plt.ylabel('Model', fontsize=14)
plt.grid(axis='x')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'model_comparison_rmse.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Created model comparison RMSE plot")

# Visualize model comparison (R²)
plt.figure(figsize=(12, 8))
sns.barplot(data=metrics_df, x='Test R²', y='Model')
plt.title('Model Comparison: Test R² (Higher is Better)', fontsize=16)
plt.xlabel('R² Score', fontsize=14)
plt.ylabel('Model', fontsize=14)
plt.grid(axis='x')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'model_comparison_r2.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Created model comparison R² plot")

# 3. FEATURE IMPORTANCE VISUALIZATION
print("Generating feature importance visualizations...")

# Get feature importance from Random Forest model
rf_importance = get_feature_importance(rf_model, cat_cols_selected, num_cols_selected)

# Visualize feature importance
plt.figure(figsize=(14, 10))
sns.barplot(data=rf_importance.head(20), x='importance', y='feature')
plt.title('Random Forest Feature Importance', fontsize=16)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.grid(axis='x')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'random_forest_feature_importance.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Created Random Forest feature importance plot")

# 4. TIME SERIES ANALYSIS
print("Generating time series visualizations...")

# Aggregate trip count by hour
hourly_counts = df.groupby('tpep_pickup_datetime_hour')['trip_duration'].count()

# Visualize hourly trip counts
plt.figure(figsize=(12, 8))
hourly_counts.plot(kind='line', marker='o')
plt.title('Trip Count by Hour of Day', fontsize=16)
plt.xlabel('Hour of Day', fontsize=14)
plt.ylabel('Number of Trips', fontsize=14)
plt.xticks(range(24))
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'hourly_trip_counts.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Created hourly trip counts plot")

# Aggregate average trip duration by hour
hourly_durations = df.groupby('tpep_pickup_datetime_hour')['trip_duration'].mean()

# Visualize hourly trip durations
plt.figure(figsize=(12, 8))
hourly_durations.plot(kind='line', marker='o')
plt.title('Average Trip Duration by Hour of Day', fontsize=16)
plt.xlabel('Hour of Day', fontsize=14)
plt.ylabel('Average Trip Duration (minutes)', fontsize=14)
plt.xticks(range(24))
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'hourly_trip_durations.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Created hourly trip durations plot")

# Aggregate trip count by day of week
dow_counts = df.groupby('tpep_pickup_datetime_dayofweek')['trip_duration'].count()

# Visualize day of week trip counts
plt.figure(figsize=(12, 8))
dow_counts.plot(kind='line', marker='o')
plt.title('Trip Count by Day of Week', fontsize=16)
plt.xlabel('Day of Week (0 = Monday, 6 = Sunday)', fontsize=14)
plt.ylabel('Number of Trips', fontsize=14)
plt.xticks(range(7))
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'dow_trip_counts.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Created day of week trip counts plot")

# 5. MAP VISUALIZATION (if coordinates are available)
print("Generating map visualizations...")

# Check if we have location data
if 'PULocationID' in df.columns and 'DOLocationID' in df.columns:
    # Create a heatmap of pickup locations
    pickup_counts = df['PULocationID'].value_counts().reset_index()
    pickup_counts.columns = ['PULocationID', 'count']

    # Visualize pickup location frequency
    plt.figure(figsize=(14, 10))
    sns.barplot(data=pickup_counts.head(20), x='count', y='PULocationID')
    plt.title('Top 20 Pickup Locations by Frequency', fontsize=16)
    plt.xlabel('Number of Pickups', fontsize=14)
    plt.ylabel('Pickup Location ID', fontsize=14)
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'pickup_location_frequency.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created pickup location frequency plot")

    # Create a heatmap of dropoff locations
    dropoff_counts = df['DOLocationID'].value_counts().reset_index()
    dropoff_counts.columns = ['DOLocationID', 'count']

    # Visualize dropoff location frequency
    plt.figure(figsize=(14, 10))
    sns.barplot(data=dropoff_counts.head(20), x='count', y='DOLocationID')
    plt.title('Top 20 Dropoff Locations by Frequency', fontsize=16)
    plt.xlabel('Number of Dropoffs', fontsize=14)
    plt.ylabel('Dropoff Location ID', fontsize=14)
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'dropoff_location_frequency.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created dropoff location frequency plot")

# 6. PREDICTION ERROR ANALYSIS
print("Generating prediction error visualizations...")

# Get the best model (Random Forest in this case)
best_model = rf_model
y_pred = best_model.predict(X_test_selected)
errors = y_test - y_pred

# Visualize prediction errors
plt.figure(figsize=(12, 8))
sns.histplot(errors, bins=50, kde=True)
plt.axvline(x=0, color='r', linestyle='--')
plt.title('Distribution of Prediction Errors (Random Forest)', fontsize=16)
plt.xlabel('Prediction Error (minutes)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'prediction_error_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Created prediction error distribution plot")

# Visualize actual vs. predicted values
plt.figure(figsize=(12, 8))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Actual vs. Predicted Trip Duration (Random Forest)', fontsize=16)
plt.xlabel('Actual Trip Duration (minutes)', fontsize=14)
plt.ylabel('Predicted Trip Duration (minutes)', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'actual_vs_predicted.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Created actual vs. predicted plot")

# Visualize prediction errors by trip distance
plt.figure(figsize=(12, 8))
error_df = pd.DataFrame({
    'trip_distance': X_test['trip_distance'] if 'trip_distance' in X_test.columns else X_test_selected.iloc[:, 0],
    'error': errors
})
sns.scatterplot(data=error_df, x='trip_distance', y='error', alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Prediction Errors by Trip Distance (Random Forest)', fontsize=16)
plt.xlabel('Trip Distance (miles)', fontsize=14)
plt.ylabel('Prediction Error (minutes)', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'prediction_errors_by_distance.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Created prediction errors by distance plot")

print(f"All required visualizations saved to {results_dir}")
