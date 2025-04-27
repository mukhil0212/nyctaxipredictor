"""
Script to generate visualizations for NYC Taxi Trip Data.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set up plotting
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Create results directory if it doesn't exist
results_dir = 'results/visualizations'
os.makedirs(results_dir, exist_ok=True)

# Load the data
print("Loading data...")
from src.data.loader import load_taxi_data
from src.data.cleaner import clean_yellow_taxi_data
from src.data.feature_engineering import engineer_features

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

# Create visualizations
print("Creating visualizations...")

# 1. Trip Duration Distribution
plt.figure(figsize=(12, 8))
sns.histplot(df['trip_duration'], bins=50, kde=True)
plt.title('Distribution of Trip Duration')
plt.xlabel('Trip Duration (minutes)')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'trip_duration_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Created trip duration distribution plot")

# 2. Trip Distance Distribution
plt.figure(figsize=(12, 8))
sns.histplot(df['trip_distance'], bins=50, kde=True)
plt.title('Distribution of Trip Distance')
plt.xlabel('Trip Distance (miles)')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'trip_distance_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Created trip distance distribution plot")

# 3. Trip Distance vs. Duration
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='trip_distance', y='trip_duration', alpha=0.5)
plt.title('Trip Distance vs. Duration')
plt.xlabel('Trip Distance (miles)')
plt.ylabel('Trip Duration (minutes)')
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'trip_distance_vs_duration.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Created trip distance vs. duration plot")

# 4. Trip Duration by Hour of Day
plt.figure(figsize=(12, 8))
sns.boxplot(data=df, x='tpep_pickup_datetime_hour', y='trip_duration')
plt.title('Trip Duration by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Trip Duration (minutes)')
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'trip_duration_by_hour.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Created trip duration by hour plot")

# 5. Trip Duration by Day of Week
plt.figure(figsize=(12, 8))
sns.boxplot(data=df, x='tpep_pickup_datetime_dayofweek', y='trip_duration')
plt.title('Trip Duration by Day of Week')
plt.xlabel('Day of Week (0 = Monday, 6 = Sunday)')
plt.ylabel('Trip Duration (minutes)')
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'trip_duration_by_day.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Created trip duration by day plot")

# 6. Trip Duration by Period of Day
plt.figure(figsize=(12, 8))
sns.boxplot(data=df, x='tpep_pickup_datetime_period', y='trip_duration')
plt.title('Trip Duration by Period of Day')
plt.xlabel('Period of Day')
plt.ylabel('Trip Duration (minutes)')
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'trip_duration_by_period.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Created trip duration by period plot")

# 7. Fare Amount Distribution
plt.figure(figsize=(12, 8))
sns.histplot(df['fare_amount'], bins=50, kde=True)
plt.title('Distribution of Fare Amount')
plt.xlabel('Fare Amount ($)')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'fare_amount_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Created fare amount distribution plot")

# 8. Fare Amount vs. Trip Distance
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='trip_distance', y='fare_amount', alpha=0.5)
plt.title('Fare Amount vs. Trip Distance')
plt.xlabel('Trip Distance (miles)')
plt.ylabel('Fare Amount ($)')
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'fare_vs_distance.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Created fare vs. distance plot")

# 9. Speed Distribution
plt.figure(figsize=(12, 8))
sns.histplot(df['speed_mph'], bins=50, kde=True)
plt.title('Distribution of Speed')
plt.xlabel('Speed (mph)')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'speed_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Created speed distribution plot")

# 10. Correlation Matrix
plt.figure(figsize=(14, 12))
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
# Exclude columns that are likely to be IDs or categorical encoded as numbers
exclude_patterns = ['ID', '_id', 'code']
numeric_cols = [col for col in numeric_cols if not any(pattern.lower() in col.lower() for pattern in exclude_patterns)]
# Limit to 15 columns for readability
if len(numeric_cols) > 15:
    numeric_cols = numeric_cols[:15]
corr_matrix = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
            annot=True, fmt='.2f', square=True, linewidths=.5)
plt.title('Correlation Matrix of Numeric Features', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Created correlation matrix plot")

print(f"All visualizations saved to {results_dir}")
