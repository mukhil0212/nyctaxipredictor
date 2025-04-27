"""
Visualization utilities for NYC Taxi Trip Data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Union
import os

def set_plotting_style():
    """Set the default plotting style for visualizations."""
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12

def plot_trip_distribution(df: pd.DataFrame, 
                          column: str,
                          bins: int = 50,
                          title: Optional[str] = None,
                          save_path: Optional[str] = None):
    """
    Plot the distribution of a trip-related column.
    
    Args:
        df: DataFrame containing taxi trip data
        column: Column to plot
        bins: Number of bins for the histogram
        title: Plot title (defaults to column name if None)
        save_path: Path to save the figure (if None, figure is displayed but not saved)
    """
    set_plotting_style()
    
    plt.figure(figsize=(12, 8))
    
    # Plot histogram with KDE
    sns.histplot(df[column], bins=bins, kde=True)
    
    # Set title and labels
    plt.title(title or f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    
    # Add descriptive statistics as text
    stats_text = (
        f"Mean: {df[column].mean():.2f}\n"
        f"Median: {df[column].median():.2f}\n"
        f"Std Dev: {df[column].std():.2f}\n"
        f"Min: {df[column].min():.2f}\n"
        f"Max: {df[column].max():.2f}"
    )
    plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                 ha='right', va='top')
    
    plt.tight_layout()
    
    # Save or show the figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def plot_time_patterns(df: pd.DataFrame,
                      datetime_col: str,
                      value_col: str,
                      agg_func: str = 'mean',
                      title: Optional[str] = None,
                      save_path: Optional[str] = None):
    """
    Plot patterns over time (hourly, daily, etc.).
    
    Args:
        df: DataFrame containing taxi trip data
        datetime_col: Column containing datetime information (e.g., 'tpep_pickup_datetime_hour')
        value_col: Column to aggregate and plot
        agg_func: Aggregation function ('mean', 'median', 'sum', 'count')
        title: Plot title
        save_path: Path to save the figure (if None, figure is displayed but not saved)
    """
    set_plotting_style()
    
    # Aggregate data
    if agg_func == 'mean':
        agg_data = df.groupby(datetime_col)[value_col].mean()
    elif agg_func == 'median':
        agg_data = df.groupby(datetime_col)[value_col].median()
    elif agg_func == 'sum':
        agg_data = df.groupby(datetime_col)[value_col].sum()
    elif agg_func == 'count':
        agg_data = df.groupby(datetime_col)[value_col].count()
    else:
        raise ValueError(f"Unsupported aggregation function: {agg_func}")
    
    plt.figure(figsize=(12, 8))
    
    # Plot the aggregated data
    agg_data.plot(kind='line', marker='o')
    
    # Set title and labels
    plt.title(title or f'{agg_func.capitalize()} {value_col} by {datetime_col}')
    plt.xlabel(datetime_col)
    plt.ylabel(f'{agg_func.capitalize()} {value_col}')
    
    plt.grid(True)
    plt.tight_layout()
    
    # Save or show the figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def plot_correlation_matrix(df: pd.DataFrame,
                           columns: Optional[List[str]] = None,
                           title: str = 'Correlation Matrix',
                           save_path: Optional[str] = None):
    """
    Plot a correlation matrix for selected columns.
    
    Args:
        df: DataFrame containing taxi trip data
        columns: List of columns to include in the correlation matrix (if None, uses all numeric columns)
        title: Plot title
        save_path: Path to save the figure (if None, figure is displayed but not saved)
    """
    set_plotting_style()
    
    # Select columns for correlation
    if columns is None:
        # Use all numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        # Exclude columns that are likely to be IDs or categorical encoded as numbers
        exclude_patterns = ['ID', '_id', 'code']
        columns = [col for col in numeric_cols if not any(pattern.lower() in col.lower() for pattern in exclude_patterns)]
    
    # Calculate correlation matrix
    corr_matrix = df[columns].corr()
    
    plt.figure(figsize=(14, 12))
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=True, fmt='.2f', square=True, linewidths=.5)
    
    plt.title(title, fontsize=16)
    plt.tight_layout()
    
    # Save or show the figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def plot_categorical_analysis(df: pd.DataFrame,
                             cat_column: str,
                             value_column: str,
                             agg_func: str = 'mean',
                             top_n: Optional[int] = None,
                             title: Optional[str] = None,
                             save_path: Optional[str] = None):
    """
    Plot analysis of a value column grouped by a categorical column.
    
    Args:
        df: DataFrame containing taxi trip data
        cat_column: Categorical column to group by
        value_column: Value column to aggregate
        agg_func: Aggregation function ('mean', 'median', 'sum', 'count')
        top_n: Show only top N categories (if None, shows all)
        title: Plot title
        save_path: Path to save the figure (if None, figure is displayed but not saved)
    """
    set_plotting_style()
    
    # Aggregate data
    if agg_func == 'mean':
        agg_data = df.groupby(cat_column)[value_column].mean().sort_values(ascending=False)
    elif agg_func == 'median':
        agg_data = df.groupby(cat_column)[value_column].median().sort_values(ascending=False)
    elif agg_func == 'sum':
        agg_data = df.groupby(cat_column)[value_column].sum().sort_values(ascending=False)
    elif agg_func == 'count':
        agg_data = df.groupby(cat_column)[value_column].count().sort_values(ascending=False)
    else:
        raise ValueError(f"Unsupported aggregation function: {agg_func}")
    
    # Limit to top N categories if specified
    if top_n is not None:
        agg_data = agg_data.head(top_n)
    
    plt.figure(figsize=(12, 8))
    
    # Create bar plot
    agg_data.plot(kind='bar')
    
    # Set title and labels
    plt.title(title or f'{agg_func.capitalize()} {value_column} by {cat_column}')
    plt.xlabel(cat_column)
    plt.ylabel(f'{agg_func.capitalize()} {value_column}')
    
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    plt.tight_layout()
    
    # Save or show the figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def plot_scatter_with_trend(df: pd.DataFrame,
                           x_column: str,
                           y_column: str,
                           hue_column: Optional[str] = None,
                           title: Optional[str] = None,
                           save_path: Optional[str] = None):
    """
    Create a scatter plot with trend line.
    
    Args:
        df: DataFrame containing taxi trip data
        x_column: Column for x-axis
        y_column: Column for y-axis
        hue_column: Column for color coding points (optional)
        title: Plot title
        save_path: Path to save the figure (if None, figure is displayed but not saved)
    """
    set_plotting_style()
    
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    if hue_column:
        scatter = sns.scatterplot(data=df, x=x_column, y=y_column, hue=hue_column, alpha=0.6)
        # Add trend line for each category
        for category in df[hue_column].unique():
            subset = df[df[hue_column] == category]
            sns.regplot(data=subset, x=x_column, y=y_column, scatter=False, line_kws={"linestyle": "--"})
    else:
        scatter = sns.scatterplot(data=df, x=x_column, y=y_column, alpha=0.6)
        # Add overall trend line
        sns.regplot(data=df, x=x_column, y=y_column, scatter=False, line_kws={"color": "red"})
    
    # Set title and labels
    plt.title(title or f'Relationship between {x_column} and {y_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    
    # Add correlation coefficient as text
    corr = df[[x_column, y_column]].corr().iloc[0, 1]
    plt.annotate(f"Correlation: {corr:.2f}", xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                 ha='left', va='top')
    
    plt.grid(True)
    plt.tight_layout()
    
    # Save or show the figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def create_dashboard(df: pd.DataFrame, 
                    taxi_type: str,
                    output_dir: str):
    """
    Create a comprehensive dashboard of visualizations for the dataset.
    
    Args:
        df: DataFrame containing taxi trip data
        taxi_type: Type of taxi data ('yellow', 'green', 'fhv', or 'fhvhv')
        output_dir: Directory to save the visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set common prefix for filenames
    prefix = f"{taxi_type}_taxi_"
    
    # 1. Trip duration distribution
    if 'trip_duration' in df.columns:
        plot_trip_distribution(
            df, 'trip_duration', bins=50,
            title=f'{taxi_type.capitalize()} Taxi Trip Duration Distribution',
            save_path=os.path.join(output_dir, f"{prefix}duration_dist.png")
        )
    
    # 2. Trip distance distribution
    distance_col = 'trip_distance' if 'trip_distance' in df.columns else 'trip_miles'
    if distance_col in df.columns:
        plot_trip_distribution(
            df, distance_col, bins=50,
            title=f'{taxi_type.capitalize()} Taxi Trip Distance Distribution',
            save_path=os.path.join(output_dir, f"{prefix}distance_dist.png")
        )
    
    # 3. Fare amount distribution
    fare_col = 'fare_amount' if 'fare_amount' in df.columns else 'base_passenger_fare'
    if fare_col in df.columns:
        plot_trip_distribution(
            df, fare_col, bins=50,
            title=f'{taxi_type.capitalize()} Taxi Fare Distribution',
            save_path=os.path.join(output_dir, f"{prefix}fare_dist.png")
        )
    
    # 4. Hourly patterns
    datetime_col = None
    if taxi_type == 'yellow':
        datetime_col = 'tpep_pickup_datetime_hour'
    elif taxi_type == 'green':
        datetime_col = 'lpep_pickup_datetime_hour'
    else:
        datetime_col = 'pickup_datetime_hour'
    
    if datetime_col in df.columns:
        # Trip count by hour
        plot_time_patterns(
            df, datetime_col, 'trip_duration', agg_func='count',
            title=f'{taxi_type.capitalize()} Taxi Trip Count by Hour',
            save_path=os.path.join(output_dir, f"{prefix}hourly_count.png")
        )
        
        # Average trip duration by hour
        if 'trip_duration' in df.columns:
            plot_time_patterns(
                df, datetime_col, 'trip_duration', agg_func='mean',
                title=f'{taxi_type.capitalize()} Taxi Average Trip Duration by Hour',
                save_path=os.path.join(output_dir, f"{prefix}hourly_duration.png")
            )
    
    # 5. Day of week patterns
    dow_col = None
    if taxi_type == 'yellow':
        dow_col = 'tpep_pickup_datetime_dayofweek'
    elif taxi_type == 'green':
        dow_col = 'lpep_pickup_datetime_dayofweek'
    else:
        dow_col = 'pickup_datetime_dayofweek'
    
    if dow_col in df.columns:
        # Trip count by day of week
        plot_time_patterns(
            df, dow_col, 'trip_duration', agg_func='count',
            title=f'{taxi_type.capitalize()} Taxi Trip Count by Day of Week',
            save_path=os.path.join(output_dir, f"{prefix}dow_count.png")
        )
    
    # 6. Correlation matrix
    if 'trip_duration' in df.columns:
        # Select relevant columns for correlation
        corr_columns = ['trip_duration']
        
        # Add distance column
        if 'trip_distance' in df.columns:
            corr_columns.append('trip_distance')
        elif 'trip_miles' in df.columns:
            corr_columns.append('trip_miles')
        
        # Add fare column
        if 'fare_amount' in df.columns:
            corr_columns.append('fare_amount')
        elif 'base_passenger_fare' in df.columns:
            corr_columns.append('base_passenger_fare')
        
        # Add other relevant columns
        for col in ['passenger_count', 'tip_amount', 'total_amount', 'speed_mph']:
            if col in df.columns:
                corr_columns.append(col)
        
        plot_correlation_matrix(
            df, corr_columns,
            title=f'{taxi_type.capitalize()} Taxi Feature Correlations',
            save_path=os.path.join(output_dir, f"{prefix}correlation.png")
        )
    
    # 7. Scatter plot of distance vs. duration
    if 'trip_duration' in df.columns and (distance_col in df.columns):
        plot_scatter_with_trend(
            df, distance_col, 'trip_duration',
            title=f'{taxi_type.capitalize()} Taxi: Trip Distance vs. Duration',
            save_path=os.path.join(output_dir, f"{prefix}distance_duration.png")
        )
    
    # 8. Payment type analysis (for yellow and green taxis)
    if 'payment_method' in df.columns and 'fare_amount' in df.columns:
        plot_categorical_analysis(
            df, 'payment_method', 'fare_amount', agg_func='mean',
            title=f'{taxi_type.capitalize()} Taxi: Average Fare by Payment Method',
            save_path=os.path.join(output_dir, f"{prefix}payment_fare.png")
        )
    
    # 9. Location analysis
    pickup_col = 'PULocationID' if 'PULocationID' in df.columns else 'PUlocationID'
    if pickup_col in df.columns and 'trip_duration' in df.columns:
        plot_categorical_analysis(
            df, pickup_col, 'trip_duration', agg_func='mean', top_n=10,
            title=f'{taxi_type.capitalize()} Taxi: Average Trip Duration by Top Pickup Locations',
            save_path=os.path.join(output_dir, f"{prefix}location_duration.png")
        )
