"""
Main script for NYC Taxi Trip Data Analysis.
"""

import os
import pandas as pd
import numpy as np
import argparse
from datetime import datetime

from src.config import RESULTS_DIR, MODELS_DIR
from src.data.loader import load_taxi_data
from src.data.cleaner import (
    clean_yellow_taxi_data, clean_green_taxi_data,
    clean_fhv_data, clean_fhvhv_data
)
from src.data.feature_engineering import engineer_features
from src.data.feature_selection import (
    select_features_mutual_info, select_features_lasso,
    compare_feature_selection_methods, get_common_features
)
from src.visualization.visualizer import create_dashboard
from src.models.trainer import (
    prepare_data_for_modeling, train_linear_regression,
    train_ridge_regression, train_random_forest,
    train_gradient_boosting, train_baseline_model,
    train_knn_regressor, train_svm_regressor, train_naive_bayes,
    evaluate_model, save_model, get_feature_importance
)
from src.models.evaluator import create_evaluation_dashboard
from src.utils.helpers import log_message, save_metrics, print_dataframe_info, sample_dataframe, timer_decorator

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='NYC Taxi Trip Data Analysis')

    parser.add_argument('--taxi-type', type=str, default='yellow',
                        choices=['yellow', 'green', 'fhv', 'fhvhv'],
                        help='Type of taxi data to analyze')

    parser.add_argument('--months', type=str, nargs='+', default=['2025-01', '2025-02'],
                        help='Months to analyze (e.g., 2025-01 2025-02)')

    parser.add_argument('--sample-size', type=int, default=100000,
                        help='Number of rows to sample for analysis (use -1 for all data)')

    parser.add_argument('--skip-eda', action='store_true',
                        help='Skip exploratory data analysis')

    parser.add_argument('--skip-modeling', action='store_true',
                        help='Skip model training and evaluation')

    parser.add_argument('--models', type=str, nargs='+',
                        default=['baseline', 'linear', 'ridge', 'random_forest', 'gradient_boosting', 'knn', 'svm', 'naive_bayes'],
                        choices=['baseline', 'linear', 'ridge', 'random_forest', 'gradient_boosting', 'knn', 'svm', 'naive_bayes'],
                        help='Models to train')

    parser.add_argument('--feature-selection', action='store_true',
                        help='Apply feature selection techniques')

    parser.add_argument('--max-features', type=int, default=20,
                        help='Maximum number of features to select when using feature selection')

    return parser.parse_args()

@timer_decorator
def run_data_pipeline(taxi_type, months, sample_size):
    """Run the data loading, cleaning, and feature engineering pipeline."""
    log_message(f"Loading {taxi_type} taxi data for months: {months}")
    df = load_taxi_data(taxi_type, months)
    print_dataframe_info(df, f"Raw {taxi_type} taxi data")

    # Sample data if requested
    if sample_size > 0 and len(df) > sample_size:
        log_message(f"Sampling {sample_size} rows from {len(df)} total rows")
        df = sample_dataframe(df, n=sample_size)

    # Clean data based on taxi type
    log_message(f"Cleaning {taxi_type} taxi data")
    if taxi_type == 'yellow':
        df_clean = clean_yellow_taxi_data(df)
    elif taxi_type == 'green':
        df_clean = clean_green_taxi_data(df)
    elif taxi_type == 'fhv':
        df_clean = clean_fhv_data(df)
    elif taxi_type == 'fhvhv':
        df_clean = clean_fhvhv_data(df)
    else:
        raise ValueError(f"Unsupported taxi type: {taxi_type}")

    print_dataframe_info(df_clean, f"Cleaned {taxi_type} taxi data")

    # Engineer features
    log_message(f"Engineering features for {taxi_type} taxi data")
    df_features = engineer_features(df_clean, taxi_type)
    print_dataframe_info(df_features, f"Feature-engineered {taxi_type} taxi data")

    return df_features

@timer_decorator
def run_eda(df, taxi_type):
    """Run exploratory data analysis and create visualizations."""
    log_message(f"Running EDA for {taxi_type} taxi data")

    # Create visualization dashboard
    output_dir = os.path.join(RESULTS_DIR, 'visualizations', taxi_type)
    create_dashboard(df, taxi_type, output_dir)

    log_message(f"EDA visualizations saved to {output_dir}")

@timer_decorator
def run_modeling(df, taxi_type, models_to_train, apply_feature_selection=False, max_features=20):
    """Train and evaluate models."""
    log_message(f"Running modeling for {taxi_type} taxi data")

    # Prepare data for modeling
    target_col = 'trip_duration'
    if target_col not in df.columns:
        log_message(f"Target column '{target_col}' not found in data. Skipping modeling.")
        return

    # Prepare data
    X_train, X_test, y_train, y_test, cat_cols, num_cols = prepare_data_for_modeling(df, target_col)
    log_message(f"Data split into train ({len(X_train)} rows) and test ({len(X_test)} rows) sets")

    # Apply feature selection if requested
    if apply_feature_selection:
        log_message(f"Applying feature selection to reduce features (max: {max_features})")

        # Compare different feature selection methods
        feature_selection_results = compare_feature_selection_methods(X_train, y_train, k=max_features)

        # Get common features selected by at least 2 methods
        selected_features = get_common_features(feature_selection_results, min_methods=2)

        # If we have too many features, use mutual information to select top max_features
        if len(selected_features) > max_features:
            log_message(f"Selected {len(selected_features)} features, reducing to {max_features} using mutual information")
            _, selected_features = select_features_mutual_info(X_train, y_train, k=max_features)

        # If we have too few features, use mutual information to select features
        if len(selected_features) < 5:
            log_message(f"Only {len(selected_features)} features selected, using mutual information to select {max_features}")
            _, selected_features = select_features_mutual_info(X_train, y_train, k=max_features)

        # Filter X_train and X_test to include only selected features
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]

        # Update categorical and numerical columns
        cat_cols = [col for col in cat_cols if col in selected_features]
        num_cols = [col for col in num_cols if col in selected_features]

        log_message(f"Feature selection complete. Selected {len(selected_features)} features.")
        log_message(f"Selected features: {selected_features}")

        # Use the selected features for modeling
        X_train = X_train_selected
        X_test = X_test_selected

    # Create directories for results
    models_output_dir = os.path.join(MODELS_DIR, taxi_type)
    eval_output_dir = os.path.join(RESULTS_DIR, 'model_evaluation', taxi_type)
    os.makedirs(models_output_dir, exist_ok=True)
    os.makedirs(eval_output_dir, exist_ok=True)

    # Train models
    trained_models = {}
    model_metrics = {}

    # First, train a baseline model for comparison
    if 'baseline' in models_to_train:
        log_message("Training Baseline model (mean prediction)")
        baseline_model, baseline_metrics = train_baseline_model(X_train, y_train, cat_cols, num_cols, strategy='mean')
        baseline_test_metrics = evaluate_model(baseline_model, X_test, y_test)
        baseline_metrics.update(baseline_test_metrics)

        # Save model and metrics
        save_model(baseline_model, 'baseline_mean', models_output_dir)
        save_metrics(baseline_metrics, os.path.join(eval_output_dir, 'baseline_mean_metrics.json'))

        # Create evaluation dashboard
        create_evaluation_dashboard(
            baseline_model, X_test, y_test, pd.DataFrame({'feature': [], 'importance': []}),
            'Baseline_Mean', eval_output_dir
        )

        trained_models['Baseline (Mean)'] = baseline_model
        model_metrics['Baseline (Mean)'] = baseline_metrics

    if 'linear' in models_to_train:
        log_message("Training Linear Regression model")
        linear_model, linear_metrics = train_linear_regression(X_train, y_train, cat_cols, num_cols)
        linear_test_metrics = evaluate_model(linear_model, X_test, y_test)
        linear_metrics.update(linear_test_metrics)

        # Save model and metrics
        save_model(linear_model, 'linear_regression', models_output_dir)
        save_metrics(linear_metrics, os.path.join(eval_output_dir, 'linear_regression_metrics.json'))

        # Get feature importance (not available for linear regression)
        linear_importance = pd.DataFrame({'feature': [], 'importance': []})

        # Create evaluation dashboard
        create_evaluation_dashboard(
            linear_model, X_test, y_test, linear_importance,
            'Linear_Regression', eval_output_dir
        )

        trained_models['Linear Regression'] = linear_model
        model_metrics['Linear Regression'] = linear_metrics

    if 'ridge' in models_to_train:
        log_message("Training Ridge Regression model")
        ridge_model, ridge_metrics = train_ridge_regression(X_train, y_train, cat_cols, num_cols)
        ridge_test_metrics = evaluate_model(ridge_model, X_test, y_test)
        ridge_metrics.update(ridge_test_metrics)

        # Save model and metrics
        save_model(ridge_model, 'ridge_regression', models_output_dir)
        save_metrics(ridge_metrics, os.path.join(eval_output_dir, 'ridge_regression_metrics.json'))

        # Get feature importance (not available for ridge regression)
        ridge_importance = pd.DataFrame({'feature': [], 'importance': []})

        # Create evaluation dashboard
        create_evaluation_dashboard(
            ridge_model, X_test, y_test, ridge_importance,
            'Ridge_Regression', eval_output_dir
        )

        trained_models['Ridge Regression'] = ridge_model
        model_metrics['Ridge Regression'] = ridge_metrics

    if 'random_forest' in models_to_train:
        log_message("Training Random Forest model")
        rf_model, rf_metrics = train_random_forest(X_train, y_train, cat_cols, num_cols)
        rf_test_metrics = evaluate_model(rf_model, X_test, y_test)
        rf_metrics.update(rf_test_metrics)

        # Save model and metrics
        save_model(rf_model, 'random_forest', models_output_dir)
        save_metrics(rf_metrics, os.path.join(eval_output_dir, 'random_forest_metrics.json'))

        # Get feature importance
        rf_importance = get_feature_importance(rf_model, cat_cols, num_cols)

        # Create evaluation dashboard
        create_evaluation_dashboard(
            rf_model, X_test, y_test, rf_importance,
            'Random_Forest', eval_output_dir
        )

        trained_models['Random Forest'] = rf_model
        model_metrics['Random Forest'] = rf_metrics

    if 'gradient_boosting' in models_to_train:
        log_message("Training Gradient Boosting model")
        gb_model, gb_metrics = train_gradient_boosting(X_train, y_train, cat_cols, num_cols)
        gb_test_metrics = evaluate_model(gb_model, X_test, y_test)
        gb_metrics.update(gb_test_metrics)

        # Save model and metrics
        save_model(gb_model, 'gradient_boosting', models_output_dir)
        save_metrics(gb_metrics, os.path.join(eval_output_dir, 'gradient_boosting_metrics.json'))

        # Get feature importance
        gb_importance = get_feature_importance(gb_model, cat_cols, num_cols)

        # Create evaluation dashboard
        create_evaluation_dashboard(
            gb_model, X_test, y_test, gb_importance,
            'Gradient_Boosting', eval_output_dir
        )

        trained_models['Gradient Boosting'] = gb_model
        model_metrics['Gradient Boosting'] = gb_metrics

    if 'knn' in models_to_train:
        log_message("Training K-Nearest Neighbors model")
        knn_model, knn_metrics = train_knn_regressor(X_train, y_train, cat_cols, num_cols)
        knn_test_metrics = evaluate_model(knn_model, X_test, y_test)
        knn_metrics.update(knn_test_metrics)

        # Save model and metrics
        save_model(knn_model, 'knn_regressor', models_output_dir)
        save_metrics(knn_metrics, os.path.join(eval_output_dir, 'knn_regressor_metrics.json'))

        # Create evaluation dashboard (KNN doesn't have feature importance)
        create_evaluation_dashboard(
            knn_model, X_test, y_test, pd.DataFrame({'feature': [], 'importance': []}),
            'KNN_Regressor', eval_output_dir
        )

        trained_models['K-Nearest Neighbors'] = knn_model
        model_metrics['K-Nearest Neighbors'] = knn_metrics

    if 'svm' in models_to_train:
        log_message("Training Support Vector Machine model")
        svm_model, svm_metrics = train_svm_regressor(X_train, y_train, cat_cols, num_cols)
        svm_test_metrics = evaluate_model(svm_model, X_test, y_test)
        svm_metrics.update(svm_test_metrics)

        # Save model and metrics
        save_model(svm_model, 'svm_regressor', models_output_dir)
        save_metrics(svm_metrics, os.path.join(eval_output_dir, 'svm_regressor_metrics.json'))

        # Create evaluation dashboard (SVM doesn't have feature importance)
        create_evaluation_dashboard(
            svm_model, X_test, y_test, pd.DataFrame({'feature': [], 'importance': []}),
            'SVM_Regressor', eval_output_dir
        )

        trained_models['Support Vector Machine'] = svm_model
        model_metrics['Support Vector Machine'] = svm_metrics

    if 'naive_bayes' in models_to_train:
        log_message("Training Naive Bayes model")
        nb_model, nb_metrics = train_naive_bayes(X_train, y_train, cat_cols, num_cols)
        nb_test_metrics = evaluate_model(nb_model, X_test, y_test)
        nb_metrics.update(nb_test_metrics)

        # Save model and metrics
        save_model(nb_model, 'naive_bayes', models_output_dir)
        save_metrics(nb_metrics, os.path.join(eval_output_dir, 'naive_bayes_metrics.json'))

        # Create evaluation dashboard (Naive Bayes doesn't have feature importance)
        create_evaluation_dashboard(
            nb_model, X_test, y_test, pd.DataFrame({'feature': [], 'importance': []}),
            'Naive_Bayes', eval_output_dir
        )

        trained_models['Naive Bayes'] = nb_model
        model_metrics['Naive Bayes'] = nb_metrics

    log_message(f"Model training and evaluation completed. Results saved to {eval_output_dir}")

    return trained_models, model_metrics

def main():
    """Main function to run the NYC Taxi Trip Data Analysis."""
    # Parse command line arguments
    args = parse_args()

    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(RESULTS_DIR, f"nyc_taxi_analysis_{args.taxi_type}_{timestamp}.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    log_message(f"Starting NYC Taxi Trip Data Analysis for {args.taxi_type} taxi", log_file)
    log_message(f"Arguments: {args}", log_file)

    try:
        # Run data pipeline
        df = run_data_pipeline(args.taxi_type, args.months, args.sample_size)

        # Run EDA if not skipped
        if not args.skip_eda:
            run_eda(df, args.taxi_type)
        else:
            log_message("Skipping EDA as requested", log_file)

        # Run modeling if not skipped
        if not args.skip_modeling:
            run_modeling(df, args.taxi_type, args.models,
                        apply_feature_selection=args.feature_selection,
                        max_features=args.max_features)
        else:
            log_message("Skipping modeling as requested", log_file)

        log_message("Analysis completed successfully", log_file)

    except Exception as e:
        log_message(f"Error during analysis: {str(e)}", log_file)
        raise

if __name__ == "__main__":
    main()
