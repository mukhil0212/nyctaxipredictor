# NYC Taxi Trip Data Analysis

## Project Overview

This project performs exploratory data analysis (EDA) and data cleaning on NYC taxi trip records to understand trip patterns, fare distribution, and factors influencing ride duration. It also builds machine learning models to predict trip duration based on available features.

## Dataset

The dataset consists of NYC taxi trip records from different taxi types:

- Yellow Taxi: Traditional yellow medallion taxis
- Green Taxi: Borough taxis that operate in the outer boroughs
- FHV: For-Hire Vehicles (e.g., community livery, black car services)
- FHVHV: High-Volume For-Hire Vehicles (e.g., Uber, Lyft)

The data is stored in Parquet format in the `nyc_datasets` directory.

## Project Structure

```
nyc_taxi_prediction/
├── nyc_datasets/           # Raw data files
├── models/                 # Saved trained models
├── results/                # Analysis results and visualizations
├── src/                    # Source code
│   ├── config.py           # Configuration settings
│   ├── main.py             # Main script to run the analysis
│   ├── data/               # Data processing modules
│   │   ├── loader.py       # Data loading utilities
│   │   ├── cleaner.py      # Data cleaning utilities
│   │   └── feature_engineering.py # Feature engineering utilities
│   ├── visualization/      # Visualization modules
│   │   └── visualizer.py   # Visualization utilities
│   ├── models/             # Model training and evaluation modules
│   │   ├── trainer.py      # Model training utilities
│   │   └── evaluator.py    # Model evaluation utilities
│   └── utils/              # Utility modules
│       └── helpers.py      # Helper functions
└── README.md               # Project documentation
```

## Features

- **Data Loading**: Load and combine taxi trip data from multiple files
- **Data Cleaning**: Handle missing values, remove outliers, and fix inconsistencies
- **Feature Engineering**: Create new features from existing data to improve model performance
- **Exploratory Data Analysis**: Visualize trip patterns, fare distribution, and other insights
- **Model Training**: Train multiple regression models to predict trip duration
- **Model Evaluation**: Evaluate model performance and visualize results

## Models

The project implements the following regression models:

- Linear Regression
- Ridge Regression
- Random Forest Regression
- Gradient Boosting Regression

## Usage

To run the analysis, use the following command:

```bash
python -m src.main --taxi-type yellow --months 2025-01 2025-02 --sample-size 100000
```

### Command Line Arguments

- `--taxi-type`: Type of taxi data to analyze (`yellow`, `green`, `fhv`, or `fhvhv`)
- `--months`: Months to analyze (e.g., `2025-01 2025-02`)
- `--sample-size`: Number of rows to sample for analysis (use `-1` for all data)
- `--skip-eda`: Skip exploratory data analysis
- `--skip-modeling`: Skip model training and evaluation
- `--models`: Models to train (`linear`, `ridge`, `random_forest`, `gradient_boosting`)

## Results

The analysis results are saved in the `results` directory:

- `visualizations/`: EDA visualizations
- `model_evaluation/`: Model evaluation metrics and visualizations

Trained models are saved in the `models` directory.

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- pyarrow (for reading Parquet files)

## Learning Opportunities

This project provides hands-on experience with:

- Working with large real-world datasets
- Handling missing and inconsistent values
- Detecting and removing anomalies
- Feature engineering techniques
- Implementing and comparing multiple regression models
- Model evaluation and interpretation
