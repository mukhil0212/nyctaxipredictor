"""
Configuration settings for the NYC Taxi Trip Data Analysis project.
"""

import os
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = PROJECT_ROOT / "nyc_datasets"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Dataset files
YELLOW_TAXI_JAN = DATA_DIR / "yellow_tripdata_2025-01.parquet"
YELLOW_TAXI_FEB = DATA_DIR / "yellow_tripdata_2025-02.parquet"
GREEN_TAXI_JAN = DATA_DIR / "green_tripdata_2025-01.parquet"
GREEN_TAXI_FEB = DATA_DIR / "green_tripdata_2025-02.parquet"
FHV_JAN = DATA_DIR / "fhv_tripdata_2025-01.parquet"
FHV_FEB = DATA_DIR / "fhv_tripdata_2025-02.parquet"
FHVHV_JAN = DATA_DIR / "fhvhv_tripdata_2025-01.parquet"
FHVHV_FEB = DATA_DIR / "fhvhv_tripdata_2025-02.parquet"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
