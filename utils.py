"""
utils.py  —  California House Price Prediction Helpers & Presets
Matches the End-to-End ML Pipeline from Project 15.4 (California Housing Dataset)
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

TARGET_COL = "median_house_value"

NUMERICAL_FEATURES = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income"
]

CATEGORICAL_FEATURES = ["ocean_proximity"]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

OCEAN_PROXIMITY_OPTIONS = [
    "<1H OCEAN",
    "INLAND",
    "NEAR OCEAN",
    "NEAR BAY",
    "ISLAND"
]

OCEAN_PROXIMITY_EMOJIS = {
    "<1H OCEAN": "🚗 <1H Ocean",
    "INLAND": "🏞️ Inland",
    "NEAR OCEAN": "🌊 Near Ocean",
    "NEAR BAY": "🌉 Near Bay",
    "ISLAND": "🏝️ Island"
}

# California Region Presets for 1-click exploration in the UI
CALIFORNIA_PRESETS = {
    "Custom Location (Manual Input)": None,
    "San Francisco - Bay Area (Near Bay)": {
        "longitude": -122.25,
        "latitude": 37.85,
        "housing_median_age": 42.0,
        "total_rooms": 2500,
        "total_bedrooms": 450,
        "population": 1100,
        "households": 420,
        "median_income": 8.35, # ~$83,500
        "ocean_proximity": "NEAR BAY"
    },
    "Silicon Valley - Palo Alto (<1H Ocean)": {
        "longitude": -122.14,
        "latitude": 37.44,
        "housing_median_age": 35.0,
        "total_rooms": 3800,
        "total_bedrooms": 600,
        "population": 1400,
        "households": 550,
        "median_income": 9.50, # ~$95,000
        "ocean_proximity": "<1H OCEAN"
    },
    "West Los Angeles - Santa Monica (Near Ocean)": {
        "longitude": -118.49,
        "latitude": 34.02,
        "housing_median_age": 38.0,
        "total_rooms": 3100,
        "total_bedrooms": 520,
        "population": 1250,
        "households": 490,
        "median_income": 7.80, # ~$78,000
        "ocean_proximity": "NEAR OCEAN"
    },
    "San Diego - Coastal (<1H Ocean)": {
        "longitude": -117.23,
        "latitude": 32.84,
        "housing_median_age": 28.0,
        "total_rooms": 2900,
        "total_bedrooms": 480,
        "population": 1300,
        "households": 460,
        "median_income": 6.20, # ~$62,000
        "ocean_proximity": "<1H OCEAN"
    },
    "Sacramento Suburbs (Inland)": {
        "longitude": -121.49,
        "latitude": 38.58,
        "housing_median_age": 22.0,
        "total_rooms": 2800,
        "total_bedrooms": 510,
        "population": 1450,
        "households": 500,
        "median_income": 3.85, # ~$38,500
        "ocean_proximity": "INLAND"
    },
    "Central Valley - Fresno (Inland)": {
        "longitude": -119.77,
        "latitude": 36.75,
        "housing_median_age": 20.0,
        "total_rooms": 2200,
        "total_bedrooms": 430,
        "population": 1500,
        "households": 440,
        "median_income": 2.50, # ~$25,000
        "ocean_proximity": "INLAND"
    },
    "Catalina Island (Island)": {
        "longitude": -118.33,
        "latitude": 33.34,
        "housing_median_age": 50.0,
        "total_rooms": 2100,
        "total_bedrooms": 410,
        "population": 750,
        "households": 320,
        "median_income": 4.10, # ~$41,000
        "ocean_proximity": "ISLAND"
    }
}


def create_preprocessor():
    """
    Creates the exact ColumnTransformer preprocessing pipeline specified in the tutorial:
    - Numerical: Median SimpleImputer + StandardScaler
    - Categorical: Most Frequent SimpleImputer + OneHotEncoder(handle_unknown='ignore')
    """
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )
    return preprocess


def build_input_df(
    longitude: float,
    latitude: float,
    housing_median_age: float,
    total_rooms: float,
    total_bedrooms: float,
    population: float,
    households: float,
    median_income: float,
    ocean_proximity: str
) -> pd.DataFrame:
    """Creates a single-row DataFrame formatted for the model pipeline."""
    return pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }])


def format_usd(val: float) -> str:
    """Format price in standard US Dollar currency string."""
    if val >= 1_000_000:
        return f"${val / 1_000_000:.2f}M"
    return f"${val:,.0f}"


def calculate_price_range(predicted_price: float, rmse: float = 46000.0):
    """Calculates approximate 68% prediction interval using model RMSE."""
    lower = max(10000.0, predicted_price - rmse)
    upper = predicted_price + rmse
    return format_usd(lower), format_usd(upper)
