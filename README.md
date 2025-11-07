# CS5228 HDB Resale Price Prediction

A machine learning project for predicting Singapore HDB resale prices using ensemble methods and feature engineering.

## Project Structure

- **Scripts/** - Jupyter notebooks for EDA, feature engineering, and model training
- **Utils/** - Helper functions for data processing and distance calculations
- **Dataset/** - Training/test data and auxiliary location datasets
- **Models/** - Saved model parameters and metrics

## Models Implemented

- Linear Regression
- Random Forest
- XGBoost
- LightGBM (with hyperparameter tuning)

## Key Features

- Geographic proximity features (malls, MRT stations, schools)
- Temporal features (year, month)
- HDB-specific attributes (flat type, floor area, lease commence date)
- Feature standardization and encoding
