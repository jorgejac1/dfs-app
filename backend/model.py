"""
model.py
---------
This module contains functions for training and predicting our DFS model.
We build a stacking ensemble using XGBRegressor, RandomForestRegressor, and LinearRegression,
and perform hyperparameter tuning via GridSearchCV.
The trained model is saved to disk.
"""

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error

MODEL_PATH = "model.pkl"

def train_model():
    # Load historical training data
    data = pd.read_csv('historical_data.csv')
    
    # Define features and target
    # These features should match the ones used in your feature extraction pipeline.
    features = [
        'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG3M', 'Usage', 'OpponentDefense', 'TeamPace',
        'InactiveAdjustment', 'HomeAwayFactor', 'RestFactor', 'LineupFactor', 'OpponentMLAdj',
        'FPPM', 'HomeAwayFantasyRatio', 'TeamLineupEfficiency', 'E_NET_RATING', 'E_USG_PCT',
        'PtsStd', 'RollingFantasyAvg', 'eFG', 'TSA'
    ]
    target = 'DKPoints'
    
    X = data[features]
    y = data[target]
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define base estimators for stacking
    estimators = [
        ('xgb', XGBRegressor(random_state=42)),
        ('rf', RandomForestRegressor(random_state=42)),
        ('lr', LinearRegression())
    ]
    
    # Define the stacking regressor with a final estimator
    stack_reg = StackingRegressor(estimators=estimators, final_estimator=LinearRegression(), cv=3)
    
    # Hyperparameter grid for GridSearchCV (tuning some parameters of the base models)
    param_grid = {
        'xgb__max_depth': [3, 5],
        'xgb__learning_rate': [0.01, 0.1],
        'xgb__n_estimators': [100, 200],
        'rf__n_estimators': [100, 200],
    }
    
    grid = GridSearchCV(stack_reg, param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    
    # Evaluate model performance on the test set
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Best Parameters: {grid.best_params_}")
    print(f"Model MSE: {mse}")
    
    # Save the best model to disk
    joblib.dump(best_model, MODEL_PATH)
    
    return best_model

def predict_model(features):
    """
    Load the trained model and predict DFS points.
    
    Parameters:
      features (pd.DataFrame): Feature vectors for the players.
      
    Returns:
      predictions (np.array): Predicted DraftKings points.
    """
    model = joblib.load(MODEL_PATH)
    preds = model.predict(features)
    return preds
