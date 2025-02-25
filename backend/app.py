"""
app.py
---------
Main Flask application that exposes the /api/predict endpoint.
It reads a CSV file from the request, computes features via utils.py,
uses the model from model.py to predict DFS projections, and returns JSON.
"""

import os, math, pandas as pd, numpy as np, joblib
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from utils import create_feature_dataframe_from_csv
from model import predict_model  # Import our ML prediction function

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
MODEL_PATH = "model.pkl"  # Must match the MODEL_PATH in model.py
limiter = Limiter(app, key_func=get_remote_address, default_limits=["100 per hour"])

@app.route('/api/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict_api():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Error reading CSV: {e}"}), 400

    required_columns = ["Position", "Name", "Game Info", "TeamAbbrev", "AvgPointsPerGame", "Salary"]
    for col in required_columns:
        if col not in df.columns:
            return jsonify({"error": f"Missing required column: {col}"}), 400

    feature_df = create_feature_dataframe_from_csv(df)
    if feature_df is None or feature_df.empty:
        return jsonify({"error": "No features computed from input data."}), 400

    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model not trained. Please train the model first."}), 400

    # Define the feature columns used in training
    feature_cols = [
        "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "MIN", "FGA", "FTA",
        "PER", "TS", "AST_TOV", "DWS", "Usage", "OpponentDefense", "AllowedPoints",
        "TeamPace", "InactiveAdjustment", "HomeAwayFactor", "RestFactor",
        "LineupFactor", "OpponentMLAdj", "FPPM",
        "HomeAwayFantasyRatio", "TeamLineupEfficiency",
        "E_NET_RATING", "E_USG_PCT", "PtsStd", "RollingFantasyAvg", "eFG", "TSA"
    ]
    X = feature_df[feature_cols]
    predictions = predict_model(X)
    uncertainty = 0.05  # Simulated ±5% uncertainty
    df = df.iloc[:len(predictions)].copy()
    df['MLProjectedDKPoints'] = np.round(predictions, 2)
    df['LowerBound'] = df['MLProjectedDKPoints'] * (1 - uncertainty)
    df['UpperBound'] = df['MLProjectedDKPoints'] * (1 + uncertainty)

    # Compute dynamic target points (5x salary ratio)
    df['Target Points'] = df['Salary'].apply(lambda s: (s / 1000) * 5)
    df['Meets Target'] = df.apply(lambda row: row['MLProjectedDKPoints'] >= row['Target Points'], axis=1)
    df['Value Score'] = df.apply(lambda row: (row['MLProjectedDKPoints'] * 1000 / row['Salary']) if row['Salary'] > 0 else 0.0, axis=1)

    projections = df[["Name", "MLProjectedDKPoints", "Salary", "Target Points", "Meets Target", "Value Score"]].to_dict(orient="records")
    top_players = df.sort_values(by='MLProjectedDKPoints', ascending=False).head(5)[["Name", "MLProjectedDKPoints", "Salary"]].to_dict(orient="records")
    top_value = df.sort_values(by='Value Score', ascending=False).head(5)[["Name", "Value Score", "MLProjectedDKPoints", "Salary"]].to_dict(orient="records")

    return jsonify({
        "projections": projections,
        "top_5_players": top_players,
        "top_5_value_plays": top_value
    })

if __name__ == '__main__':
    app.run(debug=True)
