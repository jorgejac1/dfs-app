"""
test_app.py
--------------
Unit tests for the Flask API endpoint in app.py using Flask's test client.
"""

import os, json, pandas as pd, pytest
from app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_predict_api_no_file(client):
    response = client.post('/api/predict')
    data = json.loads(response.data)
    assert "error" in data

def test_predict_api_missing_column(client, tmp_path):
    df = pd.DataFrame({
        "Position": ["PF"],
        "Name": ["Test Player"],
        "Game Info": ["MIA@MIL 02/23/2025 06:00PM ET"],
        "TeamAbbrev": ["MIA"],
        "AvgPointsPerGame": [50]
    })
    csv_path = tmp_path / "missing.csv"
    df.to_csv(csv_path, index=False)
    with open(csv_path, "rb") as f:
        response = client.post('/api/predict', data={"file": f})
    data = json.loads(response.data)
    assert "error" in data
    assert "Missing required column" in data["error"]

def test_predict_api_success(client, monkeypatch, tmp_path):
    df = pd.DataFrame({
        "Position": ["PF", "SG"],
        "Name": ["Player 1", "Player 2"],
        "Game Info": ["MIA@MIL 02/23/2025 06:00PM ET", "LAL@BOS 02/23/2025 06:00PM ET"],
        "TeamAbbrev": ["MIA", "LAL"],
        "AvgPointsPerGame": [50, 45],
        "Salary": [10000, 9500]
    })
    csv_path = tmp_path / "success.csv"
    df.to_csv(csv_path, index=False)
    
    from utils import create_feature_dataframe_from_csv
    dummy_features = pd.DataFrame([
        {"PTS": 20, "REB": 10, "AST": 5, "STL": 2, "BLK": 1, "TOV": 3,
         "FGM": 8, "FG3M": 2, "MIN": 30, "FGA": 15, "FTA": 5,
         "PER": 1.0, "TS": 0.5, "AST_TOV": 1.67, "DWS": 0.2,
         "Usage": 25, "OpponentDefense": 110, "AllowedPoints": 30,
         "TeamPace": 100, "InactiveAdjustment": 1.0, "HomeAwayFactor": 1.0,
         "RestFactor": 1.0, "LineupFactor": 1.0, "OpponentMLAdj": 1.0,
         "FPPM": 0.8, "HomeAwayFantasyRatio": 1.2, "TeamLineupEfficiency": 5.0,
         "E_NET_RATING": 105, "E_USG_PCT": 22, "PtsStd": 3.0, "RollingFantasyAvg": 45,
         "eFG": 0.55, "TSA": 17},
        {"PTS": 18, "REB": 9, "AST": 6, "STL": 3, "BLK": 1, "TOV": 4,
         "FGM": 7, "FG3M": 1, "MIN": 32, "FGA": 14, "FTA": 4,
         "PER": 0.9, "TS": 0.48, "AST_TOV": 1.5, "DWS": 0.25,
         "Usage": 23, "OpponentDefense": 108, "AllowedPoints": 28,
         "TeamPace": 102, "InactiveAdjustment": 1.0, "HomeAwayFactor": 1.05,
         "RestFactor": 1.0, "LineupFactor": 1.0, "OpponentMLAdj": 1.0,
         "FPPM": 0.75, "HomeAwayFantasyRatio": 1.1, "TeamLineupEfficiency": 4.5,
         "E_NET_RATING": 103, "E_USG_PCT": 21, "PtsStd": 2.5, "RollingFantasyAvg": 42,
         "eFG": 0.52, "TSA": 15}
    ])
    monkeypatch.setattr("utils.create_feature_dataframe_from_csv", lambda df: dummy_features)
    monkeypatch.setattr("joblib.load", lambda path: type("DummyModel", (), {"predict": lambda self, X: [55.5, 48.3]})())
    
    with open(csv_path, "rb") as f:
        response = client.post('/api/predict', data={"file": f})
    data = json.loads(response.data)
    assert "projections" in data
    assert isinstance(data["projections"], list)
