from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import json
import sys
import os

sys.path.append("src")
from simulator import predict_match, run_monte_carlo, TEAM_STATS
from updater import update_live_results

app = FastAPI(
    title="FIFA WC 2026 Prediction API",
    description="Live match predictions and tournament simulation for FIFA World Cup 2026",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PROCESSED = "data/processed"

# ── Load predictions on startup ───────────────────────────────────────────────
@app.on_event("startup")
async def load_predictions():
    global tournament_probs
    path = f"{PROCESSED}/wc2026_predictions.json"
    if os.path.exists(path):
        with open(path) as f:
            tournament_probs = json.load(f)
    else:
        tournament_probs = {}

tournament_probs = {}


# ── Models ────────────────────────────────────────────────────────────────────
class MatchPredictionResponse(BaseModel):
    home_team:     str
    away_team:     str
    home_win_prob: float
    draw_prob:     float
    away_win_prob: float
    favorite:      str
    is_knockout:   bool


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "FIFA WC 2026 Prediction API",
        "endpoints": [
            "/predict",
            "/simulate",
            "/live",
            "/teams",
            "/standings"
        ]
    }

@app.get("/predict", response_model=MatchPredictionResponse)
def predict(
    home: str = Query(..., description="Home team name e.g. Argentina"),
    away: str = Query(..., description="Away team name e.g. France"),
    knockout: bool = Query(False, description="Is this a knockout match?")
):
    """Get win/draw/loss probabilities for a match."""
    if home not in TEAM_STATS:
        raise HTTPException(status_code=404, detail=f"Team '{home}' not found")
    if away not in TEAM_STATS:
        raise HTTPException(status_code=404, detail=f"Team '{away}' not found")

    hw, d, aw = predict_match(home, away, int(knockout))

    if hw > aw:
        favorite = f"{home} (home)"
    elif aw > hw:
        favorite = f"{away} (away)"
    else:
        favorite = "Even"

    return MatchPredictionResponse(
        home_team=home,
        away_team=away,
        home_win_prob=round(hw, 4),
        draw_prob=round(d, 4),
        away_win_prob=round(aw, 4),
        favorite=favorite,
        is_knockout=knockout
    )

@app.get("/simulate")
def simulate(n: int = Query(1000, description="Number of Monte Carlo simulations")):
    """Run tournament simulation and return win probabilities."""
    if n > 10000:
        raise HTTPException(status_code=400, detail="Max 10000 simulations")
    np.random.seed(42)
    probs = run_monte_carlo(n=n)
    return {
        "simulations": n,
        "predictions": [
            {"team": team, "win_probability": round(prob, 4)}
            for team, prob in probs.items()
        ]
    }

@app.get("/live")
def live_results():
    """Fetch and return latest WC 2026 match results."""
    df = update_live_results()
    if df is None:
        raise HTTPException(status_code=503, detail="Could not fetch live data")

    finished  = df[df["status"] == "FINISHED"].to_dict(orient="records")
    scheduled = df[df["status"] == "SCHEDULED"].head(10).to_dict(orient="records")

    return {
        "finished_count":  len(finished),
        "scheduled_count": len(df[df["status"] == "SCHEDULED"]),
        "latest_results":  finished[-5:] if finished else [],
        "upcoming":        scheduled
    }

@app.get("/teams")
def list_teams():
    """List all teams with their stats."""
    return {
        "teams": [
            {"team": team, **stats}
            for team, stats in TEAM_STATS.items()
        ]
    }

@app.get("/standings")
def tournament_standings():
    """Return current tournament win probability standings."""
    if not tournament_probs:
        raise HTTPException(status_code=404, detail="Run simulator first")
    return {
        "standings": [
            {"rank": i+1, "team": team, "win_probability": round(prob, 4)}
            for i, (team, prob) in enumerate(tournament_probs.items())
        ]
    }