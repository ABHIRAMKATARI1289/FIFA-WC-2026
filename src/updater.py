import requests
import json
import os
import pandas as pd
from datetime import datetime
import pickle

API_KEY   = "e59f890b27904702a809c235a25eef9b"   # ← paste your key here
BASE_URL  = "https://api.football-data.org/v4"
PROCESSED = "data/processed"
MODELS    = "models"

HEADERS = {"X-Auth-Token": API_KEY}

# WC 2026 competition ID on football-data.org
WC_ID = 2000

def fetch_wc_matches():
    """Fetch all WC 2026 matches from API."""
    url = f"{BASE_URL}/competitions/{WC_ID}/matches"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        print(f"API error: {response.status_code} — {response.text}")
        return None
    return response.json()

def fetch_standings():
    """Fetch current group standings."""
    url = f"{BASE_URL}/competitions/{WC_ID}/standings"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        print(f"API error: {response.status_code}")
        return None
    return response.json()

def parse_matches(raw):
    """Parse API response into clean dataframe."""
    matches = []
    for m in raw.get("matches", []):
        home = m["homeTeam"]["name"]
        away = m["awayTeam"]["name"]
        status = m["status"]  # SCHEDULED, IN_PLAY, FINISHED
        stage  = m["stage"]
        date   = m["utcDate"]

        home_goals = None
        away_goals = None
        if status == "FINISHED":
            home_goals = m["score"]["fullTime"]["home"]
            away_goals = m["score"]["fullTime"]["away"]

        matches.append({
            "match_id":   m["id"],
            "date":       date,
            "stage":      stage,
            "status":     status,
            "home_team":  home,
            "away_team":  away,
            "home_goals": home_goals,
            "away_goals": away_goals,
        })
    return pd.DataFrame(matches)

def determine_outcome(row):
    if pd.isna(row["home_goals"]):
        return None
    if row["home_goals"] > row["away_goals"]:
        return "home_win"
    elif row["home_goals"] < row["away_goals"]:
        return "away_win"
    else:
        return "draw"

def update_live_results():
    """Main function — fetch latest results and save."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Fetching WC 2026 results...")

    raw = fetch_wc_matches()
    if raw is None:
        return

    df = parse_matches(raw)
    df["outcome"] = df.apply(determine_outcome, axis=1)

    finished = df[df["status"] == "FINISHED"]
    scheduled = df[df["status"] == "SCHEDULED"]

    print(f"  Finished matches:  {len(finished)}")
    print(f"  Scheduled matches: {len(scheduled)}")

    # Save to disk
    df.to_csv(f"{PROCESSED}/live_matches.csv", index=False)
    print(f"  Saved to data/processed/live_matches.csv ✅")

    # Show latest results
    if len(finished) > 0:
        print(f"\nLatest results:")
        for _, row in finished.tail(5).iterrows():
            print(f"  {row['home_team']} {int(row['home_goals'])} - {int(row['away_goals'])} {row['away_team']}")

    return df

def get_match_prediction(home, away, is_knockout=False):
    """Get win probability for a specific upcoming match."""
    # Import here to avoid circular imports
    import sys
    sys.path.append("src")
    from simulator import predict_match, get_stats

    hw, d, aw = predict_match(home, away, int(is_knockout))
    print(f"\n{home} vs {away}")
    print(f"  Home Win: {hw:.1%}")
    print(f"  Draw:     {d:.1%}")
    print(f"  Away Win: {aw:.1%}")
    return hw, d, aw

if __name__ == "__main__":
    # Step 1: fetch latest results
    df = update_live_results()

    if df is not None:
        # Step 2: show upcoming match predictions
        scheduled = df[df["status"] == "SCHEDULED"].head(5)
        if len(scheduled) > 0:
            print(f"\nPredictions for upcoming matches:")
            for _, row in scheduled.iterrows():
                stage = row["stage"]
                is_ko = 0 if "GROUP" in str(stage) else 1
                get_match_prediction(
                    row["home_team"],
                    row["away_team"],
                    is_knockout=bool(is_ko)
                )
        else:
            print("\nNo scheduled matches found yet.")