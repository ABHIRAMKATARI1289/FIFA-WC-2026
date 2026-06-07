import pandas as pd
import numpy as np
import os

PROCESSED = "data/processed"

def get_team_form(team, match_date, results_df, n=20):
    """Get a team's stats from their last n matches before match_date."""
    mask = (
        ((results_df["home_team"] == team) | (results_df["away_team"] == team)) &
        (results_df["date"] < match_date)
    )
    recent = results_df[mask].sort_values("date", ascending=False).head(n)

    if len(recent) == 0:
        return {
            "win_rate": 0.5, "goals_scored_avg": 1.0,
            "goals_conceded_avg": 1.0, "matches_played": 0
        }

    wins, goals_scored, goals_conceded = 0, 0, 0

    for _, row in recent.iterrows():
        if row["home_team"] == team:
            goals_scored += row["home_score"]
            goals_conceded += row["away_score"]
            if row["outcome"] == "home_win":
                wins += 1
        else:
            goals_scored += row["away_score"]
            goals_conceded += row["home_score"]
            if row["outcome"] == "away_win":
                wins += 1

    n_matches = len(recent)
    return {
        "win_rate": wins / n_matches,
        "goals_scored_avg": goals_scored / n_matches,
        "goals_conceded_avg": goals_conceded / n_matches,
        "matches_played": n_matches
    }


def get_h2h(home_team, away_team, match_date, results_df, n=10):
    """Head to head record between two teams before match_date."""
    mask = (
        ((results_df["home_team"] == home_team) & (results_df["away_team"] == away_team)) |
        ((results_df["home_team"] == away_team) & (results_df["away_team"] == home_team))
    ) & (results_df["date"] < match_date)

    h2h = results_df[mask].sort_values("date", ascending=False).head(n)

    if len(h2h) == 0:
        return {"h2h_home_win_rate": 0.5, "h2h_matches": 0}

    home_wins = 0
    for _, row in h2h.iterrows():
        if row["home_team"] == home_team and row["outcome"] == "home_win":
            home_wins += 1
        elif row["away_team"] == home_team and row["outcome"] == "away_win":
            home_wins += 1

    return {
        "h2h_home_win_rate": home_wins / len(h2h),
        "h2h_matches": len(h2h)
    }


def build_features(wc_matches, results):
    """Build feature matrix for all WC matches."""
    print(f"Building features for {len(wc_matches)} matches...")
    rows = []

    for i, match in wc_matches.iterrows():
        if i % 100 == 0:
            print(f"  Processing match {i}/{len(wc_matches)}...")

        home = match["home_team"]
        away = match["away_team"]
        date = match["date"]

        home_form = get_team_form(home, date, results)
        away_form = get_team_form(away, date, results)
        h2h     = get_h2h(home, away, date, results)

        row = {
            "year":           match["year"],
            "date":           date,
            "stage":          match["stage"],
            "is_knockout":    match["is_knockout"],
            "home_team":      home,
            "away_team":      away,
            "home_goals":     match["home_goals"],
            "away_goals":     match["away_goals"],
            "outcome":        match["outcome"],

            # Home team form
            "home_win_rate":           home_form["win_rate"],
            "home_goals_scored_avg":   home_form["goals_scored_avg"],
            "home_goals_conceded_avg": home_form["goals_conceded_avg"],
            "home_matches_played":     home_form["matches_played"],

            # Away team form
            "away_win_rate":           away_form["win_rate"],
            "away_goals_scored_avg":   away_form["goals_scored_avg"],
            "away_goals_conceded_avg": away_form["goals_conceded_avg"],
            "away_matches_played":     away_form["matches_played"],

            # Head to head
            "h2h_home_win_rate": h2h["h2h_home_win_rate"],
            "h2h_matches":       h2h["h2h_matches"],

            # Derived
            "goal_diff_avg": (
                home_form["goals_scored_avg"] - home_form["goals_conceded_avg"]
            ) - (
                away_form["goals_scored_avg"] - away_form["goals_conceded_avg"]
            ),
            "win_rate_diff": home_form["win_rate"] - away_form["win_rate"],
        }
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Loading processed data...")
    wc_matches = pd.read_csv(f"{PROCESSED}/wc_matches.csv", parse_dates=["date"])
    results    = pd.read_csv(f"{PROCESSED}/international_results.csv", parse_dates=["date"])

    # Filter to WC era only (1990 onwards — more reliable data)
    wc_modern = wc_matches[wc_matches["year"] >= 1990].copy()
    print(f"  WC matches (1990+): {len(wc_modern)}")
    print(f"  International results available: {len(results)}\n")

    features_df = build_features(wc_modern, results)

    out_path = f"{PROCESSED}/features.csv"
    features_df.to_csv(out_path, index=False)
    print(f"\nSaved features to {out_path}")
    print(f"Shape: {features_df.shape}")
    print(f"\nFeature columns:\n{list(features_df.columns)}")
    print(f"\nSample row:\n{features_df.iloc[0].to_dict()}")
    print(f"\nOutcome distribution:\n{features_df['outcome'].value_counts()}")