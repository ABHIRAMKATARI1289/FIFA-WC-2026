import pandas as pd
import numpy as np
import pickle
import json
import os

MODELS    = "models"
PROCESSED = "data/processed"
OUTPUT    = "data/processed"

# ── Load model ────────────────────────────────────────────────────────────────
with open(f"{MODELS}/best_model.pkl", "rb") as f:
    model = pickle.load(f)
with open(f"{MODELS}/feature_cols.pkl", "rb") as f:
    FEATURE_COLS = pickle.load(f)

# ── WC 2026 Groups ────────────────────────────────────────────────────────────
GROUPS = {
    "A": ["USA", "Panama", "Honduras", "Morocco"],
    "B": ["Spain", "Croatia", "Morocco", "Uruguay"],
    "C": ["Argentina", "Chile", "Peru", "Australia"],
    "D": ["France", "Mexico", "Poland", "Saudi Arabia"],
    "E": ["Brazil", "Colombia", "Ecuador", "Japan"],
    "F": ["England", "Netherlands", "Senegal", "Iran"],
    "G": ["Portugal", "Turkey", "Czech Republic", "Cameroon"],
    "H": ["Germany", "Belgium", "Algeria", "New Zealand"],
    "I": ["South Korea", "Ghana", "Costa Rica", "Greece"],
    "J": ["Switzerland", "Serbia", "Ivory Coast", "Canada"],
    "K": ["Denmark", "Nigeria", "Venezuela", "Qatar"],
    "L": ["Tunisia", "Sweden", "Norway", "Jamaica"],
}

# ── Team stats (form proxies for 2026) ───────────────────────────────────────
# Based on recent international results — update as tournament approaches
TEAM_STATS = {
    "Argentina":     {"win_rate": 0.72, "goals_scored": 2.3, "goals_conceded": 0.7},
    "France":        {"win_rate": 0.68, "goals_scored": 2.1, "goals_conceded": 0.9},
    "Brazil":        {"win_rate": 0.65, "goals_scored": 2.0, "goals_conceded": 0.8},
    "England":       {"win_rate": 0.63, "goals_scored": 1.9, "goals_conceded": 0.8},
    "Spain":         {"win_rate": 0.67, "goals_scored": 2.2, "goals_conceded": 0.7},
    "Germany":       {"win_rate": 0.60, "goals_scored": 1.9, "goals_conceded": 1.0},
    "Portugal":      {"win_rate": 0.65, "goals_scored": 2.1, "goals_conceded": 0.9},
    "Netherlands":   {"win_rate": 0.60, "goals_scored": 1.8, "goals_conceded": 0.9},
    "Belgium":       {"win_rate": 0.58, "goals_scored": 1.8, "goals_conceded": 1.0},
    "Croatia":       {"win_rate": 0.55, "goals_scored": 1.5, "goals_conceded": 0.9},
    "Uruguay":       {"win_rate": 0.55, "goals_scored": 1.6, "goals_conceded": 1.0},
    "Colombia":      {"win_rate": 0.55, "goals_scored": 1.7, "goals_conceded": 1.0},
    "Mexico":        {"win_rate": 0.52, "goals_scored": 1.6, "goals_conceded": 1.1},
    "USA":           {"win_rate": 0.50, "goals_scored": 1.5, "goals_conceded": 1.1},
    "Japan":         {"win_rate": 0.50, "goals_scored": 1.4, "goals_conceded": 1.0},
    "South Korea":   {"win_rate": 0.48, "goals_scored": 1.3, "goals_conceded": 1.1},
    "Morocco":       {"win_rate": 0.52, "goals_scored": 1.4, "goals_conceded": 0.9},
    "Senegal":       {"win_rate": 0.50, "goals_scored": 1.4, "goals_conceded": 1.0},
    "Switzerland":   {"win_rate": 0.52, "goals_scored": 1.5, "goals_conceded": 1.0},
    "Denmark":       {"win_rate": 0.52, "goals_scored": 1.5, "goals_conceded": 0.9},
    "Poland":        {"win_rate": 0.48, "goals_scored": 1.4, "goals_conceded": 1.1},
    "Serbia":        {"win_rate": 0.48, "goals_scored": 1.5, "goals_conceded": 1.2},
    "Turkey":        {"win_rate": 0.48, "goals_scored": 1.4, "goals_conceded": 1.1},
    "Australia":     {"win_rate": 0.45, "goals_scored": 1.2, "goals_conceded": 1.2},
    "Ecuador":       {"win_rate": 0.45, "goals_scored": 1.3, "goals_conceded": 1.2},
    "Chile":         {"win_rate": 0.45, "goals_scored": 1.3, "goals_conceded": 1.2},
    "Peru":          {"win_rate": 0.42, "goals_scored": 1.2, "goals_conceded": 1.3},
    "Canada":        {"win_rate": 0.45, "goals_scored": 1.3, "goals_conceded": 1.2},
    "Iran":          {"win_rate": 0.45, "goals_scored": 1.2, "goals_conceded": 1.1},
    "Saudi Arabia":  {"win_rate": 0.40, "goals_scored": 1.1, "goals_conceded": 1.3},
    "Cameroon":      {"win_rate": 0.42, "goals_scored": 1.2, "goals_conceded": 1.3},
    "Ghana":         {"win_rate": 0.40, "goals_scored": 1.1, "goals_conceded": 1.3},
    "Tunisia":       {"win_rate": 0.42, "goals_scored": 1.1, "goals_conceded": 1.1},
    "Algeria":       {"win_rate": 0.45, "goals_scored": 1.3, "goals_conceded": 1.1},
    "Nigeria":       {"win_rate": 0.45, "goals_scored": 1.4, "goals_conceded": 1.2},
    "Ivory Coast":   {"win_rate": 0.45, "goals_scored": 1.3, "goals_conceded": 1.2},
    "Czech Republic":{"win_rate": 0.45, "goals_scored": 1.3, "goals_conceded": 1.1},
    "Belgium":       {"win_rate": 0.58, "goals_scored": 1.8, "goals_conceded": 1.0},
    "Sweden":        {"win_rate": 0.48, "goals_scored": 1.4, "goals_conceded": 1.1},
    "Norway":        {"win_rate": 0.50, "goals_scored": 1.6, "goals_conceded": 1.1},
    "Greece":        {"win_rate": 0.42, "goals_scored": 1.2, "goals_conceded": 1.2},
    "Costa Rica":    {"win_rate": 0.40, "goals_scored": 1.1, "goals_conceded": 1.3},
    "Panama":        {"win_rate": 0.38, "goals_scored": 1.0, "goals_conceded": 1.4},
    "Honduras":      {"win_rate": 0.35, "goals_scored": 0.9, "goals_conceded": 1.5},
    "Venezuela":     {"win_rate": 0.38, "goals_scored": 1.0, "goals_conceded": 1.3},
    "Qatar":         {"win_rate": 0.35, "goals_scored": 0.9, "goals_conceded": 1.4},
    "Jamaica":       {"win_rate": 0.35, "goals_scored": 0.9, "goals_conceded": 1.4},
    "New Zealand":   {"win_rate": 0.32, "goals_scored": 0.8, "goals_conceded": 1.5},
}

DEFAULT_STATS = {"win_rate": 0.40, "goals_scored": 1.1, "goals_conceded": 1.3}

def get_stats(team):
    return TEAM_STATS.get(team, DEFAULT_STATS)

def predict_match(home, away, is_knockout=0):
    """Returns (home_win_prob, draw_prob, away_win_prob)"""
    hs = get_stats(home)
    as_ = get_stats(away)
    features = pd.DataFrame([{
        "home_win_rate":           hs["win_rate"],
        "home_goals_scored_avg":   hs["goals_scored"],
        "home_goals_conceded_avg": hs["goals_conceded"],
        "away_win_rate":           as_["win_rate"],
        "away_goals_scored_avg":   as_["goals_scored"],
        "away_goals_conceded_avg": as_["goals_conceded"],
        "h2h_home_win_rate":       0.5,
        "h2h_matches":             3,
        "goal_diff_avg":           (hs["goals_scored"] - hs["goals_conceded"]) -
                                   (as_["goals_scored"] - as_["goals_conceded"]),
        "win_rate_diff":           hs["win_rate"] - as_["win_rate"],
        "is_knockout":             is_knockout,
    }])
    probs = model.predict_proba(features)[0]
    # map to home_win, draw, away_win
    class_map = {c: i for i, c in enumerate(model.classes_)}
    return (
        probs[class_map[2]],  # home win
        probs[class_map[1]],  # draw
        probs[class_map[0]],  # away win
    )

def simulate_match(home, away, is_knockout=False):
    """Simulate one match. In knockouts, no draws — use extra time coin flip."""
    hw, d, aw = predict_match(home, away, int(is_knockout))
    if is_knockout:
        # Redistribute draw prob 50/50
        hw_adj = hw + d * 0.5
        aw_adj = aw + d * 0.5
        total  = hw_adj + aw_adj
        r = np.random.random()
        return home if r < hw_adj / total else away
    else:
        r = np.random.random()
        if r < hw:
            return "home_win"
        elif r < hw + d:
            return "draw"
        else:
            return "away_win"

def simulate_group(teams):
    """Simulate a group stage — returns top 2 teams."""
    points = {t: 0 for t in teams}
    gd     = {t: 0 for t in teams}
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            result = simulate_match(teams[i], teams[j], is_knockout=False)
            if result == "home_win":
                points[teams[i]] += 3
                gd[teams[i]]     += 1
                gd[teams[j]]     -= 1
            elif result == "away_win":
                points[teams[j]] += 3
                gd[teams[j]]     += 1
                gd[teams[i]]     -= 1
            else:
                points[teams[i]] += 1
                points[teams[j]] += 1
    ranked = sorted(teams, key=lambda t: (points[t], gd[t]), reverse=True)
    return ranked[0], ranked[1]

def simulate_tournament():
    """Simulate full WC 2026 tournament, return winner."""
    # Group stage
    group_winners  = {}
    group_runners  = {}
    for grp, teams in GROUPS.items():
        w, r = simulate_group(teams)
        group_winners[grp] = w
        group_runners[grp] = r

    # Round of 32 (48 teams, top 2 from 12 groups + 8 best 3rd place)
    # Simplified: use top 2 from each group → 24 teams → R16 pairs
    qualified = []
    for grp in sorted(GROUPS.keys()):
        qualified.append(group_winners[grp])
        qualified.append(group_runners[grp])

    # Knockout rounds
    round_teams = qualified
    while len(round_teams) > 1:
        next_round = []
        for i in range(0, len(round_teams), 2):
            if i + 1 < len(round_teams):
                winner = simulate_match(round_teams[i], round_teams[i+1], is_knockout=True)
                next_round.append(winner)
            else:
                next_round.append(round_teams[i])
        round_teams = next_round

    return round_teams[0]

def run_monte_carlo(n=10000):
    """Run n simulations, return win probabilities."""
    print(f"Running {n} Monte Carlo simulations...")
    win_counts = {}
    for i in range(n):
        if i % 1000 == 0:
            print(f"  Simulation {i}/{n}...")
        winner = simulate_tournament()
        win_counts[winner] = win_counts.get(winner, 0) + 1

    probs = {team: count / n for team, count in win_counts.items()}
    probs = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))
    return probs

if __name__ == "__main__":
    np.random.seed(42)
    probs = run_monte_carlo(n=10000)

    print("\n── WC 2026 Win Probabilities ─────────────────────────────────")
    for i, (team, prob) in enumerate(probs.items(), 1):
        bar = "█" * int(prob * 100)
        print(f"  {i:>2}. {team:<20} {prob:.1%}  {bar}")

    # Save results
    with open(f"{OUTPUT}/wc2026_predictions.json", "w") as f:
        json.dump(probs, f, indent=2)
    print(f"\nSaved to data/processed/wc2026_predictions.json ✅")