import pandas as pd
import numpy as np
import os

RAW = "data/raw"
PROCESSED = "data/processed"
os.makedirs(PROCESSED, exist_ok=True)

# ── Name standardization map ──────────────────────────────────────────────────
NAME_MAP = {
    "West Germany": "Germany",
    "Soviet Union": "Russia",
    "Czechoslovakia": "Czech Republic",
    "Yugoslavia": "Serbia",
    "Republic of Ireland": "Ireland",
    "China PR": "China",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "Trinidad and Tobago": "Trinidad & Tobago",
    "United States": "USA",
    "Türkiye": "Turkey",
    "IR Iran": "Iran",
}

def standardize(name):
    if isinstance(name, str):
        return NAME_MAP.get(name.strip(), name.strip())
    return name


# ── 1. Load WorldCupMatches ───────────────────────────────────────────────────
print("Loading WorldCupMatches.csv ...")
wc = pd.read_csv(f"{RAW}/WorldCupMatches.csv")
print(f"  Raw shape: {wc.shape}")
print(f"  Columns: {list(wc.columns)}\n")

# Keep only useful columns
wc = wc[[
    "Year", "Datetime", "Stage", "Stadium", "City",
    "Home Team Name", "Away Team Name",
    "Home Team Goals", "Away Team Goals",
    "Win conditions", "Attendance",
    "Home Team Initials", "Away Team Initials"
]].copy()

# Parse year and date
wc = wc.dropna(subset=["Year"])
wc["Year"] = wc["Year"].astype(int)
wc["date"] = pd.to_datetime(wc["Datetime"], errors="coerce")

# Standardize team names
wc["home_team"] = wc["Home Team Name"].apply(standardize)
wc["away_team"] = wc["Away Team Name"].apply(standardize)
wc["home_goals"] = pd.to_numeric(wc["Home Team Goals"], errors="coerce")
wc["away_goals"] = pd.to_numeric(wc["Away Team Goals"], errors="coerce")

# Drop rows with missing goals
wc = wc.dropna(subset=["home_goals", "away_goals"])

# Outcome from home team perspective
def get_outcome(row):
    if row["home_goals"] > row["away_goals"]:
        return "home_win"
    elif row["home_goals"] < row["away_goals"]:
        return "away_win"
    else:
        return "draw"

wc["outcome"] = wc.apply(get_outcome, axis=1)
wc["is_knockout"] = wc["Stage"].apply(
    lambda s: 0 if isinstance(s, str) and "group" in s.lower() else 1
)

wc_clean = wc[[
    "Year", "date", "Stage", "is_knockout",
    "home_team", "away_team",
    "home_goals", "away_goals", "outcome",
    "Attendance"
]].rename(columns={"Year": "year", "Stage": "stage", "Attendance": "attendance"})

print(f"  Clean WC matches: {len(wc_clean)} rows")
print(f"  Years covered: {wc_clean['year'].min()} – {wc_clean['year'].max()}\n")


# ── 2. Load international results (team form) ─────────────────────────────────
print("Loading results.csv ...")
res = pd.read_csv(f"{RAW}/results.csv")
print(f"  Raw shape: {res.shape}")
print(f"  Columns: {list(res.columns)}\n")

res["date"] = pd.to_datetime(res["date"], errors="coerce")
res["home_team"] = res["home_team"].apply(standardize)
res["away_team"] = res["away_team"].apply(standardize)
res = res.dropna(subset=["home_score", "away_score"])
res["home_score"] = res["home_score"].astype(int)
res["away_score"] = res["away_score"].astype(int)

def get_outcome_res(row):
    if row["home_score"] > row["away_score"]:
        return "home_win"
    elif row["home_score"] < row["away_score"]:
        return "away_win"
    else:
        return "draw"

res["outcome"] = res.apply(get_outcome_res, axis=1)

print(f"  Clean results: {len(res)} rows")
print(f"  Date range: {res['date'].min().date()} – {res['date'].max().date()}\n")


# ── 3. Load WorldCups summary ─────────────────────────────────────────────────
print("Loading WorldCups.csv ...")
wcs = pd.read_csv(f"{RAW}/WorldCups.csv")
print(f"  Columns: {list(wcs.columns)}")
print(wcs[["Year", "Winner", "Runners-Up", "Third", "Fourth", "GoalsScored", "QualifiedTeams"]].to_string(index=False))
print()


# ── 4. Load shootouts ─────────────────────────────────────────────────────────
print("Loading shootouts.csv ...")
shootouts = pd.read_csv(f"{RAW}/shootouts.csv")
shootouts["date"] = pd.to_datetime(shootouts["date"], errors="coerce")
shootouts["home_team"] = shootouts["home_team"].apply(standardize)
shootouts["away_team"] = shootouts["away_team"].apply(standardize)
print(f"  Shootout records: {len(shootouts)}\n")


# ── 5. Save processed files ───────────────────────────────────────────────────
print("Saving processed files ...")
wc_clean.to_csv(f"{PROCESSED}/wc_matches.csv", index=False)
res.to_csv(f"{PROCESSED}/international_results.csv", index=False)
shootouts.to_csv(f"{PROCESSED}/shootouts.csv", index=False)
wcs.to_csv(f"{PROCESSED}/world_cups_summary.csv", index=False)

print("  ✓ wc_matches.csv")
print("  ✓ international_results.csv")
print("  ✓ shootouts.csv")
print("  ✓ world_cups_summary.csv")
print("\nETL complete.")


# ── 6. Quick sanity check ─────────────────────────────────────────────────────
print("\n── Sanity check ──────────────────────────────────────────────────────")
print(f"WC matches per year:\n{wc_clean.groupby('year').size().to_string()}")
print(f"\nOutcome distribution:\n{wc_clean['outcome'].value_counts().to_string()}")
print(f"\nTop 10 teams by WC appearances (home+away):")
all_teams = pd.concat([wc_clean["home_team"], wc_clean["away_team"]])
print(all_teams.value_counts().head(10).to_string())
