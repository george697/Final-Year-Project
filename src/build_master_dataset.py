"""
build_master_dataset.py
=======================
Merges 5 seasons of Football-Data.co.uk CSVs with ClubElo ratings
to produce a single clean master dataset.

HOW TO USE:
1. Make sure the CSVs are in data/raw named:
prem2021.csv, prem2122.csv, prem2223.csv, prem2324.csv,prem2425.csv
2. Make sure soccerdata is installed, if not run: pip install -r requirements.txt
3. Run from project root: python src/build_master_dataset.py
4. Output saved to processed/master_dataset.csv

INFO:
First time running this will take a while since all the ELO needs to be fetched but
once its fetched once it will be cached on all future runs so it will run fast.
If you want to avoid this, uncomment line number : ...
it will use the premade generated csv.
"""

import pandas as pd
import numpy as np
from soccerdata import ClubElo
from pathlib import Path

#--------------------------------------------------------------------------------------------
#0. CONFIGURATION
# -------------------------------------------------------------------------------------------

RAW_DATA_DIR       = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

SEASON_FILES = [
    ("prem2021.csv", "2020-21"),
    ("prem2122.csv", "2021-22"),
    ("prem2223.csv", "2022-23"),
    ("prem2324.csv", "2023-24"),
    ("prem2425.csv", "2024-25"),
]

#Football-data and ClubElo use different name formats for some clubs
# e.g. Football-data uses "Nott'm Forest" while ClubElo uses "Forest"
#Any mismatches during testing were added here

TEAM_NAME_MAP = {
    "Nott'm Forest":    "Forest",
    "Sheffield United": "Sheffield United",
    "Sheffield Utd":    "Sheffield United",
    "Man United":       "Man United",
    "Man City":         "Man City",
    "Tottenham":        "Tottenham",
    "Newcastle":        "Newcastle",
    "Leeds":            "Leeds",
    "Leicester":        "Leicester",
    "Wolves":           "Wolves",
    "Brighton":         "Brighton",
    "Brentford":        "Brentford",
    "Bournemouth":      "Bournemouth",
    "Fulham":           "Fulham",
    "Crystal Palace":   "Crystal Palace",
    "Aston Villa":      "Aston Villa",
    "Everton":          "Everton",
    "West Ham":         "West Ham",
    "Southampton":      "Southampton",
    "Chelsea":          "Chelsea",
    "Arsenal":          "Arsenal",
    "Liverpool":        "Liverpool",
    "Luton":            "Luton",
    "Burnley":          "Burnley",
    "Ipswich":          "Ipswich",
    "West Brom":        "West Brom",
}

#--------------------------------------------------------------------------------------------
#1. Load and combine all Football-Data CSVs
#--------------------------------------------------------------------------------------------

#Only keeps the columns that are relevant to prediction. Football-Data contains
#around 80 columns including closing odds variants and referee names.

COLS_TO_KEEP = [
    "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",                  #Full-time goals and results
    "HTHG", "HTAG", "HTR",                  #Half-time scores
    "HS",  "AS",                            #Shots
    "HST", "AST",                           #Shots on target
    "HC",  "AC",                            #Corners
    "HF",  "AF",                            #Fouls
    "HY",  "AY",                            #Yellow-cards
    "HR",  "AR",                            #Red-cards
    "B365H", "B365D", "B365A",              #B365 bookmaker odds used for baseline
]

season_dfs = []

#Load each season CSV, keep only required columns, add season labels
#latin-1 encoding handles special characters in older Football-Data files
for filename, season_label in SEASON_FILES:
    filepath = RAW_DATA_DIR / filename

    if not filepath.exists():
        print(f"  WARNING: {filepath} not found - skipping.")
        continue

    df = pd.read_csv(filepath, encoding="latin-1")
    cols_present = [c for c in COLS_TO_KEEP if c in df.columns]
    missing_cols = [c for c in COLS_TO_KEEP if c not in df.columns]

    if missing_cols:
        print(f"  INFO {filename}: columns not found (will be NaN): {missing_cols}")

    df = df[cols_present].copy()
    df["Season"] = season_label #Column used later for the time-based train/test split
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTR"])

    season_dfs.append(df)

#Fail immediately with a clear message rather than producing
#an empty dataset that would silently break the whole pipeline
if not season_dfs:
    raise FileNotFoundError(
        "No CSV files loaded. Check your data/raw/ folder and filenames."
    )

matches = pd.concat(season_dfs, ignore_index=True)
print(f"\n  Total matches: {len(matches)}")

matches["Date"] = pd.to_datetime(matches["Date"], dayfirst=True)
matches = matches.sort_values("Date").reset_index(drop=True)
print(f" Date range: {matches['Date'].min().date()}"
      f" to {matches['Date'].max().date()}")

#--------------------------------------------------------------------------------------------
#2. Pre-fetch ELO for all unique match dates
#--------------------------------------------------------------------------------------------

elo_client = ClubElo()

#Fetch ELO once per match date rather than once per match
#604 unique calls across 5 seasons vs 1900 individual API calls
#soccerdata caches data instantly so future runs are instant
unique_dates = sorted(matches["Date"].dt.strftime("%Y-%m-%d").unique())
total = len(unique_dates)
print(f"  {total} unique match dates to fetch")

#Store as a dictionary keyed by (date, team_name) for fast lookup per match
elo_lookup = {}
failed_dates = []

for i, date_str in enumerate(unique_dates):
    try:
        df_elo = elo_client.read_by_date(date_str)
        df_pl  = df_elo[df_elo["league"] == "ENG-Premier League"]

        for team_name, row in df_pl.iterrows():
            elo_lookup[(date_str, team_name)] = row["elo"]

    except Exception as e:
        failed_dates.append((date_str, str(e)))


if failed_dates:
    print(f"\n  WARNING: Failed to fetch {len(failed_dates)} date(s):")
    for d, err in failed_dates[:5]:
        print(f"    {d}: {err}")

print(f"\n  ELO lookup built: {len(elo_lookup)} entries\n")

#--------------------------------------------------------------------------------------------
#3. Attach ELO to every match row
#--------------------------------------------------------------------------------------------

def lookup_elo(date, team_raw):
    """
    Look up ELO for a team on or just before a given match date.
    ClubElo updates ratings after matches are played, not on match day,
    so the exact date sometimes has no entry. So search back up to 7 days
    to find the most recent available rating. This is correct since it
    represents team's strength entering the match. Falls back to np.nan only
    if nothing found within the 7-day window. 3 Teams always end up having missing
    data, this is explained in the report and fixed in feature_engineering.py
    """
    team_clubelo = TEAM_NAME_MAP.get(team_raw, team_raw)

    #Search backwards up to 7 days from match date
    for days_back in range(0, 8):
        check_date = (date - pd.Timedelta(days=days_back)).strftime("%Y-%m-%d")
        val = elo_lookup.get((check_date, team_clubelo), None)
        if val is not None:
            return val

    return np.nan
matches["HomeElo"] = matches.apply(
    lambda row: lookup_elo(row["Date"], row["HomeTeam"]), axis=1
)
matches["AwayElo"] = matches.apply(
    lambda row: lookup_elo(row["Date"], row["AwayTeam"]), axis=1
)

#Elo difference: positive = home team stronger, negative = away team stronger
# This derived feature is the strongest single predictor in the model
matches["EloDiff"] = matches["HomeElo"] - matches["AwayElo"]

missing_home = matches[matches["HomeElo"].isna()]
missing_away = matches[matches["AwayElo"].isna()]

print(f"  Matches missing HomeElo: {len(missing_home)}")
print(f"  Matches missing AwayElo: {len(missing_away)}")

if len(missing_home) > 0 or len(missing_away) > 0:
    unmatched = pd.Series(
        list(missing_home["HomeTeam"]) + list(missing_away["AwayTeam"])
    ).unique()
    print(f"\n  WARNING: Still unmatched after 7-day lookback.")
    print(f"  Affected team names:")
    for name in sorted(unmatched):
        print(f"    '{name}'")
    print(f"\n  Affected rows:")
    print(matches[matches["HomeElo"].isna() | matches["AwayElo"].isna()]
          [["Date", "HomeTeam", "AwayTeam", "HomeElo", "AwayElo"]].to_string())
else:
    print("  All ELO ratings matched successfully\n")

#--------------------------------------------------------------------------------------------
#4. Encode the target variable
#--------------------------------------------------------------------------------------------

# H=0 (Home Win), D=1 (Draw), A=2 (Away Win)
# Numeric encoding required by scikit-learn and XGBoost classifiers
matches["Result"] = matches["FTR"].map({"H": 0, "D": 1, "A": 2})
counts = matches["FTR"].value_counts()
total  = len(matches)
print(f"\n  Result distribution across all {total} matches:")
print(f"    Home wins (H): {counts.get('H', 0)}  ({counts.get('H', 0)/total*100:.1f}%)")
print(f"    Draws     (D): {counts.get('D', 0)}  ({counts.get('D', 0)/total*100:.1f}%)")
print(f"    Away wins (A): {counts.get('A', 0)}  ({counts.get('A', 0)/total*100:.1f}%)\n")

#--------------------------------------------------------------------------------------------
#5. Save
#--------------------------------------------------------------------------------------------

output_path = PROCESSED_DATA_DIR / "master_dataset.csv"
matches.to_csv(output_path, index=False)


print(f"\n  Path:  {output_path.resolve()}")
print(f"\n  Columns:")
for col in matches.columns:
    print(f"    {col}")
