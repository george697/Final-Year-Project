"""
feature_engineering.py
=======================
Takes master_dataset.csv and adds rolling form features for each team
based on their last 5 matches before each game.

Features added:
HomeGoalsLast5      -goals scored by home team in last 5 matches
HomeConcededLast5   -goals conceded by home team in last 5 matches
HomeWinsLast5       -wins by home team in last 5 matches
HomeFormPointsLast5 -points earned by home team in last 5 matches(W=3, D=1, L=0)
AwayGoalsLast5      -goals scored by away team in last 5 matches
AwayConcededLast5   -goals conceded by away team in last 5 matches
AwayWinsLast5       -wins by away team in last 5 matches
AwayFormPointsLast5 -points earned by away team in last 5 matches
FormPointsDiff      -HomeFormPoints minus AwayFormPoints

HOW TO USE:
1. Run build_master_dataset.py first to generate master_dataset.csv
2. Run from project root: python src/feature_engineering.py
3. Output saved to data/processed/master_dataset.csv
"""

import pandas as pd
from pathlib import Path

#--------------------------------------------------------------------------------------------
#0. Load Master Dataset
#--------------------------------------------------------------------------------------------

INPUT_PATH  = Path("data/processed/master_dataset.csv")
OUTPUT_PATH = Path("data/processed/featured_dataset.csv")


df = pd.read_csv(INPUT_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print(f"  Loaded: {len(df)} matches\n")

#--------------------------------------------------------------------------------------------
#1. Build long-format table
#--------------------------------------------------------------------------------------------

# Rolling form must be computed per team across all their matches, both home and away.
# The fixture table has one row per match with separate home/away columns
# Unpivot it so each team appears once per match played.
# Compute rolling stats, then merge back onto the original fixture table.

# Home side rows - goals and result from the home team's perspective
home_rows = df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].copy()
home_rows.columns = ["Date", "Team", "Opponent", "GoalsFor", "GoalsAgainst", "MatchResult"]
home_rows["IsHome"] = 1

# Away side rows - goals are flipped so GoalsFor always mean goals scored by that team
away_rows = df[["Date", "AwayTeam", "HomeTeam", "FTAG", "FTHG", "FTR"]].copy()
away_rows.columns = ["Date", "Team", "Opponent", "GoalsFor", "GoalsAgainst", "MatchResult"]
away_rows["IsHome"] = 0

# Invert results from the away team's perspective (H in the data = loss for away team)
away_rows["MatchResult"] = away_rows["MatchResult"].map({"H": "L", "A": "W", "D": "D"})
home_rows["MatchResult"] = home_rows["MatchResult"].map({"H": "W", "A": "L", "D": "D"})

long_df = pd.concat([home_rows, away_rows], ignore_index=True)
long_df = long_df.sort_values(["Team", "Date"]).reset_index(drop=True)

# Points system mirrors the real Premier League W=3, D=1, L=0
# Used instead of wins alone since a team with 3 draws has different form to a team with 0 wins and 5 losses
# wins alone would not distinguish them
points_map = {"W": 3, "D": 1, "L": 0}
long_df["Points"] = long_df["MatchResult"].map(points_map)
long_df["Win"]    = (long_df["MatchResult"] == "W").astype(int)

#--------------------------------------------------------------------------------------------
#2. Compute rolling form features per team
#--------------------------------------------------------------------------------------------

def rolling_last5(series):
    """
    Compute rolling sum over last 5 matches, excluding current match.

    shift(1) prevents data leakage by ensuring the model never sees the result
    of the match it is predicting as part of its input. Without it the rolling window would include
    current match result.

    min_periods=1 allows partial windows at the start of the dataset rather than returning NaN when
    fewer than 5 prior matches are available.

    transform() is used instead of apply() so the result stays aligned with the original index.
    apply() caused a bug where goals were summed across all teams rather than per team independently
    """
    return series.shift(1).rolling(window=5, min_periods=1).sum()

# Apply per team group so each team's rolling window only covers their own matches
long_df["GoalsForLast5"]     = long_df.groupby("Team")["GoalsFor"].transform(rolling_last5)
long_df["GoalsAgainstLast5"] = long_df.groupby("Team")["GoalsAgainst"].transform(rolling_last5)
long_df["WinsLast5"]         = long_df.groupby("Team")["Win"].transform(rolling_last5)
long_df["FormPointsLast5"]   = long_df.groupby("Team")["Points"].transform(rolling_last5)

#--------------------------------------------------------------------------------------------
#3. Split back into home and away then merge onto fixture table
#--------------------------------------------------------------------------------------------

# Extract home team features, only rows where the team was playing at home
home_form = (
    long_df[long_df["IsHome"] == 1]
    [["Date", "Team", "GoalsForLast5", "GoalsAgainstLast5", "WinsLast5", "FormPointsLast5"]]
    .copy()
    .rename(columns={
        "Team":              "HomeTeam",
        "GoalsForLast5":     "HomeGoalsLast5",
        "GoalsAgainstLast5": "HomeConcededLast5",
        "WinsLast5":         "HomeWinsLast5",
        "FormPointsLast5":   "HomeFormPoints",
    })
)

# Extract away team features, only rows where the team was playing away
away_form = (
    long_df[long_df["IsHome"] == 0]
    [["Date", "Team", "GoalsForLast5", "GoalsAgainstLast5", "WinsLast5", "FormPointsLast5"]]
    .copy()
    .rename(columns={
        "Team":              "AwayTeam",
        "GoalsForLast5":     "AwayGoalsLast5",
        "GoalsAgainstLast5": "AwayConcededLast5",
        "WinsLast5":         "AwayWinsLast5",
        "FormPointsLast5":   "AwayFormPoints",
    })
)

df = df.merge(home_form, on=["Date", "HomeTeam"], how="left")
df = df.merge(away_form, on=["Date", "AwayTeam"], how="left")

# Derived feature encoding relative momentum between the two sides
# Positive = home team in better recent form, negative = away team in better form
df["FormPointsDiff"] = df["HomeFormPoints"] - df["AwayFormPoints"]

#--------------------------------------------------------------------------------------------
#4.Handle remaining NaNs
#--------------------------------------------------------------------------------------------

form_cols = [
    "HomeGoalsLast5", "HomeConcededLast5", "HomeWinsLast5", "HomeFormPoints",
    "AwayGoalsLast5", "AwayConcededLast5", "AwayWinsLast5", "AwayFormPoints",
    "FormPointsDiff"
]

#Form NaNs only appear at the start of the first season where no prior matches exist.
#0 is correct since there is no form to report
df[form_cols] = df[form_cols].fillna(0)

#3 matches at the start of 2020-21 have missing ELO for newly promoted clubs
# Backward fill uses the team's next available rating.
df["HomeElo"] = df.groupby("HomeTeam")["HomeElo"].bfill()
df["AwayElo"] = df.groupby("AwayTeam")["AwayElo"].bfill()
df["EloDiff"] = df["HomeElo"] - df["AwayElo"]

#Validate no NaNs remain in key model input columns before saving
key_cols = ["HomeElo", "AwayElo", "EloDiff"] + form_cols
nan_counts = df[key_cols].isna().sum()
remaining_nans = nan_counts[nan_counts > 0]

if len(remaining_nans) == 0:
    print("  No NaNs remaining in key columns\n")
else:
    print("  WARNING: NaNs still present:")
    print(remaining_nans.to_string())
    print()

#--------------------------------------------------------------------------------------------
#5. Save
#--------------------------------------------------------------------------------------------

df.to_csv(OUTPUT_PATH, index=False)

print("DONE - featured_dataset.csv saved")
print(f"\n  Path:    {OUTPUT_PATH.resolve()}")

