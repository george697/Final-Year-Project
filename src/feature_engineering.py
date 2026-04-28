import pandas as pd
from pathlib import Path

INPUT_PATH  = Path("data/processed/master_dataset.csv")
OUTPUT_PATH = Path("data/processed/featured_dataset.csv")


df = pd.read_csv(INPUT_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print(f"  Loaded: {len(df)} matches\n")


home_rows = df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].copy()
home_rows.columns = ["Date", "Team", "Opponent", "GoalsFor", "GoalsAgainst", "MatchResult"]
home_rows["IsHome"] = 1

away_rows = df[["Date", "AwayTeam", "HomeTeam", "FTAG", "FTHG", "FTR"]].copy()
away_rows.columns = ["Date", "Team", "Opponent", "GoalsFor", "GoalsAgainst", "MatchResult"]
away_rows["IsHome"] = 0
away_rows["MatchResult"] = away_rows["MatchResult"].map({"H": "L", "A": "W", "D": "D"})

home_rows["MatchResult"] = home_rows["MatchResult"].map({"H": "W", "A": "L", "D": "D"})

long_df = pd.concat([home_rows, away_rows], ignore_index=True)
long_df = long_df.sort_values(["Team", "Date"]).reset_index(drop=True)

points_map = {"W": 3, "D": 1, "L": 0}
long_df["Points"] = long_df["MatchResult"].map(points_map)
long_df["Win"]    = (long_df["MatchResult"] == "W").astype(int)

def rolling_last5(series):
    return series.shift(1).rolling(window=5, min_periods=1).sum()

long_df["GoalsForLast5"]     = long_df.groupby("Team")["GoalsFor"].transform(rolling_last5)
long_df["GoalsAgainstLast5"] = long_df.groupby("Team")["GoalsAgainst"].transform(rolling_last5)
long_df["WinsLast5"]         = long_df.groupby("Team")["Win"].transform(rolling_last5)
long_df["FormPointsLast5"]   = long_df.groupby("Team")["Points"].transform(rolling_last5)


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

df["FormPointsDiff"] = df["HomeFormPoints"] - df["AwayFormPoints"]


form_cols = [
    "HomeGoalsLast5", "HomeConcededLast5", "HomeWinsLast5", "HomeFormPoints",
    "AwayGoalsLast5", "AwayConcededLast5", "AwayWinsLast5", "AwayFormPoints",
    "FormPointsDiff"
]

df[form_cols] = df[form_cols].fillna(0)

df["HomeElo"] = df.groupby("HomeTeam")["HomeElo"].bfill()
df["AwayElo"] = df.groupby("AwayTeam")["AwayElo"].bfill()
df["EloDiff"] = df["HomeElo"] - df["AwayElo"]

key_cols = ["HomeElo", "AwayElo", "EloDiff"] + form_cols
nan_counts = df[key_cols].isna().sum()
remaining_nans = nan_counts[nan_counts > 0]

if len(remaining_nans) == 0:
    print("  No NaNs remaining in key columns\n")
else:
    print("  WARNING: NaNs still present:")
    print(remaining_nans.to_string())
    print()


df.to_csv(OUTPUT_PATH, index=False)

print("DONE - featured_dataset.csv saved")
print(f"\n  Path:    {OUTPUT_PATH.resolve()}")

