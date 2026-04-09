
import pandas as pd
import numpy as np
from soccerdata import ClubElo
from pathlib import Path

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


COLS_TO_KEEP = [
    "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",
    "HTHG", "HTAG", "HTR",
    "HS",  "AS",
    "HST", "AST",
    "HC",  "AC",
    "HF",  "AF",
    "HY",  "AY",
    "HR",  "AR",
    "B365H", "B365D", "B365A",
]

season_dfs = []

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
    df["Season"] = season_label
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTR"])

    season_dfs.append(df)

if not season_dfs:
    raise FileNotFoundError(
        "No CSV files loaded. Check your data/raw/ folder and filenames."
    )

matches = pd.concat(season_dfs, ignore_index=True)
print(f"\n  Total matches: {len(matches)}")

matches["Date"] = pd.to_datetime(matches["Date"], dayfirst=True)
matches = matches.sort_values("Date").reset_index(drop=True)


elo_client = ClubElo()

unique_dates = sorted(matches["Date"].dt.strftime("%Y-%m-%d").unique())
total = len(unique_dates)
print(f"  {total} unique match dates to fetch")

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




def lookup_elo(date, team_raw):

    team_clubelo = TEAM_NAME_MAP.get(team_raw, team_raw)

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



matches["Result"] = matches["FTR"].map({"H": 0, "D": 1, "A": 2})



output_path = PROCESSED_DATA_DIR / "master_dataset.csv"
matches.to_csv(output_path, index=False)


print(f"\n  Path:  {output_path.resolve()}")
print(f"\n  Columns:")
for col in matches.columns:
    print(f"    {col}")
