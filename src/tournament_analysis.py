"""
tournament_analysis.py
======================
Post-hoc analysis of how tournament fixture load affects XGBoost
prediction accuracy on the 2024-25 Premier League Season.

This script is not part of the main prediction pipeline. It is a separate
analysis designed to examine whether fixture congestion from European
and domestic cup competitions represents a source of prediction error that
historical match features cannot capture.

Tournament data was compiled manually for all 20 Premier League clubs and stored
in data/raw/tournament2425.csv. It covers CL, EL, ECL, EFL Cup, FA Cup, and Community
Shield fixtures for the 2024-25 season.

Five analyses are run:
    1. Accuracy by fixture situation (no load / domestic cup / European)
    2. Accuracy by which team (home vs away) had the load
    3. Accuracy by days rest since last tournament fixture
    4. Accuracy by specific tournament type
    5. Team-level accuracy correlated with European game count

HOW TO USE:
1. Run build_master_dataset.py then feature_engineering.py first
2. Run from project root: python src/tournament_analysis.py
3. Output saved to reports/tournament_analysis.txt
"""

import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier

FEATURED_PATH   = Path("data/processed/featured_dataset.csv")
TOURNAMENT_PATH = Path("data/raw/tournament2425.csv")
REPORTS_DIR     = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

#--------------------------------------------------------------------------------------------
# 0. Load data
#--------------------------------------------------------------------------------------------

df = pd.read_csv(FEATURED_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

tour = pd.read_csv(TOURNAMENT_PATH)
tour["Date"] = pd.to_datetime(tour["Date"])

print(f"  League matches loaded:      {len(df)}")
print(f"  Tournament records loaded:  {len(tour)} (team-fixture entries across all competitions)")
print(f"  Teams in tournament data:   {tour['Team'].nunique()}\n")

#--------------------------------------------------------------------------------------------
# 1. Retrain XGBoost on 2020-24, test on 2024-25
#--------------------------------------------------------------------------------------------
# XGBoost is retrained here with identical settings to model_training.py
# so tournament load results are directly comparable to the main evaluation

FEATURES = [
    "HomeElo", "AwayElo", "EloDiff",
    "HomeGoalsLast5", "HomeConcededLast5", "HomeWinsLast5", "HomeFormPoints",
    "AwayGoalsLast5", "AwayConcededLast5", "AwayWinsLast5", "AwayFormPoints",
    "FormPointsDiff",
]

train_df = df[df["Season"] != "2024-25"].copy()
test_df  = df[df["Season"] == "2024-25"].copy().reset_index(drop=True)

X_train = train_df[FEATURES]
y_train = train_df["Result"]
X_test  = test_df[FEATURES]
y_test  = test_df["Result"]

model = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="mlogloss", random_state=42, verbosity=0
)
model.fit(X_train, y_train)

# Store predictions and a binary correct/incorrect column for easy grouping later
test_df["Predicted"] = model.predict(X_test)
test_df["Correct"]   = (test_df["Predicted"] == test_df["Result"]).astype(int)

overall_acc = test_df["Correct"].mean()
print(f"  Overall XGBoost accuracy on 2024-25: {overall_acc:.4f} ({overall_acc*100:.1f}%)\n")

#--------------------------------------------------------------------------------------------
#2. Calculate tournament load per team per match
#--------------------------------------------------------------------------------------------

def get_load(team, match_date, tournament_df, window_days=14):
    """
    Look up tournament activity for a team in the 14 days before a match.

    A 14-day window captures mid-week European fixtures that typically fall between
    league gameweeks without extending so far back that fixtures from a previous round are included.

    Returns a dictionary of five metrics:
        TournamentGames     - raw count of non-league fixtures in the window
        TournamentLoad      - weighted sum using TournamentWeight column
        DaysRest            - days since most recent tournament fixture (returns window_days if no fixtures found = full rest)
        IsEuropean          - 1 if any CL/EL/ECL fixture in window, else 0
        EuropeanGames       - count of European fixtures in window
    """
    cutoff = match_date - pd.Timedelta(days=window_days)
    #Filter to this team's fixtures within the lookback window
    mask = (
        (tournament_df["Team"] == team) &
        (tournament_df["Date"] >= cutoff) &
        (tournament_df["Date"] < match_date)
    )
    recent = tournament_df[mask]

    #No tournament games in window, return full rest default
    if len(recent) == 0:
        return {
            "TournamentGames": 0,
            "TournamentLoad":  0,
            "DaysRest":        window_days,
            "IsEuropean":      0,
            "EuropeanGames":   0,
        }

    most_recent = recent["Date"].max()
    days_rest   = (match_date - most_recent).days
    #European competitions only, domestic cups excluded from IsEuropean flag
    european    = recent[recent["Tournament"].isin(["CL", "EL", "ECL"])]

    return {
        "TournamentGames": len(recent),
        "TournamentLoad":  recent["TournamentWeight"].sum(),
        "DaysRest":        days_rest,
        "IsEuropean":      1 if len(european) > 0 else 0,
        "EuropeanGames":   len(european),
    }

# Calculate load for both teams in every 2024-25 test match
home_loads, away_loads = [], []
for _, row in test_df.iterrows():
    home_loads.append(get_load(row["HomeTeam"], row["Date"], tour))
    away_loads.append(get_load(row["AwayTeam"], row["Date"], tour))

# Convert list of dictionaries to dataframes and prefix columns by team side
home_load_df = pd.DataFrame(home_loads).add_prefix("Home_")
away_load_df = pd.DataFrame(away_loads).add_prefix("Away_")

# Merge load columns onto the test dataframe alongside predictions
test_df = pd.concat([
    test_df.reset_index(drop=True),
    home_load_df.reset_index(drop=True),
    away_load_df.reset_index(drop=True)
], axis=1)

# Combined load: sum of both teams weighted tournament burden for each match
test_df["CombinedLoad"]   = test_df["Home_TournamentLoad"] + test_df["Away_TournamentLoad"]

# EitherEuropean: 1 if at least one team had a European fixture in the window
test_df["EitherEuropean"] = (
    (test_df["Home_IsEuropean"] == 1) | (test_df["Away_IsEuropean"] == 1)
).astype(int)

# BothEuropean: 1 only if both teams had European fixtures — the most congested matches
test_df["BothEuropean"]   = (
    (test_df["Home_IsEuropean"] == 1) & (test_df["Away_IsEuropean"] == 1)
).astype(int)

print(f"  Matches with ANY tournament game (14 days prior): {(test_df['CombinedLoad'] > 0).sum()}")
print(f"  Matches with European game (14 days prior):       {test_df['EitherEuropean'].sum()}")
print(f"  Matches with both teams European:                 {test_df['BothEuropean'].sum()}")
print(f"  Matches with no tournament games at all:          {(test_df['CombinedLoad'] == 0).sum()}\n")

#--------------------------------------------------------------------------------------------
# Analysis 1: Accuracy by fixture situation
#--------------------------------------------------------------------------------------------

no_load_group     = test_df[test_df["CombinedLoad"] == 0]
domestic_only     = test_df[
    (test_df["CombinedLoad"] > 0) &
    (test_df["EitherEuropean"] == 0)
]
any_european      = test_df[test_df["EitherEuropean"] == 1]
both_european     = test_df[test_df["BothEuropean"] == 1]

print(f"\n  {'Group':<35} {'Matches':>8} {'Accuracy':>10} {'vs Overall':>12}")
print(f"  {'-'*35} {'-'*8} {'-'*10} {'-'*12}")

groups_a1 = [
    ("No tournament games",          no_load_group),
    ("Domestic cup only",            domestic_only),
    ("Any European game",            any_european),
    ("Both teams had European game", both_european),
]

group_results = {}
for name, group in groups_a1:
    if len(group) == 0:
        continue
    acc  = group["Correct"].mean()
    diff = acc - overall_acc
    sign = "+" if diff >= 0 else ""
    print(f"  {name:<35} {len(group):>8} {acc:>9.4f}  {sign}{diff*100:.1f}%")
    group_results[name] = {"n": len(group), "accuracy": acc, "diff": diff}

print()

#--------------------------------------------------------------------------------------------
# Analysis 2 : Home vs away team load
#--------------------------------------------------------------------------------------------
#Tests whether the location of the load (home vs away) affects accuracy

home_loaded = test_df[test_df["Home_TournamentLoad"] > 0]
away_loaded = test_df[test_df["Away_TournamentLoad"] > 0]
neither     = test_df[
    (test_df["Home_TournamentLoad"] == 0) &
    (test_df["Away_TournamentLoad"] == 0)
]

print(f"\n  {'Group':<35} {'Matches':>8} {'Accuracy':>10} {'vs Overall':>12}")
print(f"  {'-'*35} {'-'*8} {'-'*10} {'-'*12}")

for name, group in [
    ("Home team had tournament game",    home_loaded),
    ("Away team had tournament game",    away_loaded),
    ("Neither team had tournament game", neither),
]:
    if len(group) == 0:
        continue
    acc  = group["Correct"].mean()
    diff = acc - overall_acc
    sign = "+" if diff >= 0 else ""
    print(f"  {name:<35} {len(group):>8} {acc:>9.4f}  {sign}{diff*100:.1f}%")

print()

#--------------------------------------------------------------------------------------------
# Analysis 3 : Days Rest
#--------------------------------------------------------------------------------------------
# Uses the minimum days rest across both teams as the worst-case measure
# Only includes matches where at least one team had a recent tournament game

played = test_df[test_df["CombinedLoad"] > 0].copy()

# Take the minimum rest of the two teams, worst-case fatigue for the match
played["MinDaysRest"] = played[["Home_DaysRest", "Away_DaysRest"]].min(axis=1)

# Bin rest days into four groups that reflect distinct managerial situations
bins   = [0, 3, 5, 7, 14]
labels = ["1-3 days", "4-5 days", "6-7 days", "8-14 days"]
played["RestBucket"] = pd.cut(played["MinDaysRest"], bins=bins, labels=labels)

print(f"\n  {'Rest since last tournament game':<30} {'Matches':>8} {'Accuracy':>10} {'vs Overall':>12}")
print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*12}")

for bucket in labels:
    group = played[played["RestBucket"] == bucket]
    if len(group) == 0:
        continue
    acc  = group["Correct"].mean()
    diff = acc - overall_acc
    sign = "+" if diff >= 0 else ""
    print(f"  {bucket:<30} {len(group):>8} {acc:>9.4f}  {sign}{diff*100:.1f}%")

print()

#--------------------------------------------------------------------------------------------
# Analysis 4: Accuracy by tournament type
#--------------------------------------------------------------------------------------------
# For each competition, checks whether either team played that specific tournament in the 14 days before the league match.
# Groups are not mutually exclusive, a team could have played both a CL game and an EFL Cup game in the 14-day window,
# so match counts across all five competitions will sum to more than 380.

print(f"\n  {'Tournament type':<30} {'Matches':>8} {'Accuracy':>10} {'vs Overall':>12}")
print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*12}")

for tourn in ["CL", "EL", "ECL", "EFL", "FA"]:
    tourn_name = {
        "CL":  "Champions League",
        "EL":  "Europa League",
        "ECL": "Conference League",
        "EFL": "League Cup",
        "FA":  "FA Cup"
    }[tourn]

    def had_recent_tourn_game(team, match_date, tournament):
        cutoff = match_date - pd.Timedelta(days=14)
        mask = (
            (tour["Team"] == team) &
            (tour["Tournament"] == tournament) &
            (tour["Date"] >= cutoff) &
            (tour["Date"] < match_date)
        )
        return len(tour[mask]) > 0

    # Flag each match if either team played this competition recently
    match_flags = []
    for _, row in test_df.iterrows():
        home_flag = had_recent_tourn_game(row["HomeTeam"], row["Date"], tourn)
        away_flag = had_recent_tourn_game(row["AwayTeam"], row["Date"], tourn)
        match_flags.append(home_flag or away_flag)

    group = test_df[match_flags]
    if len(group) == 0:
        print(f"  {tourn_name:<30} {'No matches found':>8}")
        continue

    acc  = group["Correct"].mean()
    diff = acc - overall_acc
    sign = "+" if diff >= 0 else ""
    print(f"  {tourn_name:<30} {len(group):>8} {acc:>9.4f}  {sign}{diff*100:.1f}%")

print()

#--------------------------------------------------------------------------------------------
# Analysis 5: Team-level accuracy
#--------------------------------------------------------------------------------------------
# Looks at each club individually to see if model accuracy correlates with European game count
# tests whether volatile non-European clubs or European clubs are driving the prediction errors

team_stats = []
all_teams  = test_df["HomeTeam"].unique()

for team in sorted(all_teams):
    # All matches where this team appeared (home or away)
    team_matches = test_df[
        (test_df["HomeTeam"] == team) | (test_df["AwayTeam"] == team)
    ]
    n        = len(team_matches)
    acc      = team_matches["Correct"].mean()

    # Average tournament load per match, sum of home and away load across season / matches
    avg_load = (
        test_df.loc[test_df["HomeTeam"] == team, "Home_TournamentLoad"].sum() +
        test_df.loc[test_df["AwayTeam"] == team, "Away_TournamentLoad"].sum()
    ) / n

    # Total European fixtures played by this team in 2024-25
    european_games = tour[
        (tour["Team"] == team) &
        (tour["Tournament"].isin(["CL", "EL", "ECL"]))
    ].shape[0]

    team_stats.append({
        "Team": team, "Matches": n, "Accuracy": acc,
        "AvgLoad": avg_load, "EuropeanGames": european_games
    })

# Sort by accuracy ascending, most mispredicted teams first
team_df = pd.DataFrame(team_stats).sort_values("Accuracy")

print(f"\n  {'Team':<20} {'Matches':>8} {'Accuracy':>10} {'Avg Load':>10} {'Euro Games':>12}")
print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*10} {'-'*12}")
for _, row in team_df.iterrows():
    print(f"  {row['Team']:<20} {row['Matches']:>8.0f} {row['Accuracy']:>9.4f}  "
          f"{row['AvgLoad']:>9.2f}  {row['EuropeanGames']:>11.0f}")

print()

#--------------------------------------------------------------------------------------------
# Save results
#--------------------------------------------------------------------------------------------

output_path = REPORTS_DIR / "tournament_analysis.txt"

with open(output_path, "w") as f:
    f.write("TOURNAMENT LOAD ANALYSIS - 2024-25 SEASON\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Overall XGBoost accuracy: {overall_acc:.4f}\n\n")

    f.write("ANALYSIS 1: ACCURACY BY FIXTURE SITUATION\n")
    f.write("-" * 40 + "\n")
    for name, res in group_results.items():
        sign = "+" if res["diff"] >= 0 else ""
        f.write(f"{name:<35} n={res['n']:>3}  acc={res['accuracy']:.4f}  "
                f"({sign}{res['diff']*100:.1f}% vs overall)\n")

    f.write("\nANALYSIS 2: HOME VS AWAY LOAD\n")
    f.write("-" * 40 + "\n")
    for name, group in [
        ("Home team had tournament game",    home_loaded),
        ("Away team had tournament game",    away_loaded),
        ("Neither team had tournament game", neither),
    ]:
        if len(group) == 0:
            continue
        acc  = group["Correct"].mean()
        diff = acc - overall_acc
        sign = "+" if diff >= 0 else ""
        f.write(f"{name:<35} n={len(group):>3}  acc={acc:.4f}  "
                f"({sign}{diff*100:.1f}% vs overall)\n")

    f.write("\nANALYSIS 3: DAYS REST\n")
    f.write("-" * 40 + "\n")
    for bucket in labels:
        group = played[played["RestBucket"] == bucket]
        if len(group) == 0:
            continue
        acc  = group["Correct"].mean()
        diff = acc - overall_acc
        sign = "+" if diff >= 0 else ""
        f.write(f"{bucket:<30} n={len(group):>3}  acc={acc:.4f}  "
                f"({sign}{diff*100:.1f}% vs overall)\n")

    f.write("\nANALYSIS 4: ACCURACY BY TOURNAMENT TYPE\n")
    f.write("-" * 40 + "\n")
    for tourn in ["CL", "EL", "ECL", "EFL", "FA"]:
        tourn_name = {
            "CL":  "Champions League",
            "EL":  "Europa League",
            "ECL": "Conference League",
            "EFL": "League Cup",
            "FA":  "FA Cup"
        }[tourn]
        match_flags = []
        for _, row in test_df.iterrows():
            cutoff = row["Date"] - pd.Timedelta(days=14)
            home_flag = len(tour[(tour["Team"] == row["HomeTeam"]) &
                                 (tour["Tournament"] == tourn) &
                                 (tour["Date"] >= cutoff) &
                                 (tour["Date"] < row["Date"])]) > 0
            away_flag = len(tour[(tour["Team"] == row["AwayTeam"]) &
                                 (tour["Tournament"] == tourn) &
                                 (tour["Date"] >= cutoff) &
                                 (tour["Date"] < row["Date"])]) > 0
            match_flags.append(home_flag or away_flag)
        group = test_df[match_flags]
        if len(group) == 0:
            continue
        acc  = group["Correct"].mean()
        diff = acc - overall_acc
        sign = "+" if diff >= 0 else ""
        f.write(f"{tourn_name:<30} n={len(group):>3}  acc={acc:.4f}  "
                f"({sign}{diff*100:.1f}% vs overall)\n")

    f.write("\nANALYSIS 5: TEAM-LEVEL BREAKDOWN\n")
    f.write("-" * 40 + "\n")
    for _, row in team_df.iterrows():
        f.write(f"{row['Team']:<20} acc={row['Accuracy']:.4f}  "
                f"avg_load={row['AvgLoad']:.2f}  euro_games={row['EuropeanGames']:.0f}\n")

print(f"  Full results saved to: {output_path.resolve()}\n")