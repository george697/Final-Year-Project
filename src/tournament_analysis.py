import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier

FEATURED_PATH   = Path("data/processed/featured_dataset.csv")
TOURNAMENT_PATH = Path("data/raw/tournament2425.csv")
REPORTS_DIR     = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(FEATURED_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

tour = pd.read_csv(TOURNAMENT_PATH)
tour["Date"] = pd.to_datetime(tour["Date"])

print(f"  League matches loaded:      {len(df)}")
print(f"  Tournament records loaded:  {len(tour)} (team-fixture entries across all competitions)")
print(f"  Teams in tournament data:   {tour['Team'].nunique()}\n")


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

test_df["Predicted"] = model.predict(X_test)
test_df["Correct"]   = (test_df["Predicted"] == test_df["Result"]).astype(int)

overall_acc = test_df["Correct"].mean()
print(f"  Overall XGBoost accuracy on 2024-25: {overall_acc:.4f} ({overall_acc*100:.1f}%)\n")


def get_load(team, match_date, tournament_df, window_days=14):
    cutoff = match_date - pd.Timedelta(days=window_days)
    mask = (
        (tournament_df["Team"] == team) &
        (tournament_df["Date"] >= cutoff) &
        (tournament_df["Date"] < match_date)
    )
    recent = tournament_df[mask]

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
    european    = recent[recent["Tournament"].isin(["CL", "EL", "ECL"])]

    return {
        "TournamentGames": len(recent),
        "TournamentLoad":  recent["TournamentWeight"].sum(),
        "DaysRest":        days_rest,
        "IsEuropean":      1 if len(european) > 0 else 0,
        "EuropeanGames":   len(european),
    }

home_loads, away_loads = [], []
for _, row in test_df.iterrows():
    home_loads.append(get_load(row["HomeTeam"], row["Date"], tour))
    away_loads.append(get_load(row["AwayTeam"], row["Date"], tour))

home_load_df = pd.DataFrame(home_loads).add_prefix("Home_")
away_load_df = pd.DataFrame(away_loads).add_prefix("Away_")

test_df = pd.concat([
    test_df.reset_index(drop=True),
    home_load_df.reset_index(drop=True),
    away_load_df.reset_index(drop=True)
], axis=1)

test_df["CombinedLoad"]   = test_df["Home_TournamentLoad"] + test_df["Away_TournamentLoad"]
test_df["EitherEuropean"] = (
    (test_df["Home_IsEuropean"] == 1) | (test_df["Away_IsEuropean"] == 1)
).astype(int)
test_df["BothEuropean"]   = (
    (test_df["Home_IsEuropean"] == 1) & (test_df["Away_IsEuropean"] == 1)
).astype(int)

print(f"  Matches with ANY tournament game (14 days prior): {(test_df['CombinedLoad'] > 0).sum()}")
print(f"  Matches with European game (14 days prior):       {test_df['EitherEuropean'].sum()}")
print(f"  Matches with both teams European:                 {test_df['BothEuropean'].sum()}")
print(f"  Matches with no tournament games at all:          {(test_df['CombinedLoad'] == 0).sum()}\n")



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


played = test_df[test_df["CombinedLoad"] > 0].copy()
played["MinDaysRest"] = played[["Home_DaysRest", "Away_DaysRest"]].min(axis=1)

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


team_stats = []
all_teams  = test_df["HomeTeam"].unique()

for team in sorted(all_teams):
    team_matches = test_df[
        (test_df["HomeTeam"] == team) | (test_df["AwayTeam"] == team)
    ]
    n        = len(team_matches)
    acc      = team_matches["Correct"].mean()
    avg_load = (
        test_df.loc[test_df["HomeTeam"] == team, "Home_TournamentLoad"].sum() +
        test_df.loc[test_df["AwayTeam"] == team, "Away_TournamentLoad"].sum()
    ) / n
    european_games = tour[
        (tour["Team"] == team) &
        (tour["Tournament"].isin(["CL", "EL", "ECL"]))
    ].shape[0]

    team_stats.append({
        "Team": team, "Matches": n, "Accuracy": acc,
        "AvgLoad": avg_load, "EuropeanGames": european_games
    })

team_df = pd.DataFrame(team_stats).sort_values("Accuracy")

print(f"\n  {'Team':<20} {'Matches':>8} {'Accuracy':>10} {'Avg Load':>10} {'Euro Games':>12}")
print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*10} {'-'*12}")
for _, row in team_df.iterrows():
    print(f"  {row['Team']:<20} {row['Matches']:>8.0f} {row['Accuracy']:>9.4f}  "
          f"{row['AvgLoad']:>9.2f}  {row['EuropeanGames']:>11.0f}")

print()


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

    f.write("\nTEAM-LEVEL BREAKDOWN\n")
    f.write("-" * 40 + "\n")
    for _, row in team_df.iterrows():
        f.write(f"{row['Team']:<20} acc={row['Accuracy']:.4f}  "
                f"avg_load={row['AvgLoad']:.2f}  euro_games={row['EuropeanGames']:.0f}\n")

print(f"  Full results saved to: {output_path.resolve()}\n")