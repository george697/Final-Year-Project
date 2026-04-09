
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.metrics import brier_score_loss

from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

INPUT_PATH   = Path("data/processed/featured_dataset.csv")
RESULTS_PATH = Path("reports/model_results.txt")
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)


FEATURES = [
    "HomeElo",
    "AwayElo",
    "EloDiff",

    "HomeGoalsLast5",
    "HomeConcededLast5",
    "HomeWinsLast5",
    "HomeFormPoints",

    "AwayGoalsLast5",
    "AwayConcededLast5",
    "AwayWinsLast5",
    "AwayFormPoints",

    "FormPointsDiff",
]

TARGET = "Result"

missing = [f for f in FEATURES if f not in df.columns]
if missing:
    raise ValueError(f"Missing feature columns: {missing}")


train_df = df[df["Season"] != "2024-25"].copy()
test_df  = df[df["Season"] == "2024-25"].copy()

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test  = test_df[FEATURES]
y_test  = test_df[TARGET]

counts = y_test.value_counts().sort_index()
labels = {0: "Home Win", 1: "Draw", 2: "Away Win"}
for code, count in counts.items():
    print(f"    {labels[code]}: {count} ({count/len(y_test)*100:.1f}%)")
print()

#BOOKMAKER

odds_cols = ["B365H", "B365D", "B365A"]

if all(c in test_df.columns for c in odds_cols):
    odds_test = test_df[odds_cols].copy().dropna()
    valid_idx = odds_test.index

    implied = 1.0 / odds_test
    implied = implied.div(implied.sum(axis=1), axis=0)

    bookie_pred = implied.values.argmax(axis=1)
    bookie_true = y_test.loc[valid_idx].values

    bookie_acc = accuracy_score(bookie_true, bookie_pred)
    bookie_f1  = f1_score(bookie_true, bookie_pred, average="macro", zero_division=0)

    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(bookie_true, classes=[0, 1, 2])
    bookie_brier = np.mean([
        brier_score_loss(y_true_bin[:, i], implied.values[:, i])
        for i in range(3)
    ])

    print(f"  Accuracy:    {bookie_acc:.4f}  ({bookie_acc*100:.1f}%)")
    print(f"  Macro F1:    {bookie_f1:.4f}")
    print(f"  Brier Score: {bookie_brier:.4f}  (lower = better)\n")
else:
    print("  Bet365 odds columns not found — skipping bookmaker baseline\n")
    bookie_acc, bookie_f1, bookie_brier = None, None, None


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


from sklearn.preprocessing import label_binarize

def evaluate_model(name, model, X_tr, X_te, y_tr, y_te):

    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)

    acc  = accuracy_score(y_te, y_pred)
    f1   = f1_score(y_te, y_pred, average="macro", zero_division=0)

    y_true_bin = label_binarize(y_te, classes=[0, 1, 2])
    brier = np.mean([
        brier_score_loss(y_true_bin[:, i], y_prob[:, i])
        for i in range(3)
    ])

    print(f"  Accuracy:    {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Macro F1:    {f1:.4f}")
    print(f"  Brier Score: {brier:.4f}  (lower = better)")

    # Compare to bookmaker
    if bookie_acc is not None:
        acc_diff = acc - bookie_acc
        print(f"\n  vs Bookmaker: accuracy {'+' if acc_diff >= 0 else ''}{acc_diff*100:.1f}%")

    print(f"\n  Classification Report:")
    print(classification_report(
        y_te, y_pred,
        target_names=["Home Win", "Draw", "Away Win"],
        zero_division=0
    ))

    print(f"  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"  {'':12} HomeWin  Draw  AwayWin")
    cm = confusion_matrix(y_te, y_pred)
    row_labels = ["HomeWin", "Draw   ", "AwayWin"]
    for label, row in zip(row_labels, cm):
        print(f"  {label}    {row[0]:>6}  {row[1]:>5}  {row[2]:>7}")
    print()

    return {"model": name, "accuracy": acc, "macro_f1": f1, "brier": brier, "trained": model}



results = []

# Model 1: Logistic Regression
lr = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
    C=1.0,
    random_state=42
)
results.append(evaluate_model(
    "Logistic Regression", lr,
    X_train_scaled, X_test_scaled, y_train, y_test
))

# Model 2: Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)
results.append(evaluate_model(
    "Random Forest", rf,
    X_train, X_test, y_train, y_test
))

# Model 3: XGBoost
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=42,
    verbosity=0
)
results.append(evaluate_model(
    "XGBoost", xgb,
    X_train, X_test, y_train, y_test
))

#summary

print(f"\n  {'Model':<25} {'Accuracy':>10} {'Macro F1':>10} {'Brier':>10}")
print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")

if bookie_acc is not None:
    print(f"  {'Bookmaker (Bet365)':<25} {bookie_acc:>9.4f}  {bookie_f1:>9.4f}  {bookie_brier:>9.4f}")

for r in results:
    print(f"  {r['model']:<25} {r['accuracy']:>9.4f}  {r['macro_f1']:>9.4f}  {r['brier']:>9.4f}")

print()

xgb_model = results[2]["trained"]
importances = pd.Series(xgb_model.feature_importances_, index=FEATURES)
importances = importances.sort_values(ascending=False)

print()
for feat, imp in importances.items():
    bar = "█" * int(imp * 200)
    print(f"  {feat:<25} {imp:.4f}  {bar}")

print()


with open(RESULTS_PATH, "w") as f:
    f.write("MODEL RESULTS SUMMARY\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"{'Model':<25} {'Accuracy':>10} {'Macro F1':>10} {'Brier':>10}\n")
    f.write(f"{'-'*25} {'-'*10} {'-'*10} {'-'*10}\n")
    if bookie_acc is not None:
        f.write(f"{'Bookmaker (Bet365)':<25} {bookie_acc:>9.4f}  {bookie_f1:>9.4f}  {bookie_brier:>9.4f}\n")
    for r in results:
        f.write(f"{r['model']:<25} {r['accuracy']:>9.4f}  {r['macro_f1']:>9.4f}  {r['brier']:>9.4f}\n")
    f.write("\n\nFeature Importances (XGBoost)\n")
    for feat, imp in importances.items():
        f.write(f"  {feat:<25} {imp:.4f}\n")

print(f"  Results saved to: {RESULTS_PATH.resolve()}")

