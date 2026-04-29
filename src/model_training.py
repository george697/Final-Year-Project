"""
model_training.py
=================
Trains and evaluates three models on the featured dataset:
- Logistic Regression   (interpretable baseline)
- Random Forest         (tree-based ensemble)
- XGBoost               (gradient boosting)

Train/test split strategy:
Train on seasons 2020-21 to 2023-24 (4 seasons, 1520 matches)
Test on season 2024-25 (1 season, 380 matches)

This is a TIME-BASED split, never train on future data.
Target variable:
Result: 0 = Home Win, 1 = Draw, 2 = Away Win

Evaluation metrics:
- Accuracy              (overall correctness)
- Macro F1 Score        (penalises poor draw prediction)
- Brier score           (probabilistic accuracy)
- Bookmaker baseline    (Bet365 odds converted to probabilities)

HOW TO USE:
1. Run build_master_dataset.py first and then feature_engineering.py to build the featured dataset.csv
2. Run from project root: python src/model_training.py
3. output saved to reports/model_results.txt
"""
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import label_binarize
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

#--------------------------------------------------------------------------------------------
#0. Load Data
#--------------------------------------------------------------------------------------------

INPUT_PATH   = Path("data/processed/featured_dataset.csv")
RESULTS_PATH = Path("reports/model_results.txt")
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

#--------------------------------------------------------------------------------------------
#1. Define Features and Target
#--------------------------------------------------------------------------------------------

# 3 ELO-based features + 9 rolling form features = 12 total
FEATURES = [
    #Team strength ratings
    "HomeElo",
    "AwayElo",
    "EloDiff",

    #Home team attacking/defensive form
    "HomeGoalsLast5",
    "HomeConcededLast5",
    #Home team results based form
    "HomeWinsLast5",
    "HomeFormPoints",

    #Away team attacking/defensive form
    "AwayGoalsLast5",
    "AwayConcededLast5",
    #Away team results-based form
    "AwayWinsLast5",
    "AwayFormPoints",

    #relative momentum (home minus away)
    "FormPointsDiff",
]

TARGET = "Result"   # 0=Home Win, 1=Draw, 2=Away Win

# Validate all features are present before training, fail immediately rather
# than producing results from an incomplete feature set
missing = [f for f in FEATURES if f not in df.columns]
if missing:
    raise ValueError(f"Missing feature columns: {missing}")

#--------------------------------------------------------------------------------------------
#2. Train-based train/test split
#--------------------------------------------------------------------------------------------
# A random split was rejected because it would allow training on future seasons and testing on past ones,
# which risks leaking team strength information across season boundaries.

train_df = df[df["Season"] != "2024-25"].copy() # 2020-21 to 2023-24
test_df  = df[df["Season"] == "2024-25"].copy() # held out test season

# X = input feature, y = target labels
X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test  = test_df[FEATURES]
y_test  = test_df[TARGET]

# Show class distribution in test set, imbalance matters for metric choice
counts = y_test.value_counts().sort_index()
labels = {0: "Home Win", 1: "Draw", 2: "Away Win"}
for code, count in counts.items():
    print(f"    {labels[code]}: {count} ({count/len(y_test)*100:.1f}%)")
print()

#--------------------------------------------------------------------------------------------
#3. Bookmaker Baseline
#--------------------------------------------------------------------------------------------
# Convert decimal odds to implied probabilities and evaluate as a prediction model
# This sets the industry benchmark all three ML models are compared against

odds_cols = ["B365H", "B365D", "B365A"]

if all(c in test_df.columns for c in odds_cols):
    odds_test = test_df[odds_cols].copy().dropna()
    valid_idx = odds_test.index

    implied = 1.0 / odds_test # Invert odds to get raw implied probabilities (e.g. odds 2.0 -> 50%)
    implied = implied.div(implied.sum(axis=1), axis=0) # normalize out the overround

    #Predict the outcome with the highest implied probability
    bookie_pred = implied.values.argmax(axis=1)
    bookie_true = y_test.loc[valid_idx].values

    bookie_acc = accuracy_score(bookie_true, bookie_pred)
    bookie_f1  = f1_score(bookie_true, bookie_pred, average="macro", zero_division=0)

    # Brier score requires one-hot encoded true labels for multi-class calculation
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

#--------------------------------------------------------------------------------------------
#4. Scale features (Needed for Logistic Regression)
#--------------------------------------------------------------------------------------------

# Logistic Regression learns weighted coefficients, features on very different scales (ELO ~1500 vs wins ~0-5)
# would cause ELO to dominate unfairly. StandardScaler transforms each feature to zero mean and unit variance.
# fit only on training data, then apply to test data using the same learned parameters.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

#--------------------------------------------------------------------------------------------
#5. Evaluate a trained model
#--------------------------------------------------------------------------------------------

def evaluate_model(name, model, X_tr, X_te, y_tr, y_te):
    """
    Train a model and evaluate it on the test set.
    Returns a dictionary of results for the summary table.

    predict()   -> final class prediction (0, 1 or 2)
    predict_proba() -> probability distribution across all three classes,
    used for Brier score calculation
    """
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)

    acc  = accuracy_score(y_te, y_pred)
    f1   = f1_score(y_te, y_pred, average="macro", zero_division=0)

    # Brier score computed per class then averaged
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

#--------------------------------------------------------------------------------------------
#6. Train and evaluate all models
#--------------------------------------------------------------------------------------------

results = []

# Model 1: Logistic Regression
#Simples model, serves as interpretable lower bound
lr = LogisticRegression(
    solver="lbfgs",     #algorithm used to train model
    max_iter=1000,      #maximum number of training steps, ensures convergence
    C=1.0,              #controls regularisation
    random_state=42     #fixes randomness so results are reproducible
)
results.append(evaluate_model(
    "Logistic Regression", lr,
    X_train_scaled, X_test_scaled, y_train, y_test
))

# Model 2: Random Forest
rf = RandomForestClassifier(
    n_estimators=200,       #number of trees
    max_depth=8,            #limits how deep each tree can go, prevents overfitting
    min_samples_leaf=10,    #minimum samples in a leaf node, stop trees from becoming too specific
    random_state=42,
    n_jobs=-1               #uses all CPU cores to make training faster.
)
results.append(evaluate_model(
    "Random Forest", rf,
    X_train, X_test, y_train, y_test
))

# Model 3: XGBoost
# Primary model, builds trees sequentially, each correcting the previous ensemble
# Shallow trees (max_depth=4) are preferred in boosting since each tree makes
# small incremental corrections rather than capturing full patterns independently
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,              #uses 80% of rows per tree
    colsample_bytree=0.8,       #uses 80% of columns (features) per tree, both reduce overfitting
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=42,
    verbosity=0
)
results.append(evaluate_model(
    "XGBoost", xgb,
    X_train, X_test, y_train, y_test
))

#--------------------------------------------------------------------------------------------
#7. Summary table
#--------------------------------------------------------------------------------------------

print(f"\n  {'Model':<25} {'Accuracy':>10} {'Macro F1':>10} {'Brier':>10}")
print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")

if bookie_acc is not None:
    print(f"  {'Bookmaker (Bet365)':<25} {bookie_acc:>9.4f}  {bookie_f1:>9.4f}  {bookie_brier:>9.4f}")

for r in results:
    print(f"  {r['model']:<25} {r['accuracy']:>9.4f}  {r['macro_f1']:>9.4f}  {r['brier']:>9.4f}")

print()

# XGBoost built-in feature importances, shows which features drove predictions most
# Ranked highest to lowest with a visual bar for quick comparison
xgb_model = results[2]["trained"]
importances = pd.Series(xgb_model.feature_importances_, index=FEATURES)
importances = importances.sort_values(ascending=False)

print()
for feat, imp in importances.items():
    bar = "█" * int(imp * 200)
    print(f"  {feat:<25} {imp:.4f}  {bar}")

print()

#--------------------------------------------------------------------------------------------
#8. Save results to file
#--------------------------------------------------------------------------------------------

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

