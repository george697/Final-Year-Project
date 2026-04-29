"""
visualisations.py
=================
Generates all six figures used in the report.

Figures 1-3 cover model evaluation results (Chapter 6):
    Figure 1 - Model performance comparison bar chart
    Figure 2 - Confusion matrices for all three models
    Figure 3 - XGBoost feature importances

Figures 4-6 cover tournament load analysis results (Chapter 7):
    Figure 4 - Accuracy by fixture situation
    Figure 5 - Accuracy by days rest since last tournament fixture
    Figure 6 - Accuracy by tournament type

HOW TO USE:
1. Run from project root: python src/visualisations.py
2. All figures saved to reports/figures/ as PNG files (directory is created automatically if it does not exist)

All result values are hardcoded from model_training.py and tournament_analysis.py output.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLOURS = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

#--------------------------------------------------------------------------------------------
# FIGURE 1: Model comparison bar chart
# Results from model_training.py,  bookmaker included as industry benchmark
#--------------------------------------------------------------------------------------------

models   = ["Bookmaker", "Logistic\nRegression", "Random\nForest", "XGBoost"]
accuracy = [0.5395, 0.5184, 0.5158, 0.5105]
macro_f1 = [0.4041, 0.3940, 0.3992, 0.4255]
brier    = [0.1929, 0.1969, 0.1975, 0.2035]

x     = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width, accuracy, width, label="Accuracy",    color=COLOURS[0], alpha=0.85)
bars2 = ax.bar(x,         macro_f1, width, label="Macro F1",    color=COLOURS[1], alpha=0.85)
bars3 = ax.bar(x + width, brier,    width, label="Brier Score", color=COLOURS[2], alpha=0.85)

ax.set_xlabel("Model", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Performance Comparison vs Bookmaker Baseline", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim(0, 0.65)
ax.axhline(y=0.5395, color="red", linestyle="--", linewidth=1.2, alpha=0.6)
ax.grid(axis="y", alpha=0.3)

# Annotate each bar with its exact value
for bar in [*bars1, *bars2, *bars3]:
    ax.annotate(f"{bar.get_height():.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure1_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Figure 1 saved")

#--------------------------------------------------------------------------------------------
# FIGURE 2: Confusion matrices
# One heatmap per model, shows draw prediction weakness
# Rows = actual class, Columns = predicted class
#--------------------------------------------------------------------------------------------

confusion_matrices = {
    "Logistic Regression": np.array([[123, 0, 32], [56, 1, 36], [59, 0, 73]]),
    "Random Forest":       np.array([[123, 3, 29], [58, 2, 33], [60, 1, 71]]),
    "XGBoost":             np.array([[119, 9, 27], [55, 8, 30], [55, 10, 67]]),
}
labels = ["Home Win", "Draw", "Away Win"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Confusion Matrices — 2024-25 Test Season", fontsize=13, fontweight="bold")

for ax, (model_name, cm) in zip(axes, confusion_matrices.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                ax=ax, cbar=False, linewidths=0.5)
    ax.set_title(model_name, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure2_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("Figure 2 saved")

#--------------------------------------------------------------------------------------------
# FIGURE 3: XGBoost feature importances
# Blue = ELO-based features, Green = form-based features
# Reversed so highest importance appears at the top
#--------------------------------------------------------------------------------------------

features = [
    "EloDiff", "AwayElo", "HomeElo", "FormPointsDiff",
    "HomeFormPoints", "HomeGoalsLast5", "AwayGoalsLast5",
    "AwayFormPoints", "AwayConcededLast5", "HomeWinsLast5",
    "HomeConcededLast5", "AwayWinsLast5"
]
importances = [
    0.1435, 0.0881, 0.0873, 0.0819,
    0.0805, 0.0780, 0.0774, 0.0771,
    0.0754, 0.0721, 0.0714, 0.0673
]
colours = ["#1565C0" if "Elo" in f else "#2E7D32" for f in features]

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(features[::-1], importances[::-1], color=colours[::-1], alpha=0.85)
ax.set_xlabel("Feature Importance", fontsize=12)
ax.set_title("XGBoost Feature Importances", fontsize=13, fontweight="bold")
ax.grid(axis="x", alpha=0.3)

for bar, val in zip(bars, importances[::-1]):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9)

elo_patch  = mpatches.Patch(color="#1565C0", alpha=0.85, label="ELO-based features")
form_patch = mpatches.Patch(color="#2E7D32", alpha=0.85, label="Form-based features")
ax.legend(handles=[elo_patch, form_patch], fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure3_feature_importances.png", dpi=150, bbox_inches="tight")
plt.close()
print("Figure 3 saved")

#--------------------------------------------------------------------------------------------
# FIGURE 4: Accuracy by fixture situation
# Green = above baseline, Red = below baseline
#--------------------------------------------------------------------------------------------

groups     = ["No\ntournament\ngames", "Domestic\ncup only",
              "Any\nEuropean\ngame", "Both teams\nhad European\ngame"]
accuracies = [0.5177, 0.5476, 0.4602, 0.4737]
ns         = [141, 126, 113, 19]
baseline   = 0.5105

bar_colours = ["#4CAF50" if a >= baseline else "#F44336" for a in accuracies]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(groups, accuracies, color=bar_colours, alpha=0.85, width=0.5)
ax.axhline(y=baseline, color="black", linestyle="--", linewidth=1.5)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("XGBoost Accuracy by Fixture Situation — 2024-25",
             fontsize=13, fontweight="bold")
ax.set_ylim(0.40, 0.60)
ax.grid(axis="y", alpha=0.3)

# Show both accuracy percentage and sample size on each bar
for bar, val, n in zip(bars, accuracies, ns):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f"{val:.1%}\n(n={n})", ha="center", va="bottom",
            fontsize=9, fontweight="bold")

above = mpatches.Patch(color="#4CAF50", alpha=0.85, label="Above baseline")
below = mpatches.Patch(color="#F44336", alpha=0.85, label="Below baseline")
ax.legend(handles=[above, below,
          plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.5,
                     label=f"Overall baseline ({baseline:.1%})")],
          fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure4_fixture_situation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Figure 4 saved")

#--------------------------------------------------------------------------------------------
# FIGURE 5: Days rest effect
# Only includes matches where at least one team had a tournament fixture
#--------------------------------------------------------------------------------------------

rest_buckets = ["1-3 days", "4-5 days", "6-7 days", "8-14 days"]
rest_acc     = [0.5059, 0.4400, 0.5455, 0.5352]
baseline     = 0.5105

bar_colours = ["#4CAF50" if a >= baseline else "#F44336" for a in rest_acc]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(rest_buckets, rest_acc, color=bar_colours, alpha=0.85, width=0.5)
ax.axhline(y=baseline, color="black", linestyle="--", linewidth=1.5)
ax.set_xlabel("Days Rest Since Last Tournament Fixture", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("XGBoost Accuracy by Days Rest — Matches with Tournament Fixtures",
             fontsize=12, fontweight="bold")
ax.set_ylim(0.35, 0.62)
ax.grid(axis="y", alpha=0.3)

for bar, val in zip(bars, rest_acc):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f"{val:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold")

above = mpatches.Patch(color="#4CAF50", alpha=0.85, label="Above baseline")
below = mpatches.Patch(color="#F44336", alpha=0.85, label="Below baseline")
ax.legend(handles=[above, below,
          plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.5,
                     label=f"Overall baseline ({baseline:.1%})")],
          fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure5_days_rest.png", dpi=150, bbox_inches="tight")
plt.close()
print("Figure 5 saved")

#--------------------------------------------------------------------------------------------
# FIGURE 6: Accuracy by tournament type
# Sample sizes shown on bars since they vary significantly across competitions
#--------------------------------------------------------------------------------------------

tourn_names = ["Champions\nLeague", "Europa\nLeague", "Conference\nLeague",
               "League\nCup", "FA\nCup"]
tourn_acc   = [0.5082, 0.3256, 0.5714, 0.5119, 0.5111]
tourn_n     = [61, 43, 21, 84, 90]
baseline    = 0.5105

bar_colours = ["#4CAF50" if a >= baseline else "#F44336" for a in tourn_acc]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(tourn_names, tourn_acc, color=bar_colours, alpha=0.85, width=0.5)
ax.axhline(y=baseline, color="black", linestyle="--", linewidth=1.5)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("XGBoost Accuracy by Tournament Type — 2024-25",
             fontsize=12, fontweight="bold")
ax.set_ylim(0.25, 0.65)
ax.grid(axis="y", alpha=0.3)

for bar, val, n in zip(bars, tourn_acc, tourn_n):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f"{val:.1%}\n(n={n})", ha="center", va="bottom",
            fontsize=9, fontweight="bold")

above = mpatches.Patch(color="#4CAF50", alpha=0.85, label="Above baseline")
below = mpatches.Patch(color="#F44336", alpha=0.85, label="Below baseline")
ax.legend(handles=[above, below,
          plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.5,
                     label=f"Overall baseline ({baseline:.1%})")],
          fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure6_tournament_type.png", dpi=150, bbox_inches="tight")
plt.close()
print("Figure 6 saved")

print(f"\nAll figures saved to {OUTPUT_DIR.resolve()}")