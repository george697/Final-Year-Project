# Predicting Football Matches Using Machine Learning

This repository contains my final-year project exploring machine learning methods (XGBoost, Random Forest, Logistic Regression) to predict Premier League match outcomes using full-season comparative data.

## Setup and Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python3.11 -m venv venv
   ```
3. Activate it:
   ```
   Mac/Linux:  source venv/bin/activate
   Windows:    venv\Scripts\activate
   ```
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Run the pipeline in order:
   ```
   python src/build_master_dataset.py
   python src/feature_engineering.py
   python src/model_training.py
   python src/tournament_analysis.py
   python src/visualisations.py
   ```

> **macOS note:** If XGBoost fails to import, install the missing dependency first:
> ```
> brew install libomp
> ```

## Structure

```
Final-Year-Project/
│
├── src/                          # Source code for data processing, modelling, and analysis
│   ├── build_master_dataset.py   # Combines raw data into a unified dataset
│   ├── feature_engineering.py    # Feature creation and transformation logic
│   ├── model_training.py         # Model training and evaluation
│   ├── tournament_analysis.py    # Analysis specific to tournament data
│   └── visualisations.py         # Scripts to generate plots and figures
│
├── data/
│   ├── raw/                      # Unprocessed datasets
│   │   ├── prem2021.csv
│   │   ├── prem2122.csv
│   │   ├── prem2223.csv
│   │   ├── prem2324.csv
│   │   ├── prem2425.csv
│   │   └── tournament2425.csv
│   └── processed/                # Cleaned and transformed datasets
│       ├── featured_dataset.csv
│       └── master_dataset.csv
│
├── reports/
│   ├── figures/                  # Generated visual outputs
│   │   ├── figure1_model_comparison.png
│   │   ├── figure2_confusion_matrices.png
│   │   ├── figure3_feature_importances.png
│   │   ├── figure4_fixture_situation.png
│   │   ├── figure5_days_rest.png
│   │   └── figure6_tournament_type.png
│   ├── model_results.txt         # Model evaluation outputs
│   └── tournament_analysis.txt
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Results Summary

| Model | Accuracy | Macro F1 | Brier |
|---|---|---|---|
| Bookmaker (Bet365) | 53.9% | 0.404 | 0.193 |
| Logistic Regression | 51.8% | 0.394 | 0.197 |
| Random Forest | 51.6% | 0.399 | 0.198 |
| XGBoost | 51.1% | 0.426 | 0.204 |

## Data Sources

- Match statistics: [Football-Data.co.uk](https://www.football-data.co.uk)
- ELO ratings: [ClubElo](http://clubelo.com) via the [soccerdata](https://github.com/probberechts/soccerdata) library