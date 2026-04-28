# Final-Year-Project
# Predicting football matches using Machine Learning

This repository contains my final-year project exploring machine learning methods (XGBoost, Random Forest, Logistic Regression) to predict Premier League match outcomes using full-season comparative data.

# Setup and Installation

1. Clone the repository
2. Create a virtual environment:
   python3.11 -m venv venv
3. Activate it:
   Mac/Linux:  source venv/bin/activate
   Windows:    venv\Scripts\activate
4. Install dependencies:
   pip install -r requirements.txt
5. Run the pipeline in order:
   python src/build_master_dataset.py
   python src/feature_engineering.py
   python src/model_training.py
   python src/tournament_analysis.py
   python src/visualisations.py

# Structure: 
Final-Year-Project/
│
├── src/                          # Source code for data processing, modelling, and analysis
│   ├── data/
│   │   ├── external/             # External/raw data sources (original datasets)
│   │   ├── raw/                  # Unprocessed datasets
│   │   │   ├── prem2021.csv
│   │   │   ├── prem2122.csv
│   │   │   ├── prem2223.csv
│   │   │   ├── prem2324.csv
│   │   │   ├── prem2425.csv
│   │   │   └── tournament2425.csv
│   │   ├── processed/            # Cleaned and transformed datasets
│   │   │   ├── featured_dataset.csv
│   │   │   └── master_dataset.csv
│   │
│   ├── reports/
│   │   ├── figures/              # Generated visual outputs
│   │   │   ├── figure1_model_comparison.png
│   │   │   ├── figure2_confusion_matrices.png
│   │   │   ├── figure3_feature_importances.png
│   │   │   ├── figure4_fixture_situation.png
│   │   │   ├── figure5_days_rest.png
│   │   │   └── figure6_tournament_type.png
│   │   ├── model_results.txt     # Model evaluation outputs
│   │   └── tournament_analysis.txt
│   │
│   ├── build_master_dataset.py   # Combines raw data into a unified dataset
│   ├── feature_engineering.py    # Feature creation and transformation logic
│   ├── model_training.py         # Model training and evaluation
│   ├── tournament_analysis.py    # Analysis specific to tournament data
│   └── visualisations.py         # Scripts to generate plots and figures
│
├── .gitignore                   # Files and directories to ignore in Git
├── LICENSE                      # Project license
├── README.md                    # Project documentation
└── requirements.txt             # Python dependencies