# Rainfall Prediction Using Five Machine Learning Regression Algorithms

A predictive analytics project that estimates **next-year annual rainfall** for Indian meteorological subdivisions from historical rainfall records (1901–2015). Five regression algorithms are trained on the same features, compared on a chronological holdout, and the best model is used to generate illustrative 2016 estimates.

> **Disclaimer:** The 2016 values are machine learning outputs from a historical-data model. They are **not** official meteorological forecasts and should not be used alone for high-impact decisions.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Methodology](#methodology)
- [Models](#models)
- [Results](#results)
- [Future-Year Prediction (2016)](#future-year-prediction-2016)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Limitations](#limitations)
- [Future Scope](#future-scope)
- [Author](#author)

---

## Overview

Rainfall affects agriculture, water-resource planning, and disaster preparedness. This project demonstrates a reproducible workflow:

1. Load and inspect the raw CSV data
2. Exploratory data analysis (distributions, monthly and regional patterns)
3. Build lag-based features and a next-year target
4. Preprocess numeric and categorical fields
5. Split the data chronologically into training and test sets
6. Train and evaluate five regression models
7. Select the best model by RMSE and generate 2016 estimates

## Dataset

| Property | Value |
| --- | --- |
| File | `rainfall_area-wt_sd_1901-2015.csv` |
| Rows | 4,116 |
| Columns | 19 |
| Period | 1901–2015 |
| Subdivisions | 36 |
| Duplicate rows | 0 |
| Missing cells | 134 (26 in `ANNUAL`) |

Columns include the subdivision name, the year, twelve monthly rainfall values (`JAN`–`DEC`), `ANNUAL`, and four seasonal aggregates (`Jan-Feb`, `Mar-May`, `Jun-Sep`, `Oct-Dec`). The target is derived from `ANNUAL`.

## Feature Engineering

Records are sorted by subdivision and year; features are built per subdivision. For a target year `t+1`:

| Feature | Description |
| --- | --- |
| `SUBDIVISION` | Categorical subdivision identifier |
| `TARGET_YEAR` | Year being predicted |
| `PREVIOUS_RAINFALL` | Annual rainfall in year `t` |
| `RAINFALL_2_YEARS_AGO` | Annual rainfall in year `t-1` |
| `THREE_YEAR_AVERAGE` | Mean annual rainfall over years `t-2`, `t-1`, `t` |

The target is the next year's `ANNUAL` rainfall (`shift(-1)` within each subdivision). After removing incomplete rows, **3,935** supervised records remain.

## Methodology

- **Preprocessing:** numeric features use median imputation and `StandardScaler`; `SUBDIVISION` uses most-frequent imputation and `OneHotEncoder`. Everything is fitted inside each model pipeline on training data only.
- **Chronological split:** the first 80% of distinct target years form the training set and the rest form the test set. This evaluates models on later years than they were trained on.

| Item | Value |
| --- | --- |
| Training target years | 1904–1992 |
| Testing target years | 1993–2015 |
| Training records | 3,119 |
| Testing records | 816 |
| Random state | 42 |

- **Metrics:** MAE, MSE, RMSE, and R². RMSE is the primary model-selection criterion.

## Models

| Algorithm | Configuration |
| --- | --- |
| Linear Regression | Ordinary least squares |
| Decision Tree Regressor | `max_depth=8`, `random_state=42` |
| Random Forest Regressor | `n_estimators=200`, `max_depth=10`, `random_state=42`, `n_jobs=-1` |
| Gradient Boosting Regressor | `n_estimators=150`, `learning_rate=0.05`, `max_depth=3`, `random_state=42` |
| K-Nearest Neighbors Regressor | `n_neighbors=5` |

## Results

MAE and RMSE are in millimetres; MSE is in squared millimetres.

| Algorithm | MAE | MSE | RMSE | R² |
| --- | --- | --- | --- | --- |
| **Random Forest** | 208.45 | 79170.98 | **281.37** | **0.8907** |
| Linear Regression | 206.04 | 79851.78 | 282.58 | 0.8898 |
| Gradient Boosting | 212.70 | 84377.77 | 290.48 | 0.8835 |
| KNN | 231.30 | 99584.50 | 315.57 | 0.8625 |
| Decision Tree | 229.46 | 102573.36 | 320.27 | 0.8584 |

**Random Forest** achieved the lowest RMSE and highest R², though its margin over Linear Regression is small. Results apply to this dataset, feature set, parameters, and test period.

### Key data insights

- Mean annual rainfall: about 1411.01 mm (median 1121.30 mm, standard deviation 903.85 mm)
- July is the wettest month on average (347.21 mm), followed by August (290.26 mm) and June (230.23 mm)
- Highest long-term mean: Arunachal Pradesh (3418.86 mm); lowest: West Rajasthan (292.67 mm)

## Future-Year Prediction (2016)

The best model (Random Forest) is refitted on all supervised records and applied to each subdivision's latest complete history. Estimates for all 36 subdivisions average 1322.78 mm (minimum 331.87 mm, maximum 3419.47 mm).

| Subdivision | Predicted rainfall 2016 (mm) |
| --- | --- |
| Coastal Karnataka | 3419.5 |
| Konkan & Goa | 2874.7 |
| Andaman & Nicobar Islands | 2824.0 |
| Kerala | 2790.6 |
| Arunachal Pradesh | 2541.3 |

## Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Jupyter Notebook / VS Code

## Project Structure

```
.
├── rainfall_area-wt_sd_1901-2015.csv   # Input dataset
├── rainfall_prediction.py              # Main Python program
├── model_comparison.csv                # Output: model metrics
├── rainfall_predictions.csv            # Output: 2016 estimates
└── README.md
```

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

Place the dataset CSV in the same folder as the script (or update the path in the `read_csv` call), then run:

```bash
python rainfall_prediction.py
```

The program prints the evaluation metrics and saves:

- `model_comparison.csv`: MAE, MSE, RMSE, and R² for all five models
- `rainfall_predictions.csv`: 2016 rainfall estimates for each subdivision

## Limitations

- Historical data only (ends in 2015); no live observations
- Predictors are limited to annual rainfall history, target year, and subdivision
- Subdivision-level data cannot capture local variation
- Evaluated on a single chronological holdout; hyperparameters are fixed, not tuned
- A high R² does not guarantee small errors for every subdivision or year

## Future Scope

- Add temperature, humidity, pressure, wind, and ocean-atmosphere indicators
- Use rolling-origin or expanding-window validation
- Tune hyperparameters with time-series-aware validation
- Compare against statistical time-series baselines
- Estimate prediction intervals
- Build a dashboard for exploring data and predictions

## Author

**Krishna Saketh Naidu**
School of Computer Science and Engineering, Lovely Professional University, Phagwara, Punjab, India
