import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# 1. LOAD DATASET

df = pd.read_csv("C:/Users/krish/Downloads/rainfall_area-wt_sd_1901-2015.csv")

print("\nFIRST FIVE ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nDATASET INFORMATION")
df.info()

print("\nSTATISTICAL SUMMARY")
print(df.describe())

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE RECORDS")
print(df.duplicated().sum())

df = df.drop_duplicates()


# 2. EXPLORATORY DATA ANALYSIS

plt.figure(figsize=(10, 5))
sns.histplot(df["ANNUAL"].dropna(), bins=30, kde=True)
plt.title("Annual Rainfall Distribution")
plt.xlabel("Annual Rainfall")
plt.show()


plt.figure(figsize=(12, 5))
sns.lineplot(data=df, x="YEAR", y="ANNUAL", errorbar=None)
plt.title("Annual Rainfall Over Years")
plt.xlabel("Year")
plt.ylabel("Rainfall")
plt.show()


plt.figure(figsize=(10, 5))
sns.boxplot(x=df["ANNUAL"])
plt.title("Annual Rainfall Boxplot")
plt.show()


monthly_columns = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"
]

monthly_average = df[monthly_columns].mean()

plt.figure(figsize=(12, 5))
sns.barplot(x=monthly_average.index, y=monthly_average.values)
plt.title("Average Monthly Rainfall")
plt.xlabel("Month")
plt.ylabel("Rainfall")
plt.show()


plt.figure(figsize=(12, 7))
sns.heatmap(
    df[monthly_columns + ["ANNUAL"]].corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Rainfall Correlation Heatmap")
plt.show()


subdivision_average = df.groupby(
    "SUBDIVISION"
)["ANNUAL"].mean().sort_values(ascending=False)

plt.figure(figsize=(13, 6))
subdivision_average.plot(kind="bar")
plt.title("Average Rainfall by Subdivision")
plt.xlabel("Subdivision")
plt.ylabel("Average Rainfall")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


print("\nEDA COMPLETED")


# 3. DATA PREPROCESSING AND FEATURE ENGINEERING

df = df.sort_values(
    ["SUBDIVISION", "YEAR"]
).copy()

grouped = df.groupby("SUBDIVISION")

df["PREVIOUS_RAINFALL"] = df["ANNUAL"]

df["RAINFALL_2_YEARS_AGO"] = grouped["ANNUAL"].shift(1)

df["THREE_YEAR_AVERAGE"] = grouped["ANNUAL"].transform(
    lambda x: x.rolling(window=3).mean()
)

df["TARGET_RAINFALL"] = grouped["ANNUAL"].shift(-1)

df["TARGET_YEAR"] = grouped["YEAR"].shift(-1)


features = [
    "SUBDIVISION",
    "TARGET_YEAR",
    "PREVIOUS_RAINFALL",
    "RAINFALL_2_YEARS_AGO",
    "THREE_YEAR_AVERAGE"
]

target = "TARGET_RAINFALL"

data = df[features + [target]].dropna().copy()

X = data[features]
y = data[target]


# 4. DATA PREPROCESSING PIPELINE

numeric_features = [
    "TARGET_YEAR",
    "PREVIOUS_RAINFALL",
    "RAINFALL_2_YEARS_AGO",
    "THREE_YEAR_AVERAGE"
]

categorical_features = ["SUBDIVISION"]


numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# 5. TRAIN AND TEST SPLIT (CHRONOLOGICAL)

years = sorted(data["TARGET_YEAR"].unique())

split_index = int(len(years) * 0.8)

split_year = years[split_index]

train_data = data[
    data["TARGET_YEAR"] < split_year
]

test_data = data[
    data["TARGET_YEAR"] >= split_year
]

X_train = train_data[features]
X_test = test_data[features]

y_train = train_data[target]
y_test = test_data[target]

print("\nTRAINING RECORDS:", len(X_train))
print("TESTING RECORDS:", len(X_test))
print("TESTING START YEAR:", split_year)


# 6. FIVE MACHINE LEARNING ALGORITHMS

models = {

    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=8,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),

    "KNN": KNeighborsRegressor(
        n_neighbors=5
    )
}


# 7. MODEL TRAINING AND EVALUATION

results = []
trained_models = {}

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    mse = mean_squared_error(y_test, predictions)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[name] = pipeline

    print("\nMODEL:", name)
    print("MAE:", round(mae, 2))
    print("MSE:", round(mse, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 SCORE:", round(r2, 4))


# 8. MODEL COMPARISON

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="RMSE",
    ascending=True
)

print("\nMODEL PERFORMANCE COMPARISON")
print(results_df.to_string(index=False))


# 9. RMSE COMPARISON GRAPH

plt.figure(figsize=(10, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="RMSE"
)

plt.title("RMSE Comparison of Five ML Models")
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()


# 10. R2 SCORE COMPARISON GRAPH

plt.figure(figsize=(10, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="R2 Score"
)

plt.title("R2 Score Comparison")
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()


# 11. SELECT BEST MODEL

best_model_name = results_df.iloc[0]["Model"]

print("\nBEST PERFORMING MODEL:", best_model_name)


# 12. TRAIN BEST MODEL ON COMPLETE DATASET

best_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", models[best_model_name])
])

best_model.fit(X, y)


# 13. NEXT YEAR RAINFALL PREDICTION

future_data = df.dropna(
    subset=[
        "PREVIOUS_RAINFALL",
        "RAINFALL_2_YEARS_AGO",
        "THREE_YEAR_AVERAGE"
    ]
).copy()

future_data = future_data.sort_values(
    ["SUBDIVISION", "YEAR"]
)

future_data = future_data.groupby(
    "SUBDIVISION"
).tail(1)


future_X = pd.DataFrame({

    "SUBDIVISION": future_data["SUBDIVISION"],

    "TARGET_YEAR": future_data["YEAR"] + 1,

    "PREVIOUS_RAINFALL": future_data["PREVIOUS_RAINFALL"],

    "RAINFALL_2_YEARS_AGO": future_data["RAINFALL_2_YEARS_AGO"],

    "THREE_YEAR_AVERAGE": future_data["THREE_YEAR_AVERAGE"]

})


future_predictions = best_model.predict(future_X)


prediction_results = future_X[
    ["SUBDIVISION", "TARGET_YEAR"]
].copy()

prediction_results["PREDICTED_RAINFALL"] = future_predictions


print("\nNEXT YEAR RAINFALL PREDICTION")

print(prediction_results.to_string(index=False))


# 14. SAVE RESULTS

results_df.to_csv(
    "model_comparison.csv",
    index=False
)

prediction_results.to_csv(
    "rainfall_predictions.csv",
    index=False
)


print("\nPROJECT COMPLETED SUCCESSFULLY!")

print("Model Comparison Saved Successfully")

print("Rainfall Predictions Saved Successfully")
