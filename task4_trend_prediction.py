# ============================================================
# AUSPIFY TECHNOLOGIES - DATA SCIENCE INTERNSHIP
# TASK 4 - TREND PREDICTION ANALYSIS
# Dataset: Netflix Titles
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# ============================================================
# 1. FILE PATHS
# ============================================================

input_file = Path("output/netflix_cleaned.csv")

output_folder = Path("output/task4")

output_folder.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 70)
print("STEP 1: LOADING CLEANED DATASET")
print("=" * 70)

df = pd.read_csv(input_file)

print("\nDataset loaded successfully.")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ============================================================
# 3. CHECK REQUIRED COLUMN
# ============================================================

if "release_year" not in df.columns:

    raise ValueError(
        "The dataset does not contain the 'release_year' column."
    )


# ============================================================
# 4. PREPARE RELEASE YEAR DATA
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: PREPARING RELEASE-YEAR DATA")
print("=" * 70)


# Convert release_year to numeric

df["release_year"] = pd.to_numeric(
    df["release_year"],
    errors="coerce"
)


# Remove rows where release year is missing

df = df.dropna(
    subset=["release_year"]
).copy()


# Convert year to integer

df["release_year"] = df[
    "release_year"
].astype(int)


print("\nRelease year range:")

print(
    df["release_year"].min(),
    "to",
    df["release_year"].max()
)


# ============================================================
# 5. COUNT TITLES BY YEAR
# ============================================================

yearly_content = (
    df.groupby("release_year")
    .size()
    .reset_index(name="title_count")
)


# Sort by year

yearly_content = yearly_content.sort_values(
    "release_year"
).reset_index(drop=True)


print("\nYearly content data:")

print(
    yearly_content.head(10)
)


print("\nLatest years:")

print(
    yearly_content.tail(10)
)


# Save yearly data

yearly_content.to_csv(
    output_folder / "yearly_content_trend.csv",
    index=False
)


# ============================================================
# 6. HISTORICAL TREND VISUALIZATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 3: VISUALIZING HISTORICAL TREND")
print("=" * 70)


plt.figure(
    figsize=(12, 6)
)


sns.lineplot(
    data=yearly_content,
    x="release_year",
    y="title_count",
    marker="o"
)


plt.title(
    "Netflix Content Release Trend by Year"
)


plt.xlabel(
    "Release Year"
)


plt.ylabel(
    "Number of Titles"
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    output_folder /
    "01_historical_release_trend.png",
    dpi=300
)


plt.show()

plt.close()


# ============================================================
# 7. CREATE TRAINING DATA
# ============================================================

print("\n" + "=" * 70)
print("STEP 4: PREPARING DATA FOR FORECASTING")
print("=" * 70)


X = yearly_content[
    ["release_year"]
]

y = yearly_content[
    "title_count"
]


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

# Use the last 20% of chronological observations
# as the test set.

split_index = int(
    len(yearly_content) * 0.80
)


X_train = X.iloc[
    :split_index
]


X_test = X.iloc[
    split_index:
]


y_train = y.iloc[
    :split_index
]


y_test = y.iloc[
    split_index:
]


print("\nTraining observations:")

print(
    len(X_train)
)


print("\nTesting observations:")

print(
    len(X_test)
)


print("\nTraining years:")

print(
    X_train["release_year"].min(),
    "to",
    X_train["release_year"].max()
)


print("\nTesting years:")

print(
    X_test["release_year"].min(),
    "to",
    X_test["release_year"].max()
)


# ============================================================
# 9. BUILD LINEAR REGRESSION MODEL
# ============================================================

print("\n" + "=" * 70)
print("STEP 5: BUILDING FORECASTING MODEL")
print("=" * 70)


model = LinearRegression()


model.fit(
    X_train,
    y_train
)


print("\nLinear Regression model trained successfully.")


# ============================================================
# 10. PREDICT TEST DATA
# ============================================================

y_test_pred = model.predict(
    X_test
)


# Prevent negative predictions

y_test_pred = np.maximum(
    y_test_pred,
    0
)


# ============================================================
# 11. MODEL EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 6: MODEL EVALUATION")
print("=" * 70)


mae = mean_absolute_error(
    y_test,
    y_test_pred
)


rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_test_pred
    )
)


r2 = r2_score(
    y_test,
    y_test_pred
)


print("\nMean Absolute Error (MAE):")

print(
    round(mae, 2)
)


print("\nRoot Mean Squared Error (RMSE):")

print(
    round(rmse, 2)
)


print("\nR² Score:")

print(
    round(r2, 4)
)


# ============================================================
# 12. CREATE TEST RESULTS TABLE
# ============================================================

test_results = pd.DataFrame({

    "year":
        X_test["release_year"].values,

    "actual_titles":
        y_test.values,

    "predicted_titles":
        np.round(
            y_test_pred,
            2
        )

})


test_results.to_csv(
    output_folder /
    "test_predictions.csv",
    index=False
)


# ============================================================
# 13. FUTURE YEARS
# ============================================================

print("\n" + "=" * 70)
print("STEP 7: FORECASTING FUTURE CONTENT")
print("=" * 70)


last_year = int(
    yearly_content[
        "release_year"
    ].max()
)


future_years = np.arange(
    last_year + 1,
    last_year + 6
)


future_X = pd.DataFrame({

    "release_year":
        future_years

})


future_predictions = model.predict(
    future_X
)


# Prevent negative predictions

future_predictions = np.maximum(
    future_predictions,
    0
)


future_predictions = np.round(
    future_predictions
).astype(int)


future_forecast = pd.DataFrame({

    "year":
        future_years,

    "predicted_titles":
        future_predictions

})


print("\nFuture forecast:")

print(
    future_forecast
)


future_forecast.to_csv(
    output_folder /
    "future_forecast.csv",
    index=False
)


# ============================================================
# 14. COMBINE HISTORICAL + FUTURE DATA
# ============================================================

historical_plot = yearly_content[
    ["release_year", "title_count"]
].copy()


historical_plot = historical_plot.rename(
    columns={
        "release_year": "year",
        "title_count": "actual_titles"
    }
)


future_plot = future_forecast.copy()


# ============================================================
# 15. VISUALIZE FORECAST
# ============================================================

print("\n" + "=" * 70)
print("STEP 8: VISUALIZING FUTURE PREDICTIONS")
print("=" * 70)


plt.figure(
    figsize=(14, 7)
)


# Historical data

plt.plot(
    historical_plot["year"],
    historical_plot["actual_titles"],
    marker="o",
    label="Historical"
)


# Future predictions

plt.plot(
    future_plot["year"],
    future_plot["predicted_titles"],
    marker="o",
    linestyle="--",
    label="Forecast"
)


# Separate historical and future region

plt.axvline(
    x=last_year,
    linestyle=":",
    label="Forecast Start"
)


plt.title(
    "Netflix Content Trend and Future Forecast"
)


plt.xlabel(
    "Year"
)


plt.ylabel(
    "Number of Titles"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    output_folder /
    "02_future_content_forecast.png",
    dpi=300
)


plt.show()

plt.close()


# ============================================================
# 16. ACTUAL VS PREDICTED TEST DATA
# ============================================================

plt.figure(
    figsize=(12, 6)
)


plt.plot(
    X_test["release_year"],
    y_test,
    marker="o",
    label="Actual"
)


plt.plot(
    X_test["release_year"],
    y_test_pred,
    marker="o",
    linestyle="--",
    label="Predicted"
)


plt.title(
    "Actual vs Predicted Netflix Content"
)


plt.xlabel(
    "Release Year"
)


plt.ylabel(
    "Number of Titles"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    output_folder /
    "03_actual_vs_predicted.png",
    dpi=300
)


plt.show()

plt.close()


# ============================================================
# 17. MODEL INFORMATION
# ============================================================

model_information = pd.DataFrame({

    "Metric": [

        "Model",

        "Training observations",

        "Testing observations",

        "Training start year",

        "Training end year",

        "Testing start year",

        "Testing end year",

        "MAE",

        "RMSE",

        "R2 Score"

    ],

    "Value": [

        "Linear Regression",

        len(X_train),

        len(X_test),

        X_train["release_year"].min(),

        X_train["release_year"].max(),

        X_test["release_year"].min(),

        X_test["release_year"].max(),

        round(mae, 2),

        round(rmse, 2),

        round(r2, 4)

    ]

})


model_information.to_csv(
    output_folder /
    "model_evaluation.csv",
    index=False
)


# ============================================================
# 18. TREND INTERPRETATION
# ============================================================

slope = model.coef_[0]

intercept = model.intercept_


print("\n" + "=" * 70)
print("STEP 9: TREND INTERPRETATION")
print("=" * 70)


print(
    "\nModel slope:",
    round(slope, 2)
)


print(
    "\nModel intercept:",
    round(intercept, 2)
)


if slope > 0:

    print(
        "\nInterpretation:"
    )

    print(
        "The fitted linear trend is upward."
    )

elif slope < 0:

    print(
        "\nInterpretation:"
    )

    print(
        "The fitted linear trend is downward."
    )

else:

    print(
        "\nInterpretation:"
    )

    print(
        "The fitted linear trend is approximately flat."
    )


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TASK 4 COMPLETED")
print("=" * 70)


print(
    "\nHistorical trend data saved to:"
)

print(
    output_folder /
    "yearly_content_trend.csv"
)


print(
    "\nTest predictions saved to:"
)

print(
    output_folder /
    "test_predictions.csv"
)


print(
    "\nFuture forecast saved to:"
)

print(
    output_folder /
    "future_forecast.csv"
)


print(
    "\nModel evaluation saved to:"
)

print(
    output_folder /
    "model_evaluation.csv"
)


print(
    "\nAll visualizations saved to:"
)

print(
    output_folder
)