# ============================================================
# AUSPIFY TECHNOLOGIES - DATA SCIENCE INTERNSHIP
# TASK 1 - DATA CLEANING & PREPROCESSING
# Dataset: Netflix Titles
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path


# ------------------------------------------------------------
# 1. DEFINE FILE PATHS
# ------------------------------------------------------------

input_file = Path("dataset/Dataset.csv")
output_folder = Path("output")

# Create output folder if it does not exist
output_folder.mkdir(exist_ok=True)

output_file = output_folder / "netflix_cleaned.csv"


# ------------------------------------------------------------
# 2. IMPORT DATASET
# ------------------------------------------------------------

print("=" * 60)
print("STEP 1: LOADING DATASET")
print("=" * 60)

df = pd.read_csv(input_file)

print("Dataset loaded successfully.")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumn names:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 3. INSPECT DATASET
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 2: DATASET INSPECTION")
print("=" * 60)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nStatistical summary:")
print(df.describe(include="all"))


# ------------------------------------------------------------
# 4. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 3: MISSING VALUE ANALYSIS")
print("=" * 60)

missing_values = df.isnull().sum()

print("\nMissing values in each column:")
print(missing_values)

print("\nTotal missing values:")
print(missing_values.sum())


# ------------------------------------------------------------
# 5. CHECK DUPLICATE RECORDS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 4: DUPLICATE RECORD ANALYSIS")
print("=" * 60)

duplicate_count = df.duplicated().sum()

print("Duplicate rows found:", duplicate_count)


# ------------------------------------------------------------
# 6. REMOVE DUPLICATE RECORDS
# ------------------------------------------------------------

if duplicate_count > 0:

    df = df.drop_duplicates()

    print("Duplicate rows removed.")

else:

    print("No duplicate rows found.")


# ------------------------------------------------------------
# 7. CLEAN TEXT COLUMNS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 5: TEXT DATA CLEANING")
print("=" * 60)

text_columns = df.select_dtypes(include=["object"]).columns

for column in text_columns:

    # Convert values to string while preserving missing values
    df[column] = df[column].astype("string").str.strip()

    # Replace empty strings with missing values
    df[column] = df[column].replace("", pd.NA)

print("Text columns cleaned.")


# ------------------------------------------------------------
# 8. HANDLE MISSING VALUES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 6: HANDLING MISSING VALUES")
print("=" * 60)

# Columns where missing values can reasonably be represented
# as "Not Available"
categorical_columns = [
    "director",
    "cast",
    "country",
    "rating"
]

for column in categorical_columns:

    if column in df.columns:
        df[column] = df[column].fillna("Not Available")


# Description is also text.
if "description" in df.columns:
    df["description"] = df["description"].fillna("Not Available")


# date_added is converted to datetime below, so we do not
# insert an artificial date for missing values.


print("Categorical/text missing values handled.")


# ------------------------------------------------------------
# 9. CONVERT DATE COLUMN
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 7: DATE TRANSFORMATION")
print("=" * 60)

if "date_added" in df.columns:

    df["date_added"] = pd.to_datetime(
        df["date_added"],
        errors="coerce"
    )

    print("date_added converted to datetime format.")


# ------------------------------------------------------------
# 10. CREATE DATE FEATURES
# ------------------------------------------------------------

if "date_added" in df.columns:

    df["date_added_year"] = df["date_added"].dt.year
    df["date_added_month"] = df["date_added"].dt.month
    df["date_added_month_name"] = df["date_added"].dt.month_name()

    print("Created:")
    print("- date_added_year")
    print("- date_added_month")
    print("- date_added_month_name")


# ------------------------------------------------------------
# 11. CLEAN AND TRANSFORM DURATION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 8: DURATION TRANSFORMATION")
print("=" * 60)

if "duration" in df.columns:

    # Extract numerical duration
    df["duration_value"] = pd.to_numeric(
        df["duration"]
        .astype("string")
        .str.extract(r"(\d+)")[0],
        errors="coerce"
    )

    # Extract duration unit
    df["duration_unit"] = (
        df["duration"]
        .astype("string")
        .str.extract(r"(\D+)$")[0]
        .str.strip()
    )

    print("Duration transformed into:")
    print("- duration_value")
    print("- duration_unit")


# ------------------------------------------------------------
# 12. STANDARDIZE TYPE COLUMN
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 9: CATEGORICAL DATA TRANSFORMATION")
print("=" * 60)

if "type" in df.columns:

    df["type"] = df["type"].str.title()

    print("Content type standardized.")


# ------------------------------------------------------------
# 13. CHECK RELEASE YEAR
# ------------------------------------------------------------

if "release_year" in df.columns:

    df["release_year"] = pd.to_numeric(
        df["release_year"],
        errors="coerce"
    )

    print("release_year converted to numeric format.")


# ------------------------------------------------------------
# 14. FINAL DATA QUALITY CHECK
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 10: FINAL DATA QUALITY CHECK")
print("=" * 60)

print("\nFinal dataset shape:")
print(df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nDuplicate rows after cleaning:")
print(df.duplicated().sum())


# ------------------------------------------------------------
# 15. SAVE CLEANED DATASET
# ------------------------------------------------------------

df.to_csv(output_file, index=False)

print("\n" + "=" * 60)
print("TASK 1 COMPLETED")
print("=" * 60)

print("Cleaned dataset saved successfully.")
print("File:", output_file)

print("\nFinal number of rows:", len(df))
print("Final number of columns:", len(df.columns))