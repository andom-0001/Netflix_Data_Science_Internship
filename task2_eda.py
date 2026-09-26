# ============================================================
# AUSPIFY TECHNOLOGIES - DATA SCIENCE INTERNSHIP
# TASK 2 - EXPLORATORY DATA ANALYSIS (EDA)
# Dataset: Netflix Titles
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------

input_file = Path("output/netflix_cleaned.csv")

output_folder = Path("output/task2")

output_folder.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD CLEANED DATASET
# ------------------------------------------------------------

print("=" * 60)
print("STEP 1: LOADING CLEANED DATASET")
print("=" * 60)

df = pd.read_csv(input_file)

print("Dataset loaded successfully.")

print("\nRows:", df.shape[0])
print("Columns:", df.shape[1])


# ------------------------------------------------------------
# 3. DATASET STRUCTURE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 2: DATASET STRUCTURE")
print("=" * 60)

print("\nColumn names:")

for column in df.columns:
    print("-", column)


print("\nDataset information:")

print(df.info())


# ------------------------------------------------------------
# 4. STATISTICAL ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 3: STATISTICAL ANALYSIS")
print("=" * 60)

print("\nNumerical statistics:")

print(df.describe())


# ------------------------------------------------------------
# 5. CONTENT TYPE DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 4: CONTENT TYPE DISTRIBUTION")
print("=" * 60)

type_counts = df["type"].value_counts()

print("\nNumber of titles by type:")

print(type_counts)


# Calculate percentages

type_percentage = df["type"].value_counts(normalize=True) * 100

print("\nPercentage distribution:")

print(type_percentage.round(2))


# ------------------------------------------------------------
# 6. VISUALIZATION - CONTENT TYPE
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="type"
)

plt.title("Netflix Content Distribution by Type")

plt.xlabel("Content Type")

plt.ylabel("Number of Titles")

plt.tight_layout()

plt.savefig(
    output_folder / "01_content_type_distribution.png",
    dpi=300
)

plt.show()

plt.close()


# ------------------------------------------------------------
# 7. RELEASE YEAR ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 5: RELEASE YEAR ANALYSIS")
print("=" * 60)

release_year_counts = (
    df["release_year"]
    .value_counts()
    .sort_index()
)

print("\nTitles by release year:")

print(release_year_counts)


# ------------------------------------------------------------
# 8. VISUALIZATION - RELEASE YEAR
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    release_year_counts.index,
    release_year_counts.values
)

plt.title("Netflix Titles by Release Year")

plt.xlabel("Release Year")

plt.ylabel("Number of Titles")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    output_folder / "02_release_year_trend.png",
    dpi=300
)

plt.show()

plt.close()


# ------------------------------------------------------------
# 9. TOP COUNTRIES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 6: TOP COUNTRIES")
print("=" * 60)


# Some records contain multiple countries.
# We use the first listed country for this analysis.

country_data = (
    df["country"]
    .dropna()
    .astype(str)
    .str.split(",")
    .str[0]
    .str.strip()
)

country_counts = country_data.value_counts().head(10)


print("\nTop 10 countries:")

print(country_counts)


# ------------------------------------------------------------
# 10. VISUALIZATION - TOP COUNTRIES
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.barplot(
    x=country_counts.values,
    y=country_counts.index
)

plt.title("Top 10 Countries by Number of Netflix Titles")

plt.xlabel("Number of Titles")

plt.ylabel("Country")

plt.tight_layout()

plt.savefig(
    output_folder / "03_top_10_countries.png",
    dpi=300
)

plt.show()

plt.close()


# ------------------------------------------------------------
# 11. TOP CATEGORIES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 7: TOP CATEGORIES")
print("=" * 60)


# Some titles have multiple categories.
# We use the first listed category for this analysis.

category_data = (
    df["listed_in"]
    .dropna()
    .astype(str)
    .str.split(",")
    .str[0]
    .str.strip()
)

category_counts = category_data.value_counts().head(10)


print("\nTop 10 categories:")

print(category_counts)


# ------------------------------------------------------------
# 12. VISUALIZATION - TOP CATEGORIES
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.barplot(
    x=category_counts.values,
    y=category_counts.index
)

plt.title("Top 10 Netflix Categories")

plt.xlabel("Number of Titles")

plt.ylabel("Category")

plt.tight_layout()

plt.savefig(
    output_folder / "04_top_10_categories.png",
    dpi=300
)

plt.show()

plt.close()


# ------------------------------------------------------------
# 13. RATINGS ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 8: RATINGS ANALYSIS")
print("=" * 60)

rating_counts = (
    df["rating"]
    .value_counts()
    .head(10)
)

print("\nTop ratings:")

print(rating_counts)


# ------------------------------------------------------------
# 14. VISUALIZATION - RATINGS
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.barplot(
    x=rating_counts.values,
    y=rating_counts.index
)

plt.title("Netflix Titles by Rating")

plt.xlabel("Number of Titles")

plt.ylabel("Rating")

plt.tight_layout()

plt.savefig(
    output_folder / "05_ratings_distribution.png",
    dpi=300
)

plt.show()

plt.close()


# ------------------------------------------------------------
# 15. MOVIE VS TV SHOW BY YEAR
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 9: MOVIE VS TV SHOW TREND")
print("=" * 60)

type_year = (
    df.groupby(
        ["release_year", "type"]
    )
    .size()
    .unstack(fill_value=0)
)


print(type_year.tail(10))


# ------------------------------------------------------------
# 16. VISUALIZATION - MOVIE VS TV SHOW TREND
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

if "Movie" in type_year.columns:

    plt.plot(
        type_year.index,
        type_year["Movie"],
        label="Movie"
    )


if "TV Show" in type_year.columns:

    plt.plot(
        type_year.index,
        type_year["TV Show"],
        label="TV Show"
    )


plt.title("Movies vs TV Shows by Release Year")

plt.xlabel("Release Year")

plt.ylabel("Number of Titles")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    output_folder / "06_movie_vs_tv_show_trend.png",
    dpi=300
)

plt.show()

plt.close()


# ------------------------------------------------------------
# 17. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("EDA SUMMARY")
print("=" * 60)

print("\nTotal titles:", len(df))

print("\nContent type distribution:")

print(type_counts)

print("\nTop 10 countries:")

print(country_counts)

print("\nTop 10 categories:")

print(category_counts)

print("\nTop ratings:")

print(rating_counts)


# ------------------------------------------------------------
# 18. SAVE ANALYSIS RESULTS
# ------------------------------------------------------------

summary = pd.DataFrame({
    "Metric": [
        "Total Titles",
        "Movies",
        "TV Shows",
        "Unique Countries",
        "Unique Ratings",
        "Earliest Release Year",
        "Latest Release Year"
    ],

    "Value": [
        len(df),
        (df["type"] == "Movie").sum(),
        (df["type"] == "TV Show").sum(),
        df["country"].nunique(),
        df["rating"].nunique(),
        df["release_year"].min(),
        df["release_year"].max()
    ]
})


summary.to_csv(
    output_folder / "eda_summary.csv",
    index=False
)


# ------------------------------------------------------------
# END
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TASK 2 COMPLETED")
print("=" * 60)

print("\nCharts saved in:")

print(output_folder)