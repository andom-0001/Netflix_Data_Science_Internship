# ============================================================
# AUSPIFY TECHNOLOGIES - DATA SCIENCE INTERNSHIP
# TASK 3 - RECOMMENDATION SYSTEM ANALYSIS
# Dataset: Netflix Titles
# ============================================================

import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------

input_file = Path("output/netflix_cleaned.csv")

output_folder = Path("output/task3")

output_folder.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

print("=" * 60)
print("STEP 1: LOADING DATASET")
print("=" * 60)

df = pd.read_csv(input_file)

print("Dataset loaded successfully.")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ------------------------------------------------------------
# 3. SELECT RELEVANT FEATURES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 2: SELECTING CONTENT FEATURES")
print("=" * 60)

# Features that describe the Netflix content
feature_columns = [
    "type",
    "director",
    "cast",
    "country",
    "rating",
    "listed_in",
    "description"
]

# Keep only columns that actually exist
feature_columns = [
    column for column in feature_columns
    if column in df.columns
]

print("Features used:")

for column in feature_columns:
    print("-", column)


# ------------------------------------------------------------
# 4. HANDLE MISSING VALUES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 3: PREPARING TEXT DATA")
print("=" * 60)

for column in feature_columns:

    df[column] = (
        df[column]
        .fillna("")
        .astype(str)
    )


# ------------------------------------------------------------
# 5. CREATE COMBINED CONTENT FEATURE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 4: CREATING CONTENT FEATURES")
print("=" * 60)

df["content_features"] = (
    df[feature_columns]
    .agg(" ".join, axis=1)
)

print("Combined content feature created.")


# ------------------------------------------------------------
# 6. TEXT PREPROCESSING
# ------------------------------------------------------------

df["content_features"] = (
    df["content_features"]
    .str.lower()
    .str.replace(r"[^a-zA-Z0-9\s]", " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

print("Text preprocessing completed.")


# ------------------------------------------------------------
# 7. TF-IDF VECTORIZATION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 5: TF-IDF FEATURE EXTRACTION")
print("=" * 60)

tfidf = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = tfidf.fit_transform(
    df["content_features"]
)

print("TF-IDF matrix created.")

print("Matrix shape:", tfidf_matrix.shape)


# ------------------------------------------------------------
# 8. COSINE SIMILARITY
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 6: CALCULATING CONTENT SIMILARITY")
print("=" * 60)

cosine_sim = cosine_similarity(tfidf_matrix)

print("Cosine similarity calculated.")

print("Similarity matrix shape:", cosine_sim.shape)


# ------------------------------------------------------------
# 9. CREATE TITLE INDEX
# ------------------------------------------------------------

indices = pd.Series(
    df.index,
    index=df["title"].str.lower()
).drop_duplicates()


# ------------------------------------------------------------
# 10. RECOMMENDATION FUNCTION
# ------------------------------------------------------------

def recommend_titles(title, number_of_recommendations=10):

    title_key = title.lower().strip()

    # Check if title exists
    if title_key not in indices:

        print("\nTitle not found in dataset.")

        return pd.DataFrame()

    # Get index of selected title
    idx = indices[title_key]

    # Get similarity scores
    similarity_scores = list(
        enumerate(cosine_sim[idx])
    )

    # Sort by similarity
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Remove the selected title itself
    similarity_scores = [
        item
        for item in similarity_scores
        if item[0] != idx
    ]

    # Select top recommendations
    top_scores = similarity_scores[
        :number_of_recommendations
    ]

    recommendation_indices = [
        item[0]
        for item in top_scores
    ]

    recommendation_scores = [
        item[1]
        for item in top_scores
    ]

    recommendations = df.loc[
        recommendation_indices,
        ["title", "type", "listed_in", "rating"]
    ].copy()

    recommendations["similarity_score"] = (
        recommendation_scores
    )

    recommendations["similarity_percentage"] = (
        recommendations["similarity_score"] * 100
    ).round(2)

    return recommendations.reset_index(drop=True)


# ------------------------------------------------------------
# 11. TEST RECOMMENDATION SYSTEM
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 7: GENERATING RECOMMENDATIONS")
print("=" * 60)

# Select a title automatically from the dataset
sample_title = df["title"].dropna().iloc[0]

print("\nSelected title:")

print(sample_title)


recommendations = recommend_titles(
    sample_title,
    10
)


print("\nRecommended titles:")

if not recommendations.empty:

    print(
        recommendations.to_string(index=False)
    )

else:

    print("No recommendations generated.")


# ------------------------------------------------------------
# 12. SAVE RECOMMENDATIONS
# ------------------------------------------------------------

if not recommendations.empty:

    output_file = (
        output_folder /
        "recommendations.csv"
    )

    recommendations.to_csv(
        output_file,
        index=False
    )

    print("\nRecommendations saved to:")

    print(output_file)


# ------------------------------------------------------------
# 13. CREATE EVALUATION SAMPLE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 8: RECOMMENDATION QUALITY ANALYSIS")
print("=" * 60)

if not recommendations.empty:

    average_similarity = (
        recommendations["similarity_percentage"]
        .mean()
    )

    print(
        "Average similarity of top recommendations:",
        round(average_similarity, 2),
        "%"
    )

    print(
        "\nHighest similarity:",
        round(
            recommendations[
                "similarity_percentage"
            ].max(),
            2
        ),
        "%"
    )

    print(
        "Lowest similarity:",
        round(
            recommendations[
                "similarity_percentage"
            ].min(),
            2
        ),
        "%"
    )


# ------------------------------------------------------------
# 14. SAVE MODEL INFORMATION
# ------------------------------------------------------------

model_info = pd.DataFrame({
    "Parameter": [
        "Feature extraction",
        "Similarity method",
        "Number of titles",
        "Number of TF-IDF features"
    ],

    "Value": [
        "TF-IDF",
        "Cosine Similarity",
        len(df),
        tfidf_matrix.shape[1]
    ]
})

model_info.to_csv(
    output_folder / "model_information.csv",
    index=False
)


# ------------------------------------------------------------
# 15. FINAL MESSAGE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TASK 3 COMPLETED")
print("=" * 60)

print(
    "\nThe Netflix content recommendation system "
    "was successfully created."
)