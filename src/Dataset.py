
import pandas as pd

# Load the original dataset
file_path = "data/labeled_sentiment_output_codemixed.csv"

df = pd.read_csv(file_path, encoding="utf-8-sig")

print("Original dataset shape:", df.shape)

# Correct the column names
df.columns = df.iloc[0]

# Remove the first row because it contains column names
df = df.iloc[1:].reset_index(drop=True)

print("\nCorrected dataset shape:", df.shape)

print("\nCorrect column names:")
print(df.columns.tolist())

print("\nFirst five records:")
print(df.head().to_string())

print("\nData types:")
print(df.dtypes)

print("\nMissing values in important columns:")
print(df[["text", "sentiment", "translated_text",
          "cleaned_text"]].isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# Sentiment distribution
print("\nSentiment distribution:")
print(df["sentiment"].value_counts(dropna=False))
