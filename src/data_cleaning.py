
import pandas as pd

# Load dataset
df = pd.read_csv(
    "data/labeled_sentiment_output_codemixed.csv",
    encoding="utf-8-sig"
)

# Correct column headers
df.columns = df.iloc[0]
df = df.iloc[1:].reset_index(drop=True)

print("Dataset loaded:", df.shape)

# Remove repeated header rows
df = df[df["sentiment"] != "sentiment"]

# Keep only important columns
df = df[["text", "sentiment", "translated_text", "cleaned_text"]]

# Remove extra spaces
for column in df.columns:
    df[column] = df[column].apply(
        lambda x: x.strip() if isinstance(x, str) else x
    )

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check empty original text
print("\nEmpty original text:",
      (df["text"].isna() | (df["text"] == "")).sum())

# Check exact duplicate records
print("\nExact duplicate rows:", df.duplicated().sum())

# Check duplicate original texts
print("\nDuplicate original texts:",
      df.duplicated(subset=["text"]).sum())

# Check whether identical texts have conflicting labels
label_counts = df.groupby("text")["sentiment"].nunique()

print("\nTexts with conflicting sentiment labels:",
      (label_counts > 1).sum())

# Check sentiment distribution
print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

print("\nCleaning inspection completed!")
