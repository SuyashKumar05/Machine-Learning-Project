import pandas as pd
import os

# 1. Load dataset
df = pd.read_csv(
    "data/labeled_sentiment_output_codemixed.csv",
    encoding="utf-8-sig"
)

# 2. Correct column headers
df.columns = df.iloc[0]
df = df.iloc[1:].reset_index(drop=True)

# 3. Remove repeated header rows
df = df[df["sentiment"] != "sentiment"].copy()

# 4. Keep important columns
df = df[["text", "sentiment", "translated_text", "cleaned_text"]]

# 5. Remove extra spaces
for column in df.columns:
    df[column] = df[column].apply(
        lambda x: x.strip() if isinstance(x, str) else x
    )

# 6. Remove empty original text
before = len(df)

df = df.dropna(subset=["text"])
df = df[df["text"] != ""]

print("Empty text records removed:", before - len(df))

# 7. Find conflicting sentiment labels
label_counts = df.groupby("text")["sentiment"].nunique()

conflicting_texts = label_counts[label_counts > 1].index

print("Conflicting text groups:", len(conflicting_texts))

# 8. Remove all records belonging to conflicting text groups
before = len(df)

df = df[~df["text"].isin(conflicting_texts)].copy()

print("Records removed due to conflicting labels:", before - len(df))

# 9. Remove duplicate original texts
before = len(df)

df = df.drop_duplicates(subset=["text"], keep="first")

print("Duplicate text records removed:", before - len(df))

# 10. Check final dataset
print("\nFinal dataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nFinal sentiment distribution:")
print(df["sentiment"].value_counts())

# 11. Save cleaned dataset
os.makedirs("outputs", exist_ok=True)

df.to_csv(
    "outputs/cleaned_sentiment_dataset.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nCleaned dataset saved successfully!")