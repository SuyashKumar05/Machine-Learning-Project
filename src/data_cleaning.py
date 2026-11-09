import pandas as pd
import os


df = pd.read_csv(
    "data/labeled_sentiment_output_codemixed.csv",
    encoding="utf-8-sig"
)

#column headers
df.columns = df.iloc[0]
df = df.iloc[1:].reset_index(drop=True)

#Remove header rows
df = df[df["sentiment"] != "sentiment"].copy()

print("Records after removing repeated headers:", len(df))


df = df[["text", "sentiment", "translated_text", "cleaned_text"]]

print("Records after selecting columns:", len(df))

#Remove extra spaces
for column in df.columns:
    df[column] = df[column].apply(
        lambda x: x.strip() if isinstance(x, str) else x
    )

print("Records after stripping spaces:", len(df))

#remove empty original text
before = len(df)

df = df[df["text"].notna()]
df = df[df["text"].str.strip() != ""]

print("Empty text records removed:", before - len(df))

#conflicting sentiment labels
label_counts = df.groupby("text")["sentiment"].nunique()

conflicting_texts = label_counts[label_counts > 1].index

print("Conflicting text groups:", len(conflicting_texts))

#remove all records conflicting text groups
before = len(df)

df = df[~df["text"].isin(conflicting_texts)].copy()

print("Records removed due to conflicting labels:", before - len(df))

#rem duplicate original texts
before = len(df)

df = df.drop_duplicates(subset=["text"], keep="first")

print("Duplicate text records removed:", before - len(df))

#final dataset
print("\nFinal dataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nFinal sentiment distribution:")
print(df["sentiment"].value_counts())

#cleaned dataset
os.makedirs("outputs", exist_ok=True)

df.to_csv(
    "outputs/cleaned_sentiment_dataset.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nCleaned dataset saved successfully!")