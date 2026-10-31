
import pandas as pd
import os
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

# 1. Load cleaned dataset
df = pd.read_csv("outputs/cleaned_sentiment_dataset.csv")

print("Original dataset shape:", df.shape)

# 2. Remove any missing text or sentiment records
df = df.dropna(subset=["text", "sentiment"])

# 3. Basic text preprocessing
def preprocess_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["processed_text"] = df["text"].apply(preprocess_text)

# 4. Separate input and output
X = df["processed_text"]
y = df["sentiment"]

# 5. Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 6. Apply TF-IDF
tfidf = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 7. Display TF-IDF information
print("\nTF-IDF Feature Extraction Completed!")
print("Training matrix shape:", X_train_tfidf.shape)
print("Testing matrix shape:", X_test_tfidf.shape)
print("Total vocabulary features:", len(tfidf.vocabulary_))

# 8. Save extracted features
os.makedirs("outputs/features", exist_ok=True)

save_npz(
    "outputs/features/X_train_tfidf.npz",
    X_train_tfidf
)

save_npz(
    "outputs/features/X_test_tfidf.npz",
    X_test_tfidf
)

y_train.to_csv("outputs/features/y_train.csv", index=False)
y_test.to_csv("outputs/features/y_test.csv", index=False)

joblib.dump(
    tfidf,
    "outputs/features/tfidf_vectorizer.pkl"
)

print("\nAll TF-IDF files saved successfully!")
