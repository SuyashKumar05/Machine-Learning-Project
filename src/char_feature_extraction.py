
import pandas as pd
import os
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

# 1. Load cleaned dataset
df = pd.read_csv("outputs/cleaned_sentiment_dataset.csv")

# 2. Remove missing text and sentiment
df = df.dropna(subset=["text", "sentiment"])

# 3. Simple text preprocessing
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["text"] = df["text"].apply(clean_text)

# 4. Separate text and sentiment
X = df["text"]
y = df["sentiment"]

# 5. Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 6. Create character-level TF-IDF
vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=20000,
    min_df=2
)

# 7. Fit vectorizer on training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# 8. Transform testing data
X_test_tfidf = vectorizer.transform(X_test)

# 9. Create output folders
os.makedirs("outputs/features", exist_ok=True)

# 10. Save feature matrices
save_npz("outputs/features/X_train_char_tfidf.npz", X_train_tfidf)
save_npz("outputs/features/X_test_char_tfidf.npz", X_test_tfidf)

# 11. Save target labels
y_train.to_csv("outputs/features/y_train_char.csv", index=False)
y_test.to_csv("outputs/features/y_test_char.csv", index=False)

# 12. Save vectorizer
joblib.dump(
    vectorizer,
    "outputs/features/char_tfidf_vectorizer.pkl"
)

# 13. Display results
print("Character-level TF-IDF completed successfully!")

print("\nOriginal dataset shape:", df.shape)
print("Training data shape:", X_train_tfidf.shape)
print("Testing data shape:", X_test_tfidf.shape)

print("\nNumber of character features:", len(vectorizer.get_feature_names_out()))

print("\nSaved files successfully in outputs/features/")
