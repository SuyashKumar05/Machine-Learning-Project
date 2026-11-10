
import pandas as pd
import os
import re
import joblib

from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("outputs/cleaned_sentiment_dataset.csv")

df = df.dropna(subset=["text", "sentiment"])

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["text"] = df["text"].apply(clean_text)

X = df["text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_test_char = load_npz("outputs/features/X_test_char_tfidf.npz")
y_test_saved = pd.read_csv("outputs/features/y_test_char.csv").iloc[:, 0]

vectorizer = joblib.load("outputs/features/char_tfidf_vectorizer.pkl")
model = joblib.load("outputs/tuned_models/svm_char_tuned.pkl")

if len(X_test) != X_test_char.shape[0]:
    raise ValueError("Test text and feature matrix sizes do not match.")

if y_test.reset_index(drop=True).astype(str).tolist() != y_test_saved.astype(str).tolist():
    raise ValueError("Test labels do not match the saved test labels.")

if X_test_char.shape[1] != len(vectorizer.get_feature_names_out()):
    raise ValueError("Feature matrix and vectorizer do not match.")

predictions = model.predict(X_test_char)

results = pd.DataFrame({
    "text": X_test.tolist(),
    "actual_sentiment": y_test.tolist(),
    "predicted_sentiment": predictions
})

results["is_correct"] = (
    results["actual_sentiment"] == results["predicted_sentiment"]
)

misclassified = results[results["is_correct"] == False]

os.makedirs("outputs/evaluation", exist_ok=True)

results.to_csv(
    "outputs/evaluation/character_svm_error_analysis.csv",
    index=False
)

misclassified.to_csv(
    "outputs/evaluation/character_svm_misclassified.csv",
    index=False
)

report = classification_report(
    y_test,
    predictions,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

report_df.to_csv(
    "outputs/evaluation/character_svm_error_summary.csv"
)

cm = confusion_matrix(
    y_test,
    predictions,
    labels=model.classes_
)

print("\nCharacter SVM Error Analysis Completed!")

print("\nTotal test records:", len(results))
print("Correct predictions:", results["is_correct"].sum())
print("Incorrect predictions:", len(misclassified))

print("\nAccuracy:", round(results["is_correct"].mean() * 100, 2), "%")

print("\nConfusion Matrix:")
print(pd.DataFrame(
    cm,
    index=model.classes_,
    columns=model.classes_
))

print("\nMisclassified records by actual sentiment:")
print(misclassified["actual_sentiment"].value_counts())

print("\nFiles saved in outputs/evaluation/")
