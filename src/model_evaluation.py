
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

#output folders
os.makedirs("outputs/evaluation", exist_ok=True)
os.makedirs("outputs/graphs/confusion_matrices", exist_ok=True)


y_test_word = pd.read_csv("outputs/features/y_test.csv").iloc[:, 0]
y_test_char = pd.read_csv("outputs/features/y_test_char.csv").iloc[:, 0]

# test feature matrices
from scipy.sparse import load_npz

X_test_word = load_npz("outputs/features/X_test_tfidf.npz")
X_test_char = load_npz("outputs/features/X_test_char_tfidf.npz")

#all saved models
models = {
    "Word NB": (
        "outputs/models/naive_bayes.pkl",
        X_test_word,
        y_test_word
    ),

    "Word Logistic Regression": (
        "outputs/models/logistic_regression.pkl",
        X_test_word,
        y_test_word
    ),

    "Word SVM": (
        "outputs/models/svm.pkl",
        X_test_word,
        y_test_word
    ),

    "Character NB": (
        "outputs/models/naive_bayes_char.pkl",
        X_test_char,
        y_test_char
    ),

    "Character Logistic Regression": (
        "outputs/models/logistic_regression_char.pkl",
        X_test_char,
        y_test_char
    ),

    "Character SVM": (
        "outputs/models/svm_char.pkl",
        X_test_char,
        y_test_char
    ),

    "Tuned Word NB": (
        "outputs/tuned_models/naive_bayes_tuned.pkl",
        X_test_word,
        y_test_word
    ),

    "Tuned Word Logistic Regression": (
        "outputs/tuned_models/logistic_regression_tuned.pkl",
        X_test_word,
        y_test_word
    ),

    "Tuned Word SVM": (
        "outputs/tuned_models/svm_tuned.pkl",
        X_test_word,
        y_test_word
    ),

    "Tuned Character Logistic Regression": (
        "outputs/tuned_models/logistic_regression_char_tuned.pkl",
        X_test_char,
        y_test_char
    ),

    "Tuned Character SVM": (
        "outputs/tuned_models/svm_char_tuned.pkl",
        X_test_char,
        y_test_char
    )
}

#evaluate each model
results = []

for model_name, (model_path, X_test, y_test) in models.items():

    print("\nEvaluating:", model_name)

    # Load saved model
    model = joblib.load(model_path)

    # Predict sentiment
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, average="macro", zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, average="macro", zero_division=0
    )

    macro_f1 = f1_score(
        y_test, y_pred, average="macro", zero_division=0
    )

    weighted_f1 = f1_score(
        y_test, y_pred, average="weighted", zero_division=0
    )

    results.append({
        "Model": model_name,
        "Accuracy": round(accuracy * 100, 2),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "Macro F1": round(macro_f1, 4),
        "Weighted F1": round(weighted_f1, 4)
    })

  
    cm = confusion_matrix(
        y_test, y_pred,
        labels=["Negative", "Neutral", "Positive"]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Negative", "Neutral", "Positive"]
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title(model_name + " Confusion Matrix")
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")

    plt.savefig(
        f"outputs/graphs/confusion_matrices/{safe_name}.png"
    )

    plt.close()

    # Save classification 
    report = classification_report(
        y_test,
        y_pred,
        labels=["Negative", "Neutral", "Positive"],
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        f"outputs/evaluation/{safe_name}_classification_report.csv"
    )

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Macro F1:", round(macro_f1, 4))

#comparison
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Macro F1",
    ascending=False
)

results_df.to_csv(
    "outputs/evaluation/comprehensive_model_comparison.csv",
    index=False
)

print("\n==== FINAL MODEL COMPARISON ====")
print(results_df.to_string(index=False))

# 7. Identify best model
best_model = results_df.iloc[0]

print("\n========== BEST MODEL ==========")
print("Model:", best_model["Model"])
print("Accuracy:", best_model["Accuracy"], "%")
print("Macro F1:", best_model["Macro F1"])

# accuracy comparison
plt.figure(figsize=(13, 7))

plt.bar(
    results_df["Model"],
    results_df["Accuracy"]
)

plt.xticks(rotation=70, ha="right")
plt.ylabel("Accuracy (%)")
plt.xlabel("Models")
plt.title("Accuracy Comparison of All Sentiment Models")
plt.tight_layout()

plt.savefig(
    "outputs/graphs/comprehensive_accuracy_comparison.png"
)

plt.close()

#Macro F1 comparison
plt.figure(figsize=(13, 7))

plt.bar(
    results_df["Model"],
    results_df["Macro F1"]
)

plt.xticks(rotation=70, ha="right")
plt.ylabel("Macro F1 Score")
plt.xlabel("Models")
plt.title("Macro F1 Comparison of All Sentiment Models")
plt.tight_layout()

plt.savefig(
    "outputs/graphs/comprehensive_macro_f1_comparison.png"
)

plt.close()

print("\nEvaluation completed successfully!")
print("Comparison CSV saved in outputs/evaluation/")
print("Confusion matrices saved in outputs/graphs/confusion_matrices/")
