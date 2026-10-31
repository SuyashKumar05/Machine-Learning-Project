
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib

from scipy.sparse import load_npz

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Load TF-IDF features
X_train = load_npz("outputs/features/X_train_tfidf.npz")
X_test = load_npz("outputs/features/X_test_tfidf.npz")

# 2. Load sentiment labels
y_train = pd.read_csv("outputs/features/y_train.csv").iloc[:, 0]
y_test = pd.read_csv("outputs/features/y_test.csv").iloc[:, 0]

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# 3. Create machine learning models
models = {
    "Naive Bayes": MultinomialNB(),

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "SVM": LinearSVC()
}

# Create output folders
os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/graphs", exist_ok=True)

results = []

# 4. Train and evaluate each model
for name, model in models.items():

    print("\n------------------------------")
    print("Model:", name)
    print("------------------------------")

    # Train model
    model.fit(X_train, y_train)

    # Predict test data
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", round(accuracy * 100, 2), "%")

    # Display precision, recall and F1-score
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(
        model,
        "outputs/models/" + name.lower().replace(" ", "_") + ".pkl"
    )

    # Create confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=["Negative", "Neutral", "Positive"]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Negative", "Neutral", "Positive"]
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title(name + " - Confusion Matrix")
    plt.tight_layout()

    filename = name.lower().replace(" ", "_")
    plt.savefig(
        "outputs/graphs/" + filename + "_confusion_matrix.png"
    )
    plt.show()

    # Store accuracy
    results.append({
        "Model": name,
        "Accuracy": round(accuracy * 100, 2)
    })

results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(results_df)

results_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

print("\nAll models trained successfully!")
print("Models saved in outputs/models/")
print("Comparison saved in outputs/model_comparison.csv")
