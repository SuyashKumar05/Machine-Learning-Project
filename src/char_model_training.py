
import pandas as pd
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

import matplotlib.pyplot as plt

# 1. Load character-level TF-IDF features
X_train = load_npz("outputs/features/X_train_char_tfidf.npz")
X_test = load_npz("outputs/features/X_test_char_tfidf.npz")

y_train = pd.read_csv("outputs/features/y_train_char.csv").iloc[:, 0]
y_test = pd.read_csv("outputs/features/y_test_char.csv").iloc[:, 0]

# 2. Define models
models = {
    "Naive Bayes": MultinomialNB(alpha=1.0),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "SVM": LinearSVC(random_state=42)
}

# 3. Create output folders
os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/graphs", exist_ok=True)

results = []

# 4. Train and evaluate each model
for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("Accuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    # Save trained model
    filename = name.lower().replace(" ", "_")

    joblib.dump(
        model,
        f"outputs/models/{filename}_char.pkl"
    )

    # Save confusion matrix
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=model.classes_
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=model.classes_
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title(f"{name} - Character TF-IDF")
    plt.tight_layout()

    plt.savefig(
        f"outputs/graphs/{filename}_char_confusion_matrix.png"
    )

    plt.close()

    # Store results
    results.append({
        "Model": name,
        "Accuracy": round(accuracy * 100, 2)
    })

# 5. Save model comparison
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

results_df.to_csv(
    "outputs/char_model_comparison.csv",
    index=False
)

print("\nCharacter-level Model Comparison:")
print(results_df)

print("\nAll character-level models trained successfully!")
