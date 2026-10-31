
import pandas as pd
import joblib
import os

from scipy.sparse import load_npz

from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score, classification_report

# 1. Load TF-IDF features
X_train = load_npz("outputs/features/X_train_tfidf.npz")
X_test = load_npz("outputs/features/X_test_tfidf.npz")

y_train = pd.read_csv("outputs/features/y_train.csv").iloc[:, 0]
y_test = pd.read_csv("outputs/features/y_test.csv").iloc[:, 0]

# 2. Define models and parameter combinations
models = {
    "Naive Bayes": (
        MultinomialNB(),
        {"alpha": [0.1, 0.5, 1.0, 2.0]}
    ),

    "Logistic Regression": (
        LogisticRegression(max_iter=1500),
        {"C": [0.1, 0.5, 1.0, 2.0, 5.0]}
    ),

    "SVM": (
        LinearSVC(),
        {"C": [0.1, 0.5, 1.0, 2.0, 5.0]}
    )
}

results = []

os.makedirs("outputs/tuned_models", exist_ok=True)

# 3. Perform hyperparameter tuning
for name, (model, params) in models.items():

    print("\n==============================")
    print("Tuning:", name)
    print("==============================")

    grid = GridSearchCV(
        estimator=model,
        param_grid=params,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    # Best model
    best_model = grid.best_estimator_

    # Test the best model
    predictions = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("Best parameters:", grid.best_params_)
    print("Best cross-validation F1:", round(grid.best_score_, 4))
    print("Test accuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    # Save tuned model
    filename = name.lower().replace(" ", "_") + "_tuned.pkl"

    joblib.dump(
        best_model,
        "outputs/tuned_models/" + filename
    )

    results.append({
        "Model": name,
        "Best Parameters": str(grid.best_params_),
        "CV Macro F1": round(grid.best_score_, 4),
        "Test Accuracy": round(accuracy * 100, 2)
    })

# 4. Save comparison
results_df = pd.DataFrame(results)

print("\n========== TUNED MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

results_df.to_csv(
    "outputs/tuned_model_comparison.csv",
    index=False
)

print("\nHyperparameter tuning completed!")
