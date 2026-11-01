
import pandas as pd
import os
import joblib

from scipy.sparse import load_npz

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# 1. Load character-level TF-IDF features
X_train = load_npz("outputs/features/X_train_char_tfidf.npz")
X_test = load_npz("outputs/features/X_test_char_tfidf.npz")

y_train = pd.read_csv("outputs/features/y_train_char.csv").iloc[:, 0]
y_test = pd.read_csv("outputs/features/y_test_char.csv").iloc[:, 0]

# 2. Create output folder
os.makedirs("outputs/tuned_models", exist_ok=True)

# 3. Define parameter grids
models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, random_state=42),
        {"C": [0.1, 0.5, 1, 2, 5]}
    ),

    "SVM": (
        LinearSVC(random_state=42),
        {"C": [0.01, 0.1, 0.5, 1, 2]}
    )
}

results = []

# 4. Tune each model
for name, (model, parameters) in models.items():

    print("\nTuning:", name)

    grid = GridSearchCV(
        model,
        parameters,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    # 5. Evaluate on test data
    predictions = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("Best Parameters:", grid.best_params_)
    print("Best CV Macro F1:", round(grid.best_score_, 4))
    print("Test Accuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    # 6. Save tuned model
    filename = name.lower().replace(" ", "_")

    joblib.dump(
        best_model,
        f"outputs/tuned_models/{filename}_char_tuned.pkl"
    )

    results.append({
        "Model": name,
        "Best Parameters": str(grid.best_params_),
        "CV Macro F1": round(grid.best_score_, 4),
        "Test Accuracy": round(accuracy * 100, 2)
    })

# 7. Save comparison
results_df = pd.DataFrame(results)

results_df.to_csv(
    "outputs/char_tuned_model_comparison.csv",
    index=False
)

print("\nCharacter-Level Hyperparameter Tuning Results:")
print(results_df.to_string(index=False))

print("\nHyperparameter tuning completed successfully!")
