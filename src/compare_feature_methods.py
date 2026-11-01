
import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Store word-level and character-level results
data = {
    "Model": [
        "Naive Bayes",
        "Logistic Regression",
        "SVM"
    ],
    "Word-Level TF-IDF": [
        60.24, 61.85, 59.92
    ],
    "Character-Level TF-IDF": [
        57.37, 62.59, 61.74
    ]
}

df = pd.DataFrame(data)

# 2. Calculate improvement
df["Difference"] = (
    df["Character-Level TF-IDF"] -
    df["Word-Level TF-IDF"]
).round(2)

# 3. Save comparison results
os.makedirs("outputs/graphs", exist_ok=True)

df.to_csv(
    "outputs/feature_method_comparison.csv",
    index=False
)

# 4. Create comparison graph
df.plot(
    x="Model",
    y=["Word-Level TF-IDF", "Character-Level TF-IDF"],
    kind="bar",
    figsize=(10, 6)
)

plt.title("Word-Level vs Character-Level TF-IDF")
plt.ylabel("Accuracy (%)")
plt.xlabel("Machine Learning Models")
plt.xticks(rotation=0)
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()

plt.savefig(
    "outputs/graphs/feature_method_comparison.png"
)

plt.show()

# 5. Display comparison
print("\nFeature Extraction Method Comparison:")
print(df.to_string(index=False))

print("\nComparison completed successfully!")
