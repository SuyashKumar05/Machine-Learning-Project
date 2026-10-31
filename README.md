# Multilingual Sentiment Analysis of Indian Languages Using Machine Learning

A classic machine learning pipeline that sorts code-mixed text into **positive**, **negative** or **neutral** sentiment. It starts from a raw labelled CSV and ends with tuned, comparable models, with every step in its own script so each stage can be rerun on its own.

> **A note on scope.** The dataset is code-mixed, but its exact language coverage and origin haven't been verified yet. Until they are, this repo makes no claim about support for any specific Indian language.

## What this project does

- Inspects the dataset and its sentiment labels
- Cleans it: blank rows, duplicate texts and texts carrying conflicting labels are removed
- Explores class balance, text length and word count
- Turns text into TF-IDF features (unigrams and bigrams)
- Trains and compares three classifiers
- Evaluates them with accuracy, precision, recall, macro F1 and confusion matrices
- Tunes each model with grid search

## Built with

| Area | Tools |
|---|---|
| Language | Python |
| Data handling | Pandas, NumPy |
| Plots | Matplotlib, Seaborn |
| Modelling | Scikit-learn (TF-IDF, Naive Bayes, Logistic Regression, Linear SVM) |
| Workflow | VS Code, Git, GitHub |

## Repository layout

```text
Machine-Learning-Project/
├── data/
│   └── labeled_sentiment_output_codemixed.csv
├── outputs/
│   ├── graphs/
│   ├── features/
│   ├── models/
│   └── tuned_models/
├── src/
│   ├── Dataset.py
│   ├── data_cleaning.py
│   ├── EDA.py
│   ├── Feature_extraction.py
│   ├── Model_training.py
│   └── hyperparameter_tuning.py
├── notebooks/
├── report/
├── .gitignore
└── README.md
```

Everything under `outputs/` is generated, so it isn't committed. Running the scripts recreates it.

## How the pipeline works

| Step | Script | What happens |
|---|---|---|
| 1. Inspect | `Dataset.py` | Looks at the structure of the data and the sentiment labels |
| 2. Clean | `data_cleaning.py` | Drops blank rows, duplicate texts and texts with conflicting labels |
| 3. Explore | `EDA.py` | Charts sentiment distribution, text length and word count |
| 4. Featurise | `Feature_extraction.py` | Builds TF-IDF features from unigrams and bigrams |
| 5. Train | `Model_training.py` | Fits Multinomial Naive Bayes, Logistic Regression and Linear SVM |
| 6. Tune | `hyperparameter_tuning.py` | Grid search with three-fold cross-validation |

## Results

### Baseline models

| Model | Test accuracy |
|---|---:|
| Multinomial Naive Bayes | 60.24% |
| Logistic Regression | 61.85% |
| Linear SVM | 59.92% |

### After hyperparameter tuning

| Model | Best setting | Test accuracy |
|---|---|---:|
| Naive Bayes | `alpha = 0.5` | 60.39% |
| Logistic Regression | `C = 1.0` | 61.85% |
| Linear SVM | `C = 0.1` | **61.99%** |

### Reading the numbers

- The tuned Linear SVM is the best model so far at **61.99%** test accuracy.
- Tuning mattered most for the SVM, which gained about 2 points. Naive Bayes barely moved (+0.15) and Logistic Regression didn't change.
- All three models land within roughly two points of each other, so the ranking is a weak signal. Repeating the comparison over several splits would show how much of it is chance.
- Accuracy alone hides how each class is doing. Check the per-class precision, recall and confusion matrices before drawing conclusions.

## Running it yourself

Clone the repository and move into it:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Machine-Learning-Project
```

Install the dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy joblib
```

Put the original dataset in `data/` under the filename shown in the layout above, then run the scripts from the project root, in this order:

```bash
python src/Dataset.py
python src/data_cleaning.py
python src/EDA.py
python src/Feature_extraction.py
python src/Model_training.py
python src/hyperparameter_tuning.py
```

## What's next

- Add character-level TF-IDF and compare it against word-level features, since spelling varies a lot in code-mixed text
- Try more algorithms
- Look closely at transliteration and spelling variants
- Test transformer-based multilingual models
- Improve the weaker sentiment classes
- Verify the dataset's language coverage so the project can state which languages it handles

## Author

**Suyash Kumar**, machine learning project on sentiment analysis.