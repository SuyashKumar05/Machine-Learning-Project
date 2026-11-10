
# Multilingual Sentiment Analysis of Indian Languages Using Machine Learning

A machine learning project that classifies code-mixed text into three sentiment categories: **Positive, Negative, and Neutral**.

The project follows a complete machine learning pipeline, starting with raw labelled text data and covering data cleaning, exploratory data analysis, feature extraction, model training, hyperparameter tuning, model comparison, and error analysis.

> **Dataset scope:** The dataset contains code-mixed text. However, its exact language coverage and origin have not been independently verified. Therefore, this project does not claim support for any specific Indian language.

## 1. Project Objectives

- Analyse sentiment in code-mixed text.
- Clean and prepare the raw dataset.
- Explore sentiment distribution and text characteristics.
- Extract features using word-level and character-level TF-IDF.
- Train and compare multiple machine learning classifiers.
- Improve model performance using hyperparameter tuning.
- Evaluate models using accuracy, precision, recall, and F1-score.
- Analyse misclassified records to understand model limitations.

## 2. Dataset Description

The project uses a labelled CSV dataset containing code-mixed text and sentiment information.

**Dataset characteristics:**

| Attribute | Description |
|---|---|
| Original dataset records | 81,430 |
| Records after cleaning | 54,752 |
| Sentiment classes | Negative, Neutral, Positive |
| Selected columns | text, sentiment, translated_text, cleaned_text |
| Data format | CSV |

### Final sentiment distribution

| Sentiment | Records |
|---|---:|
| Negative | 19,802 |
| Neutral | 19,385 |
| Positive | 15,565 |
| **Total** | **54,752** |

## 3. Data Preprocessing

The raw dataset was cleaned before performing exploratory data analysis and model development.

The following operations were performed:

- Corrected the dataset header.
- Removed repeated header records.
- Removed records with missing or empty original text.
- Removed text groups having conflicting sentiment labels.
- Removed duplicate original text records.
- Standardised text values by removing unnecessary whitespace.

**Cleaning summary:**

| Cleaning operation | Records removed |
|---|---:|
| Empty text records | 112 |
| Conflicting sentiment records | 2,982 |
| Duplicate text records | 23,584 |
| **Final cleaned dataset** | **54,752** |

The cleaned dataset contains four selected columns: text, sentiment, translated_text, and cleaned_text.

## 4. Exploratory Data Analysis (EDA)

EDA was performed to understand the dataset and identify text characteristics.

The following visualisations were generated:

- Sentiment distribution.
- Text length distribution.
- Word count distribution.

### Text statistics

| Statistic | Text Length | Word Count |
|---|---:|---:|
| Mean | 73.68 | 12.09 |
| Median | 55 | 9 |
| Maximum | 2,017 | 301 |

The sentiment distribution shows that Negative and Neutral records are slightly more frequent than Positive records.

EDA graphs are stored in `outputs/graphs/`.

## 5. Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) was used to convert text into numerical features.

Two feature extraction approaches were implemented.

### Word-level TF-IDF

- Uses word unigrams and bigrams.
- N-gram range: (1, 2).
- Maximum features: 20,000.
- Minimum document frequency: 2.

### Character-level TF-IDF

- Uses character n-grams.
- N-gram range: (2, 5).
- Maximum features: 20,000.
- Minimum document frequency: 2.
- Analyzer: character.

The dataset was divided into training and testing sets using an 80:20 split with stratified sampling and random_state=42.

| Dataset | Records |
|---|---:|
| Training | 43,801 |
| Testing | 10,951 |

The vectorizers were fitted only on training data to avoid data leakage.

## 6. Machine Learning Models

The following classifiers were implemented:

1. Multinomial Naive Bayes
2. Logistic Regression
3. Linear Support Vector Machine (SVM)

The models were trained using both word-level and character-level TF-IDF features.

Hyperparameter tuning was performed using GridSearchCV with three-fold cross-validation.

### Word-level baseline results

| Model | Test Accuracy |
|---|---:|
| Multinomial Naive Bayes | 60.24% |
| Logistic Regression | 61.85% |
| Linear SVM | 59.92% |

### Word-level tuned results

| Model | Best Parameter | Test Accuracy |
|---|---|---:|
| Naive Bayes | alpha = 0.5 | 60.39% |
| Logistic Regression | C = 1.0 | 61.85% |
| Linear SVM | C = 0.1 | 61.99% |

### Character-level baseline results

| Model | Test Accuracy | Macro F1 |
|---|---:|---:|
| Naive Bayes | 57.37% | 0.5710 |
| Logistic Regression | 62.59% | 0.6239 |
| SVM | 61.74% | 0.6160 |

### Character-level tuned results

| Model | Best Parameter | Test Accuracy | Macro F1 |
|---|---|---:|---:|
| Logistic Regression | C = 1.0 | 62.59% | 0.6239 |
| SVM | C = 0.1 | **62.68%** | **0.6246** |

The tuned Character SVM achieved the highest test accuracy among the evaluated models. However, its improvement over Character Logistic Regression was small.

## 7. Comprehensive Model Evaluation

The models were evaluated using multiple performance metrics:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Weighted F1-score

### Model comparison

| Model | Accuracy | Precision | Recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| Tuned Character SVM | **62.68%** | 0.6285 | 0.6228 | **0.6246** | 0.6262 |
| Character Logistic Regression | 62.59% | 0.6271 | 0.6222 | 0.6239 | 0.6254 |
| Tuned Character Logistic Regression | 62.59% | 0.6271 | 0.6222 | 0.6239 | 0.6254 |
| Tuned Word SVM | 61.99% | 0.6220 | 0.6157 | 0.6177 | 0.6193 |
| Tuned Word Logistic Regression | 61.85% | 0.6202 | 0.6146 | 0.6165 | 0.6180 |
| Word Logistic Regression | 61.85% | 0.6202 | 0.6146 | 0.6165 | 0.6180 |
| Character SVM | 61.74% | 0.6169 | 0.6153 | 0.6160 | 0.6172 |
| Tuned Word Naive Bayes | 60.39% | 0.6057 | 0.6018 | 0.6031 | 0.6035 |
| Word Naive Bayes | 60.24% | 0.6059 | 0.5991 | 0.6012 | 0.6019 |
| Word SVM | 59.92% | 0.5990 | 0.5975 | 0.5981 | 0.5992 |
| Character Naive Bayes | 57.37% | 0.5818 | 0.5678 | 0.5710 | 0.5726 |

The tuned Character SVM achieved the best overall performance based on accuracy and macro F1-score.

## 8. Character SVM Error Analysis

Further error analysis was performed on the tuned Character SVM using all 10,951 test records.

### Prediction summary

| Metric | Result |
|---|---:|
| Total test records | 10,951 |
| Correct predictions | 6,864 |
| Incorrect predictions | 4,087 |
| Accuracy | 62.68% |

### Confusion matrix

| Actual / Predicted | Negative | Neutral | Positive |
|---|---:|---:|---:|
| Negative | 2,665 | 861 | 435 |
| Neutral | 907 | 2,422 | 548 |
| Positive | 689 | 647 | 1,777 |

The largest individual confusion occurred when Neutral records were classified as Negative (907 records).

This analysis helps identify classification errors and understand the limitations of the model beyond overall accuracy.

### Generated error-analysis files

| File | Description |
|---|---|
| character_svm_error_analysis.csv | Predictions for all test records |
| character_svm_misclassified.csv | Incorrectly classified records |
| character_svm_error_summary.csv | Classification metrics for each class |

These files are stored in `outputs/evaluation/`.

## 9. Project Workflow

The overall project follows this sequence:

1. Dataset inspection
2. Data cleaning and preprocessing
3. Exploratory data analysis
4. Word-level TF-IDF feature extraction
5. Character-level TF-IDF feature extraction
6. Model training
7. Hyperparameter tuning
8. Comprehensive model evaluation
9. Error analysis
10. Model comparison and reporting

## 10. Repository Structure

```text
Machine-Learning-Project/
│
├── data/
│   └── labeled_sentiment_output_codemixed.csv
│
├── outputs/
│   ├── evaluation/
│   │   ├── character_svm_error_analysis.csv
│   │   ├── character_svm_error_summary.csv
│   │   ├── character_svm_misclassified.csv
│   │   └── model classification reports
│   │
│   ├── graphs/
│   ├── features/
│   ├── models/
│   ├── tuned_models/
│   └── model comparison CSV files
│
├── src/
│   ├── Dataset.py
│   ├── data_cleaning.py
│   ├── EDA.py
│   ├── Feature_extraction.py
│   ├── Model_training.py
│   ├── hyperparameter_tuning.py
│   ├── char_feature_extraction.py
│   ├── char_model_training.py
│   ├── char_hyperparameter_tuning.py
│   ├── compare_feature_methods.py
│   ├── model_evaluation.py
│   └── character_svm_error_analysis.py
│
├── report/
├── .gitignore
└── README.md
```

Datasets, generated feature matrices, and trained model binaries are excluded from Git. Selected graphs, evaluation reports, and model comparison results are tracked in the repository.

## 11. Installation and Execution

### Clone the repository

```bash
git clone https://github.com/SuyashKumar05/Machine-Learning-Project.git

cd Machine-Learning-Project
```

### Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy joblib
```

### Dataset preparation

Place the original dataset in the `data/` directory using the filename:

`labeled_sentiment_output_codemixed.csv`

### Run the project

Execute the following scripts from the project root directory.

```bash
# Dataset inspection and cleaning
python src/Dataset.py
python src/data_cleaning.py

# Exploratory data analysis
python src/EDA.py

# Word-level feature extraction and modelling
python src/Feature_extraction.py
python src/Model_training.py
python src/hyperparameter_tuning.py

# Character-level feature extraction and modelling
python src/char_feature_extraction.py
python src/char_model_training.py
python src/char_hyperparameter_tuning.py

# Model comparison and evaluation
python src/compare_feature_methods.py
python src/model_evaluation.py

# Character SVM error analysis
python src/character_svm_error_analysis.py
```

**Note:** The scripts depend on outputs generated by earlier stages. Run them in the appropriate order. The original dataset and required trained model files must be available.

## 12. Limitations and Future Scope

### Current limitations

- The exact language coverage of the dataset has not been verified.
- The dataset contains code-mixed text, which can include spelling and transliteration variations.
- The best model achieves approximately 62.68% accuracy, leaving room for improvement.
- Some Neutral and Negative records are frequently confused.
- Classical machine learning models may not fully capture contextual meaning and linguistic relationships.

### Future improvements

- Explore additional machine learning algorithms.
- Investigate transliteration and spelling variations.
- Experiment with multilingual transformer models such as mBERT and XLM-R.
- Improve performance on weaker sentiment classes.
- Verify the dataset's language coverage and provenance.
- Explore additional text preprocessing and feature engineering techniques.

## 13. Author

**Suyash Kumar**

M.Tech – Data Science and Analytics
Lovely Professional University

Project: Multilingual Sentiment Analysis of Indian Languages Using Machine Learning

GitHub: [SuyashKumar05](https://github.com/SuyashKumar05)
