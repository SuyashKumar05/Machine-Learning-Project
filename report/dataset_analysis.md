# Dataset Analysis and Data Preparation

# 1. Dataset Overview

The dataset used in this project is named `labeled_sentiment_output_codemixed.csv`.

It is used to develop a machine learning model for classifying text into three sentiment categories:

- Positive
- Negative
- Neutral

The dataset contains code-mixed text and associated sentiment labels. It also includes translated and cleaned text fields, along with token columns.

The language coverage and original linguistic sources of the dataset still need verification. Therefore, multilingual coverage should not be assumed solely from the dataset filename.

The original dataset contained 81,430 records after correcting the header and removing repeated header rows.

After data cleaning and duplicate removal, 54,752 records remained for analysis and modelling.

## 2. Dataset Structure

The original CSV file initially contained 92 columns.

The first row contained the actual column names rather than being treated as a normal data record. The header was corrected during data preparation.

The main columns selected for this project were:
| Column | Description |
| text | Original input text |
| sentiment | Sentiment category |
| translated_text | Translated version of the text |
| cleaned_text | Preprocessed text |

The remaining token columns were not required for the current TF-IDF modelling pipeline.

## 3. Data Cleaning

Data cleaning was performed using Python and Pandas.

The following operations were carried out:

1. Corrected the CSV header using the first row.
2. Removed repeated header records.
3. Selected the four relevant columns.
4. Removed leading and trailing whitespace.
5. Removed records with blank original text.
6. Identified text groups associated with conflicting sentiment labels and removed those records.
7. Removed duplicate original text entries while keeping the first occurrence.

### Cleaning Summary

| Cleaning operation | Records removed |
|---|---:|
| Blank original text | 112 |
| Conflicting sentiment groups | 2,982 |
| Duplicate original text | 23,584 |
| Final cleaned records | 54,752 |


## 4. Missing Value Analysis

After cleaning, the dataset contained the following missing values:

| Column | Missing values |
|---|---:|
| text | 0 |
| sentiment | 0 |
| translated_text | 9 |
| cleaned_text | 146 |

The original text and sentiment columns had no missing values in the final dataset.

The missing values in translated_text and cleaned_text were retained because the current modelling pipeline uses the original text column.

## 5. Sentiment Distribution

The cleaned dataset contains three sentiment categories.

| Sentiment | Number of records |

| Negative | 19,802 |
| Neutral | 19,385 |
| Positive | 15,565 |
| **Total** | **54,752** |

Negative sentiment represents approximately 36.17% of the dataset, Neutral represents 35.41%, and Positive represents 28.43%.

The three classes are reasonably represented, although Positive has fewer records than the other two classes.

The sentiment distribution graph is available at:

`outputs/graphs/sentiment_distribution.png`

## 6. Text Length Analysis

Exploratory Data Analysis was performed to understand the length of the text records.

### Character Length

| Statistic | Value |
|---|---:|
| Mean | 73.68 |
| Median | 55 |
| Minimum | 1 |
| Maximum | 2,017 |

### Word Count

| Statistic | Value |
|---|---:|
| Mean | 12.09 |
| Median | 9 |
| Minimum | 1 |
| Maximum | 301 |

The mean is greater than the median for both character length and word count. This indicates a right-skewed distribution, where a smaller number of longer text records increase the average.

The corresponding graphs are:

- `outputs/graphs/text_length_distribution.png`
- `outputs/graphs/word_count_distribution.png`

## 7. Observations from EDA

The following observations were made:

1. The dataset contains three sentiment categories.
2. Negative and Neutral records have similar representation.
3. Positive sentiment has comparatively fewer records.
4. Most text records are relatively short.
5. A smaller number of records contain considerably longer text.
6. The dataset contains code-mixed text, which may present challenges related to spelling variations and language mixing.

## 8. Conclusion

The dataset was inspected, cleaned, and prepared for machine learning experiments.

After data preparation, 54,752 records remained with four selected columns. Exploratory Data Analysis helped understand sentiment distribution, missing values, and text length characteristics.

The prepared dataset was subsequently used for TF-IDF feature extraction and machine learning classification.

Further investigation of the dataset's language composition and source information is required to establish its suitability for evaluating multilingual sentiment analysis of Indian languages.