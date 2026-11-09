
import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("outputs/cleaned_sentiment_dataset.csv")

print("Dataset shape:", df.shape)

#folder for graphs
os.makedirs("outputs/graphs", exist_ok=True)

#Sentiment distribution
sentiment_counts = df["sentiment"].value_counts()

print("\nSentiment Distribution:")
print(sentiment_counts)

plt.figure(figsize=(7, 5))
sentiment_counts.plot(kind="bar")

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("outputs/graphs/sentiment_distribution.png")
plt.show()

#calculate text length
df["text_length"] = df["text"].astype(str).apply(len)

print("\nText Length Statistics:")
print(df["text_length"].describe())

plt.figure(figsize=(7, 5))
plt.hist(df["text_length"], bins=30)

plt.title("Distribution of Text Length")
plt.xlabel("Number of Characters")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig("outputs/graphs/text_length_distribution.png")
plt.show()

# Calculate word count
df["word_count"] = df["text"].astype(str).apply(
    lambda x: len(x.split())
)

print("\nWord Count Statistics:")
print(df["word_count"].describe())

plt.figure(figsize=(7, 5))
plt.hist(df["word_count"], bins=30)

plt.title("Distribution of Word Count")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig("outputs/graphs/word_count_distribution.png")
plt.show()

print("\nEDA completed successfully!")
print("Graphs saved in outputs/graphs/")
