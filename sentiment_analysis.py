# ============================================================
# Sentiment Analysis System – NLP Text Classifier
# Author : Shanthipriya Jana
# GitHub : github.com/shanthipriyajana/sentiment-analysis-nlp
# Description: Classifies text reviews as Positive / Negative
#              / Neutral using classic NLP + ML techniques.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# ML
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# Download required NLTK data
nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)


# ────────────────────────────────────────────────────────────
# 1. SAMPLE DATASET
#    In a real project replace this with a CSV file, e.g.:
#    df = pd.read_csv("reviews.csv")
# ────────────────────────────────────────────────────────────
data = {
    "review": [
        # Positive
        "This product is absolutely amazing and works perfectly",
        "I love it, great quality and fast delivery",
        "Excellent service, very happy with my purchase",
        "Wonderful experience, highly recommend to everyone",
        "Best product I have ever bought, totally worth it",
        "Outstanding quality, will definitely buy again",
        "Super happy with this, exceeded my expectations",
        "Fantastic product, arrived on time and in perfect condition",
        "Very satisfied, the product is exactly as described",
        "Great value for money, works like a charm",
        # Negative
        "Terrible quality, broke after one day of use",
        "Very disappointed, does not work as advertised",
        "Worst purchase ever, complete waste of money",
        "Poor quality and horrible customer service",
        "Stopped working within a week, totally useless",
        "Damaged product received, very unhappy",
        "Not worth the price at all, very bad quality",
        "Awful experience, will never buy from here again",
        "Product is fake and nothing like the pictures",
        "Extremely bad, do not buy this",
        # Neutral
        "The product is okay, nothing special",
        "It works fine but the packaging could be better",
        "Average quality, meets basic expectations",
        "Decent product for the price",
        "It is acceptable but I expected more features",
        "Works as described, nothing extraordinary",
        "Not bad but not great either",
        "Product is fine, delivery was a bit slow",
        "Okay product, might work for some people",
        "Average experience overall",
    ],
    "sentiment": (
        ["Positive"] * 10 +
        ["Negative"] * 10 +
        ["Neutral"]  * 10
    ),
}

df = pd.DataFrame(data)
print("=" * 55)
print("   Sentiment Analysis – NLP Text Classifier")
print("=" * 55)
print(f"\nDataset shape : {df.shape}")
print(f"Class distribution :\n{df['sentiment'].value_counts()}\n")


# ────────────────────────────────────────────────────────────
# 2. TEXT PREPROCESSING
# ────────────────────────────────────────────────────────────
stop_words = set(stopwords.words("english"))
stemmer    = PorterStemmer()

def preprocess(text: str) -> str:
    """Lowercase → tokenise → remove stopwords → stem."""
    tokens = word_tokenize(text.lower())
    tokens = [stemmer.stem(t) for t in tokens
              if t.isalpha() and t not in stop_words]
    return " ".join(tokens)

df["clean_review"] = df["review"].apply(preprocess)
print("Sample preprocessed text:")
for i in range(3):
    print(f"  Original : {df['review'][i]}")
    print(f"  Cleaned  : {df['clean_review'][i]}\n")


# ────────────────────────────────────────────────────────────
# 3. FEATURE EXTRACTION – TF-IDF
# ────────────────────────────────────────────────────────────
vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))

X = vectorizer.fit_transform(df["clean_review"])
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"Train samples : {X_train.shape[0]}")
print(f"Test  samples : {X_test.shape[0]}\n")


# ────────────────────────────────────────────────────────────
# 4. TRAIN & COMPARE MODELS
# ────────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes":         MultinomialNB(),
    "SVM (LinearSVC)":     LinearSVC(random_state=42),
}

results = {}
print("-" * 55)
print(f"{'Model':<25} {'Accuracy':>10}")
print("-" * 55)

for name, model in models.items():
    model.fit(X_train, y_train)
    preds    = model.predict(X_test)
    acc      = accuracy_score(y_test, preds)
    results[name] = {"model": model, "preds": preds, "accuracy": acc}
    print(f"{name:<25} {acc*100:>9.2f}%")

print("-" * 55)

# Pick best model
best_name  = max(results, key=lambda n: results[n]["accuracy"])
best_model = results[best_name]["model"]
best_preds = results[best_name]["preds"]
print(f"\nBest model : {best_name} ({results[best_name]['accuracy']*100:.2f}%)\n")


# ────────────────────────────────────────────────────────────
# 5. DETAILED EVALUATION
# ────────────────────────────────────────────────────────────
print("Classification Report:")
print(classification_report(y_test, best_preds,
                             target_names=["Negative", "Neutral", "Positive"]))


# ────────────────────────────────────────────────────────────
# 6. VISUALISATIONS
# ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("Sentiment Analysis – Model Insights", fontsize=14, fontweight="bold")

# 6a. Sentiment distribution
sentiment_counts = df["sentiment"].value_counts()
colors_pie = ["#4CAF50", "#F44336", "#2196F3"]
axes[0].pie(sentiment_counts, labels=sentiment_counts.index,
            autopct="%1.1f%%", colors=colors_pie, startangle=90)
axes[0].set_title("Class Distribution")

# 6b. Model accuracy comparison
model_names = list(results.keys())
accuracies  = [results[n]["accuracy"] * 100 for n in model_names]
bar_colors  = ["#2196F3" if n != best_name else "#4CAF50" for n in model_names]
bars = axes[1].bar(model_names, accuracies, color=bar_colors, edgecolor="white")
axes[1].set_ylim(0, 110)
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("Model Comparison")
for bar, acc in zip(bars, accuracies):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1, f"{acc:.1f}%", ha="center", fontsize=9)
axes[1].tick_params(axis="x", rotation=15)

# 6c. Confusion matrix
cm = confusion_matrix(y_test, best_preds,
                      labels=["Negative", "Neutral", "Positive"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["Negative", "Neutral", "Positive"])
disp.plot(ax=axes[2], colorbar=False, cmap="Blues")
axes[2].set_title(f"Confusion Matrix\n({best_name})")

plt.tight_layout()
plt.savefig("sentiment_analysis_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nChart saved → sentiment_analysis_results.png")


# ────────────────────────────────────────────────────────────
# 7. PREDICT NEW REVIEWS  (demo)
# ────────────────────────────────────────────────────────────
def predict_sentiment(text: str) -> str:
    cleaned  = preprocess(text)
    features = vectorizer.transform([cleaned])
    return best_model.predict(features)[0]

print("\n" + "=" * 55)
print("  Live Prediction Demo")
print("=" * 55)
test_reviews = [
    "This is the best thing I have ever bought!",
    "Complete waste of money, very disappointed.",
    "It works fine, nothing special about it.",
]
for review in test_reviews:
    label = predict_sentiment(review)
    icon  = {"Positive": "✅", "Negative": "❌", "Neutral": "➖"}[label]
    print(f"{icon} [{label:8s}]  {review}")
