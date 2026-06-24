# Sentiment Analysis System – NLP Text Classifier

> Automatically classifies text reviews as **Positive**, **Negative**, or **Neutral**  
> using Natural Language Processing (NLP) and Machine Learning.

---

## What This Project Does

Ever wondered how companies like Amazon or Flipkart understand thousands of customer reviews at once?  
This project does exactly that — it reads a piece of text and tells you whether the sentiment is positive, negative, or neutral, **automatically**.

---

## Tech Stack

| Category       | Tools Used                              |
|----------------|-----------------------------------------|
| Language       | Python 3.x                              |
| NLP            | NLTK (tokenisation, stopwords, stemming)|
| Features       | TF-IDF Vectorizer (Scikit-learn)        |
| ML Models      | Logistic Regression, Naive Bayes, SVM   |
| Evaluation     | Accuracy, Precision, Recall, F1-Score   |
| Visualisation  | Matplotlib                              |
| Data Handling  | Pandas, NumPy                           |

---

## How It Works (Step by Step)

```
Raw Text Review
      │
      ▼
1. Preprocessing
   - Lowercase conversion
   - Tokenisation (split into words)
   - Stopword removal (remove "the", "is", "a" etc.)
   - Stemming (e.g. "running" → "run")
      │
      ▼
2. Feature Extraction
   - TF-IDF Vectorizer converts text into numbers
   - Uses unigrams + bigrams (1–2 word combinations)
      │
      ▼
3. Model Training
   - Logistic Regression
   - Naive Bayes
   - SVM (LinearSVC)
   - Best model selected based on accuracy
      │
      ▼
4. Prediction & Evaluation
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix
      │
      ▼
5. Output: Positive / Negative / Neutral
```

---

## Project Structure

```
sentiment-analysis-nlp/
│
├── sentiment_analysis.py       # Main source code
├── sentiment_analysis_results.png
├── sentiment_analysis_Dashboard.png
└── README.md                   # This file
```

---

## Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/shanthipriyajana/sentiment-analysis-nlp.git
cd sentiment-analysis-nlp
```

### 2. Install dependencies
```bash
pip install numpy pandas matplotlib scikit-learn nltk
```

### 3. Run the project
```bash
python sentiment_analysis.py
```

---

## Sample Output

```
Model                     Accuracy
-----------------------------------------
Logistic Regression         87.50%
Naive Bayes                 75.00%
SVM (LinearSVC)             87.50%
-----------------------------------------
Best model : Logistic Regression (87.50%)

Live Prediction Demo
===========================================
✅ [Positive]  This is the best thing I have ever bought!
❌ [Negative]  Complete waste of money, very disappointed.
➖ [Neutral  ]  It works fine, nothing special about it.
```

---

## What I Learned

- Text preprocessing using NLTK (tokenisation, stopwords, stemming)
- Converting text to numerical features using TF-IDF
- Training and comparing multiple ML classifiers
- Evaluating models using classification metrics
- Visualising results with Matplotlib

---

## Author

**Shanthipriya Jana**  
MCA Student | Python Developer | AI/ML Enthusiast  
📧 shanthipriyajana@gmail.com  
🔗 [linkedin.com/in/shanthipriya-jana](https://www.linkedin.com/in/shanthipriya-jana-8629b6398/?trk=opento_sprofile_pfeditor)
