# Sentiment Analysis System – NLP Text Classifier

> Automatically classifies text reviews as **Positive**, **Negative**, or **Neutral** using Natural Language Processing (NLP) and Machine Learning.

---

## What This Project Does

This project analyzes text reviews and automatically predicts whether the sentiment expressed is **Positive**, **Negative**, or **Neutral**.

It demonstrates the complete NLP workflow, including text preprocessing, feature extraction, machine learning model training, evaluation, and sentiment prediction.

---

## Screenshots

### Dashboard View

![Dashboard](sentiment_analysis_dashboard.png)

### Results View

![Results](sentiment_analysis_Dashboard_results.png)

---

## Tech Stack

| Category           | Tools Used                            |
| ------------------ | ------------------------------------- |
| Language           | Python 3.x                            |
| NLP                | NLTK                                  |
| Feature Extraction | TF-IDF Vectorizer                     |
| Machine Learning   | Logistic Regression, Naive Bayes, SVM |
| Data Handling      | Pandas, NumPy                         |
| Visualization      | Matplotlib                            |
| Version Control    | Git & GitHub                          |

---

## Key Features

* Sentiment Classification (Positive, Negative, Neutral)
* Text Preprocessing using NLTK
* TF-IDF Feature Extraction
* Multiple Machine Learning Models Comparison
* Performance Evaluation Metrics
* Interactive Dashboard Interface
* Real-Time Sentiment Prediction

---

## How It Works

```text
Raw Text Review
      │
      ▼
Text Preprocessing
      │
      ▼
TF-IDF Feature Extraction
      │
      ▼
Model Training
(Logistic Regression, Naive Bayes, SVM)
      │
      ▼
Model Evaluation
      │
      ▼
Sentiment Prediction
(Positive / Negative / Neutral)
```

---

## Project Structure

```text
sentiment-analysis-NLP/
│
├── index.html
├── sentiment_analysis.py
├── sentiment_analysis_dashboard.png
├── sentiment_analysis_Dashboard_results.png
├── sentiment_analysis_results.png
└── README.md
```

---

## Installation & Usage

### Clone Repository

```bash
git clone https://github.com/shanthipriyajana-web/sentiment-analysis-NLP.git
cd sentiment-analysis-NLP
```

### Install Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn nltk
```

### Run Project

```bash
python sentiment_analysis.py
```

---

## Sample Output

```text
Model                     Accuracy
-----------------------------------------
Logistic Regression         87.50%
Naive Bayes                 75.00%
SVM (LinearSVC)             87.50%
-----------------------------------------
Best Model : Logistic Regression

Positive : This is the best product I have ever used.
Negative : Complete waste of money.
Neutral  : It works as expected.
```

---

## Skills Demonstrated

* Python Programming
* Natural Language Processing (NLP)
* Machine Learning
* Scikit-Learn
* NLTK
* Data Analysis
* Feature Engineering
* Model Evaluation
* Data Visualization
* Git & GitHub

---

## What I Learned

* Text preprocessing using NLTK
* Tokenization and Stopword Removal
* Stemming Techniques
* TF-IDF Feature Extraction
* Machine Learning Model Training
* Classification Metrics Evaluation
* Data Visualization with Matplotlib

---

## Future Enhancements

* Deploy using Flask or FastAPI
* Add Deep Learning Models (LSTM/BERT)
* Build REST APIs for Predictions
* Support Multiple Languages
* Improve Model Accuracy with Larger Datasets

---

## Author

**Shanthipriya Jana**

MCA Graduate | Python Developer | AI/ML Enthusiast

📧 [shanthipriyajana@gmail.com](mailto:shanthipriyajana@gmail.com)

🔗 LinkedIn: https://www.linkedin.com/in/shanthipriya-jana-8629b6398/
