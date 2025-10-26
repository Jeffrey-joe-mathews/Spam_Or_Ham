# 📧 Spam or Ham - SMS Classification with NLP

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-v1.x-orange.svg)](https://scikit-learn.org/stable/)
[![NLTK](https://img.shields.io/badge/NLTK-v3.x-green.svg)](https://www.nltk.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning pipeline to accurately classify SMS messages as **Spam** or **Ham** using Natural Language Processing (NLP) techniques.

---

## ✨ Features

-   **Data Preprocessing**: Tokenization, lowercasing, lemmatization, and stopword removal.
-   **TF-IDF Vectorization**: Converts text into numerical features.
-   **Multinomial Naive Bayes**: Robust classification model.
-   **Threshold Tuning**: Optimized for high spam recall.
-   **Ready-to-use Model**: `spam_NLPClassifier.pkl` for quick predictions.

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone [https://github.com/Jeffrey-joe-mathews/Spam_Or_Ham.git](https://github.com/Jeffrey-joe-mathews/Spam_Or_Ham.git)
cd Spam_Or_Ham
````

### 2\. Install Dependencies

```bash
pip install pandas numpy scikit-learn nltk joblib
```

### 3\. Run the Notebook

Explore `main.ipynb` to see the full data flow, training, and evaluation.

-----

## 🎯 Model Performance

| Metric          | Value  |
| :-------------- | :----- |
| Accuracy        | 97%    |
| Spam Precision  | 0.93   |
| Spam Recall     | 0.85   |
| Spam F1-Score   | 0.89   |

*Tuned for high spam detection (recall) with minimal false positives.*

-----

## 🛠️ Usage Example

Predicting a new message is simple:

```python
import joblib

# Load the trained model and vectorizer
vectorizer, model = joblib.load("spam_NLPClassifier.pkl")

# New message to classify
message = ["Congratulations! You've won a free prize! Text WIN to 12345."]

# Transform and predict
X = vectorizer.transform(message)
prediction = model.predict(X)

print("Spam" if prediction[0] == 1 else "Ham")
# Output: Spam
```

-----

## 📂 Project Structure

```
SPAM_OR_HAM_NLP/
├── main.ipynb                  # Main Jupyter Notebook with code
├── README.md                   # This README file
├── spam_NLPClassifier.pkl      # Pre-trained model & TF-IDF vectorizer
└── spam_or_ham_dataset/
    └── spam.csv                # Original dataset
```

-----

## 📚 Dataset

This project utilizes the [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) from the UCI Machine Learning Repository. It contains 5,572 SMS messages labeled as `ham` (legitimate) or `spam`.

-----

## 💡 Future Enhancements

  - Implement **N-grams** or **Word Embeddings** for richer text representation.
  - Experiment with advanced models (e.g., **SVM, Random Forest, BERT**).
  - Address class imbalance with techniques like oversampling.
  - Develop a **web application** or **API** for real-time spam detection.

-----

## 🧑‍💻 Author

**Jeffrey Joe Mathews**
[](https://github.com/Jeffrey-joe-mathews)

-----

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://opensource.org/licenses/MIT) file for details.

```
