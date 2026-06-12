# 🧠 Emotion Detection Using Machine Learning

[![Streamlit App](https://emotion-detection-using-machine-learning.streamlit.app)

## 🌐 Live Demo

🔗 **Try the Application Here:**
https://emotion-detection-using-machine-learning.streamlit.app

---

## 📌 Project Overview

Emotion Detection is a Natural Language Processing (NLP) project that predicts human emotions from textual input using Machine Learning techniques.

The system analyzes user-entered text and classifies it into one of six emotions:

* 😊 Joy
* 😢 Sadness
* 😠 Anger
* 😨 Fear
* ❤️ Love
* 😲 Surprise

The project combines text preprocessing, TF-IDF feature extraction, and machine learning models to achieve accurate emotion classification.

---

## 🚀 Features

✅ Multi-Class Emotion Classification

✅ Text Cleaning and Preprocessing

✅ TF-IDF Feature Extraction

✅ Multiple Model Comparison

✅ Hyperparameter Tuning

✅ Real-Time Emotion Prediction

✅ Interactive Streamlit Web Application

✅ Model and Vectorizer Persistence

---

## 📂 Dataset

The dataset contains thousands of labeled text samples representing different emotions.

### Emotion Classes

| Label | Emotion  |
| ----- | -------- |
| 0     | Sadness  |
| 1     | Anger    |
| 2     | Love     |
| 3     | Surprise |
| 4     | Fear     |
| 5     | Joy      |

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* NLTK
* Scikit-Learn
* XGBoost
* LightGBM
* Matplotlib
* Seaborn
* Streamlit
* Pickle

---

## 🔄 Project Workflow

### 1. Data Loading

Loaded the emotion dataset containing text samples and emotion labels.

### 2. Data Cleaning

Performed:

* Duplicate removal
* Lowercase conversion
* Punctuation removal
* Stopword removal
* Stemming

### 3. Text Preprocessing

Applied:

* Regular Expressions
* NLTK Stopwords
* Porter Stemmer

### 4. Feature Extraction

Used TF-IDF Vectorizer to convert textual data into numerical features.

### 5. Train-Test Split

* Training Data: 80%
* Testing Data: 20%

### 6. Model Training

The following models were trained and evaluated:

* Logistic Regression
* Linear SVM
* Multinomial Naive Bayes
* XGBoost Classifier
* LightGBM Classifier

### 7. Hyperparameter Tuning

Optimized XGBoost using RandomizedSearchCV.

### 8. Model Evaluation

Performance metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC Curve

### 9. Deployment

The best-performing model was integrated into a Streamlit application for real-time predictions.

---

## 📊 Model Performance

### Best Model

🏆 XGBoost Classifier

### Accuracy

📈 Approximately **88% Accuracy**

The XGBoost model achieved the highest performance among all tested machine learning models.

---

## 📈 Visualizations Included

* Confusion Matrix
* ROC Curve
* Actual vs Predicted Distribution
* Prediction Comparison Graph

---

## 🧪 Sample Predictions

| Input Text                       | Predicted Emotion |
| -------------------------------- | ----------------- |
| I am feeling very happy today    | Joy               |
| I miss my old friends            | Sadness           |
| I am angry about this decision   | Anger             |
| I am scared about tomorrow       | Fear              |
| I love spending time with family | Love              |
| This is unbelievable             | Surprise          |

---

## 🌐 Streamlit Application

The web application allows users to:

* Enter custom text
* Analyze emotions instantly
* View predicted emotion results
* Experience an interactive user interface

### Run Locally

Clone the repository:

```bash
git clone https://github.com/Sanjayduduka45/Emotion-Detection-Using-Machine-Learning
```

Move into the project folder:

```bash
cd emotion-detection-using-machine-learning
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```text
Emotion-Detection-Using-Machine-Learning/
│
├── app.py
├── sentiment.ipynb
├── requirements.txt
├── README.md
│
├── xgb_model.pkl
├── tfidf.pkl
├── emotions.pkl
│
└── Dataset/
    └── train.txt
```

---

## 💾 Saved Artifacts

### Trained Model

```python
xgb_model.pkl
```

### TF-IDF Vectorizer

```python
tfidf.pkl
```

### Emotion Mapping Dictionary

```python
emotions.pkl
```

---

## 🔮 Future Improvements

* Deep Learning Models (LSTM, BERT)
* Confidence Score Prediction
* Multi-Language Support
* Cloud Deployment
* Emotion Analytics Dashboard
* Enhanced User Interface

---

## 🎯 Key Learning Outcomes

This project helped in understanding:

* Natural Language Processing
* Text Preprocessing Techniques
* Feature Engineering with TF-IDF
* Machine Learning Classification
* Hyperparameter Optimization
* Model Evaluation Metrics
* Streamlit Deployment

---

## 👨‍💻 Author

**Sanjay Duduka**

Aspiring Data Scientist | Machine Learning Enthusiast 

### Connect With Me

* LinkedIn: https://www.linkedin.com/in/sanjayduduka/
* GitHub: https://github.com/Sanjayduduka45

---

## ⭐ Live Application

https://emotion-detection-using-machine-learning.streamlit.app

If you found this project useful, consider giving the repository a ⭐ on GitHub.
