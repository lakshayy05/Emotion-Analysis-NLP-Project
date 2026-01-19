# Emotion Analysis NLP Project 📝

A Natural Language Processing (NLP) web application that detects emotions (Joy, Sadness, Anger, Fear, Love, Surprise) from text. This project uses a **Hybrid Approach**, combining a Machine Learning model (Logistic Regression) with smart Rule-Based Overrides to ensure high accuracy for specific keywords.

## 🚀 Project Overview

Sentiment analysis often struggles with nuanced emotions like distinguishing "Love" from "Joy." This tool solves that by using a trained ML model for general context and a rule-based layer to capture specific intent (e.g., romantic keywords).

* **Backend:** Scikit-Learn (TF-IDF + Logistic Regression)
* **Frontend:** Streamlit
* **Architecture:** Hybrid (ML + Keyword Rules)

## 📂 Project Structure

```text
NLP-Emotion-Analysis/
│
├── app.py                       # 🖥️ Frontend: Streamlit Web App (with Hybrid Logic)
├── Emoji_analysis_finalproject.ipynb # 📓 Backend: Model Training & EDA
├── nlp_model.pkl                # 🧠 Artifact: Trained Logistic Regression Model
├── nlp_vectorizer.pkl           # 🔠 Artifact: TF-IDF Vectorizer
├── requirements.txt             # ⚙️ Dependencies
└── README.md                    # 📄 Documentation

📊 How It Works
Vectorization: The app converts user text into numbers using TF-IDF (Term Frequency-Inverse Document Frequency).
ML Prediction: A Logistic Regression model predicts the probability of each emotion.
Smart Overrides: A custom logic layer checks for high-impact keywords (e.g., "romantic", "horror") to correct the model if it misses obvious cues.
Result: The final emotion is displayed with a confidence chart.

🛠️ Tech Stack
Language: Python 3.13.3
Libraries: Scikit-learn, Pandas, NumPy, Joblib
Web Framework: Streamlit

📸 Usage Example
Input: "I am so scared of the dark."
Output: 😨 Fear (Detected via Keywords/Model)

Input: "I feel really romantic today."
Output: ❤️ Love (Detected via Rule-Based Override)
