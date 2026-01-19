import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(page_title="Emotion Analysis NLP", layout="centered")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        vectorizer = joblib.load("nlp_vectorizer.pkl")
        model = joblib.load("nlp_model.pkl")
        return vectorizer, model
    except FileNotFoundError:
        return None, None

vectorizer, model = load_assets()

# --- ERROR HANDLING ---
if vectorizer is None or model is None:
    st.error("🚨 Error: Missing Files!")
    st.stop()

# --- EMOJI MAPPING ---
# 0: Anger, 1: Fear, 2: Joy, 3: Love, 4: Sadness, 5: Surprise
emoji_map = {
    0: "😠 Anger",
    1: "😨 Fear",
    2: "😂 Joy",
    3: "❤️ Love",
    4: "😔 Sadness",
    5: "😮 Surprise"
}

# --- UI LAYOUT ---
st.title("📝 Text Emotion Analyzer")
st.markdown("Type a sentence below to detect the underlying emotional tone.")

user_text = st.text_area("Enter text:", height=100, placeholder="I feel romantic too")

# --- PREDICTION LOGIC ---
if st.button("Analyze Emotion"):
    if not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        try:
            # 1. Vectorize input
            text_vectorized = vectorizer.transform([user_text])
            
            # 2. Get Model Prediction (The "Raw" Guess)
            prediction = model.predict(text_vectorized)[0]
            
            # Get probabilities if available
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(text_vectorized)
            else:
                probs = None

            # ==========================================================
            # 🛠️ RULE-BASED OVERRIDES (The Fix)
            # ==========================================================
            lower_text = user_text.lower()
            
            # FIXED SYNTAX: added "word" between "for" and "in"
            romantic_keywords = ["romantic", "romance", "kiss", "love", "crush", "heart", "date"]
            fear_keywords = ["scared", "terrified", "spooky", "horror", "frightened"]

            # Force "Love" (Class 3) 
            if any(word in lower_text for word in romantic_keywords):
                prediction = 3  
            
            # Force "Fear" (Class 1) 
            elif any(word in lower_text for word in fear_keywords):
                prediction = 1  
            
            # ==========================================================

            # 3. Display Result
            result_text = emoji_map.get(prediction, f"Unknown Code: {prediction}")
            st.success(f"### Detected Emotion: {result_text}")

            # 4. Probability Chart (Optional)
            if probs is not None:
                st.markdown("#### Confidence Scores (Raw Model)")
                class_names = [emoji_map.get(c, str(c)) for c in model.classes_]
                prob_df = pd.DataFrame(probs, columns=class_names)
                st.bar_chart(prob_df.T)

        except Exception as e:
            st.error(f"An error occurred: {e}")