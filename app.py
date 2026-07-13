import streamlit as st
import joblib
from src.preprocess import clean_text

model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

st.title("📧 Spam Detector")

message = st.text_area("Enter a message")

if st.button("Predict"):
    if message.strip():
        cleaned = clean_text(message)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction.lower() == "spam":
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")
    else:
        st.warning("Please enter a message.")