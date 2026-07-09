import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from preprocess import clean_text

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

df = pd.read_csv("dataset/spam.csv")
df["clean"] = df["message"].apply(clean_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean"])
y = df["label"]

model = MultinomialNB()
model.fit(X, y)

joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Model trained and saved successfully!")