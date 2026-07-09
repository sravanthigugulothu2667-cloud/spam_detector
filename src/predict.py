import joblib
from preprocess import clean_text

model=joblib.load("../model.pkl")
vectorizer=joblib.load("../vectorizer.pkl")

text=input("Enter message: ")
clean=clean_text(text)
X=vectorizer.transform([clean])
print("Prediction:",model.predict(X)[0])
