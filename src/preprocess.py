import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

def clean_text(text):
    words = word_tokenize(text)

    words = [
        word.lower()
        for word in words
        if word.isalnum()
    ]

    words = [
        word for word in words
        if word not in stopwords.words('english')
    ]

    return " ".join(words)