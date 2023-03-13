import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy

def clean_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.casefold() not in stop_words]
    # Lemmatize using spaCy
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(' '.join(filtered_words))
    lemmatized_words = [token.lemma_ for token in doc]
    return ' '.join(lemmatized_words)
