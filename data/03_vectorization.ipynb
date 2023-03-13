import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Load data
df = pd.read_csv('data/complaints_clean.csv')

# Vectorize the text using Bag of Words (BoW)
bow_vectorizer = CountVectorizer()
bow_vector = bow_vectorizer.fit_transform(df['clean_text'])

# Vectorize the text using Term Frequency-Inverse Document Frequency (TF-IDF)
tfidf_vectorizer = TfidfVectorizer()
tfidf_vector = tfidf_vectorizer.fit_transform(df['clean_text'])

# Save vectorized data to file
pd.DataFrame.sparse.from_spmatrix(bow_vector).to_csv('data/bow_vector.csv', index=False)
pd.DataFrame.sparse.from_spmatrix(tfidf_vector).to_csv('data/tfidf_vector.csv', index=False)
