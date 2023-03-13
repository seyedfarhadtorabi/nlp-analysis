import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load vectorized data
bow_vector = pd.read_csv('data/bow_vector.csv').to_numpy()
tfidf_vector = pd.read_csv('data/tfidf_vector.csv').to_numpy()

# Load cleaned data
df = pd.read_csv('data/complaints_clean.csv')

# Define function to extract topics using Non-negative Matrix Factorization (NMF)
def extract_topics_nmf(vector, vectorizer):
    nmf = NMF(n_components=10, random_state=42)
    nmf.fit(vector)
    feature_names = vectorizer.get_feature_names()
    for idx, topic in enumerate(nmf.components_):
        print(f'Topic #{idx}:')
        print([feature_names[i] for i in topic.argsort()[:-11:-1]])

# Define function to extract topics using Latent Dirichlet Allocation (LDA)
def extract_topics_lda(vector, vectorizer):
    lda = LatentDirichletAllocation(n_components=10, random_state=42)
    lda.fit(vector)
    feature_names = vectorizer.get_feature_names()
    for idx, topic in enumerate(lda.components_):
        print(f'Topic #{idx}:')
        print([feature_names[i] for i in topic.argsort()[:-11:-1]])

# Extract topics using NMF and BoW vector
extract_topics_nmf(bow_vector, bow_vectorizer)

# Extract topics using LDA and TF-IDF vector
extract_topics_lda(tfidf_vector, tfidf_vectorizer)

# Generate word cloud of the most common words in the corpus
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['clean_text']))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

# Visualize topics using pyLDAvis
pyLDAvis.enable_notebook()
pyLDAvis.sklearn.prepare(lda, tfidf_vector, tfidf_vectorizer)
