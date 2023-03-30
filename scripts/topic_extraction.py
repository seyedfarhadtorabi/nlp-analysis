import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pyLDAvis
import pyLDAvis.sklearn
# Load vectorized data
bow_vector = pd.read_csv('data/bow_vector.csv').to_numpy()
tfidf_vector = pd.read_csv('data/tfidf_vector.csv').to_numpy()

# Load cleaned data
df = pd.read_csv('data/complaints_clean.csv')

# Define vectorizers and fit it to the cleaned text data
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
tfidf_vector = tfidf_vectorizer.fit_transform(df['clean_text'])  # Fit and transform the cleaned text data
bow_vectorizer = CountVectorizer(stop_words='english')
bow_vector = bow_vectorizer.fit_transform(df['clean_text'])
lda = LatentDirichletAllocation(n_components=10, random_state=42)
lda.fit(tfidf_vector)

# Define function to extract topics using Non-negative Matrix Factorization (NMF)
def extract_topics_nmf(vector, vectorizer):
    nmf = NMF(n_components=10, random_state=42)
    nmf.fit(vector)
    feature_names = vectorizer.get_feature_names()
    for idx, topic in enumerate(nmf.components_):
        print(f'Topic #{idx}:')
        topic_indices = topic.argsort()[:-11:-1]
        topic_words = [feature_names[i] for i in topic_indices if i < len(feature_names)]
        print(topic_words)

# Define function to extract topics using Latent Dirichlet Allocation (LDA)
def extract_topics_lda(vector, vectorizer):
    lda = LatentDirichletAllocation(n_components=10, random_state=42)
    lda.fit(vector)
    feature_names = vectorizer.get_feature_names()
    for idx, topic in enumerate(lda.components_):
        print(f'Topic #{idx}:')
        topic_indices = topic.argsort()[:-11:-1]
        topic_words = [feature_names[i] for i in topic_indices if i < len(feature_names)]
        print(topic_words)


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


pyLDAvis.enable_notebook()
vis_data = pyLDAvis.sklearn.prepare(lda, tfidf_vector, tfidf_vectorizer)
pyLDAvis.save_html(vis_data, 'data/lda_visualization.html')

