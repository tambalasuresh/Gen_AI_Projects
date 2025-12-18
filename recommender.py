import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("tmdb_5000_movies.xls")

def extract(text):
    try:
        return " ".join([i['name'] for i in ast.literal_eval(text)])
    except:
        return ""

df['tags'] = (
    df['overview'].fillna("") + " " +
    df['genres'].apply(extract) + " " +
    df['keywords'].apply(extract)
).str.lower()

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
vectors = vectorizer.fit_transform(df['tags'])
similarity = cosine_similarity(vectors)

def recommend(movie, n=5):
    idx = df[df['title'] == movie].index[0]
    scores = similarity[idx]
    movies = sorted(
        list(enumerate(scores)),
        reverse=True,
        key=lambda x: x[1]
    )[1:n+1]

    return [df.iloc[i[0]].title for i in movies]
