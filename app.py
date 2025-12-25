import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# import matplotlib.pyplot as plt
# import numpy as np

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered"
)

# st.write("Hello World")
# st.markdown("**Bold Text**")
# st.code("print('Hello')\n"
#         "print('hii')")


# uploaded_file = st.file_uploader("Upload CSV")

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     st.dataframe(df)


# option = st.sidebar.selectbox(
#     "Choose option",
#     ["Home", "Analysis", "Prediction0","suresh"]
# )

# col1, col2 = st.columns(2)

# col1.write("Left Side")
# col2.write("Right Side")


# df = pd.DataFrame({
#     "Name": ["A", "B", "C","D"],
#     "Score": [90, 85, 88,100]
# })

# styled_df = df.style.background_gradient(
#     subset=["Score"],
#     cmap="Greens"
# )

# st.dataframe(styled_df)


# name = st.text_input("Enter Your Name")

# def loop(name):
#     result = ""
#     for i in range(len(name)):
#         result += name[i] + "\n"
#     st.code(result)

# if name:
#     loop(name)

# data = np.random.randn(10)

# st.line_chart(data)
# st.bar_chart(data)
# # st.pie_chart(data)

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("tmdb_5000_movies.xls")

df = load_data()

# -----------------------------
# Clean Required Columns
# -----------------------------
def extract_names(text):
    try:
        return " ".join([i["name"] for i in ast.literal_eval(text)])
    except:
        return ""

df["genres"] = df["genres"].apply(extract_names)
df["keywords"] = df["keywords"].apply(extract_names)
df["overview"] = df["overview"].fillna("")

# -----------------------------
# Create Tags Feature
# -----------------------------
df["tags"] = (
    df["overview"] + " " +
    df["genres"] + " " +
    df["keywords"]
).str.lower()

# -----------------------------
# Vectorization & Similarity
# -----------------------------
@st.cache_data
def create_similarity(tags):
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )
    vectors = vectorizer.fit_transform(tags)
    return cosine_similarity(vectors)

similarity = create_similarity(df["tags"])

# -----------------------------
# Recommendation Function
# -----------------------------
def recommend(movie_name, n=5):
    index = df[df["title"] == movie_name].index[0]
    scores = similarity[index]

    movie_list = sorted(
        list(enumerate(scores)),
        reverse=True,
        key=lambda x: x[1]
    )[1:n+1]

    return df.iloc[[i[0] for i in movie_list]][
        ["title", "vote_average", "popularity"]
    ]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎬 Movie Recommendation System")
# st.title("Suresh T")
st.write("Content-based recommendations using genres, keywords & overview")

movie = st.selectbox(
    "Select a movie:",
    sorted(df["title"].values)
)

if st.button("Recommend 🎯"):
    results = recommend(movie)

    st.subheader("Recommended Movies")
    for _, row in results.iterrows():
        st.write(
            f"⭐ **{row['title']}** "
            f"| Rating: {row['vote_average']} "
            f"| Popularity: {round(row['popularity'], 1)}"
        )

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("About")
st.sidebar.info(
    "This recommender uses TF-IDF and cosine similarity "
    "based on movie overview, genres, and keywords."
)
