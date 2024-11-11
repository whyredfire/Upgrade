import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

df = pd.read_csv("Final_data.csv")

df["genres"] = df["genres"].fillna("").str.split(",")
df["tags"] = df["tags"].fillna("").str.split(",")

mlb_genres = MultiLabelBinarizer()
mlb_tags = MultiLabelBinarizer()
genres_matrix = mlb_genres.fit_transform(df["genres"])
tags_matrix = mlb_tags.fit_transform(df["tags"])

tfidf = TfidfVectorizer(stop_words="english")
summary_matrix = tfidf.fit_transform(df["summary"].fillna(""))

content_matrix = np.hstack((genres_matrix, tags_matrix, summary_matrix.toarray()))
content_similarity = cosine_similarity(content_matrix, content_matrix)

user_movie_matrix = np.random.rand(100, len(df))
svd = TruncatedSVD(n_components=50)
user_factors = svd.fit_transform(user_movie_matrix)
movie_factors = svd.components_.T


def get_hybrid_recommendations(
    movie_id, user_id=0, content_weight=0.5, collaborative_weight=0.5
):
    collaborative_score = np.dot(user_factors[user_id], movie_factors[movie_id])

    content_score = content_similarity[movie_id]

    hybrid_scores = (content_weight * content_score) + (
        collaborative_weight * collaborative_score
    )

    recommended_indices = hybrid_scores.argsort()[::-1][:10]
    return df["title"].iloc[recommended_indices]
