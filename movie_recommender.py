import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie dataset
data = {
    "title": ["Batman", "Superman", "Iron Man", "Avengers"],
    "genre": ["action hero", "action hero", "action technology", "action team"]
}

df = pd.DataFrame(data)

# Convert text to vectors
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df["genre"])

# Compute similarity
similarity = cosine_similarity(tfidf_matrix)

print("Movie Similarity Matrix:")
print(similarity)
