import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel, cosine_similarity

# Load data
credits = pd.read_csv("tmdb_5000_credits.csv")
movies = pd.read_csv("tmdb_5000_movies.csv")

# Preprocess and clean the data
movies = movies[movies['overview'].notna()] 
credits_column_renamed = credits.rename(index=str, columns={"movie_id": "id"})
movies_merge = movies.merge(credits_column_renamed, on='id')
movies_cleaned = movies_merge.drop(columns=['homepage', 'title_x', 'title_y', 'status', 'production_countries'])

# Create tags from genres and overview
movies_cleaned['tags'] = movies_cleaned['genres'] + ' ' + movies_cleaned['overview']

# Generate TF-IDF matrix based on tags
tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                      ngram_range=(1, 3), stop_words='english')
tfv_matrix = tfv.fit_transform(movies_cleaned['tags'])

# Compute content-based similarity using sigmoid kernel
content_similarity = sigmoid_kernel(tfv_matrix, tfv_matrix)

# Compute collaborative-like similarity using cosine similarity
collaborative_similarity = cosine_similarity(tfv_matrix, tfv_matrix)

# Hybrid similarity: average of content-based and collaborative-like similarities
hybrid_similarity = (content_similarity + collaborative_similarity) / 2

# Map movie titles to indices for fast lookup
indices = pd.Series(movies_cleaned.index, index=movies_cleaned['original_title']).drop_duplicates()

# Recommend movies based on a single movie title
def give_recommendations_by_title(title, similarity_matrix=hybrid_similarity):
    idx = indices.get(title, None)
    if idx is None:
        return []
    
    # Calculate similarity scores for all movies
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    
    return [i[0] for i in sim_scores]

# Recommend movies based on tags
def give_recommendations_by_tags(input_tags, similarity_matrix=hybrid_similarity):
    input_tags_matrix = tfv.transform([input_tags])
    sim_scores = sigmoid_kernel(input_tags_matrix, tfv_matrix)[0]
    return sim_scores.argsort()[::-1][1:11].tolist()

# Main function to recommend movies based on both title and tags
def recommend(input_text):
    parts = input_text.split()
    title_candidate = parts[0] if parts else None
    tags = ' '.join(parts[1:]) if len(parts) > 1 else None

    if title_candidate in indices:
        title_indices = give_recommendations_by_title(title_candidate)
    else:
        title_indices = []

    if tags:
        tag_indices = give_recommendations_by_tags(tags)
    elif not title_indices:
        tag_indices = give_recommendations_by_tags(input_text)
    else:
        tag_indices = []

    combined_indices = title_indices + tag_indices
    unique_indices = list(dict.fromkeys(combined_indices))

    recommended_titles = movies_cleaned['original_title'].iloc[unique_indices].values[:5]
    return '\n\n'.join(recommended_titles)

# Get input from the user
user_input = input("Enter a movie title followed by tags, or just tags (e.g., 'Inception sci-fi action'): ")

# Display recommendations
print("\nThe recommended movies are:\n")
print(recommend(user_input))
