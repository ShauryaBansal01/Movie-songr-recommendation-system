import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

# Load data
credits = pd.read_csv("tmdb_5000_credits.csv")
movies = pd.read_csv("tmdb_5000_movies.csv")

# Preprocess and clean the data
movies = movies[movies['overview'].notna()] 
credits_column_renamed = credits.rename(index=str, columns={"movie_id": "id"})
movies_merge = movies.merge(credits_column_renamed, on='id')
movies_cleaned = movies_merge.drop(columns=['homepage', 'title_x', 'title_y', 'status', 'production_countries'])

# Assuming tags are created from genres, overview, or other keywords
movies_cleaned['tags'] = movies_cleaned['genres'] + ' ' + movies_cleaned['overview']  # Simplified for demo purposes

# Generate TF-IDF matrix based on tags
tfv = TfidfVectorizer(min_df=3,  
                      max_features=None,
                      strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
                      ngram_range=(1, 3),
                      stop_words='english')
tfv_matrix = tfv.fit_transform(movies_cleaned['tags'])  # Using tags instead of overview

# Compute similarity matrix
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

# Map movie titles to indices for fast lookup
indices = pd.Series(movies_cleaned.index, index=movies_cleaned['original_title']).drop_duplicates()

# Recommend movies based on a single movie title
def give_recommendations_by_title(title, sig=sig):
    """
    Recommend movies based on a single movie title.

    Parameters:
    title (str): The title of the movie provided by the user.

    Returns:
    movie_indices (list): List of recommended movie indices based on title.
    """
    # Get the index of the movie that matches the title
    idx = indices.get(title, None)
    if idx is None:
        return []
    
    # Calculate similarity scores for all movies
    sig_scores = list(enumerate(sig[idx]))
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
    sig_scores = sig_scores[1:11]  # Get top 10 similar movies
    
    # Return movie indices
    return [i[0] for i in sig_scores]

# Recommend movies based on multiple tags
def give_recommendations_by_tags(input_tags, sig=sig):
    """
    Recommend movies based on tags.

    Parameters:
    input_tags (str): A space-separated string of tags input by the user.

    Returns:
    movie_indices (list): List of recommended movie indices based on tags.
    """
    # Transform input tags to match with TF-IDF matrix structure
    input_tags_matrix = tfv.transform([input_tags])
    
    # Calculate similarity scores between input tags and all movies
    sig_scores = sigmoid_kernel(input_tags_matrix, tfv_matrix)[0]
    
    # Sort movies by similarity scores in descending order
    return sig_scores.argsort()[::-1][1:11].tolist()  # Get top 10 matches excluding the input itself

# Main function to recommend movies based on both title and tags
def recommend(input_text):
    """
    Recommend movies based on both a movie title and tags.

    Parameters:
    input_text (str): The movie title and tags input by the user.

    Returns:
    recommendations (str): Comma-separated string of recommended movie titles.
    """
    # Split the input by spaces; first word is the title, remaining are tags
    parts = input_text.split()
    title = parts[0]  # First word as title
    tags = ' '.join(parts[1:]) if len(parts) > 1 else None  # Remaining words as tags
    
    title_indices = give_recommendations_by_title(title) if title else []
    tag_indices = give_recommendations_by_tags(tags) if tags else []

    # Combine and prioritize indices from title and tags
    combined_indices = title_indices + tag_indices
    unique_indices = list(dict.fromkeys(combined_indices))  # Preserve order and remove duplicates

    # Get movie titles and join them with commas
    recommended_titles = movies_cleaned['original_title'].iloc[unique_indices].values
    return ', '.join(recommended_titles)

# Get input from the user (both a movie title and tags)
user_input = input("Enter a movie title followed by tags (e.g., 'Inception sci-fi action adventure'): ")

# Display recommendations based on input
print("\nThe recommended movies are:\n")
print(recommend(user_input))
