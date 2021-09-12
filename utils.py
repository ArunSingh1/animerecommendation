import pandas as pd
import numpy as np

#Load data 
data = pd.read_csv("./data/anime_list.csv")
#df = data[['mal_id', 'title', 'synopsis', 'score', 'scored_by', 'rating']]
#df = data[['mal_id', 'title', 'synopsis', 'score', 'scored_by', 'rating','genres' ]]
df = data[['mal_id', 'title', 'synopsis', 'score', 'scored_by', 'rating','genres', 'popularity','status','type','studios' ]]

#create tfidf matrix
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
#Replace NaN with an empty string
df['synopsis'] = df['synopsis'].fillna('')

#Construct the required TF-IDF matrix by applying the fit_transform method on the overview feature
tfidf_matrix = tfidf.fit_transform(df['synopsis'])

#Output the shape of tfidf_matrix
tfidf_matrix

#print(tfidf_matrix.shape)

# Import linear_kernel to compute the dot product
from sklearn.metrics.pairwise import linear_kernel
#Construct a reverse mapping of indices and movie titles, and drop duplicate titles, if any
indices = pd.Series(df.index, index=df['title']).drop_duplicates()
#print(indices)

titles = data['title']

# Function that takes in movie title as input and gives recommendations 
def content_recommender(title, tfidf_matrix=tfidf_matrix, df=df, indices=indices):
    # Obtain the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    # And convert it into a list of tuples as described above
    sim_scores = list(enumerate(linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()))

    # Sort the movies based on the cosine similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies. Ignore the first movie.
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies df[['title','synopsis']].iloc[movie_indices] 
    # df['title'].iloc[movie_indices]
    return df[['title','synopsis','genres', 'popularity','status','type','studios','rating' ]].iloc[movie_indices]