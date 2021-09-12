import pandas as pd
import numpy as np

#Load data 
data = pd.read_csv("./data/anime_list.csv")
df = data[['mal_id', 'title', 'synopsis', 'score', 'scored_by', 'rating']]

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