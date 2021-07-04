"""Recommendation model"""

# Apriori - Association rule learning: people who buy something also buy something else
# This dataset is a basket items, you want to see what products are the most associated.
# If they buy one product what other product will they buy?
# We are interested in Support = how many ppl bought this product / total no of transactions
# I think the support number may not be 100% correct in the tables below
# Confidence = how many bought this product given they bought a different product
# Lift = Confidence / Support (strength of each rule)

"""Importing the libraries"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

"""Import the required datasets"""

# Data Preprocessing
movie_lookup = pd.read_csv(r'files\movies.csv')
ratings = pd.read_csv(r'files\ratings.csv')
print(movie_lookup)
ratings.head()

movie_lookup.shape # 58,098 rows and 3 columns
movie_lookup.columns # 3 columns are movieId, title and genre
ratings.shape # 27,753,444 rows and 4 columns
ratings.columns # columns include userId, movieId (we will join on this col), rating and timestamp

"""Join the two tables"""
movie_rec_data = pd.merge(ratings,movie_lookup,how='left', on=['movieId'])
movie_rec_data.shape # 27,753,444 rows and 6 columns
movie_rec_data.columns
movie_rec_data.head()

"""Recommender system we will build will not be time dependent"""
movie_rec_data.drop('timestamp',axis=1, inplace=True)
movie_rec_data.shape  # 27,753,444 rows and 5 columns


"""#Typical queries used to see what your data looks like - always carry this out before completing any analysis
    on your data"""
movie_rec_data.head()
movie_rec_data.info()
movie_rec_data['rating'].describe() # average rating is 3.53, with 75% of ratings between 3 and 4
movie_rec_data.columns
movie_rec_data.isnull().sum() # there are no null values in the data which is great for us

"""#Inspect the data"""
"""Highest and lowest rated movies"""
avg_score = movie_rec_data[['title','rating']].groupby('title').mean('rating').reset_index()
avg_score.sort_values('rating', ascending=False)


""""""
sns.set()
_=sns.histplot(data=movie_rec_data, x = 'rating')
plt.plot()

"""Most paired movies"""
movie_rec_data[['userId','movieId']]

