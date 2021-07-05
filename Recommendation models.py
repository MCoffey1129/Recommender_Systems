"""Recommendation models"""

"""The code below looks at three different approaches for recommending movies from a 
   subset of the "movielens" dataset (please note this analysis needs to be reviewed):

   1. Apriori model (usually used for market based analysis):
      Uses support, confidence and lift (three terms which are
      outlined below) to determine how two products are associated with each other
   
   2. Collaborative filtering:
      Collaborative filtering leverages the power of the crowd. The intuition behind
      collaborative filtering is that if a user A likes products X and Y, and if
      another user B likes product X, there is a high chance that he will
      like the product Y as well.  

   3. Content based filtering:
      In content-based filtering, the similarity between different products is
      calculated e.g. if user 1 rates a movie of
      a specific genre really highly we would recommend a movie of the same genre
      which received a very high overall average rating
      """


############################################################################################################
                                # 1. Initial data analysis
############################################################################################################

"""Importing the libraries"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from apyori import apriori

"""Import the required datasets
   update the below code to wherever you have saved the datasets"""

movie_lookup = pd.read_csv(r'files\movies.csv')
ratings = pd.read_csv(r'files\ratings_reduced.csv')
print(movie_lookup)
ratings.head()

movie_lookup.shape # 58,098 rows and 3 columns
movie_lookup.columns # 3 columns are movieId, title and genre
ratings.shape # 199,999 rows and 4 columns
ratings.columns # columns include userId, movieId (we will join the two tables using this), rating and timestamp

"""Join the two tables"""
movie_rec_data = pd.merge(ratings,movie_lookup,how='left', on=['movieId'])
movie_rec_data.shape # 199,999 rows and 6 columns as expected
movie_rec_data.columns


"""Recommender system we will build will not be time dependent"""
movie_rec_data.drop('timestamp',axis=1, inplace=True)
movie_rec_data.shape  # 199,999 rows rows and 5 columns


"""#Typical queries used to see what your data looks like - always carry this out before completing any analysis
    on your data"""
movie_rec_data.head()
movie_rec_data.info()
movie_rec_data['rating'].describe() # average rating is 3.49, with 50% of ratings between 3 and 4
movie_rec_data.columns
movie_rec_data.isnull().sum() # there are no null values in the data

"""#Inspect the data"""
"""Highest and lowest rated movies"""
avg_score = movie_rec_data[['title','rating']].groupby('title').mean('rating').reset_index()
avg_score.sort_values('rating', ascending=False)


"""Plot the percentage of each rating
   As you can see from the plot ratings are in increments of 0.5 with over 25% of cases given a rating of 4"""
sns.set()
# plt.clf()
_=sns.histplot(data=movie_rec_data, x = 'rating', stat="probability", bins=np.arange(0.5, 6.0, 0.5)-0.25)
_= plt.xlabel("Rating")
_= plt.title('Ratings', fontsize=20)
_= plt.yticks([0, 0.05, 0.10, 0.15, 0.20, 0.25], ["0%", "5%", "10%", "15%", "20%", "25%"])
plt.plot()


############################################################################################################
                                # 2. Apriori model

# For the Apriori model- assume people who watch one movie are also going to watch another (association rule learning).
# If a person watches one movie what other movie will they watch?
# We are going to change the definition of the dataset a little, instead of using
# the rating we are going to simply assume that if a user left a rating that means they have seen the movie.
# We are going to ignore whether the person gave the movie a negative or positive review.
# Apriori models (more complex versions) are really useful for companies like Amazon who recommend other
# goods to buy at the checkout, these products are determined based off of the lift factor (see below).
############################################################################################################

# We are interested in three terms:
# Support = how many users watched a certain pair of movies divided by the total no of users
# Confidence = how many people watched movie B given they watched movie A
# Lift = Confidence / Support (strength of each rule)
# how much more likely to watch movie B are people who watched movie A than the general population of users


"""Set up the data for the Apriori model"""
movies_watched = []
list_j = []
for j in range(1, 2026): # no of unique users + 1
    df_tmp = movie_rec_data.loc[movie_rec_data['userId'] == j]
    list_j = list(df_tmp['title'])
    movies_watched.append(list_j)

print(movies_watched)
print(len(movies_watched)) # 2025 as expected
print(len(movies_watched[0])) # The first user rated 16 movies


# Training the Apriori model on the dataset
# Movie ratings is our list of lists above.

# min_support parameter will make sure that we only look at pairs of entries which appear at least 10 times (0.005)

# min_confidence parameter means the model only looks at cases that have an association confidence >20%
# i.e. 15% of users who watch movie A also watch movie B, this association will be excluded from the model

# The min_lift sets out that we will only take into account cases which have a lift factor of greater than 3
# that is cases which are 3 times more likely to watch a movie than the general population of users

# min_length and max_length will give us an association between two movies. If we put min of 2 and max of 10
# it will look for associations for between 2 and 10 movies.


rules = apriori(transactions = movies_watched, min_support = 0.005, min_confidence = 0.2
                , min_lift = 3, min_length = 2, max_length = 2)

# Visualising the results

# Displaying the first results coming directly from the output of the apriori function
results = list(rules)
print(results)

# Putting the results into a Pandas DataFrame
def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
movies_apriori = pd.DataFrame(inspect(results),
                                  columns = ['movie_watched', 'movie_recommendation', 'Support', 'Confidence', 'Lift'])

# Displaying the results non sorted
print(movies_apriori)



# Displaying the results sorted by descending lifts
# 0.59% of users watched both "Friday the 13th Part V" and "Friday the 13th Part VII" (12 out of the 2025 users have
# watched both)
# 85.71% of users who watched "Friday the 13th Part V" also watched "Part VII" (out of the 14 people who watched
# part V, 12 watched part VII)
# This translates to a Lift factor of 144.64 (85.71%/0.59%)

# Summary - 12 users out of the 2,025 users watched both movies which is very low. However we know that of the 14
# people who watched "Friday the 13th Part V" 12 of them watched "Friday the 13th Part VII"
# that leads to a lift factor of greater than 144.64, users who watch "Friday the 13th Part V" are 144.64 times
# more likely to watch "Friday the 13th Part VII" than the general population of users
# This is a very simple example of how Amazon, Netflix, Facebook, Youtube etc. can recommend different products for you
# to buy or what videos you will watch
movies_apriori.nlargest(n = 50, columns = 'Lift')


"""Write out the results in csv format to your desktop"""
movies_apriori.to_csv(r'Files\movies_apriori.csv', index=False, header=True)

# Example:
# You have just watched "Jurassic Park (1993)" assuming the only information that Netflix have to suggest other
# movie titles to you are based on the information we have above what are they most likely to recommend to you?
# Ans = Last Action Hero (1993), Speed (1994), Lost World: Jurassic Park, The (1997) and Maverick (1994)

print(movies_apriori[resultsinDataFrame['movie_watched'] == 'Jurassic Park (1993)'].nlargest(n=10,columns='Lift'))


##################################################################################################################
                            # 3. Collaborative Filtering
##################################################################################################################

"""Create a relationship between different users and ratings
Most are blank because most people have not rated the movie"""
movie_matrix = movie_rec_data.pivot_table(index='userId', columns='title', values='rating')
movie_matrix.head()

"""Unfortunately because we have used such a small dataset most of the data is Null.
 For now we will take the average of each of the columns.
 However this is not best practice and we do that here purely for the purposes
 of completing this exercise!!!!!!"""
column_means = movie_matrix.mean()
movie_matrix.fillna(column_means,inplace=True)


"""Extract the Jurassic Park info"""
jp_user_rating = movie_matrix['Jurassic Park (1993)']
jp_user_rating.dropna(inplace=True)

"""Finding the correlation of Jurassic Park with different movies"""
similar_to_jp = movie_matrix.corrwith(jp_user_rating)
similar_to_jp.head()

"""Sort the data, please note we can also put in conditional statements here to only take cases
   which have a certain number of reviews or a corrlation above a certain threshold"""
corr_jp = pd.DataFrame(similar_to_jp, columns=['correlation'])
corr_jp.dropna(inplace=True)
corr_jp = corr_jp.sort_values(by='correlation', ascending=False)

corr_jp.head(30)
# Movies such as Speed, Independence day, Terminator would be recommended based on ratings

# Please note "Speed (1994)" would be the second highest rated movie based on the Apriori model
# and would be the second highest based off of ratings correlation

##################################################################################################################
                # 4. Content-based filtering - An extremely simple example
##################################################################################################################

# Genre of Jurassic Park
movie_rec_data.loc[movie_rec_data['title'] == 'Jurassic Park (1993)']  # Action|Adventure|Sci-Fi|Thriller

genre_match = movie_rec_data.loc[movie_rec_data['genres'] == 'Action|Adventure|Sci-Fi|Thriller']
genre_avg = genre_match[['title','rating']].groupby('title').mean()
genre_avg.sort_values(by='rating', ascending=False)

# Movies with the highest average rating in the genre "Action|Adventure|Sci-Fi|Thriller" are:
# You Only Live Twice (1967), Mad Max: Fury Road (2015), Road Warrior, The (Mad Max 2) (1981) and
# Star Trek: First Contact (1996)