import pandas as pd
import numpy as np


r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('/home/frozenwolfy/Desktop/Ed/ML/Movie Recommender System/ml-100k/u.data', sep='\t', names=r_cols, usecols=range(3), encoding="ISO-8859-1")

# print(ratings.head())


m_cols = ['movie_id', 'title']
movies = pd.read_csv('/home/frozenwolfy/Desktop/Ed/ML/Movie Recommender System/ml-100k/u.item', sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")

# print(movies.head())
# print(ratings.head())

ratings = pd.merge(movies, ratings)                 #merging the movies and the ratings data set

#######PIVOT TABLE########
#######for user based find corr btw the rows and for item based find corr btw the columns#####
################################################################################################
userRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')
print(userRatings.head(10))

#finding correlation btw the items of the pandas data DataFrame using the user ratings as the
#weights for calculating the correaltion of the sequence.

##item based collaborative filtering

corrMatrix = userRatings.corr()
print(corrMatrix.head())

#correaltion method used is pearson correlation method

##############LEARN TYPES OF CORRELATION METHODS############

#min_periods is the minimum number of rating required for the corresponding movie to be taken
#consideration

##increasing the calue of min_periods will decrease the accuracy of your algorithm

corrMatrix = userRatings.corr(method='pearson', min_periods=100)
# corrMatrix.head()

#dropping all the non-integer values
myRatings = userRatings.loc[0].dropna()
print(myRatings.head())

simCandidates = pd.Series()
for i in range(0, len(myRatings.index)):
    print ("Adding sims for " + myRatings.index[i] + "...")
    # Retrieve similar movies to this one that I rated
    sims = corrMatrix[myRatings.index[i]].dropna()
    # Now scale its similarity by how well I rated this movie
    sims = sims.map(lambda x: x * myRatings[i])
    # Add the score to the list of similarity candidates
    simCandidates = simCandidates.append(sims)

#Glance at our results so far:
print ("sorting...")
simCandidates.sort_values(inplace = True, ascending = False)
print (simCandidates.head(10))

simCandidates = simCandidates.groupby(simCandidates.index).sum()


simCandidates.sort_values(inplace = True, ascending = False)
simCandidates.head(10)

filteredSims = simCandidates.drop(myRatings.index)
filteredSims.head(10)  
