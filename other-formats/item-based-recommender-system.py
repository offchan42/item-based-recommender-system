
# coding: utf-8

# # How to run this program
# First, please make sure that you have **python3** installed (preferably **Anaconda** package).
# 
# Then use **jupyter notebook** to run the **.ipynb** file.
# 
# If you have any missing python modules, please install them using **pip install**.

# In[1]:

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# # Loading data
# Load the MovieLens data (educational version)

# In[2]:

get_ipython().run_cell_magic('time', '', 'data_path = "ml-latest-small/ratings.csv"\ndata = pd.read_csv(data_path, sep=\',\', header=0)\ndata = data[[\'userId\', \'movieId\', \'rating\']]')


# In[3]:

user_ids = sorted(set(data['userId']))
movie_ids = sorted(set(data['movieId']))
n_users = len(user_ids)
n_movies = len(movie_ids)

print("Number of users: {}\nNumber of movies: {}".format(n_users, n_movies))


# In[4]:

# show how many users have rated a certain movie
vector_sizes = data.groupby('movieId')['userId'].nunique().sort_values(ascending=False)
print(vector_sizes)
print('On average, each movie is rated {} times'.format(vector_sizes.mean()))


# In[5]:

# show examples of the data
data


# # Mean centering
# Subtract mean of rating for each user

# In[6]:

get_ipython().run_cell_magic('time', '', "# find the mean of each user\nuser_group = data.groupby(by='userId')\nuser_means = user_group['rating'].agg(['mean', 'count'])")


# In[7]:

# create a new column named "meanCenteredRating"

# this function takes in ratings of one user and return mean_centered ratings of that user
mean_centering = lambda ratings: ratings - ratings.mean()
data['meanCenteredRating'] = user_group['rating'].transform(mean_centering)
data


# In[8]:

# show user means
user_means


# # Splitting data
# Split the data into training set and test set. Prepare the training set as a user-item ratings matrix.

# In[10]:

# randomly split the data set
test_size = 0.05
data_train, data_test = train_test_split(data, test_size=test_size, random_state=42)

# show shape of the data
data_train.shape, data_test.shape


# In[11]:

get_ipython().run_cell_magic('time', '', '# build userId to row mapping dictionary\nuser2row = dict()\nrow2user = dict()\nfor i, user_id in enumerate(user_ids):\n    user2row[user_id] = i\n    row2user[i] = user_id\n\n# build movieId to column mapping dictionary\nmovie2col = dict()\ncol2movie = dict()\nfor i, movie_id in enumerate(movie_ids):\n    movie2col[movie_id] = i\n    col2movie[i] = movie_id')


# In[12]:

get_ipython().run_cell_magic('time', '', "# turn ratings data in table format into a user-item rating matrix\n# the field will be filled with NaN if user didn't provide a rating\ndef data_to_matrix(data):\n    mat = np.full((n_users, n_movies), np.nan, dtype=np.float32)\n    for idx, row in data.iterrows():\n        mat[user2row[row['userId']], movie2col[row['movieId']]] = row['meanCenteredRating']\n    return mat\n\n# prepare the data as a user-item rating matrix for the next step\ntrain_ratings = data_to_matrix(data_train)")


# # Compute similarity matrix
# Build the item-item similarity matrix. This section takes most of the processing time.

# In[13]:

# create a blank similarity matrix containing zeros
get_ipython().magic('time sim_matrix = np.empty((n_movies, n_movies), dtype=np.float32)')
sim_matrix.shape


# In[14]:

# remove co-elements from 2 vectors if at least one of them is NaN
def remove_nans(a, b):
    # assuming that a and b are 1-d vectors, create a new axis for both of them
    a = a[..., np.newaxis]
    b = b[..., np.newaxis]
    concat = np.concatenate([a, b], axis=1)
    nonan = concat[~np.isnan(concat).any(axis=1)]
    return nonan[:, 0], nonan[:, 1]

# show examples of how to use remove_nans()
a = np.array([-1,2     ,np.nan,4])
b = np.array([-2,np.nan,3     ,5])
remove_nans(a, b)


# In[15]:

# calculate a similarity value given 2 vectors
# the output is a value between -1 and 1
# min_co_elements is the number that determine whether to output NaN
# or output the similarity value, if co-elements are too low, the similarity
# will not be a good estimate, e.g. if there is only 1 co-element then the output
# will only be either -1 or 1, that's sometimes not desirable, so a threshold should be given
def calsim(item1, item2, min_co_elements=1):
    item1, item2 = remove_nans(item1, item2)
    if item1.size == 0 or item1.size < min_co_elements: # item1 and item2 must have the same size at this point
        return np.nan
#     print(item1.size)
    dot = item1.dot(item2)
    # find magnitude A.K.A. length of the vector by taking sqrt of the sum of squares of each element
    norm1 = np.linalg.norm(item1)
    norm2 = np.linalg.norm(item2)
    return dot / (norm1 * norm2)

# show example of how to use calsim()
calsim(a, b)


# In[17]:

# either load or run the next cell to compute similarity matrix
sim_matrix = np.load('sim_matrix.npy')


# In[ ]:

get_ipython().run_cell_magic('time', '', '# calculate all the similarities\nfor item1 in range(n_movies):\n    item1vector = train_ratings[:, item1]\n    for item2 in range(item1, n_movies):\n        item2vector = train_ratings[:, item2]\n        sim = calsim(item1vector, item2vector, min_co_elements=2)\n        sim_matrix[item1, item2] = sim\n        sim_matrix[item2, item1] = sim\n    if (item1+1) % 50 == 0 or item1+1 == n_movies:\n        print("Progress: {}/{} ({:.2f} %) items calculated".format(item1+1, n_movies, (item1+1)*100/n_movies))')


# In[ ]:

get_ipython().run_cell_magic('time', '', "# this sim matrix takes a lot of time to compute,\n# so saving it to the disk will help saving time in the future\nnp.save('sim_matrix', sim_matrix)")


# In[18]:

print('Fractions of similarity matrix that are NaN:', np.isnan(sim_matrix).mean())


# # Recommendation
# Test recommendation using the item-item similarity matrix built previously.
# 
# 1. We first need to define a predict() function then use it repeatedly to predict rating of every movie of a given user.
# 2. We then sort the predictions and show movies with top predictions

# In[19]:

# define a predict function which receives row and column in the ratings matrix
# then output a rating value (without mean addition), or np.nan if there are no co-items
# user_item is a tuple (user_row, movie_column)
# sim_threshold is the similarity threshold of each item,
# if the item exceeds this value, it will be chosen for averaging the outcome
def predict(ratings, user_item, sim_threshold, debug=True):
    desired_user, desired_item = user_item
    rating_sum = 0.
    total_sim = 0.
    for item in range(ratings.shape[1]):
        s = sim_matrix[item, desired_item]
        rating = ratings[desired_user, item]
        if np.isnan(s) or s < sim_threshold or item == desired_item or np.isnan(rating):
            continue
        rating_sum += s * rating
        total_sim += s
        if debug:
            print('sim and rating of item {}:'.format(item), s, rating)
    return rating_sum / total_sim if total_sim else np.nan


# In[20]:

# this is the similarity threshold value, as the only hyperparameter available
sim_threshold = 0.


# In[21]:

predict(train_ratings, (0, 30), sim_threshold), train_ratings[0, 30]


# In[22]:

# load the movie names
movie_file = "ml-latest-small/movies.csv"
movie_df = pd.read_csv(movie_file, header=0)
movie_df.head()


# In[23]:

# desired_user is the user row that we want to recommend
# return recommended item indices sorted by rating descendingly, and the associated score
def recommend(ratings, desired_user, sim_threshold):
    scores = []
    for item in range(ratings.shape[1]):
        score = ratings[desired_user, item]
        if np.isnan(score):
            score = predict(ratings, (desired_user, item), sim_threshold, debug=False)
        else:
            score = -np.infty # we don't want to recommend movies that user have rated
        scores.append(score)
    scores = np.array(scores)
    scores_argsort = np.argsort(scores)[::-1]
    scores_sort = np.sort(scores)[::-1]
    
    # numpy will put nan into the back of the array after sort
    # when we reverse the array, nan will be at the front
    # we want to move nan into the back again
    # so we use a numpy trick which rolls the array value
    # source: https://stackoverflow.com/a/35038821/2593810
    no_of_nan = np.count_nonzero(np.isnan(scores))
    scores_argsort = np.roll(scores_argsort, -no_of_nan)
    scores_sort = np.roll(scores_sort, -no_of_nan)
    return scores_argsort, scores_sort

def recommend_msg(user_row, scores_argsort, scores_sort, how_many=10):
    m = user_means.loc[row2user[user_row]]['mean']
    print('User mean rating:', m)
    msg = pd.DataFrame(columns=['movieId', 'title', 'genres', 'rating'])
    for i in range(how_many):
        col = scores_argsort[i]
        movie_id = col2movie[col]
        movie = movie_df.loc[movie_df['movieId'] == movie_id].iloc[0]
        msg.loc[i+1] = [movie_id, movie['title'], movie['genres'], scores_sort[i] + m]
    msg['movieId'] = msg['movieId'].astype(np.int32)
    return msg


# In[24]:

get_ipython().run_cell_magic('time', '', 'user = 0 # the given user\nscores_argsort, scores_sort = recommend(train_ratings, user, sim_threshold)')


# In[25]:

scores_argsort, scores_sort


# In[26]:

recommend_msg(user, scores_argsort, scores_sort, how_many=10)


# # Evaluation
# Evaluate the error on the test set. The error metric chosen in our work is **MAE**.
# 1. We need to predict mean centered ratings of every (user,movie) pair in the test data
# 2. Take the difference between the true ratings and the predicted ratings
# 3. Take the absolute
# 4. Take the mean
# 
# And that's how the error is computed.

# In[27]:

# first, let's take a look at some of the test data
data_test.head()


# In[28]:

# predict ratings for the given data table
def predict_table(data_test, sim_threshold, show_progress=True):
    n_test = data_test.shape[0]
    predictions = np.empty((n_test,))
    i = 0
    for idx, row in data_test.iterrows():
        pred = predict(train_ratings, (user2row[row['userId']], movie2col[row['movieId']]), sim_threshold, debug=False)
        predictions[i] = pred
        if show_progress and ((i+1) % 100 == 0 or i+1 == n_test):
            print("Progress: {}/{} ({:.2f} %) ratings predicted".format(i+1, n_test, (i+1)*100/n_test))
        i += 1
    if show_progress:
        print("Progress: {}/{} ({:.2f} %) ratings predicted".format(i+1, n_test, (i+1)*100/n_test))
    return predictions

def eval_error(data_test, predictions):
    return np.abs(data_test['meanCenteredRating'] - predictions).mean()


# In[29]:

get_ipython().run_cell_magic('time', '', '# predicting ratings for every (user,movie) pair in the test data\npredictions = predict_table(data_test, sim_threshold)')


# In[30]:

data_test['prediction'] = predictions
data_test.head()


# In[31]:

data_test['abs_error'] = np.abs(data_test['meanCenteredRating'] - data_test['prediction'])
data_test.head()


# In[32]:

# mean absolute error
mae = data_test['abs_error'].mean()
mae


# # Error Optimization
# Find the best set of hyperparameters that yields the lowest error on the test set.
# In this work, we use **sim_threshold (similarity threshold)** as the only hyperparameter of the system.
# 
# We can find the best **sim_threshold** by iteratively
# 1. varying its value
# 2. predict outcome on the test set
# 3. evaluate the error
# 4. if the error is less than the least error found so far, save current **sim_threshold** as the best candidate
# 
# Repeat this cycle until enough satisfaction is achieved.

# In[34]:

# define a set of avaialble similarity thresholds
candidate_sim_thresholds = np.linspace(-1, 0.75, num=16)
candidate_sim_thresholds


# In[35]:

get_ipython().run_cell_magic('time', '', "errors = np.empty_like(candidate_sim_thresholds, dtype=np.float32)\nfor i, sim_threshold in enumerate(candidate_sim_thresholds):\n    print('Current similarity threshold:', sim_threshold)\n    predictions = predict_table(data_test, sim_threshold, show_progress=False)\n    error = eval_error(data_test, predictions)\n    print('Error:', error)\n    errors[i] = error")


# In[36]:

best_error_idx = np.argmin(errors)
best_error = errors[best_error_idx]
best_sim_threshold = candidate_sim_thresholds[best_error_idx]
errors


# In[37]:

print('Optimal similarity threshold:', best_sim_threshold)
print('Optimal error:', best_error)


# Plot the error as a function of **sim_threshold**.

# In[38]:

plt.plot(candidate_sim_thresholds, errors, 'r*--')
plt.xlabel('similarity threshold')
plt.ylabel('MAE (mean absolute error)')
plt.grid()
plt.title('error as a function of sim_threshold')
plt.show()


# # Inference on the Real World
# This is the last step, all we have done to this point is now on production.
# 
# **Our task:** Given a user, recommend some movies.

# In[39]:

# choose a user_id from the data
user_id = 500


# In[51]:

# we are tring to show what movies the user have rated in the past
# we are going to sort the records by rating,
# so we can compare the result to the recommendation provided by the system visually
def get_ratings_of_user(user_id):
    user_records = data_train.loc[data_train['userId'] == user_id].sort_values(by='rating', ascending=False)
    user_records.drop(['userId', 'meanCenteredRating'], axis=1, inplace=True)
    get_movie = lambda movie_id: movie_df.loc[movie_df['movieId'] == movie_id].iloc[0]
    user_records['title'] = user_records['movieId'].apply(lambda movie_id: get_movie(movie_id)['title'])
    user_records['genres'] = user_records['movieId'].apply(lambda movie_id: get_movie(movie_id)['genres'])
    return user_records

get_ratings_of_user(user_id)


# In[40]:

get_ipython().run_cell_magic('time', '', 'user_row = user2row[user_id]\nscores_argsort, scores_sort = recommend(train_ratings, user_row, best_sim_threshold)')


# In[52]:

# recommend movies that the user have NOT rated yet
recommend_msg(user_row, scores_argsort, scores_sort, how_many=20)

