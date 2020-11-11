import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import scipy.sparse as sp
from scipy.sparse.linalg import svds

sns.set_style('white')


column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=column_names)
print(df.head())

movie_titles = pd.read_csv("Movie_Id_Titles")
movie_titles.head()

df = pd.merge(df,movie_titles,on='item_id')
df.head()

n_users = df.user_id.nunique()
n_items = df.item_id.nunique()

print('Num. of Users: '+ str(n_users))
print('Num of Movies: '+str(n_items))

# print(df.groupby('title')['rating'].mean().sort_values(ascending=False).head())
# print(df.groupby('title')['rating'].count().sort_values(ascending=False).head())

train_data, test_data = train_test_split(df, test_size=0.25)
train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]  

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]

user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis]) 
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])     
    return pred

item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')

def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten() 
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

print('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
print('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))

sparsity=round(1.0-len(df)/float(n_users*n_items),3)
print('The sparsity level of MovieLens100K is ' +  str(sparsity*100) + '%')

u, s, vt = svds(train_data_matrix, k = 20)
s_diag_matrix=np.diag(s)
X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
print('User-based CF MSE: ' + str(rmse(X_pred, test_data_matrix)))

# ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
# ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())

# plt.figure(figsize=(10,4))
# ratings['num of ratings'].hist(bins=70)

# plt.figure(figsize=(10,4))
# ratings['rating'].hist(bins=70)

# sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)

# moviemat = df.pivot_table(index='user_id',columns='title',values='rating')

# print(ratings.sort_values('num of ratings',ascending=False).head(10))

# starwars_user_ratings = moviemat['Star Wars (1977)']
# liarliar_user_ratings = moviemat['Liar Liar (1997)']
# print(starwars_user_ratings.head())

# similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
# similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)

# corr_starwars = pd.DataFrame(similar_to_starwars,columns=['Correlation'])
# corr_starwars.dropna(inplace=True)

# print(corr_starwars.sort_values('Correlation',ascending=False).head(10))

# corr_starwars = corr_starwars.join(ratings['num of ratings'])
# print(corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False).head())

# corr_liarliar = pd.DataFrame(similar_to_liarliar,columns=['Correlation'])
# corr_liarliar.dropna(inplace=True)
# corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
# print(corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation',ascending=False).head())