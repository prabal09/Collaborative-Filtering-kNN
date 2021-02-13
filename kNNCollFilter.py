# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 06:49:59 2020

@author: prabal
"""

#imports
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import numpy as np

#path_movies = 'D:\\EnjoyAlgorithm\\ml-latest-small\\movies.csv'
#path_ratings = 'D:\\EnjoyAlgorithm\\ml-latest-small\\ratings.csv'
#
#df_movies = pd.read_csv(path_movies,
#    usecols=['movieId', 'title'],
#    dtype={'movieId': 'int32', 'title': 'str'})
#df_ratings = pd.read_csv(path_ratings,
#    usecols=['userId', 'movieId', 'rating'],
#    dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
## filter data
#movie_rating_thres = 50
#user_rating_thres = 50
#model = NearestNeighbors()
#n_neighbors,algorithm,metric,n_jobs = 10, 'brute', 'cosine', -1  #k = 10,20,30
#model.set_params(**{
#            'n_neighbors': n_neighbors,
#            'algorithm': algorithm,
#            'metric': metric,
#            'n_jobs': n_jobs})
##print(glob.glob(path_movies+'\*'))
##print(glob.glob(path_ratings+'\*'))
#df_movies_cnt = pd.DataFrame(
#    df_ratings.groupby('movieId').size(),
#    columns=['count'])
#popular_movies = list(set(df_movies_cnt.query('count >= @movie_rating_thres').index))  # noqa

#Preprocessing

	
# dense to sparse
#from numpy import array
#from scipy.sparse import csr_matrix
#A = array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
#C = csr_matrix(A)
#D = C.todense()
#
#print(A)
#print(C)
#print(D)

from difflib import SequenceMatcher
def seqmatch(string1,string2):
    return SequenceMatcher(None,string1, string2).ratio()



def PreProcess(df_movies,df_ratings,mf=None):
    df_mc = pd.DataFrame(df_ratings.groupby('movieId').count())
    if mf !=None:
        df_mc = df_mc[df_mc['userId']>mf]
    else:
        df_mc = df_mc[df_mc['userId']>50]
    idlist = df_mc.index;idlist = list(idlist)
    df_filter = df_ratings[df_ratings.movieId.isin(idlist).values]
    movie_user_mat = df_filter.pivot(
        index='movieId', columns='userId', values='rating').fillna(0)
    return movie_user_mat
#movie_user_mat = PreProcess(df_movies,df_ratings)
#csr_mat = csr_matrix(movie_user_mat.values)    

def MatchSequence(watched_movie,hashmap):
    match_seq = []
    # get match
    for title, idx in hashmap.items():
        sim = SequenceMatcher(None,title.lower(), watched_movie.lower()).ratio()
        #print(sim)
        #ratio = fuzz.ratio(title.lower(), fav_movie.lower())
        if sim >= 0.3:
            match_seq.append((title, idx, sim))
            match_seq = sorted(match_seq, key=lambda x: x[2])[::-1]
    if len(match_seq)>20:
        match_seq = match_seq[:20]
    return match_seq

def recommend(model,csr_mat,hashmap,watched_movie,n_recommendations = 10):
    model.fit(csr_mat)
    # get input movie index
    #print('You have input movie:', watched_movie)
    match_seq = MatchSequence(watched_movie,hashmap);
    best = {}
    if match_seq != []:
        for l in range(len(match_seq)):
            idx = match_seq[l][1]
            nearest, indices = model.kneighbors(csr_mat[idx],
                        n_neighbors=n_recommendations)
            for n,i in zip(nearest[0],indices[0]):
                best[n]=i
    best_item = list(sorted(best.items()))
    best_item = best_item[0:11]
    print('If you viewed {} you may also like'.format(watched_movie))
    cc = 1
    for key,value in hashmap.items():
        if value in dict(best_item).values():
            print('{0}:{1}'.format(cc,key))
            cc+=1

    # inference



if __name__ =="__main__":
    path_movies = 'D:\\EnjoyAlgorithm\\ml-latest-small\\movies.csv'
    path_ratings = 'D:\\EnjoyAlgorithm\\ml-latest-small\\ratings.csv'
    
    df_movies = pd.read_csv(path_movies,
        usecols=['movieId', 'title'],
        dtype={'movieId': 'int32', 'title': 'str'})
    df_ratings = pd.read_csv(path_ratings,
        usecols=['userId', 'movieId', 'rating'],
        dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
    movie_user_mat = PreProcess(df_movies,df_ratings)
    csr_mat = csr_matrix(movie_user_mat.values) 
    hashmap = {movie: i for i, movie in enumerate(list
            (df_movies.set_index('movieId').loc[movie_user_mat.index].title))} 

    model = NearestNeighbors()
    n_neighbors,metric = 10, 'cosine' #k = 10,20,30
    model.set_params(**{
                'n_neighbors': n_neighbors,
                'metric': metric})
    watched_movie = input('Enter A Movie you liked: ')
    recommend(model,csr_mat,hashmap,watched_movie)