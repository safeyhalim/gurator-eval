'''
Created on 8 Mar 2020
@author: Safey A.Halim
'''

import pandas as pd
from entities.data_set import DataSet

class DataSetParser(object):
    @staticmethod
    def parse_dataset():
        # Loading the MovieLens dataset
        # ratings = pd.read_csv('../dataset/movie_ratings.data', sep='\t', names=['user', 'item', 'rating', 'timestamp'])
        groups = pd.read_csv('../dataset/user_groups.data', sep='\t')
        personalities = pd.read_csv('../dataset/personality.data', sep='\t', names=['user', 'personality'])
        social_context = pd.read_csv('../dataset/social_contexts_edited.data', sep='\t')
        individual_ratings = pd.read_csv('../dataset/u.data', sep='\t', names=['user', 'item', 'rating'])
        # Note: The first column is the group ratings DataFrame is 'user' although it refers to the group IDs. 
        # This is the case because the ratings DataFrame in Lenskit is expected to have the format [user, item, rating]
        group_ratings = pd.read_csv('../dataset/g.data', sep='\t', names=['user', 'item', 'rating'])
        return DataSet(individual_ratings, group_ratings, personalities, social_context, groups)