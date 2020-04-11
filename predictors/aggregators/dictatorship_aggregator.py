'''
Created on 1 Apr 2020

@author: Safey A.Halim
'''
from predictors.aggregators.base_aggregator import BaseAggregator
import pandas as pd
import numpy as np

class DictatorshipAggregator(BaseAggregator):
    
    def __init__(self, params):
        '''
        Args:
            params: a dictionary with the following arguments:
            social_context: A DataFrame that represents the social context from the dataset
            social_attribute: A string which is the social attribute based on which the dictator in the group should be chosen
        '''
        self.social_context = params['social_context']
        self.social_attribute = params['social_attribute']
        
    # Overriden method
    def calculate_group_predictions(self, individual_predictions):
        dictator = self._get_dictator(individual_predictions.user.unique())
        items = individual_predictions.item.unique()
        dictator_ratings = []
        for item in items:
            dictator_rating = individual_predictions.loc[(individual_predictions.item == item) & (individual_predictions.user == dictator), 'rating'].values[0]
            dictator_ratings.append(dictator_rating)
        return pd.Series(dictator_ratings, index=items) 
    
    def _get_dictator(self, users:np.ndarray):
        social_context = self.social_context
        sum_dict = {}
        for user in users:
            co_members = users.tolist()
            co_members.remove(user)
            social_attribute_values = social_context.loc[(social_context['to'] == user) & (social_context['from'].isin(co_members)), self.social_attribute]
            sum_dict[user] = social_attribute_values.sum()
            
        return max(sum_dict, key = sum_dict.get)
        