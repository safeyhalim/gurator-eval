'''
Created on 3 Apr 2020

@author: Safey A.Halim
'''

from predictors.aggregators.aggregator_factory import AggregatorFactory
from predictors.aggregators.dictatorship_aggregator import DictatorshipAggregator
from pandas import DataFrame
import string


class DictatorshipAggregatorFactory(AggregatorFactory):
    
    def __init__(self, social_attribute:string, social_context:DataFrame):
        self.social_attribute = social_attribute
        self.social_context = social_context
        
    # Overriden method
    def create_aggregator(self):
        params = {'social_context':self.social_context, 'social_attribute':self.social_attribute}
        return DictatorshipAggregator(params)
