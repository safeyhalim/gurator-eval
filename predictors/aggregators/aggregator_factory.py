'''
Created on 19 Mar 2020

@author: Safey A.Halim
'''
from predictors.aggregators.average_aggregator import AverageAggregator
from predictors.aggregators.aggregation import Aggregation
from predictors.aggregators.least_misery_aggregator import LeastMiseryAggregator
from predictors.aggregators.most_pleasure_aggregator import MostPleasureAggregator


class AggregatorFactory(object):

    def __init__(self, aggregation: Aggregation):
        self.aggregation = aggregation
        
    def create_aggregator(self):
        if self.aggregation == Aggregation.AVG:
            return AverageAggregator()
        if self.aggregation == Aggregation.LEAST_MISERY:
            return LeastMiseryAggregator()
        if self.aggregation == Aggregation.MOST_PLEASURE:
            return MostPleasureAggregator()
        raise Exception('Unknown aggregation method')
