'''
Created on 19 Mar 2020

@author: Safey A.Halim
'''
from predictors.aggregators.average_aggregator import AverageAggregator
from predictors.aggregators.aggregation import Aggregation

class AggregatorFactory(object):
    
    @staticmethod
    def create_aggregator(aggregation:Aggregation):
        if aggregation == Aggregation.AVG:
            return AverageAggregator()
        else:
            raise Exception('Unknown aggregation method')