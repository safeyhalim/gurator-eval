'''
Created on 19 Mar 2020

@author: Safey A.Halim
'''
from abc import abstractmethod

class BaseAggregator(object):
    
    @abstractmethod
    def calculate_group_predictions(self, individual_predictions):
        """
        Calculates the aggregated predictions for the group based on the individual 
        predictions from the members of that group. Implementing classes determine
        the algorithms according to which the predictions are aggregated based on the
        Social Choice theory
        
        Args:
            individual_predictions: A DataFrame in the format [user, item, rating] represents the individual
            predictions of each user in the group for each item
        Returns: 
            A Series in the format [item, rating] represents the predicted rating for the group for each of the items
        """
    
        