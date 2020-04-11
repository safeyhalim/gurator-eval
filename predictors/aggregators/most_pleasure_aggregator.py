'''
Created on 31 Mar 2020

@author: Safey A.Halim
'''

from predictors.aggregators.base_aggregator import BaseAggregator
import pandas as pd

class MostPleasureAggregator(BaseAggregator):
    
    # Overriden method
    def calculate_group_predictions(self, individual_predictions):
        items = individual_predictions.item.unique()
        avg_ratings = []
        for item in items:
            item_predictions = individual_predictions.loc[individual_predictions['item'] == item]
            avg_ratings.append(item_predictions['rating'].max())
            
        return pd.Series(avg_ratings, index=items)