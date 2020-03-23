'''
Created on 23 Mar 2020

@author: Safey A.Halim
'''
from lenskit.algorithms.basic import Fallback
import pandas as pd
import logging
from predictors.social_predictor_optimized import SocialPredictorOptimized

_logger = logging.getLogger(__name__)
class GroupPredictorFallback(Fallback):
    """
    GroupPredictorFallback extends basic.Fallback class. This is a workaround to be able to call the 
    SocialPredictorOptimized class from a group predictor class (e.g. GroupPredictor),
    because the SocialPredictorOptimized depends on passing the co_group_members
    to the predict_for_user method which is not possible in the current implementation of the basic.Fallback class.
    That's why the method predict_for_user_in_group is added to this class for that particular purpose.
    """
    
    def predict_for_user_in_group(self, user, items, co_group_members, ratings=None):
        remaining = pd.Index(items)
        preds = None

        for algo in self.algorithms:
            _logger.debug('predicting for %d items for user %s', len(remaining), user)
            # If the algorithm is an instance of SocialPredictorOptimized class, 
            # then it needs the co_group_members for the user to whom it's predicting, 
            # otherwise, the predict_for_user method is called as usual
            if type(algo) == SocialPredictorOptimized: 
                aps = algo.predict_for_user_in_group(user, items, co_group_members, ratings) 
            else:
                aps = algo.predict_for_user(user, remaining, ratings=ratings)
            aps = aps[aps.notna()]
            if preds is None:
                preds = aps
            else:
                preds = pd.concat([preds, aps])
            remaining = remaining.difference(preds.index)
            if len(remaining) == 0:
                break

        return preds.reindex(items)
        