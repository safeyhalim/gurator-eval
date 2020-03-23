'''
Created on 5 Mar 2020

@author: Safey A.Halim
'''
from lenskit.algorithms.item_knn import ItemItem
import pandas as pd


class GroupPredictor(ItemItem):

    def __init__(self, nnbrs, groups, individual_ratings, single_user_predictor, aggregator, min_nbrs=1, min_sim=1.0e-6, save_nbrs=None,
                 center=True, aggregate='weighted-average'):
        super(GroupPredictor, self).__init__(nnbrs, min_nbrs, min_sim, save_nbrs, center, aggregate)
        self.groups = groups
        self.individual_ratings = individual_ratings
        self.aggregator = aggregator
        self.single_user_predictor = single_user_predictor
    
    # Overriden method    
    def fit(self, ratings):
        super(GroupPredictor, self).fit(ratings)
        self.single_user_predictor.fit(self.individual_ratings)
        return self    
        
    # Overriden method
    def predict_for_user(self, user, items, ratings=None):
        '''
        The method predicts ratings a for group.
        This method is overriden from the Lenskit library's Predictor class which
        is supposed to predict ratings for individual users. That's why the 
        method's name is misleading: although the names is predict_for_user, 
        it actually predicts for a group whose ID is indicated by the method argument user
        
        Args:
            user: represents the ID of the group to which we should predict 
            (Lenskit knows only user ratings/predictions, so will have to use that)
            items: the list of items to predict ratings for the user
            ratings: ratings matrix
        '''
        return self._predict_for_group(user, items)
    
    def _predict_for_group(self, group, items):
        users = self._get_users_in_group(group)
        individual_users_predictions = pd.DataFrame(columns=['user', 'item', 'rating'])
        for user in users: 
            co_group_members = list(users)
            co_group_members.remove(user)
            results_for_user = self.single_user_predictor.predict_for_user_in_group(user, items, co_group_members)  # returns a Series of [item, rating] for that user
            individual_users_predictions = pd.concat([individual_users_predictions, self._convert_predictions_into_df(user, results_for_user)])
        return self.aggregator.calculate_group_predictions(individual_users_predictions)
    
    def _convert_predictions_into_df(self, user, predictions):
        predictions_df = pd.DataFrame({'user': user, 'item': predictions.index, 'rating': predictions.values})
        return predictions_df
    
    def _get_users_in_group(self, group):
        return self.groups.loc[self.groups['group_id'] == group, 'user_id'].tolist()
        
