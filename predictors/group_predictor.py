'''
Created on 5 Mar 2020

@author: Safey A.Halim
'''
from predictors.social_predictor_optimized import SocialPredictorOptimized

class GroupPredictor(SocialPredictorOptimized):
    
    def __init__(self, nnbrs, groups, social_context, personalities, social_attributes, all_items, individual_ratings, social_attributes_indices=None, min_nbrs=1, min_sim=1.0e-6, save_nbrs=None,
                 center=True, aggregate='weighted-average'):
        self.individual_ratings = individual_ratings
        super(GroupPredictor, self).__init__(nnbrs, groups, social_context,
                                              personalities, social_attributes,
                                               all_items, social_attributes_indices,
                                                min_nbrs, min_sim, save_nbrs, 
                                                center, aggregate)
        
    
    # Overriden method    
    def fit(self, ratings):
        super(GroupPredictor, self).fit(ratings)
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
        return self._predict_for_group(user, items, ratings)
    
    
    def _predict_for_group(self, group, items, ratings=None):
        users = self._get_users_in_group(group) 
        
    
    def _get_users_in_group(self, group):
        return self.groups.loc[self.groups['group_id'] == group, 'user_id'].tolist()
        
        