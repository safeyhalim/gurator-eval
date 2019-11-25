from lenskit.algorithms.item_knn import ItemItem
import pandas as pd

class SocialPredictor(ItemItem):
    def __init__(self, nnbrs, groups, social_context, personalities, social_attributes, min_nbrs=1, min_sim=1.0e-6, save_nbrs=None,
                 center=True, aggregate='weighted-average'):
        super(self.__class__, self).__init__(nnbrs, min_nbrs, min_sim, save_nbrs, center, aggregate)
        self.groups = groups
        self.social_context = social_context
        self.personalities = personalities
        self.social_attributes = social_attributes
        self.temp_ratings = pd.DataFrame(columns=['user', 'item', 'prediction'])
        
    
    def fit(self, ratings):
        super(self.__class__, self).fit(ratings)
        return self
    
    
    def predict_for_user(self, user, items, ratings=None):
        co_groups_members = self._get_co_groups_members(user)
        if len(co_groups_members) == 0: # No groups for this user, prediction is the same as Item-Item Knn
            return super(self.__class__, self).predict_for_user(user, items, ratings)
        
        # Calculate the baseline predictions (Item-Item Knn) for the user and add them to the temp_ratings df
        self._add_baseline_predictions_to_ratings(user, items, ratings)
              
        # Calculate the baseline predictions (Item-Item Knn) for the user's co-groups members
        self._predict_co_groups_members_ratings(items, ratings, co_groups_members)
        
        # Calculate social prediction for user for each item
        results = pd.Series(index=self.item_index_)
        for item in items:
            social_prediction_value = self._calculate_social_prediction_for_item(user, item, co_groups_members, ratings)
            results.at[item] = social_prediction_value
        
        return results
    
    def __str__(self):
        return ItemItem.__str__(self)
    
    
    def _get_co_groups_members(self, user):
        groups = self.groups
        user_groups = groups.loc[groups['user_id'] == user, 'group_id'].tolist()
        co_groups_members = []
        for group in user_groups:
            co_groups_members = co_groups_members + groups.loc[groups['group_id'] == group, 'user_id'].tolist()
        
        co_groups_members = self._clear_co_groups_members_list(user, co_groups_members)    
        return co_groups_members
    
    def _clear_co_groups_members_list(self, user, co_groups_members):
        # Remove all occurrences of the user from the list of co-groups members
        while user in co_groups_members: 
            co_groups_members.remove(user)
        # Remove all duplicates in case one (or more) user is a member in more than one group of that user
        return list(set(co_groups_members))
    
    
    def _predict_co_groups_members_ratings(self, items, ratings, co_groups_members):
        for member in co_groups_members:
            self._add_baseline_predictions_to_ratings(member, items, ratings)
            
    
    def _add_baseline_predictions_to_ratings(self, user, items, ratings):
        if user in self.temp_ratings.user.values: # If already predicted, don't do anything
            return
        predictions = super(self.__class__, self).predict_for_user(user, items, ratings)
        predictions_df = pd.DataFrame({'user' : user, 'item' : predictions.index, 'prediction' : predictions.values})
        self.temp_ratings = pd.concat([self.temp_ratings, predictions_df], ignore_index=True)
        
        
    def _calculate_social_prediction_for_item(self, u, item, co_groups_members, ratings):
        sum_social_attr = 0
        sum_pred = 0
        social_attribute = self.social_attributes[0]
        for v in co_groups_members:
            social_attr_val = self._get_social_attribute_value(u, v, social_attribute)
            predvi = self._get_baseline_prediction_from_ratings(v, item, ratings)
            pv = self._get_personality_value(v)
            sum_social_attr += social_attr_val
            sum_pred += ((predvi + pv) * social_attr_val)
            
        social_prediction_value =  (1 / sum_social_attr) * sum_pred
        if social_prediction_value != None:
            return social_prediction_value
        return self._get_baseline_predictions_from_ratings(u, item)
            

    def _get_personality_value(self, v):
        p = self.personalities
        personality_row = p[(p['user'] == v)]
        if personality_row.empty:
            return 0.5
        return personality_row['personality'].values[0]
        
        
    def _get_social_attribute_value(self, u, v, attribute):
        sc = self.social_context
        social_attrs_row = sc[(sc['from'] == u) & (sc['to'] == v)]
        if social_attrs_row.empty:
            return 0.25
        social_attr_value = social_attrs_row[attribute].values[0]
        if social_attr_value == 0:
            return 0.01 # to avoid division by zero
        return social_attr_value
        
    def _get_baseline_prediction_from_ratings(self, user, item, ratings):
        r = self.temp_ratings
        prediction_row = r[(r['user'] == user) & (r['item'] == item)]
        if prediction_row.empty: # Because of the candidate selector, ratings for a certain user may be missing some items. Those missing ratings will be then added individually
            return self._add_single_item_baseline_prediction_to_ratings_and_return_it(user, item, ratings)
        return prediction_row['prediction'].values[0]
    
    
    def _add_single_item_baseline_prediction_to_ratings_and_return_it(self, user, item, ratings):
        prediction_value = super(self.__class__, self).predict_for_user(user, [item], ratings)
        prediction_value_df = pd.DataFrame({'user' : user, 'item' : item, 'prediction' : prediction_value.values})
        self.temp_ratings.append(prediction_value_df, ignore_index = True)
        return prediction_value.values[0]
        
        
        
        