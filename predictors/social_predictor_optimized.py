from lenskit.algorithms.item_knn import ItemItem
import pandas as pd
import numpy as np
from numba import prange, njit
import math

# Globals
all_predictions = np.array([])  # column structure: [user, item, prediction]
social_context = []
SOCIAL_CAPITAL = 2
TIE_STRENGTH = 3
SOCIAL_SIMILARITY = 4
SOCIAL_CONTEXT_SIMILARITY = 5
SYMPATHY = 6
DOMAIN_EXPERTISE = 7
SOCIAL_HIERARCHY = 8
RELATIONSHIP = 9
RELATIONSHIP_EDITED = 10


@njit(nogil=True)    
def _do_social_predictions(user, items, co_groups_members, personalities, social_attribute_indices, all_predictions, social_context):
    initial_values = np.full((items.size, 1), np.nan, dtype=np.float_)
    results = np.column_stack((items, initial_values))
    for i in prange(items.shape[0]):
        item = items[i]
        # social_prediction_value = _calculate_social_prediction_for_item(user, item, co_groups_members, personalities, social_attribute_indices, all_predictions, social_context)
        social_prediction_value = _calculate_social_prediction_for_item_with_whole_social_context(user, item, co_groups_members, personalities, social_attribute_indices, all_predictions, social_context)
        results[results[:, 0] == item, 1] = social_prediction_value
    return results, all_predictions

@njit(nogil=True) 
def _calculate_social_prediction_for_item(u, item, co_groups_members, personalities, social_attribute, all_predictions, social_context):
    sum_social_attr = 0
    sum_pred = 0
    group_members_baseline_predictions = _get_all_nonnan_baseline_predictions_for_item(co_groups_members, item, all_predictions)
    if np.all(group_members_baseline_predictions == 0):
        return _get_baseline_predictions_from_ratings(u, item, all_predictions)
           
    for v in group_members_baseline_predictions[:, 0]:
        social_attr_val = _get_social_attribute_value(u, v, social_attribute, social_context)
        row = group_members_baseline_predictions[np.where(group_members_baseline_predictions[:, 0] == v)]
        predvi = row[0, 1]
        pv = _get_personality_value(v, personalities)
        sum_social_attr += social_attr_val
        sum_pred += ((predvi + pv) * social_attr_val)
    return (1 / sum_social_attr) * sum_pred

@njit(nogil=True) 
def _calculate_social_prediction_for_item_with_whole_social_context(u, item, co_groups_members, personalities, social_attribute_indices, all_predictions, social_context):
    sum_social_context = 0
    sum_pred = 0
    social_context_towards_self = social_attribute_indices.shape[0]
    group_members_baseline_predictions = _get_all_nonnan_baseline_predictions_for_item(co_groups_members, item, all_predictions)
    if np.all(group_members_baseline_predictions == 0):
        return _get_baseline_predictions_from_ratings(u, item, all_predictions)
           
    for v in group_members_baseline_predictions[:, 0]:
        social_attr_vals = np.empty(social_attribute_indices.shape[0], np.float_)
        for social_attribute_index in social_attribute_indices:
            social_attr_vals = np.append(social_attr_vals, _get_social_attribute_value(u, v, social_attribute_index, social_context))
        social_context_val = np.sum(social_attr_vals)
        row = group_members_baseline_predictions[np.where(group_members_baseline_predictions[:, 0] == v)]
        predvi = row[0, 1]
        pv = _get_personality_value(v, personalities)
        sum_social_context += social_context_val
        sum_pred += ((predvi + pv) * social_context_val)
    return (1 / (social_context_towards_self + sum_social_context)) * sum_pred

@njit(nogil=True) 
def _get_all_nonnan_baseline_predictions_for_item(users, item, all_predictions):
    preds_arr = np.zeros((1, 2))
    for i in range(len(users)):
        user = users[i]
        pred_value = _get_baseline_predictions_from_ratings(user, item, all_predictions)
        if math.isnan(pred_value) == False:
            if np.all(preds_arr == 0):
                preds_arr = np.asarray([[user, pred_value]])
            else:
                preds_arr = np.concatenate((preds_arr, np.asarray([[user, pred_value]])), axis=0)
    return preds_arr

@njit(nogil=True)
def _get_personality_value(v, personalities):
    p = personalities
    personality_row = p[np.where((p[:, 0] == v))]
    if personality_row.size == 0:
        return 0.5
    return personality_row[0, 1]

@njit(nogil=True)    
def _get_social_attribute_value(u, v, sc_index, social_context):
    sc = social_context
    social_attrs_row = sc[np.where((sc[:, 0] == u) * (sc[:, 1] == v))]
    if social_attrs_row.size == 0:
        return 0.25
    social_attr_value = social_attrs_row[0, sc_index]
    if social_attr_value == 0:
        return 0.01  # to avoid division by zero
    return social_attr_value

@njit(nogil=True)
def _get_baseline_predictions_from_ratings(user, item, all_predictions):
    r = all_predictions
    prediction_row = r[np.where((r[:, 0] == user) * (r[:, 1] == item))]
    return prediction_row[0, 2]


class SocialPredictorOptimized(ItemItem):

    def __init__(self, nnbrs, groups, social_context, personalities, social_attributes, all_items, social_attributes_indices=None, min_nbrs=1, min_sim=1.0e-6, save_nbrs=None,
                 center=True, aggregate='weighted-average'):
        super(SocialPredictorOptimized, self).__init__(nnbrs, min_nbrs, min_sim, save_nbrs, center, aggregate)
        self.groups = groups
        self.personalities = personalities.values if isinstance(personalities, pd.DataFrame) else personalities
        self.social_attributes = social_attributes
        self.social_attributes_indices = self._get_social_attributes_indices(social_context, social_attributes) if social_attributes_indices is None else social_attributes_indices
        self.social_context = social_context.values if isinstance(social_context, pd.DataFrame) else social_context
        self.all_items = all_items
        
    # Overriden method    
    def fit(self, ratings):
        super(SocialPredictorOptimized, self).fit(ratings)
        return self
    
    # Overriden method
    def predict_for_user(self, user, items, ratings=None):
        co_groups_members = self._get_co_groups_members(user)
        return self.predict_for_user_in_group(user, items, ratings, co_groups_members)
    
    def predict_for_user_in_group(self, user, items, co_group_members, ratings=None):
        if len(co_group_members) == 0:  # No groups for this user, prediction is the same as Item-Item Knn
            return super(SocialPredictorOptimized, self).predict_for_user(user, items, ratings)
        
        # Calculate the baseline predictions (Item-Item Knn) for the user and add them to the all_predictions df
        self._add_baseline_predictions_to_ratings(user, self.all_items, ratings)
              
        # Calculate the baseline predictions (Item-Item Knn) for the user's co-groups members
        self._predict_co_groups_members_ratings(self.all_items, ratings, co_group_members)
        
        # Calculate social prediction for user for each item
        global all_predictions
        co_group_members = np.asarray(co_group_members)
        social_attribute_indices = np.asarray(self.social_attributes_indices)
        results, all_predictions = _do_social_predictions(user, np.asarray(items),
                                                        co_group_members,
                                                        self.personalities, social_attribute_indices, all_predictions,
                                                        self.social_context)
        results = pd.Series(index=results[:, 0], data=results[:, 1])
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
        global all_predictions
        if all_predictions.size != 0 and user in all_predictions[:, 0]:  # If already predicted, don't do anything
            return
        predictions = super(SocialPredictorOptimized, self).predict_for_user(user, items, ratings)
        user_column = np.full((predictions.size, 1), user)
        predictions_arr = np.column_stack((user_column, predictions.index.to_numpy(), predictions.to_numpy()))
        if all_predictions.size == 0:
            all_predictions = predictions_arr
        else:
            all_predictions = np.concatenate((all_predictions, predictions_arr), axis=0)
    
    def _get_social_attributes_indices(self, social_context, attributes):
        global SOCIAL_CAPITAL, TIE_STRENGTH, SOCIAL_SIMILARITY, SOCIAL_CONTEXT_SIMILARITY, SYMPATHY, DOMAIN_EXPERTISE, SOCIAL_HIERARCHY, RELATIONSHIP, RELATIONSHIP_EDITED
        attr_idx = np.array([], dtype=np.int_)
        for attribute in attributes:
            attr_idx = np.append(attr_idx, social_context.columns.get_loc(attribute))
        return attr_idx
        
