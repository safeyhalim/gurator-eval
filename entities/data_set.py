'''
Created on 8 Mar 2020

@author: Safey A.Halim
'''

class DataSet(object):
    def __init__(self, individual_ratings, group_ratings, internal_group_ratings, external_group_ratings, personalities, social_context, groups):
        self.individual_ratings = individual_ratings
        self.group_ratings = group_ratings
        self.internal_group_ratings = internal_group_ratings
        self.external_group_ratings = external_group_ratings
        self.personalities = personalities
        self.social_context = social_context
        self.groups = groups
        
        