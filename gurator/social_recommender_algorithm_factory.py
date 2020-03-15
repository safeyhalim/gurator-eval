'''
Created on 8 Mar 2020

@author: Safey A.Halim
'''
from lenskit.algorithms import als, basic, item_knn as knn
from helpers import social_relationship_preprocessor
from entities.data_set import DataSet
from predictors.group_predictor import GroupPredictor
from predictors.social_predictor_optimized import SocialPredictorOptimized

social_attribute_dict = {'trst':'tie_strength', 'socsim':'social_similarity',
                          'domex':'domain_expertise', 'hierch':'social_hierarchy',
                          'socap':'social_capital', 'soxsim':'social_context_similarity',
                          'symp':'sympathy', 'rel':'relationship_edited'}

class SocialRecommenderAlgorithmFactory(object):
    @staticmethod
    def create_social_recommender_algorithm(algo_name, data_set:DataSet, is_group_recommender):
        social_attribute = SocialRecommenderAlgorithmFactory._get_social_attribute(algo_name)
        if social_attribute == None:
            return SocialRecommenderAlgorithmFactory._create_non_social_recommender_algorithm(algo_name)
        return SocialRecommenderAlgorithmFactory._do_create_social_recommender(social_attribute, data_set, is_group_recommender)
        
    @staticmethod
    def _do_create_social_recommender(social_attribute, data_set:DataSet, is_group_recommender):
        social_context = SocialRecommenderAlgorithmFactory._prepare_social_context(data_set)
        groups = data_set.groups
        personalities = data_set.personalities
        individual_ratings = data_set.individual_ratings
        group_ratings = data_set.group_ratings
        if is_group_recommender:
            items = group_ratings['item'].unique()
            return SocialRecommenderAlgorithmFactory._create_recommender_algorithm_with_fallback(GroupPredictor(20, groups,
                                                                                                             social_context, 
                                                                                                             personalities, 
                                                                                                             [social_attribute],
                                                                                                              items, individual_ratings))
        items = individual_ratings['item'].unique()            
        return SocialRecommenderAlgorithmFactory._create_recommender_algorithm_with_fallback(SocialPredictorOptimized(20, groups,
                                                                                                             social_context, 
                                                                                                             personalities, 
                                                                                                             [social_attribute],
                                                                                                              items))
        
    @staticmethod
    def _get_social_attribute(algo_name):
        if algo_name == 'ii' or algo_name == 'als':
            return None
        return social_attribute_dict[algo_name]
    
    @staticmethod
    def _create_non_social_recommender_algorithm(algo_name):
        if algo_name == 'ii':
            return SocialRecommenderAlgorithmFactory._create_recommender_algorithm_with_fallback(knn.ItemItem(20))
        elif algo_name == 'als':
            return als.BiasedMF(50)
        
    @staticmethod
    def _create_recommender_algorithm_with_fallback(algo):
        base = basic.Bias(damping=3)
        algo = basic.Fallback(algo, base)
        return algo
    
    @staticmethod
    def _prepare_social_context(data_set):
        social_context = social_relationship_preprocessor.remove_social_relationship_field(data_set.social_context)
        social_relationship_preprocessor.set_social_relationships_weights(social_context)
        return social_context
        
        