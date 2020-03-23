'''
Created on 8 Mar 2020

@author: Safey A.Halim
'''
from lenskit.algorithms import als, basic, item_knn as knn
from helpers import social_relationship_preprocessor
from entities.data_set import DataSet
from predictors.group_predictor import GroupPredictor
from predictors.social_predictor_optimized import SocialPredictorOptimized
from predictors.group_predictor_fallback import GroupPredictorFallback
from predictors.aggregators.aggregation import Aggregation
from predictors.aggregators.aggregator_factory import AggregatorFactory

social_attribute_dict = {'trst':'tie_strength', 'socsim':'social_similarity',
                          'domex':'domain_expertise', 'hierch':'social_hierarchy',
                          'socap':'social_capital', 'soxsim':'social_context_similarity',
                          'symp':'sympathy', 'rel':'relationship_edited'}

NEIGHBORS = 20
NUM_FEATURES = 50
DAMPING_FACTOR = 3


class SocialRecommenderAlgorithmFactory(object):

    @staticmethod
    def create_recommender_algorithm(algo_name, data_set:DataSet, aggregation):
        social_attribute = SocialRecommenderAlgorithmFactory._get_social_attribute(algo_name)
        if social_attribute == None:
            algo = SocialRecommenderAlgorithmFactory._create_non_social_recommender_algorithm(algo_name, aggregation)
        else:
            algo = SocialRecommenderAlgorithmFactory._create_social_recommender_algorithm(social_attribute, data_set, aggregation)
        if aggregation == Aggregation.NONE:
            return algo
        return GroupPredictor(NEIGHBORS, data_set.groups, data_set.individual_ratings, algo, AggregatorFactory.create_aggregator(aggregation))
        
    @staticmethod
    def _create_social_recommender_algorithm(social_attribute, data_set:DataSet, aggregation):
        groups, personalities, _, _, social_context = SocialRecommenderAlgorithmFactory._get_data_set_values(data_set)
        social_context = SocialRecommenderAlgorithmFactory._prepare_social_context(social_context)
        items = SocialRecommenderAlgorithmFactory._get_items(data_set, aggregation)
        return SocialRecommenderAlgorithmFactory._create_recommender_algorithm_with_fallback(SocialPredictorOptimized(NEIGHBORS, groups,
                                                                                                             social_context,
                                                                                                             personalities,
                                                                                                             [social_attribute],
                                                                                                              items), aggregation)
        
    @staticmethod
    def _get_social_attribute(algo_name):
        if algo_name == 'ii' or algo_name == 'als':
            return None
        return social_attribute_dict[algo_name]
    
    @staticmethod
    def _create_non_social_recommender_algorithm(algo_name, aggregation):
        if algo_name == 'ii':
            algo = knn.ItemItem(NEIGHBORS)
        elif algo_name == 'als':
            algo = als.BiasedMF(NUM_FEATURES)
        return SocialRecommenderAlgorithmFactory._create_recommender_algorithm_with_fallback(algo, aggregation)

    @staticmethod
    def _create_recommender_algorithm_with_fallback(algo, aggregation):
        base = basic.Bias(damping=DAMPING_FACTOR)
        if aggregation != Aggregation.NONE:
            algo = GroupPredictorFallback(algo, base)
        else:
            algo = basic.Fallback(algo, base)
        return algo
    
    @staticmethod
    def _prepare_social_context(social_context):
        social_context = social_relationship_preprocessor.remove_social_relationship_field(social_context)
        social_relationship_preprocessor.set_social_relationships_weights(social_context)
        return social_context
        
    @staticmethod
    def _get_data_set_values(data_set:DataSet):
        return data_set.groups, data_set.personalities, data_set.individual_ratings, data_set.group_ratings, data_set.social_context
    
    @staticmethod
    def _get_items(data_set:DataSet, aggregation):
        if aggregation != Aggregation.NONE:
            return data_set.group_ratings['item'].unique()
        return data_set.individual_ratings['item'].unique()
