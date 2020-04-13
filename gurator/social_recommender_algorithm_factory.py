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
from predictors.aggregators.dictatorship_aggregator_factory import DictatorshipAggregatorFactory
from gurator.algorithm_wrapper import AlgorithmWrapper
from _cffi_backend import string

social_attribute_dict = {'trst':'tie_strength', 'socsim':'social_similarity',
                          'domex':'domain_expertise', 'hierch':'social_hierarchy',
                          'socap':'social_capital', 'soxsim':'social_context_similarity',
                          'symp':'sympathy', 'rel':'relationship_edited'}

NEIGHBORS = 20
NUM_FEATURES = 50
DAMPING_FACTOR = 3


class SocialRecommenderAlgorithmFactory(object):
    
    @staticmethod
    def create_recommender_algorithms(algo_names:[string], data_set:DataSet, aggregation) -> [AlgorithmWrapper]:
        if algo_names[0] == 'all':
            return SocialRecommenderAlgorithmFactory._do_create_recommender_algorithms(list(social_attribute_dict.keys()) + ['ii'], data_set, aggregation)
        return SocialRecommenderAlgorithmFactory._do_create_recommender_algorithms(algo_names, data_set, aggregation)
                
    @staticmethod
    def create_dictatorship_recommender_algorithms(data_set:DataSet) -> [AlgorithmWrapper]:
        '''
            Creates a list that contains a basic item-item collaborative 
            filtering group recommender with aggregation strategy "Average", 
            and other basic item-item collaborative group recommender algorithms
            with dictatorship aggregation strategy for each. For each dictatorship aggregation strategies, 
            a different social context attribute will be the parameter based on which the dictator will be chosen.
            The aim here will be to compare, for each algorithm, the influence of
            the aggregation strategy alone on the outcome of the recommender and 
            compare each of the dictatorship based algorithm against the 
            first item-item algorithm with Average aggregation strategy 
            (which is the baseline algorithm for this use case).
        '''
        social_context = SocialRecommenderAlgorithmFactory._prepare_social_context(data_set.social_context)
        social_attribute_keys = list(social_attribute_dict.keys())
        algo_wrappers = []
        # First create the baseline line algorithm: An item-item collaborative filtering group recommender with aggregation strategy: average
        algo_wrappers.append(AlgorithmWrapper('ii', SocialRecommenderAlgorithmFactory.create_recommender_algorithm('ii', data_set, Aggregation.AVG)))
        # Then, create a list of item-item group recommenders with dictatorship 
        # aggregation strategy, for each a different social context attribute 
        # is used to chose the dictator while aggregating the group predictions
        for social_attribute_key in social_attribute_keys:
            algo_name = 'ii-' + social_attribute_key
            algo = SocialRecommenderAlgorithmFactory._create_non_social_recommender_algorithm('ii', Aggregation.DICTATORSHIP)
            dictatorship_aggregator = DictatorshipAggregatorFactory(social_attribute_dict[social_attribute_key], social_context).create_aggregator()
            group_predictor = GroupPredictor(NEIGHBORS, data_set.groups, data_set.individual_ratings, algo, dictatorship_aggregator)
            algo_wrappers.append(AlgorithmWrapper(algo_name, group_predictor))
        return algo_wrappers
    
    @staticmethod
    def create_full_social_context_and_ii_algorithms(data_set:DataSet, aggregation:Aggregation) -> [AlgorithmWrapper]:
        '''
            Creates a list of AlgoWrapper objects that contains two group recommender algorithms with the passed aggregation strategy.
            The first is a baseline item-item collaborative filtering. The second is a full social context recommender algorithm
            which will generate predictions based on ALL the social context attributes defined in the social_attribute_dict
        '''
        aggregator = AggregatorFactory(aggregation).create_aggregator()
        ii_algo = GroupPredictor(NEIGHBORS, data_set.groups, data_set.individual_ratings, 
                                 SocialRecommenderAlgorithmFactory._create_non_social_recommender_algorithm('ii', aggregation), aggregator)
        social_context_attributes = list(social_attribute_dict.values())
        social_context_algo = GroupPredictor(NEIGHBORS, data_set.groups, data_set.individual_ratings, 
                                 SocialRecommenderAlgorithmFactory._create_social_recommender_algorithm(social_context_attributes, data_set, aggregation), aggregator)
        return [AlgorithmWrapper('ii', ii_algo), AlgorithmWrapper('socntxt', social_context_algo)]
    
    @staticmethod
    def create_full_social_context_and_ii_and_trust_algorithms(data_set:DataSet, aggregation:Aggregation) -> [AlgorithmWrapper]:
        '''
            Creates a list of AlgoWrapper objects that contains three group recommender algorithms with the passed aggregation strategy.
            The first is a baseline item-item collaborative filtering, the second is a social recommender algorithm with tie strength 
            (proxy for trust) as the social context attribute based on which the recommender generates predictions, and the third is 
            a full social context recommender algorithm which will generate predictions based on ALL the social context attributes 
            defined in the social_attribute_dict (also including tie strength as a proxy for trust)
        '''
        aggregator = AggregatorFactory(aggregation).create_aggregator()
        tie_strength_algo = GroupPredictor(NEIGHBORS, data_set.groups, data_set.individual_ratings, 
                                 SocialRecommenderAlgorithmFactory._create_social_recommender_algorithm([social_attribute_dict['trst']], data_set, aggregation), aggregator)
        algorithm_wrappers = SocialRecommenderAlgorithmFactory.create_full_social_context_and_ii_algorithms(data_set, aggregation)
        algorithm_wrappers.append(AlgorithmWrapper('trst', tie_strength_algo))
        return algorithm_wrappers
        
        
    @staticmethod
    def _do_create_recommender_algorithms(algo_names, data_set:DataSet, aggregation):
        algo_wrappers = []
        for algo_name in algo_names:
            algo_wrappers.append(AlgorithmWrapper(algo_name, SocialRecommenderAlgorithmFactory.create_recommender_algorithm(algo_name, data_set, aggregation)))
        return algo_wrappers
            
        
    @staticmethod
    def create_recommender_algorithm(algo_name, data_set:DataSet, aggregation):
        social_attribute = SocialRecommenderAlgorithmFactory._get_social_attribute(algo_name)
        if social_attribute == None:
            algo = SocialRecommenderAlgorithmFactory._create_non_social_recommender_algorithm(algo_name, aggregation)
        else:
            algo = SocialRecommenderAlgorithmFactory._create_social_recommender_algorithm([social_attribute], data_set, aggregation)
        if aggregation == Aggregation.NONE:
            return algo
        return GroupPredictor(NEIGHBORS, data_set.groups, data_set.individual_ratings, algo, AggregatorFactory(aggregation).create_aggregator())
        
    @staticmethod
    def _create_social_recommender_algorithm(social_attribute:[string], data_set:DataSet, aggregation):
        groups, personalities, _, _, social_context = SocialRecommenderAlgorithmFactory._get_data_set_values(data_set)
        social_context = SocialRecommenderAlgorithmFactory._prepare_social_context(social_context)
        items = SocialRecommenderAlgorithmFactory._get_items(data_set, aggregation)
        return SocialRecommenderAlgorithmFactory._create_recommender_algorithm_with_fallback(SocialPredictorOptimized(NEIGHBORS, groups,
                                                                                                             social_context,
                                                                                                             personalities,
                                                                                                             social_attribute,
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
