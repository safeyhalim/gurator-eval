import sys
from lenskit import batch, util
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender
from helpers.input_parser import InputParser
from helpers.dataset_parser import DataSetParser
from gurator.social_recommender_algorithm_factory import SocialRecommenderAlgorithmFactory
from entities.data_set import DataSet
from helpers.output_generator import OutputGenerator
from predictors.aggregators.aggregation import Aggregation
from gurator.ratings_type import RatingsType

AGGREGATION = Aggregation.AVG  
RATINGS_TYPE = RatingsType.EXTERNAL_GROUPS

N = 3 #number of items to recommend for user

def main():
    algo_name = InputParser.parse_input(sys.argv)
    data_set = DataSetParser.parse_dataset()
    algo_wrappers = create_recommender_algorithms(algo_name, data_set)
    ratings = get_ratings(data_set)
    # Generate recommendations
    all_recs, test_data = recommend(algo_wrappers, ratings)
    # Generate predictions
    preds = predict(algo_wrappers, ratings) 
    # Export output
    OutputGenerator.generate_output(all_recs, test_data, preds, AGGREGATION, N, RATINGS_TYPE, get_algo_name_for_output(algo_name))
    

def create_recommender_algorithms(algo_name, data_set:DataSet):
    if AGGREGATION == Aggregation.DICTATORSHIP:
        return SocialRecommenderAlgorithmFactory.create_dictatorship_recommender_algorithms(data_set)
    if algo_name == 'full-soc':
        return SocialRecommenderAlgorithmFactory.create_full_social_context_and_ii_algorithms(data_set, AGGREGATION)
    if algo_name == 'full-soc-trst':
        return SocialRecommenderAlgorithmFactory.create_full_social_context_and_ii_and_trust_algorithms(data_set, AGGREGATION)
    return SocialRecommenderAlgorithmFactory.create_recommender_algorithms([algo_name], data_set, AGGREGATION)
        
def get_ratings(data_set:DataSet):
    if RATINGS_TYPE == RatingsType.GROUPS:
        return data_set.group_ratings
    elif RATINGS_TYPE == RatingsType.INTERNAL_GROUPS:
        return data_set.internal_group_ratings
    elif RATINGS_TYPE == RatingsType.EXTERNAL_GROUPS:
        return data_set.external_group_ratings
    return data_set.individual_ratings


def get_algo_name_for_output(algo_name):
    if algo_name == 'full-soc' or algo_name == 'full-soc-trst':
        return algo_name
    return None
    
    
def recommend(algo_wrappers, ratings):
    all_recs = []
    test_data = []
    for train, test in xf.partition_users(ratings[['user', 'item', 'rating']], 5, xf.SampleFrac(0.2)):
        test_data.append(test)
        for algo_wrapper in algo_wrappers:
            all_recs.append(do_recommend(algo_wrapper, train, test))
    return all_recs, test_data

    
def do_recommend(algo_wrapper, train, test):
    fittable = util.clone(algo_wrapper.algo)
    fittable = Recommender.adapt(fittable)
    fittable.fit(train)
    users = test.user.unique()
    # now we run the recommender
    recs = batch.recommend(fittable, users, N)
    # add the algorithm name for analyzability
    recs['Algorithm'] = algo_wrapper.name
    return recs


def predict(algo_wrappers, ratings):
    all_preds = []
    for algo_wrapper in algo_wrappers:
        algo_wrapper.algo.fit(ratings)
        preds = batch.predict(algo_wrapper.algo, ratings)
        preds['Algorithm'] = algo_wrapper.name
        all_preds.append(preds)
    return all_preds
        
    
if __name__ == '__main__':
    main()