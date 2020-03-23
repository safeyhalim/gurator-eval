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

OUTPUT_DIR = '../output/'

def main():
    algo_name = InputParser.parse_input(sys.argv)
    data_set = DataSetParser.parse_dataset()
    aggregation = Aggregation.AVG
    
    algo = SocialRecommenderAlgorithmFactory.create_recommender_algorithm(algo_name, data_set, aggregation)
    ratings = get_ratings(data_set, aggregation)
    
    # Generate recommendations
    all_recs, test_data = recommend(algo, algo_name, ratings)
    # Generate predictions
    preds = predict(algo, ratings) 
    # Export output
    OutputGenerator.generate_output(all_recs, test_data, preds, algo_name, aggregation)


def get_ratings(data_set:DataSet, aggregation):
    if aggregation != Aggregation.NONE:
        return data_set.group_ratings
    return data_set.individual_ratings    
    
def recommend(algo, algo_name, ratings):
    all_recs = []
    test_data = []
    for train, test in xf.partition_users(ratings[['user', 'item', 'rating']], 5, xf.SampleFrac(0.2)):
        test_data.append(test)
        all_recs.append(do_recommend(algo_name, algo, train, test))
        
    return all_recs, test_data

    
def do_recommend(aname, algo, train, test):
    fittable = util.clone(algo)
    fittable = Recommender.adapt(fittable)
    fittable.fit(train)
    users = test.user.unique()
    # now we run the recommender
    recs = batch.recommend(fittable, users, 100)
    # add the algorithm name for analyzability
    recs['Algorithm'] = aname
    return recs


def predict(algo, ratings):
    algo.fit(ratings)
    return batch.predict(algo, ratings)
        
    
if __name__ == '__main__':
    main()