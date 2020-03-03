import sys
import os
from lenskit import batch, util
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, als, basic, item_knn as knn
import pandas as pd
from predictors.social_predictor_optimized import SocialPredictorOptimized
from predictors.social_predictor import SocialPredictor
from helpers import social_relationship_preprocessor

OUTPUT_DIR = '../output/'

def main():
    algo_name = parse_input()
    # Loading the dataset
    # ratings = pd.read_csv('../dataset/movie_ratings.data', sep='\t', names=['user', 'item', 'rating', 'timestamp'])
    ratings = pd.read_csv('../dataset/u.data', sep='\t', names=['user', 'item', 'rating'])
    if algo_name == 'ii':
        algo = create_recommender_algorithm_with_fallback(knn.ItemItem(20))
    elif algo_name == 'als':
        algo = als.BiasedMF(50)
    elif algo_name == 'trst':
        algo = create_social_recommender_algorithm('tie_strength', ratings)
    elif algo_name == 'socsim':
        algo = create_social_recommender_algorithm('social_similarity', ratings)
    elif algo_name == 'domex':
        algo = create_social_recommender_algorithm('domain_expertise', ratings)
    elif algo_name == 'hierch':
        algo = create_social_recommender_algorithm('social_hierarchy', ratings)
    elif algo_name == 'socap':
        algo = create_social_recommender_algorithm('social_capital', ratings)
    elif algo_name == 'soxsim':
        algo = create_social_recommender_algorithm('social_context_similarity', ratings)
    elif algo_name == 'symp':
        algo = create_social_recommender_algorithm('sympathy', ratings)
    elif algo_name == 'rel':
        algo = create_social_recommender_algorithm('relationship_edited', ratings)
        
    # Generate recommendations
    all_recs, test_data = recommend(algo, algo_name, ratings)
    # Generate predictions
    preds = predict(algo, ratings)
    
    # Export output
    export_to_csv(all_recs, test_data, preds, algo_name)


def parse_input():
    if len(sys.argv) == 1:
        print("Error: Missing the algorithm name")
        exit()
    algo_name = sys.argv[1]
    if algo_name == 'ii'or algo_name == 'als' or algo_name == 'trst' or algo_name == 'socsim' or algo_name == 'domex' or algo_name == 'hierch' or algo_name == 'socap' or algo_name == 'soxsim' or algo_name == 'symp' or algo_name == 'rel':
        return algo_name
    print("Error: Unknown algorithm name")
    exit()
    

def create_social_recommender_algorithm(social_attribute, ratings):
    groups = pd.read_csv('../dataset/user_groups.data', sep='\t')
    personalities = pd.read_csv('../dataset/personality.data', sep='\t', names=['user', 'personality'])
    social_context = pd.read_csv('../dataset/social_contexts_edited.data', sep='\t')
    social_context = social_relationship_preprocessor.remove_social_relationship_field(social_context)
    social_relationship_preprocessor.set_social_relationships_weights(social_context)
    return create_recommender_algorithm_with_fallback(SocialPredictorOptimized(20, groups, social_context, personalities, [social_attribute], ratings['item'].unique()))
    
def create_recommender_algorithm_with_fallback(algo):
    base = basic.Bias(damping=3)
    algo = basic.Fallback(algo, base)
    return algo
    
    
def recommend(algo, algo_name, ratings):
    all_recs = []
    test_data = []
    for train, test in xf.partition_users(ratings[['user', 'item', 'rating']], 5, xf.SampleFrac(0.2)):
        test_data.append(test)
        all_recs.append(do_recommend(algo_name, algo, train, test))
        
    return all_recs, test_data


def predict(algo, ratings):
    algo.fit(ratings)
    return batch.predict(algo, ratings)

    
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


def export_to_csv(recs, test_data, preds, algo_name):
    dir_path = OUTPUT_DIR + algo_name + '/'
    create_output_dir_if_not_exists(dir_path)
    do_export_to_csv(recs, dir_path + 'recs.csv')
    do_export_to_csv(test_data, dir_path + 'testdata.csv')
    do_export_to_csv(preds, dir_path + 'preds.csv')
    
    
def do_export_to_csv(obj, file_name):
    if isinstance(obj, pd.DataFrame) == False:
        obj = pd.concat(obj, ignore_index=True)
    obj.to_csv(file_name, index=False)
    
    
def create_output_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    
if __name__ == '__main__':
    main()