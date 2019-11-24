import sys
import os
from lenskit import batch, util
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, als, item_knn as knn
import pandas as pd
from predictors.trust_predictor import TrustPredictor


OUTPUT_DIR = '../output/'

def main():
    algo_name = parse_input()
    # Loading the dataset
    # ratings = pd.read_csv('../dataset/movie_ratings.data', sep='\t', names=['user', 'item', 'rating', 'timestamp'])
    ratings = pd.read_csv('../dataset/u.data', sep='\t', names=['user', 'item', 'rating'])
    if algo_name == 'ii':
        algo = knn.ItemItem(20)
    elif algo_name == 'als':
        algo = als.BiasedMF(50)
    if algo_name == 'trst':
        groups = pd.read_csv('../dataset/user_groups.data', sep='\t')
        personalities = pd.read_csv('../dataset/personality.data', sep='\t', names=['user', 'personality'])
        social_context = pd.read_csv('../dataset/social_contexts.data', sep='\t')
        algo = TrustPredictor(20, groups, social_context, personalities)
    
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
    if algo_name == 'ii'or algo_name == 'als' or algo_name == 'trst':
        return algo_name
    print("Error: Unknown algorithm name")
    exit()
    
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