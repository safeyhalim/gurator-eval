'''
Created on 15 Mar 2020

@author: Safey A.Halim
'''

import pandas as pd
import os
from predictors.aggregators.aggregation import Aggregation
from gurator.ratings_type import RatingsType

OUTPUT_DIR = '../output/'

class OutputGenerator(object):

    @staticmethod
    def generate_output(recs, test_data, preds, aggregation, n, ratings_type, special_name=None):
        dir_path = OutputGenerator._generate_dir_path(aggregation, n, special_name, ratings_type)
        OutputGenerator._create_output_dir_if_not_exists(dir_path)
        OutputGenerator._do_export_to_csv(recs, dir_path + 'recs.csv')
        OutputGenerator._do_export_to_csv(test_data, dir_path + 'testdata.csv')
        OutputGenerator._do_export_to_csv(preds, dir_path + 'preds.csv')
    
    
    @staticmethod
    def _do_export_to_csv(obj, file_name):
        if isinstance(obj, pd.DataFrame) == False:
            obj = pd.concat(obj, ignore_index=True)
        obj.to_csv(file_name, index=False)
    
    @staticmethod
    def _create_output_dir_if_not_exists(dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            
    
    @staticmethod
    def _generate_dir_path(aggregation, n, special_name, ratings_type):
        n_str = str(n)
        ratings_type_str = OutputGenerator._get_ratings_type_str(ratings_type)
        if ratings_type_str is not None:
            n_str += ratings_type_str
        if special_name is not None:
            n_str += special_name
        if aggregation == Aggregation.NONE:
            return OUTPUT_DIR + 'single-' + n_str + '/'
        return OUTPUT_DIR + aggregation.name + '-' + n_str + '/'
    
    
    @staticmethod
    def _get_ratings_type_str(ratings_type):
        if ratings_type == RatingsType.INTERNAL_GROUPS:
            return 'in'
        if ratings_type == RatingsType.EXTERNAL_GROUPS:
            return 'ex'
        return None
            
        