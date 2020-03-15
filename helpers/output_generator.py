'''
Created on 15 Mar 2020

@author: Safey A.Halim
'''

import pandas as pd
import os

OUTPUT_DIR = '../output/'

class OutputGenerator(object):

    @staticmethod
    def generate_output(recs, test_data, preds, algo_name):
        dir_path = OUTPUT_DIR + algo_name + '/'
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
        