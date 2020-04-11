'''
Created on 8 Mar 2020

@author: Safey A.Halim
'''

algo_names = ['all',
              'full-soc', 
              'ii', 
              'als',
               'trst',
                'socsim',
                 'domex',
                  'hierch',
                   'socap',
                    'soxsim',
                     'symp',
                      'rel']
class InputParser(object):
    @staticmethod
    def parse_input(cmd_input):
        if len(cmd_input) == 1:
            print("Error: Missing the algorithm name")
            exit()
        algo_name = cmd_input[1]
        if algo_name in algo_names:
            return algo_name
        print("Error: Unknown algorithm name")
        exit()
        