import numpy as np

def calculate_sparsity(ratings):
    """
    Calculates the sparsity level of the ratings dataset. 
    The sparsity level is defined as: (number of observed ratings)/(number of all possible ratings)
    it's indicates how sparse is the ratings dataset
    
    Args: 
        ratings: Pandas DataFrame of the ratings dataset
        
    Returns: Floating point number represents the sparsity level of the ratings dataset
    """
    ratings_matrix = ratings.pivot(index='item', columns='user', values='rating')
    total_num_ratings = ratings_matrix.size
    
    sparsity = _get_observed_ratings(ratings_matrix) / total_num_ratings
    return sparsity



def _get_observed_ratings(ratings_matrix):
    """
    Calculates the number of non-null ratings (observed ratings) from a matrix 
    formed Pandas DataFrame
    
    Args: 
        ratings_matrix: Pandas DataFrame in matrix form (DataFrame on which the pivot
        method was called)
    
    Returns: number of non-null (observed) ratings in the matrix
    """
    num_ratings_per_user = np.logical_not(ratings_matrix.isnull()).sum()
    return num_ratings_per_user.sum()
    