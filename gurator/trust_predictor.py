from lenskit.algorithms.item_knn import ItemItem
import pandas as pd

ratings = pd.read_csv('../social_contexts.data', sep='\t')
print(ratings.head(5))

class TrustPredictor(ItemItem):
    def __init__(self, nnbrs, groups, social_context, personalities, min_nbrs=1, min_sim=1.0e-6, save_nbrs=None,
                 center=True, aggregate='weighted-average'):
        super(self.__class__, self).__init__(nnbrs, min_nbrs, min_sim, save_nbrs, center, aggregate)
        self.groups = groups
        self.social_context = social_context
        self.personalities = personalities
        
        