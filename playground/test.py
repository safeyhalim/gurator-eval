from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, als, item_knn as knn
from lenskit import topn
import pandas as pd
from matplotlib import pyplot as plt
from lenskit.metrics.topn import ndcg


def eval(aname, algo, train, test):
    fittable = util.clone(algo)
    fittable = Recommender.adapt(fittable)
    fittable.fit(train)
    users = test.user.unique()
    # now we run the recommender
    recs = batch.recommend(fittable, users, 100)
    # add the algorithm name for analyzability
    recs['Algorithm'] = aname
    return recs

ratings = pd.read_csv('u.data', sep='\t',
                      names=['user', 'item', 'rating', 'timestamp'])

print('Sample of the data set\n{}'.format(ratings.head()))


algo_ii = knn.ItemItem(20)
algo_als = als.BiasedMF(50)

all_recs = []
test_data = []
for train, test in xf.partition_users(ratings[['user', 'item', 'rating']], 5, xf.SampleFrac(0.2)):
    test_data.append(test)
    all_recs.append(eval('ItemItem', algo_ii, train, test))
    all_recs.append(eval('ALS', algo_als, train, test))

all_recs = pd.concat(all_recs, ignore_index=True)
print('All recommendations\n{}'.format(all_recs.head()))

test_data = pd.concat(test_data, ignore_index=True)
rla = topn.RecListAnalysis()
rla.add_metric(topn.ndcg)
results = rla.compute(all_recs, test_data)
print('Evaluation\n{}'.format(results.head()))

print('Mean nDCG per algorithm\n{}'.format(results.groupby('Algorithm').ndcg.mean()))

ndcg_groupedby_alg = results.groupby('Algorithm').ndcg.mean().plot.bar()
plt.savefig('sample_plot.svg', format = 'svg')
plt.show()
