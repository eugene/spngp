import numpy as np
import pandas as pd
from learnspngp import build, query, build_bins
from spngp import structure

np.random.seed(58)

data = pd.read_csv('datasets/ccpp/ccpp.csv')
data = pd.DataFrame(data).dropna() # miss = data.isnull().sum()/len(data)
dmean, dstd = data.mean(), data.std()
data = (data-dmean)/dstd

# GPSPN on full data
train = data.sample(frac=0.8, random_state=58)
test  = data.drop(train.index)
x, y = train.iloc[:, :-1].values, train.iloc[:, -1].values.reshape(-1,1)

opts = {
    'min_samples':          0,
    'X':                    x, 
    'qd':                   4, 
    'max_depth':            4, 
    'max_samples':     10**10, 
    'log':               True,
    'jump':              True,
    'reduce_branching':  True
}
root_region, gps_ = build_bins(**opts)
#root_region, gps_ = build(X=x, delta_divisor=3, max_depth=2)
root, gps         = structure(root_region, gp_types=['rbf'])

for i, gp in enumerate(gps):
    idx = query(x, gp.mins, gp.maxs)
    gp.x, gp.y = x[idx], y[idx]
    print(f"Training GP {i+1}/{len(gps)} ({len(idx)})")
    gp.init(cuda=True)

root.update()

for smudge in np.arange(0, 0.5, 0.05):
    mu_s, cov_s = root.forward(test.iloc[:, 0:4].values, smudge=smudge)
    mu_s = (mu_s.ravel() * dstd.iloc[-1]) + dmean.iloc[-1]
    mu_t = (test.iloc[:, -1]*dstd.iloc[-1]) + dmean.iloc[-1]
    sqe = (mu_s - mu_t.values)**2
    rmse = np.sqrt(sqe.sum()/len(test))
    print(f"SPN-GP (smudge={round(smudge, 4)}) \t RMSE: {rmse}")
