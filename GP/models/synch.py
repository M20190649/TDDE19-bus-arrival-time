import gpflow
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import pickle

def __build(X, Y, use_priors):
   with gpflow.defer_build():
        rbf = gpflow.kernels.RBF(2, ARD=True)
        m = gpflow.models.GPR(X, Y, rbf)
    
        if use_priors:
            m.clear()
            m.kern.lengthscales.prior = gpflow.priors.Gamma(5, 0.012)
            m.kern.variance.prior = gpflow.priors.Gamma(5., 8.)
            m.likelihood.variance.prior = gpflow.priors.Gamma(5., 0.000012)

        return m

     
def train(X_unnorm, Y, duplicate_data, use_priors):
    scaler = StandardScaler().fit(X_unnorm)
    X = scaler.transform(X_unnorm)
   
    if duplicate_data:
        u, s, vh = np.linalg.svd(X)
        v1 = 1.5*vh[1,:]
        X = np.vstack((X, X+v1, X-v1))
        Y = np.vstack((Y, Y, Y))
        
    m = __build(X, Y, use_priors)
    m.compile()
    opt = gpflow.train.ScipyOptimizer()
    opt.minimize(m)
    print(m.as_pandas_table())
    return m, X, Y

def load(X, Y, path):
    m = __build(X, Y, False)    
    with open(path, 'rb') as handle:
        params = pickle.load(handle)
        m.assign(params)
    return m
        
def save(path, m):
    with open(path, 'wb') as handle:
        pickle.dump(m.read_trainables(), handle, protocol=pickle.HIGHEST_PROTOCOL)

