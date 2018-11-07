""" Module for training, saving and loading the synchronisation GP. """
from functools import reduce
import math
import pickle
import gpflow
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

def train(train_x, train_y, valid_x, valid_y, n_runs):

    # Scale data to zero mean and unit variance
    train_x_norm = __normalise(train_x)
    valid_x_norm = __normalise(valid_x)
    d = train_x.shape[1]
    
    model = __build(train_x_norm, train_y)

    # These are drawn from our priors
    ls = np.random.gamma(2, 0.9, (n_runs, d)) # was 2, 0.009 before, but it was shit. Needs to be investigated
    sigmafs = np.random.gamma(10, 0.04, n_runs)
    sigmas = np.random.gamma(5, 0.0000022, n_runs)

    def run(old_best, n):
        l, sigmaf, sigma = ls[n, :], sigmafs[n], sigmas[n]
        new_model = __set_params(model, l, sigmaf, sigma)
        mae = __eval(new_model, valid_x_norm, valid_y)
        return (mae, l, sigmaf, sigma) if  mae > old_best[0] else old_best

    mae_max, l_max, sigmaf_max, sigma_max = reduce(run, range(n_runs), (-math.inf,))
    return __set_params(model, l_max, sigmaf_max, sigma_max), mae_max
#    mae_max, l_max, sigmaf_max, sigma_max = __max_under(results, itemgetter(0))

def plot_predictions(model, data_x, data_y):
    """
    Plots the predictions of the posterior mean function
    against the true values of provided data.
    """
    mean, _var = model.predict_y(__normalise(data_x))
    df = pd.DataFrame(np.hstack([data_x, data_y, mean]), columns=list('xypm'))
    ax = sns.lineplot(x='p', y='m', data=df)
    ax.set(xlabel='Ground truth', ylabel='Prediction')
    return ax

def plot_posterior_mean(model, data):
    """
    Plots the posterior mean of the model as a heatmap
    in the region of the provided data, with the data overlayed
    as a scatterplot.
    """

    data_norm = __normalise(data)
    
    padding = 0 # Padding > 0 does not work when overlaying the data in scatterplot
    norm_latmax = data_norm[:, 0].max() + padding
    norm_latmin = data_norm[:, 0].min() - padding
    norm_longmax = data_norm[:, 1].min() - padding
    norm_longmin = data_norm[:, 1].max() + padding

    d = 200 # resolution of the heatmap
    xlist = np.linspace(norm_latmin, norm_latmax, d)
    ylist = np.linspace(norm_longmin, norm_longmax, d)
    xx, yy = np.meshgrid(xlist, ylist)
    grid = np.array([np.reshape(xx,(-1,)).T,np.reshape(yy,(-1,))]).T
    mean, _var = model.predict_y(grid)
    
    lat_unnorm = data['lat']
    lon_unnorm = data['lon']
    lat_grid = ((lat_unnorm-lat_unnorm.min())/(lat_unnorm.max() - lat_unnorm.min()))*d
    lon_grid = ((lon_unnorm-lon_unnorm.min())/(lon_unnorm.max() - lon_unnorm.min()))*d
    df_grid = pd.DataFrame({'lat':lat_grid, 'lon': lon_grid})

    hm_grid = mean.reshape(d,d)[::-1] # Flip in y direction
    ax = sns.heatmap(hm_grid, vmin=0, vmax=1, cmap="YlGnBu")
    ax.invert_yaxis()
    sns.scatterplot(x='lat', y='lon', data=df_grid, color=".2", axes=ax)

    return ax

    # scaler = StandardScaler().fit(grid)
    # grid_norm = scaler.transform(grid)
    # mean, _var = model.predict_y(grid_norm)
    # hm_grid = mean.reshape(d, d)[::-1] # Reshape to correct dim and flip in y direction
    # ax = sns.heatmap(hm_grid, vmin=0, vmax=1, cmap="YlGnBu")
    # ax.invert_yaxis() # Invert Y-axis to get same orientation as the data
    # return ax
    
def load(X, Y, path):
    model = __build(X, Y)
    with open(path, 'rb') as handle:
        params = pickle.load(handle)
        model.assign(params)
    return model

def save(path, model):
    with open(path, 'wb') as handle:
        pickle.dump(model.read_trainables(), handle, protocol=pickle.HIGHEST_PROTOCOL)

def __max_under(xs, f):
    max_under_f = lambda x, y: x if f(x) > f(y) else y
    return reduce(max_under_f, xs, (-math.inf,))

def __build(X, Y):
    with gpflow.defer_build():
        rbf = gpflow.kernels.RBF(X.shape[1], ARD=True)
        model = gpflow.models.GPR(X, Y, rbf)
    return model

def __mape(pred_y, true_y):
    return np.mean(np.abs((true_y - pred_y) / true_y)) * 100

def __normalise(data):
    scaler = StandardScaler().fit(data)
    return scaler.transform(data)
    
#MAPE gives undefined results for data points where we
#have arrived and the remaining time is 0 so we
#evaluate the model and store its parameters and MAE
def __eval(model, valid_x, valid_y):
    gpflow.train.ScipyOptimizer().minimize(model)
    pred_y, var = model.predict_y(valid_x)
    return mean_absolute_error(valid_y, pred_y)

def __set_params(model, l, sigmaf, sigma):
    model.clear()
    model.kern.lengthscales = l
    model.kern.variance = sigmaf
    model.likelihood.variance = sigma
    model.compile()
    return model
        
