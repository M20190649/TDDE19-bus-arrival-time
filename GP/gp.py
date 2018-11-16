""" Module for training, plotting, saving and loading GPs.
"""

from functools import reduce
import os
import math
import pickle
import gpflow
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def build(X, Y, l_prior, sigmaf_prior, sigma_prior):
    X_norm = __normalise(X)
    Y_norm = __normalise(Y)
    d = X.shape[1]

    with gpflow.defer_build():
        rbf = gpflow.kernels.RBF(d, ARD=True)
        model = gpflow.models.GPR(X_norm, Y_norm, rbf)
        model.kern.lengthscales.prior = l_prior
        model.kern.variance.prior = sigmaf_prior
        model.likelihood.variance.prior = sigma_prior

    return model

def train(model, n_runs, draw_l, draw_sigmaf, draw_sigma):
    """
    Trains n_runs times with random restarts and returns a model with the parametrisation 
    that maximises the log likelihod. The draw_* arguments are functions that 
    draw from your chosen prior to be used as seed values for the random restarts.
    """

    def run(old_best, _n):
        l, sigmaf, sigma = draw_l(), draw_sigmaf(), draw_sigma()
        set_params(model, l, sigmaf, sigma)
        gpflow.train.ScipyOptimizer().minimize(model)
        loglik = model.compute_log_likelihood()
        return (loglik, l, sigmaf, sigma) if  loglik > old_best[0] else old_best

    _loglik_max, l_max, sigmaf_max, sigma_max = reduce(run, range(n_runs), (-math.inf,))
    set_params(model, l_max, sigmaf_max, sigma_max)


### SAVE AND LOAD STUFFS ###

save_dir = r'./gps/'

def __gp_path(gp_name, seg_n):
    return save_dir + gp_name + '-' + str(seg_n) + '.pkl'

def save(model, gp_name, seg_n):
    """
    Saves a models parameters to be loaded later. The seg_n parameter makes sure that the
    "same" gp can be saved in segments.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path = __gp_path(gp_name, seg_n)
    with open(path, 'wb') as handle:
        pickle.dump(model.read_trainables(), handle, protocol=pickle.HIGHEST_PROTOCOL)
            
def load(X, Y, gp_name, seg_n):
    """
    Creates a new model and sets its parameters to the ones loaded from provided information.
    The seg_n parameter makes sure that the "same" gp can be saved in segments.
    """

    model = build(X, Y, None, None, None)
    path = __gp_path(gp_name, seg_n)
    try:
        with open(path, 'rb') as handle:
            params = pickle.load(handle)
            model.assign(params)
        return model
    
    except:
        raise ValueError('GP not found')

def plot_predictions(model, data_x, data_y):
    """
    Plots the predictions of the models posterior mean function
    against the true values of provided data.
    """
    mean, _var = model.predict_y(__normalise(data_x))
    df = pd.DataFrame(np.hstack([data_x, data_y, mean]), columns=list('xypm'))
    ax = sns.lineplot(x='p', y='m', data=df)
    ax.set(xlabel='Ground truth', ylabel='Prediction')
    return ax

def plot_posterior_mean(model, X, Y):
    """
    Plots the posterior mean of the model as a heatmap
    in the region of the provided data, with the data overlayed
    as a scatterplot. Assumes the data to be an n by 2 matrix of coordinates.
    """

    X_norm = __normalise(X)
    
    padding = 0 # Padding > 0 does not work when overlaying the data in scatterplot
    norm_latmax = X_norm[:, 0].max() + padding
    norm_latmin = X_norm[:, 0].min() - padding
    norm_longmax = X_norm[:, 1].min() - padding
    norm_longmin = X_norm[:, 1].max() + padding

    d = 200 # resolution of the heatmap
    xlist = np.linspace(norm_latmin, norm_latmax, d)
    ylist = np.linspace(norm_longmin, norm_longmax, d)
    xx, yy = np.meshgrid(xlist, ylist)
    grid = np.array([np.reshape(xx,(-1,)).T, np.reshape(yy,(-1,))]).T
    mean, _var = model.predict_y(grid)
    
    lat_unnorm = X[:, 0]
    lon_unnorm = X[:, 1]
    lat_grid = ((lat_unnorm-lat_unnorm.min())/(lat_unnorm.max() - lat_unnorm.min()))*d
    lon_grid = ((lon_unnorm-lon_unnorm.min())/(lon_unnorm.max() - lon_unnorm.min()))*d
    progress = Y[:, 0]
    df_grid = pd.DataFrame({'lat':lat_grid, 'lon': lon_grid, 'progress': progress})

    hm_grid = mean.reshape(d,d)[::-1] # Flip in y direction
    cmap = "YlGnBu"
    ax = sns.heatmap(hm_grid, vmin=0, vmax=1, cmap=cmap)
    ax.invert_yaxis()
    sns.scatterplot(x='lat', y='lon', hue='progress', palette=cmap, data=df_grid, axes=ax)
    return ax


def gamma_prior(alpha, theta):
    return gpflow.priors.Gamma(alpha, theta)
         
def __mape(pred_y, true_y):
    """
    MAPE will break when true data has 0 in it, which ours does so it is not used.
    """
    return np.mean(np.abs((true_y - pred_y) / true_y)) * 100

def __normalise(data):
    scaler = StandardScaler().fit(data)
    return scaler.transform(data)

def eval(model, metric, valid_x, valid_y):
    """
    Evaluates the model using provided metric.
    """
    gpflow.train.ScipyOptimizer().minimize(model)
    pred_y, _var = model.predict_y(valid_x)
    return metric(valid_y, pred_y)

def set_params(model, l, sigmaf, sigma):
    model.clear()
    model.kern.lengthscales = l
    model.kern.variance = sigmaf
    model.likelihood.variance = sigma
    model.compile()
