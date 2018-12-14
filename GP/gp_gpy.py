

"""Convenience functions for GPs using GPy."""

import json
import os
import glob
import re
import pickle
import psycopg2 as pg
from psycopg2.extras import DictCursor
from typing import NamedTuple, List, Dict
import numpy as np
import pandas as pd

import GPy
from GPy.models import GPRegression
from sklearn.preprocessing import StandardScaler

DB_NAME = 'gp'
DB_USER = 'gp_user'
DB_PW = 'gp_pw'
LIKELIHOOD = 'likelikood'
SYNCHRONISATION = 'synch'
PREDICTION = 'pred'

class GP(NamedTuple):
    gp_type: str
    model: GPRegression
    X_scaler: StandardScaler
    Y_scaler: StandardScaler
    route_n: int
    traj_n: int
    seg_n: int
    
#    X: np.ndarray
#    Y: np.ndarray

#    name: str

## FUNCTIONS ACTING ON GP ##

def loglik(gp: GP):
    return gp.model.log_likelihood()

def train(gp: GP, n_restarts: int, messages=False):
    gp.model.optimize_restarts(n_restarts, messages=messages)

def predict(gp: GP, X: np.ndarray) -> np.ndarray:
    """
    Wraps the GPy predict function. Scales the data before predicting.
    """
    X_scaled = gp.X_scaler.transform(X)
    Y_scaled, var = gp.model.predict(X_scaled)
    return (gp.Y_scaler.inverse_transform(Y_scaled), var)

def plot(gp: GP):
    gp.model.plot()

def set_params(gp: GP, params: np.ndarray):
    gp.model[:] = params
    gp.model.update_model(True)
    return gp

def build_synch(X: np.ndarray,
                Y: np.ndarray,
                route_n: int,
                seg_n: int) -> GP:

    return build(SYNCHRONISATION, X, Y, route_n, 0, seg_n)

def build(gp_type: str,
          X: np.ndarray,
          Y: np.ndarray,
          route_n: int,
          traj_n: int,
          seg_n: int) -> GP:
    """
    Creates a wrapper data type arounnd a GPy regression model.
    This is done to enable saving of more information together with the models when
    saving them. Also scales the data. The results are terrible without scaling it.
    """
    X_scaler = StandardScaler()
    Y_scaler = StandardScaler()
    X_scaler.fit(X)
    Y_scaler.fit(Y)
    k = GPy.kern.RBF(input_dim=X.shape[1], ARD=False)
    model = GPRegression(X_scaler.transform(X), Y_scaler.transform(Y), k)
    return GP(gp_type, model, X_scaler, Y_scaler, \
              int(route_n), int(traj_n), int(seg_n))

## PRIORS ##

def set_kern_ls_prior(gp: GP, prior):
    gp.model.kern.lengthscale.set_prior(prior)

def set_kern_var_prior(gp: GP, prior):
    gp.model.kern.variance.set_prior(prior)

def set_lik_var_prior(gp: GP, prior):
    gp.model.likelihood.variance.set_prior(gp, prior)

def gamma_prior(mean: float, var: float):
    return GPy.priors.Gamma.from_EV(mean, var)






## SAVE AND LOAD STUFF ##

def __gp_path(name: str) -> str:
    gp_dir = r'./gps'
    return gp_dir + '/' + name + '/'

def __gp_file_name(route_n: int, traj_n: int, seg_n: int) -> str:
    return str(int(route_n)) \
        + '.' + str(int(traj_n)) \
        + '.' + str(int(seg_n))

def __gp_model_file(route_n: int, traj_n: int, seg_n: int) -> str:
    return __gp_file_name(route_n, traj_n, seg_n) + '.npy'

def __gp_data_file(route_n: int, traj_n: int, seg_n: int) -> str:
    return __gp_file_name(route_n, traj_n, seg_n) + '.pkl'

def __gp_file_info(file_path: str) -> (int, int, int):
    m = re.match(r'./gps/*/(\d+).(\d+).(\d+).*', file_path)
    return m.group(1), m.group(2), m.group(3)

def create_db() -> int:
    """
    Creates a database for storing the GPs in.
    Needs to be run before save and load can be done.
    """
    conn = pg.connect('dbname={} user={} password={}' \
                      .format(DB_NAME, DB_USER, DB_PW))
    print("Connection opened...", conn)

def acquire_db_conn():
    return pg.connect('dbname={} user={} password={}' \
                      .format(DB_NAME, DB_USER, DB_PW))

def save_synch(gp: GP, conn) -> None:
    """
    Special case of the save function for saving synch GPs. Synch GPs
    are uniquely defined by (route, segment), while GPs for likelihood 
    and prediction are defined by (route, trajectory, segment).
    """
    with conn.cursor() as cur:
        cur.execute('''
            DELETE FROM gp
            WHERE route = %s
            AND   segment = %s
            AND   type = 'synch'
       ;''', (gp.route_n, gp.seg_n))

        cur.execute('''
            INSERT INTO gp (route, trajectory, segment, type, model, featurescaler, targetscaler)
            VALUES (%(route)s, %(traj)s, %(seg)s, %(type)s, %(model)s, %(x_scaler)s, %(y_scaler)s)
        ''', {'route': gp.route_n,
              'traj': gp.traj_n,
              'seg': gp.seg_n,
              'type': 'synch',
              'model': json.dumps(gp.model.to_dict()),
              'x_scaler': pickle.dumps(gp.X_scaler),
              'y_scaler': pickle.dumps(gp.Y_scaler)})

        conn.commit()

def save(gp: GP, conn) -> None:
    """
    Saves the GP to a database using the provided connection.
    """
    with conn.cursor() as cur:
        cur.execute('''
            DELETE FROM gp
            WHERE route = %s
            AND   segment = %s
            AND   trajectory = %s
            AND   type = %s
       ;''', (gp.route_n, gp.seg_n, gp.traj_n, gp.gp_type))

        cur.execute('''
            INSERT INTO gp (route, trajectory, segment, type, model, featurescaler, targetscaler)
            VALUES (%(route)s, %(traj)s, %(seg)s, %(type)s, %(model)s, %(x_scaler)s, %(y_scaler)s)
        ''', {'route': gp.route_n,
              'traj': gp.traj_n,
              'seg': gp.seg_n,
              'type': gp.gp_type,
              'model': json.dumps(gp.model.to_dict()),
              'x_scaler': pickle.dumps(gp.X_scaler),
              'y_scaler': pickle.dumps(gp.Y_scaler)})

        conn.commit()

def save_data(data, path):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_data(path):
    with open(path, 'rb') as handle:
        X, Y = pickle.load(handle)
    return X, Y

def load_params(path):
    return np.load(path)

def save_params(params, path):
    return np.save(path, params)

# def load_synch(route_n: int, seg_n: int, version: str) -> GP:
#    return load('synch-' + str(version), route_n, 0, seg_n)

def load_synch(route_n: int, seg_n: int, conn) -> GP:
    """
    Special case of the plain load function for loading a synchronisation GP.
    """
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute('''
            SELECT model, featurescaler, targetscaler
            FROM gp 
            WHERE route = %s
            AND   segment = %s
            AND   type = 'synch'
        ;''', (int(route_n), int(seg_n)))
        res = cur.fetchone()
    
    model = GPRegression.from_dict(dict(res['model']))
    X_scaler = pickle.loads(res['featurescaler'])
    Y_scaler = pickle.loads(res['targetscaler'])
    return GP(SYNCHRONISATION, model, X_scaler, Y_scaler, route_n, 0, seg_n)

def load(gp_type: str, route_n: int, traj_n: int, seg_n: int, conn) -> GP:
    """
    Loads a model that has previously been saved for the provided
    name, route, traj, seg using the provided connection.
    """
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute('''
            SELECT model, featurescaler, targetscaler
            FROM gp 
            WHERE route = %s
            AND   trajectory = %s
            AND   segment = %s
            AND   type = %s
        ;''', (int(route_n), int(traj_n), int(seg_n), gp_type))
        res = cur.fetchone()

    model = GPRegression.from_dict(dict(res['model']))
    X_scaler = pickle.loads(res['featurescaler'])
    Y_scaler = pickle.loads(res['targetscaler'])
    return GP(gp_type, model, X_scaler, Y_scaler, route_n, traj_n, seg_n)

def load_trajs(gp_type: str, route_n: int, seg_n: int, conn) -> List[GP]:
    """
    Loads all GPs with the given type, trained on trajectories
    for provided route and segment.
    """
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute('''
            SELECT model, trajectory, featurescaler, targetscaler
            FROM gp 
            WHERE route = %s
            AND   segment = %s
            AND   type = %s
        ;''', (int(route_n), int(seg_n), gp_type))
        rows = cur.fetchall()

    def make_model(row):
        model = GPRegression.from_dict(dict(row['model']))
        X_scaler = pickle.loads(row['featurescaler'])
        Y_scaler = pickle.loads(row['targetscaler'])
        traj_n = row['trajectory']
        return GP(gp_type, model, X_scaler, Y_scaler, route_n, traj_n, seg_n)
    
    return [make_model(row) for row in rows]


#    spara arrival time och modellen bara
    # Find all .npy files in name save dir
#    params = [load_gp(path) for path in file_paths]
 #   return {r: {t: {s: p for _, _, s, p in params}
  #              for _, t, _, _ in params}
   #         for r, _, _, _ in params}
