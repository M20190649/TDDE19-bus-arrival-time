""" Module for plotting the bus data.
"""
import seaborn as sns
import matplotlib.pyplot as plt

def traj_segment_grid(data, x, y, hue=None):
    segs = data.seg.unique()
    n_cols = 2
    n_segs = len(segs)
    for n in range(1, n_segs+1, 2):
        _fig, ax = plt.subplots(nrows=1, ncols=n_cols, figsize=(12, 6))
        seg1 = data[data.seg == n]
        sns.scatterplot(x=x, y=y, data=seg1, ax=ax[0], hue=hue)
        seg2 = data[data.seg == n+1]
        sns.scatterplot(x=x, y=y, data=seg2, ax=ax[1], hue=hue)
        ax[0].set_aspect('equal', 'datalim')
        ax[1].set_aspect('equal', 'datalim')

def traj_segments(data):
    segs = data.seg.unique()
    _fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    ax.set_aspect('equal', 'datalim')
    for seg_n in segs:
        seg = data[data.seg == seg_n]
        sns.scatterplot(x="lat", y="lon", data=seg, ax=ax)

def traj_progress(data):
    segs = data.seg.unique()
    _fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    index_data = data.copy()
    
    index_data['index'] = data.index
    for seg_n in segs:
        seg = index_data[index_data.seg == seg_n]
        sns.scatterplot(x="index", y="progress", data=seg, ax=ax)

