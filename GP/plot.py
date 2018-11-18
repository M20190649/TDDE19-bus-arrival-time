""" Module for plotting the bus data.
"""
import seaborn as sns
import matplotlib.pyplot as plt

def traj_segment_grid(data):
    segs = data.seg.unique()
    n_cols = 2
    n_segs = len(segs)
    for n in range(1, n_segs+1, 2):
        _fig, ax = plt.subplots(nrows=1, ncols=n_cols, figsize=(12, 6))
        seg1 = data[data.seg == n]
        sns.scatterplot(x="lat", y="lon", data=seg1, ax=ax[0])
        seg2 = data[data.seg == n+1]
        sns.scatterplot(x="lat", y="lon", data=seg2, ax=ax[1])

def traj_segments(data):
    segs = data.seg.unique()
    _fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
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
