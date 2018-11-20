"""Functions for working with the synchronisation GPs."""

import gp_gpy as gp
from gp_gpy import GP
from typing import NamedTuple, List, Dict
import numpy as np
import pandas as pd

def build(data: pd.DataFrame,
          X: List[str],
          Y: List[str],
          seg_n: int) -> GP:
    """
    Builds a synchronisation GP for a segment.
    """
    return gp.build(data, X, Y, 'synch', 0, 0, seg_n)
