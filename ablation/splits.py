"""
Held-out test split + StratifiedKFold CV split logic for the ablation
study. Operates on an already-loaded label array -- no file I/O.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import N_SPLITS, RANDOM_STATE, TEST_FRAC


def make_split(y_cls: np.ndarray):
    """
    One held-out test split + N_SPLITS StratifiedKFold CV folds over the
    remaining pool -- the SAME row split can be reused for classification
    and regression so results stay directly comparable.

    Returns:
        pool_idx, test_idx : np.ndarray of row indices
        folds               : list[(train_idx, val_idx)]
    """
    idx = np.arange(len(y_cls))
    pool_idx, test_idx = train_test_split(
        idx, test_size=TEST_FRAC, stratify=y_cls, random_state=RANDOM_STATE
    )

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds = []
    for tr_rel, va_rel in skf.split(pool_idx, y_cls[pool_idx]):
        folds.append((pool_idx[tr_rel], pool_idx[va_rel]))

    return pool_idx, test_idx, folds
