"""
File-level held-out test set + K-fold GROUP cross-validation split logic.

Operates only on filenames / grouping keys -- never opens or reads file
contents. Wire this up to your own file-reading layer to turn the returned
filename lists into actual samples.
"""

import re

import numpy as np
from sklearn.model_selection import GroupKFold

from .config import N_SPLITS, SPLIT_SEED, TEST_FRAC


def extract_velocities_from_filename(filename: str):
    """
    Parse Vsg and Vsl from a filename like:
        'Dispersed_Vsg=0.50_Vsl=1.20_run3.xlsx'
    Returns (vsg, vsl) as floats.
    """
    vsg_match = re.search(r"Vsg=([\d.]+)", filename)
    vsl_match = re.search(r"Vsl=([\d.]+)", filename)
    vsg = float(vsg_match.group(1)) if vsg_match else 0.0
    vsl = float(vsl_match.group(1)) if vsl_match else 0.0
    return vsg, vsl


def extract_group_id(filename: str) -> str:
    """
    Group key used for GroupKFold: files that share the same (Vsg, Vsl)
    set-point are near-duplicate operating conditions (typically repeat
    runs at that flow rate). Keeping every repeat of the same set-point
    together -- either all in the held-out test carve-out, or all on the
    same side of a CV fold -- stops the model from being validated on a
    condition it effectively saw (a near-duplicate replicate of) during
    training.
    """
    vsg_match = re.search(r"Vsg=([\d.]+)", filename)
    vsl_match = re.search(r"Vsl=([\d.]+)", filename)
    vsg_str = vsg_match.group(1) if vsg_match else "NA"
    vsl_str = vsl_match.group(1) if vsl_match else "NA"
    return f"Vsg={vsg_str}_Vsl={vsl_str}"


def split_class_files_kfold(
    files: list,
    seed: int = SPLIT_SEED,
    n_splits: int = N_SPLITS,
    test_frac: float = TEST_FRAC,
) -> tuple:
    """
    Given a list of filenames belonging to ONE class, return:
        test_files : list[str]
        folds      : list[(train_files, val_files)], length n_splits

    Grouping (see extract_group_id) ensures every fold's train/val split
    and the held-out test carve-out are disjoint at the set-point level,
    not just the file level.
    """
    n = len(files)
    if n == 0:
        return [], [([], []) for _ in range(n_splits)]

    file_groups = {f: extract_group_id(f) for f in files}
    unique_groups = sorted(set(file_groups.values()))

    rng = np.random.RandomState(seed)
    shuffled_groups = list(unique_groups)
    rng.shuffle(shuffled_groups)

    n_groups = len(shuffled_groups)
    n_test_groups = int(round(n_groups * test_frac))
    # keep at least 1 group in the CV pool if possible
    n_test_groups = min(n_test_groups, max(0, n_groups - 1))
    test_groups = set(shuffled_groups[:n_test_groups])
    pool_groups = set(shuffled_groups[n_test_groups:])

    test_files = [f for f in files if file_groups[f] in test_groups]
    pool_files = [f for f in files if file_groups[f] in pool_groups]

    if len(pool_files) == 0:
        return test_files, [([], []) for _ in range(n_splits)]

    pool_group_list = [file_groups[f] for f in pool_files]
    n_unique_pool_groups = len(set(pool_group_list))

    actual_splits = min(n_splits, n_unique_pool_groups) if n_unique_pool_groups >= 2 else 1

    folds = []
    pool_arr = np.array(pool_files, dtype=object)
    groups_arr = np.array(pool_group_list, dtype=object)

    gkf = GroupKFold(n_splits=actual_splits)
    for train_idx, val_idx in gkf.split(pool_arr, groups=groups_arr):
        folds.append((list(pool_arr[train_idx]), list(pool_arr[val_idx])))

    # Pad out to n_splits with empty folds if there weren't enough groups
    while len(folds) < n_splits:
        folds.append(([], []))

    return test_files, folds
