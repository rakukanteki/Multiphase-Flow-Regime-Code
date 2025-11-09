"""
Helper utility functions
"""
import random
import numpy as np
import torch
import os
import re


def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def extract_velocities_from_filename(filename):
    """Extract Vsg and Vsl from filename"""
    vsg_match = re.search(r'Vsg=([\d.]+)', filename)
    vsl_match = re.search(r'Vsl=([\d.]+)', filename)
    vsg = float(vsg_match.group(1).rstrip('.')) if vsg_match else 0.0
    vsl = float(vsl_match.group(1).rstrip('.')) if vsl_match else 0.0
    return vsg, vsl


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2 + 1e-8)


def euclidean_distance(vec1, vec2):
    """Calculate Euclidean distance between two vectors"""
    return np.linalg.norm(vec1 - vec2)