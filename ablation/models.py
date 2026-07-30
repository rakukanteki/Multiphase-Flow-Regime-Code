"""
Model zoo for the classical-ML ablation study.
"""

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .config import RANDOM_STATE


def get_classifiers():
    return {
        "RF":  RandomForestClassifier(n_estimators=300, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1),
        "GBM": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE),
        "SVM": SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=RANDOM_STATE),
        "kNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "LR":  LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "DT":  DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
    }


def get_regressors():
    """Each jointly predicts [Vsg, Vsl] (2 targets)."""
    return {
        "RF":  RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "GBM": MultiOutputRegressor(GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE)),
        "SVM": MultiOutputRegressor(SVR(kernel="rbf", C=10.0, gamma="scale")),
        "kNN": KNeighborsRegressor(n_neighbors=5, weights="distance"),
        "LR":  LinearRegression(),
        "DT":  DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE),
    }
