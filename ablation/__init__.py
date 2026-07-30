"""
Classical-ML ablation study (RF / GBM / SVM / kNN / LR / DT) for flow-regime
classification and Vsg/Vsl regression on pressure time-series.

Excludes data loading (bring your own X, y arrays), print statements, and
plotting -- kept: model zoo, split logic, and the KFold train/evaluate loop.
"""
