"""
KFold train/evaluate loops for the classical-ML ablation study.

Accepts already-loaded feature/label arrays (X, y_*) -- no file reading.
No print statements, no plotting.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.preprocessing import StandardScaler

from .config import CLASS_NAMES
from .models import get_classifiers, get_regressors
from .stats import confidence_interval


def run_classification_ablation(X, y_cls, pool_idx, test_idx, folds):
    """
    For each classifier in the model zoo: run StratifiedKFold CV over the
    pool, then fit once on the full pool and evaluate on the held-out test
    set. Returns per-model CV/test summaries and per-class TEST metrics.
    """
    X_test, y_test = X[test_idx], y_cls[test_idx]
    X_pool, y_pool = X[pool_idx], y_cls[pool_idx]

    cv_rows = []
    test_summary_rows = []
    per_class_rows = []
    fitted_models = {}

    for name, base_model in get_classifiers().items():
        fold_accs, fold_f1s = [], []
        fold_class_precision = {c: [] for c in CLASS_NAMES}
        fold_class_recall = {c: [] for c in CLASS_NAMES}
        fold_class_f1 = {c: [] for c in CLASS_NAMES}

        for i, (tr_idx, va_idx) in enumerate(folds):
            X_tr, y_tr = X[tr_idx], y_cls[tr_idx]
            X_va, y_va = X[va_idx], y_cls[va_idx]

            scaler = StandardScaler().fit(X_tr)
            X_tr_s, X_va_s = scaler.transform(X_tr), scaler.transform(X_va)

            model = get_classifiers()[name]
            model.fit(X_tr_s, y_tr)
            pred = model.predict(X_va_s)

            acc = accuracy_score(y_va, pred)
            f1 = f1_score(y_va, pred, average="macro")
            fold_accs.append(acc)
            fold_f1s.append(f1)
            cv_rows.append({"model": name, "fold": i + 1, "val_accuracy": acc, "val_f1_macro": f1})

            fold_report = classification_report(
                y_va, pred, labels=list(range(len(CLASS_NAMES))),
                target_names=CLASS_NAMES, output_dict=True, zero_division=0,
            )
            for c in CLASS_NAMES:
                fold_class_precision[c].append(fold_report[c]["precision"])
                fold_class_recall[c].append(fold_report[c]["recall"])
                fold_class_f1[c].append(fold_report[c]["f1-score"])

        # Fit final model on the FULL pool, evaluate once on the untouched test set
        scaler = StandardScaler().fit(X_pool)
        X_pool_s, X_test_s = scaler.transform(X_pool), scaler.transform(X_test)
        final_model = get_classifiers()[name]
        final_model.fit(X_pool_s, y_pool)
        test_pred = final_model.predict(X_test_s)

        test_acc = accuracy_score(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred, average="macro")
        acc_ci_mean, acc_ci_half = confidence_interval(fold_accs)
        f1_ci_mean, f1_ci_half = confidence_interval(fold_f1s)

        report_dict = classification_report(
            y_test, test_pred, labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES, output_dict=True, digits=4, zero_division=0,
        )

        for c in CLASS_NAMES:
            p_mean, p_ci = confidence_interval(fold_class_precision[c])
            r_mean, r_ci = confidence_interval(fold_class_recall[c])
            f_mean, f_ci = confidence_interval(fold_class_f1[c])
            per_class_rows.append({
                "model": name,
                "class": c,
                "cv_precision_mean": p_mean, "cv_precision_ci95": p_ci,
                "cv_recall_mean": r_mean, "cv_recall_ci95": r_ci,
                "cv_f1_mean": f_mean, "cv_f1_ci95": f_ci,
                "test_precision": report_dict[c]["precision"],
                "test_recall": report_dict[c]["recall"],
                "test_f1": report_dict[c]["f1-score"],
                "test_support": report_dict[c]["support"],
            })

        fitted_models[name] = {"model": final_model, "scaler": scaler}
        test_summary_rows.append({
            "model": name,
            "cv_accuracy_mean": np.mean(fold_accs), "cv_accuracy_std": np.std(fold_accs),
            "cv_accuracy_ci95": acc_ci_half,
            "cv_f1_macro_mean": np.mean(fold_f1s), "cv_f1_macro_std": np.std(fold_f1s),
            "cv_f1_macro_ci95": f1_ci_half,
            "test_accuracy": test_acc, "test_f1_macro": test_f1,
        })

    return {
        "cv_rows": cv_rows,
        "test_summary_rows": test_summary_rows,
        "per_class_rows": per_class_rows,
        "fitted_models": fitted_models,
    }


def run_regression_ablation(X, y_vsg, y_vsl, pool_idx, test_idx, folds):
    """
    For each regressor in the model zoo: run KFold CV over the pool
    (jointly predicting [Vsg, Vsl]), then fit once on the full pool and
    evaluate on the held-out test set.
    """
    X_test = X[test_idx]
    y_test = np.stack([y_vsg[test_idx], y_vsl[test_idx]], axis=1)
    X_pool = X[pool_idx]
    y_pool = np.stack([y_vsg[pool_idx], y_vsl[pool_idx]], axis=1)

    cv_rows = []
    summary_rows = []
    fitted_models = {}

    for name, base_model in get_regressors().items():
        fold_vsg_mae, fold_vsl_mae = [], []

        for i, (tr_idx, va_idx) in enumerate(folds):
            X_tr = X[tr_idx]
            X_va = X[va_idx]
            y_tr = np.stack([y_vsg[tr_idx], y_vsl[tr_idx]], axis=1)
            y_va = np.stack([y_vsg[va_idx], y_vsl[va_idx]], axis=1)

            scaler = StandardScaler().fit(X_tr)
            X_tr_s, X_va_s = scaler.transform(X_tr), scaler.transform(X_va)

            model = get_regressors()[name]
            model.fit(X_tr_s, y_tr)
            pred = model.predict(X_va_s)

            vsg_mae = mean_absolute_error(y_va[:, 0], pred[:, 0])
            vsl_mae = mean_absolute_error(y_va[:, 1], pred[:, 1])
            fold_vsg_mae.append(vsg_mae)
            fold_vsl_mae.append(vsl_mae)
            cv_rows.append({"model": name, "fold": i + 1, "val_vsg_mae": vsg_mae, "val_vsl_mae": vsl_mae})

        scaler = StandardScaler().fit(X_pool)
        X_pool_s, X_test_s = scaler.transform(X_pool), scaler.transform(X_test)
        final_model = get_regressors()[name]
        final_model.fit(X_pool_s, y_pool)
        test_pred = final_model.predict(X_test_s)

        vsg_mae = mean_absolute_error(y_test[:, 0], test_pred[:, 0])
        vsg_rmse = float(np.sqrt(mean_squared_error(y_test[:, 0], test_pred[:, 0])))
        vsl_mae = mean_absolute_error(y_test[:, 1], test_pred[:, 1])
        vsl_rmse = float(np.sqrt(mean_squared_error(y_test[:, 1], test_pred[:, 1])))

        vsg_ci_mean, vsg_ci_half = confidence_interval(fold_vsg_mae)
        vsl_ci_mean, vsl_ci_half = confidence_interval(fold_vsl_mae)

        fitted_models[name] = {"model": final_model, "scaler": scaler}
        summary_rows.append({
            "model": name,
            "cv_vsg_mae_mean": np.mean(fold_vsg_mae), "cv_vsg_mae_std": np.std(fold_vsg_mae),
            "cv_vsg_mae_ci95": vsg_ci_half,
            "cv_vsl_mae_mean": np.mean(fold_vsl_mae), "cv_vsl_mae_std": np.std(fold_vsl_mae),
            "cv_vsl_mae_ci95": vsl_ci_half,
            "test_vsg_mae": vsg_mae, "test_vsg_rmse": vsg_rmse,
            "test_vsl_mae": vsl_mae, "test_vsl_rmse": vsl_rmse,
        })

    return {
        "cv_rows": cv_rows,
        "summary_rows": summary_rows,
        "fitted_models": fitted_models,
    }
