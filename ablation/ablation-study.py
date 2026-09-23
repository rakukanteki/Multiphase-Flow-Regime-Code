import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats as sp_stats

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config -- EDIT THESE to match your dataset
# ---------------------------------------------------------------------------
BASE_DIR = r"D:\\Research\\Multiphase-papers\\Multiphase-papers\\Paper1\\Project\\Experiments"
SUB_FOLDERS = ["Dispersed-Flow", "Plug-Flow", "Slug-Flow"]  # label 0, 1, 2
CLASS_NAMES = ["Dispersed Flow", "Plug Flow", "Slug Flow"]

TEST_FRAC = 0.15      # held-out final test fraction
N_SPLITS = 5           # number of cross-validation folds
RANDOM_STATE = 42
FINAL_FOLD_IDX = 2      # zero-based: Fold 3, matching the final MTPINN test model

RESULTS_DIR = Path("results/ml_ablation")
MODELS_DIR = Path("models/ml_ablation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading -- one row (raw pressure time series) per Excel file
# ---------------------------------------------------------------------------
def extract_velocities_from_filename(filename: str):
    """Parse Vsg and Vsl from a filename like 'Dispersed_Vsg=0.50_Vsl=1.20_run3.xlsx'."""
    vsg_match = re.search(r"Vsg=([\d.]+)", filename)
    vsl_match = re.search(r"Vsl=([\d.]+)", filename)
    vsg = float(vsg_match.group(1)) if vsg_match else 0.0
    vsl = float(vsl_match.group(1)) if vsl_match else 0.0
    return vsg, vsl


def load_data():
    """
    Load every usable Excel recording once and preserve its unique run ID.

    IMPORTANT: this loader does NOT create a split. Splits are generated later
    with the exact same (Vsg, Vsl)-grouped logic as the original MTPINN
    implementation. The classical ML feature representation remains unchanged:
    the raw pressure traces are edge-padded to the maximum observed length.
    """
    pressure_data = []
    metadata = []

    for label_idx, folder in enumerate(SUB_FOLDERS):
        folder_path = os.path.join(BASE_DIR, folder)
        if not os.path.isdir(folder_path):
            print(f"[WARNING] Folder not found: {folder_path} -- skipping.")
            continue

        xlsx_files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith(".xlsx"))
        print(f"  {folder}: {len(xlsx_files)} files")

        for fname in xlsx_files:
            fpath = os.path.join(folder_path, fname)
            try:
                df = pd.read_excel(fpath)
            except Exception as exc:
                print(f"    [ERROR] Cannot read {fname}: {exc}")
                continue

            if "Pressure (barA)" not in df.columns:
                print(f"    [WARNING] 'Pressure (barA)' not found in {fname} -- skipping.")
                continue

            pressure = df["Pressure (barA)"].dropna().values.astype(np.float32)
            # Match the original MTPINN file-acceptance threshold.
            if len(pressure) < 8:
                print(f"    [WARNING] {fname} has too few samples ({len(pressure)}) -- skipping.")
                continue

            vsg, vsl = extract_velocities_from_filename(fname)
            pressure_data.append(pressure)
            metadata.append({
                "label": label_idx,
                "vsg": vsg,
                "vsl": vsl,
                "run_id": f"{folder}/{fname}",
            })

    if not pressure_data:
        raise SystemExit(
            "\n[FATAL] No files were loaded. Check BASE_DIR and SUB_FOLDERS."
        )

    max_len = max(len(p) for p in pressure_data)
    print(f"  Max pressure series length: {max_len} samples")

    X_list = []
    for p in pressure_data:
        if len(p) < max_len:
            X_list.append(np.pad(p, (0, max_len - len(p)), mode="edge"))
        else:
            X_list.append(p[:max_len])

    X = np.asarray(X_list, dtype=np.float32)
    y_cls = np.asarray([m["label"] for m in metadata], dtype=np.int64)
    y_vsg = np.asarray([m["vsg"] for m in metadata], dtype=np.float32)
    y_vsl = np.asarray([m["vsl"] for m in metadata], dtype=np.float32)
    run_ids = np.asarray([m["run_id"] for m in metadata], dtype=object)

    print(f"\nLoaded {len(X)} usable files total, {X.shape[1]} features (time samples) each.")
    return X, y_cls, y_vsg, y_vsl, run_ids


def extract_group_id(filename: str) -> str:
    """
    EXACT grouping rule used by the original MTPINN notebook.
    Repeated files sharing the same filename-encoded (Vsg, Vsl) set point
    are assigned to the same group.
    """
    vsg_match = re.search(r"Vsg=([\d.]+)", filename)
    vsl_match = re.search(r"Vsl=([\d.]+)", filename)
    vsg_str = vsg_match.group(1) if vsg_match else "NA"
    vsl_str = vsl_match.group(1) if vsl_match else "NA"
    return f"Vsg={vsg_str}_Vsl={vsl_str}"


def split_class_files_kfold(
    files,
    seed: int = RANDOM_STATE,
    n_splits: int = N_SPLITS,
    test_frac: float = TEST_FRAC,
):
    """Exact per-class held-out-test + GroupKFold logic from Last_Final_V6.ipynb."""
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
    n_test_groups = min(n_test_groups, max(0, n_groups - 1))

    test_groups = set(shuffled_groups[:n_test_groups])
    pool_groups = set(shuffled_groups[n_test_groups:])

    test_files = [f for f in files if file_groups[f] in test_groups]
    pool_files = [f for f in files if file_groups[f] in pool_groups]

    if not pool_files:
        return test_files, [([], []) for _ in range(n_splits)]

    pool_group_list = [file_groups[f] for f in pool_files]
    n_unique_pool_groups = len(set(pool_group_list))
    actual_splits = min(n_splits, n_unique_pool_groups) if n_unique_pool_groups >= 2 else 1

    if actual_splits < 2:
        raise RuntimeError(
            f"Need at least two unique operating-condition groups for GroupKFold; got {n_unique_pool_groups}."
        )

    folds = []
    pool_arr = np.asarray(pool_files, dtype=object)
    groups_arr = np.asarray(pool_group_list, dtype=object)

    gkf = GroupKFold(n_splits=actual_splits)
    for train_rel, val_rel in gkf.split(pool_arr, groups=groups_arr):
        folds.append((list(pool_arr[train_rel]), list(pool_arr[val_rel])))

    while len(folds) < n_splits:
        folds.append(([], []))

    return test_files, folds


def make_master_split(run_ids):
    """
    Recreate the ORIGINAL MTPINN split and map its filenames to rows in this
    baseline dataset. No train_test_split/StratifiedKFold is used.
    """
    id_to_idx = {rid: i for i, rid in enumerate(run_ids)}
    if len(id_to_idx) != len(run_ids):
        raise RuntimeError("Duplicate run IDs detected.")

    test_run_ids = []
    fold_train_ids = [[] for _ in range(N_SPLITS)]
    fold_val_ids = [[] for _ in range(N_SPLITS)]
    file_manifest = {}

    for folder in SUB_FOLDERS:
        folder_path = os.path.join(BASE_DIR, folder)
        xlsx_files = sorted(
            f for f in os.listdir(folder_path) if f.lower().endswith(".xlsx")
        )

        test_files, class_folds = split_class_files_kfold(xlsx_files)
        file_manifest[folder] = {
            "test": test_files,
            "folds": [
                {"train": tr_files, "val": va_files}
                for tr_files, va_files in class_folds
            ],
        }

        test_run_ids.extend([f"{folder}/{f}" for f in test_files])
        for fold_idx, (tr_files, va_files) in enumerate(class_folds):
            fold_train_ids[fold_idx].extend([f"{folder}/{f}" for f in tr_files])
            fold_val_ids[fold_idx].extend([f"{folder}/{f}" for f in va_files])

    required_ids = set(test_run_ids)
    for i in range(N_SPLITS):
        required_ids.update(fold_train_ids[i])
        required_ids.update(fold_val_ids[i])

    missing = sorted(required_ids - set(id_to_idx))
    if missing:
        preview = "\n".join(missing[:20])
        raise RuntimeError(
            "The master split contains files that were not loaded by the baseline loader. "
            "To guarantee identical MTPINN/baseline partitions, resolve these files first.\n"
            f"Missing ({len(missing)}):\n{preview}"
        )

    test_idx = np.asarray([id_to_idx[r] for r in test_run_ids], dtype=int)
    folds = []
    for i in range(N_SPLITS):
        tr_idx = np.asarray([id_to_idx[r] for r in fold_train_ids[i]], dtype=int)
        va_idx = np.asarray([id_to_idx[r] for r in fold_val_ids[i]], dtype=int)
        folds.append((tr_idx, va_idx))

    # The CV pool is the union of one fold's train and validation rows.
    pool_idx = np.asarray(sorted(set(folds[0][0]) | set(folds[0][1])), dtype=int)
    return pool_idx, test_idx, folds, file_manifest


def validate_master_split(y_cls, run_ids, pool_idx, test_idx, folds):
    """Hard checks required before any baseline is trained."""
    print("\n" + "=" * 78)
    print("MASTER SPLIT VALIDATION")
    print("=" * 78)

    # These are the expected counts from the original MTPINN execution.
    expected_test_n = 56
    expected_support = [18, 19, 19]
    expected_fold_sizes = [(255, 66), (256, 65), (257, 64), (258, 63), (258, 63)]

    support = np.bincount(y_cls[test_idx], minlength=len(CLASS_NAMES)).tolist()
    assert len(test_idx) == expected_test_n, (
        f"Expected {expected_test_n} held-out runs, got {len(test_idx)}. "
        "You are not using the same dataset state as Last_Final_V6.ipynb."
    )
    assert support == expected_support, f"Expected test support {expected_support}, got {support}."
    assert set(pool_idx).isdisjoint(set(test_idx)), "Pool/test overlap detected."

    # Group identity is class-scoped because the original MTPINN split is built
    # independently inside each flow-regime folder.
    def scoped_group(i):
        rid = str(run_ids[int(i)])
        folder, fname = rid.split("/", 1)
        return f"{folder}|{extract_group_id(fname)}"

    test_groups = {scoped_group(i) for i in test_idx}

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        assert (len(tr_idx), len(va_idx)) == expected_fold_sizes[fold_idx], (
            f"Fold {fold_idx + 1}: expected {expected_fold_sizes[fold_idx]}, "
            f"got {(len(tr_idx), len(va_idx))}."
        )
        assert set(tr_idx).isdisjoint(set(va_idx)), f"Fold {fold_idx + 1}: train/val overlap."
        assert set(tr_idx).isdisjoint(set(test_idx)), f"Fold {fold_idx + 1}: train/test overlap."
        assert set(va_idx).isdisjoint(set(test_idx)), f"Fold {fold_idx + 1}: val/test overlap."

        train_groups = {scoped_group(i) for i in tr_idx}
        val_groups = {scoped_group(i) for i in va_idx}
        assert train_groups.isdisjoint(val_groups), f"Fold {fold_idx + 1}: operating-condition group leakage train/val."
        assert train_groups.isdisjoint(test_groups), f"Fold {fold_idx + 1}: operating-condition group leakage train/test."
        assert val_groups.isdisjoint(test_groups), f"Fold {fold_idx + 1}: operating-condition group leakage val/test."

    print(f"[OK] Held-out test = {len(test_idx)}")
    print(f"[OK] Test support = {support}  ({', '.join(CLASS_NAMES)})")
    for i, (tr_idx, va_idx) in enumerate(folds):
        print(f"[OK] Fold {i + 1}: train={len(tr_idx)}, val={len(va_idx)}")
    print("[OK] No file overlap and no (Vsg,Vsl)-group overlap across train/val/test.")
    print(f"[OK] Final head-to-head test training partition = Fold {FINAL_FOLD_IDX + 1} train set")


def save_split_manifest(run_ids, y_cls, y_vsg, y_vsl, test_idx, folds):
    """Save exact run IDs and fold assignments for reviewer reproducibility."""
    test_set = set(test_idx.tolist())
    val_fold_by_idx = {}
    for fold_no, (_, va_idx) in enumerate(folds, start=1):
        for idx in va_idx:
            if int(idx) in val_fold_by_idx:
                raise RuntimeError(f"Run appears in validation more than once: {run_ids[int(idx)]}")
            val_fold_by_idx[int(idx)] = fold_no

    rows = []
    for i, rid in enumerate(run_ids):
        fname = os.path.basename(rid)
        rows.append({
            "run_id": rid,
            "flow_regime": CLASS_NAMES[int(y_cls[i])],
            "Vsg_m_s": float(y_vsg[i]),
            "Vsl_m_s": float(y_vsl[i]),
            "condition_group": extract_group_id(fname),
            "held_out_test": "Yes" if i in test_set else "No",
            "validation_fold": "" if i in test_set else val_fold_by_idx.get(i, ""),
        })

    manifest_df = pd.DataFrame(rows).sort_values(
        ["held_out_test", "validation_fold", "flow_regime", "run_id"],
        ascending=[False, True, True, True],
    )
    out_path = RESULTS_DIR / "split_manifest.csv"
    manifest_df.to_csv(out_path, index=False)
    print(f"[OK] Saved exact split manifest: {out_path}")
    return manifest_df


# ---------------------------------------------------------------------------
# Confidence-interval helpers (used by the comparison bar plots)
# ---------------------------------------------------------------------------
def confidence_interval(values, confidence: float = 0.95):
    """Mean + half-width of a (small-sample) t-distribution CI across CV folds."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(np.mean(values)) if n else 0.0
    if n <= 1:
        return mean, 0.0
    std = float(np.std(values, ddof=1))
    sem = std / np.sqrt(n)
    t_crit = float(sp_stats.t.ppf((1 + confidence) / 2.0, n - 1))
    return mean, t_crit * sem


def clipped_asymmetric_error(mean: float, half_width: float, lo: float = 0.0, hi: float = 1.0):
    """
    Asymmetric (lower_err, upper_err) lengths so mean-lower_err and
    mean+upper_err never leave [lo, hi] -- an accuracy/F1 bar's error
    bar can never imply less than 0% or more than 100%.
    """
    lower_err = mean - max(lo, mean - half_width)
    upper_err = min(hi, mean + half_width) - mean
    return max(0.0, lower_err), max(0.0, upper_err)


# ---------------------------------------------------------------------------
# Model zoo
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Classification ablation
# ---------------------------------------------------------------------------
def run_classification_ablation(X, y_cls, pool_idx, test_idx, folds):
    print("\n" + "=" * 70)
    print("  CLASSIFICATION ABLATION  (flow regime)")
    print("=" * 70)

    X_test, y_test = X[test_idx], y_cls[test_idx]

    # Final held-out comparison uses exactly the same Fold-3 training files
    # as the MTPINN checkpoint selected for test evaluation.
    final_train_idx = folds[FINAL_FOLD_IDX][0]
    X_final_train = X[final_train_idx]
    y_final_train = y_cls[final_train_idx]

    cv_rows = []
    test_summary_rows = []
    per_class_rows = []

    for name in get_classifiers():
        print(f"\n  -- {name} --")
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
            print(f"    fold {i + 1}: val_acc={acc:.4f}  val_f1_macro={f1:.4f}")
            cv_rows.append({"model": name, "fold": i + 1, "val_accuracy": acc, "val_f1_macro": f1})

            fold_report = classification_report(
                y_va, pred, labels=list(range(len(CLASS_NAMES))),
                target_names=CLASS_NAMES, output_dict=True, zero_division=0,
            )
            for c in CLASS_NAMES:
                fold_class_precision[c].append(fold_report[c]["precision"])
                fold_class_recall[c].append(fold_report[c]["recall"])
                fold_class_f1[c].append(fold_report[c]["f1-score"])

        # Final head-to-head test: same Fold-3 training partition + same 56-run test set as MTPINN.
        scaler = StandardScaler().fit(X_final_train)
        X_final_train_s = scaler.transform(X_final_train)
        X_test_s = scaler.transform(X_test)
        final_model = get_classifiers()[name]
        final_model.fit(X_final_train_s, y_final_train)
        test_pred = final_model.predict(X_test_s)

        test_acc = accuracy_score(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred, average="macro")
        acc_ci_mean, acc_ci_half = confidence_interval(fold_accs)
        f1_ci_mean, f1_ci_half = confidence_interval(fold_f1s)
        print(f"    CV mean:  acc={np.mean(fold_accs):.4f} +/- {np.std(fold_accs):.4f}   "
              f"f1_macro={np.mean(fold_f1s):.4f} +/- {np.std(fold_f1s):.4f}")
        print(f"    TEST:     acc={test_acc:.4f}   f1_macro={test_f1:.4f}")

        report_dict = classification_report(
            y_test, test_pred, labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES, output_dict=True, digits=4, zero_division=0,
        )
        report_text = classification_report(
            y_test, test_pred, labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES, digits=4, zero_division=0,
        )
        with open(RESULTS_DIR / f"classification_report_{name}.txt", "w") as fh:
            fh.write(f"Model: {name}\n")
            fh.write(f"CV accuracy: {np.mean(fold_accs):.4f} +/- {np.std(fold_accs):.4f} "
                     f"(95% CI +/- {acc_ci_half:.4f})\n")
            fh.write(f"CV f1_macro: {np.mean(fold_f1s):.4f} +/- {np.std(fold_f1s):.4f} "
                     f"(95% CI +/- {f1_ci_half:.4f})\n\n")
            fh.write("TEST SET CLASSIFICATION REPORT\n")
            fh.write(report_text)

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

        cm = confusion_matrix(y_test, test_pred, labels=list(range(len(CLASS_NAMES))))
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES,
                    yticklabels=CLASS_NAMES, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{name} -- Confusion Matrix (TEST)")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"confusion_matrix_{name}.png", dpi=150)
        plt.close(fig)

        joblib.dump({"model": final_model, "scaler": scaler}, MODELS_DIR / f"{name}_classifier.joblib")

        test_summary_rows.append({
            "model": name,
            "cv_accuracy_mean": np.mean(fold_accs), "cv_accuracy_std": np.std(fold_accs),
            "cv_accuracy_ci95": acc_ci_half,
            "cv_f1_macro_mean": np.mean(fold_f1s), "cv_f1_macro_std": np.std(fold_f1s),
            "cv_f1_macro_ci95": f1_ci_half,
            "test_accuracy": test_acc, "test_f1_macro": test_f1,
        })

    cv_df = pd.DataFrame(cv_rows)
    summary_df = pd.DataFrame(test_summary_rows).sort_values("test_accuracy", ascending=False)
    per_class_df = pd.DataFrame(per_class_rows)
    cv_df.to_csv(RESULTS_DIR / "classification_cv_folds.csv", index=False)
    summary_df.to_csv(RESULTS_DIR / "classification_summary.csv", index=False)
    per_class_df.to_csv(RESULTS_DIR / "classification_per_class.csv", index=False)

    # ---- Accuracy comparison bar plot, 95% CI error bars clipped to [0, 1] ----
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(summary_df))
    cv_means = summary_df["cv_accuracy_mean"].values
    cv_ci = summary_df["cv_accuracy_ci95"].values
    cv_err_lower, cv_err_upper = zip(*[
        clipped_asymmetric_error(m, h, lo=0.0, hi=1.0) for m, h in zip(cv_means, cv_ci)
    ])
    ax.bar(x - 0.2, cv_means, width=0.4, yerr=[cv_err_lower, cv_err_upper],
           label="CV val accuracy (95% CI)", color="tab:blue", capsize=4)
    ax.bar(x + 0.2, summary_df["test_accuracy"], width=0.4,
           label="Held-out test accuracy", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["model"])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Classification Ablation -- Accuracy by Model (95% CI, clipped to [0, 1])")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "accuracy_comparison.png", dpi=150)
    plt.close(fig)

    # ---- Per-class F1 (TEST set) grouped bar plot ----
    fig, ax = plt.subplots(figsize=(10, 5))
    models = summary_df["model"].tolist()
    n_models = len(models)
    n_classes = len(CLASS_NAMES)
    width = 0.8 / n_models
    x_base = np.arange(n_classes)
    for j, m in enumerate(models):
        vals = [per_class_df[(per_class_df["model"] == m) & (per_class_df["class"] == c)]["test_f1"].iloc[0]
                for c in CLASS_NAMES]
        ax.bar(x_base + (j - (n_models - 1) / 2) * width, vals, width=width, label=m)
    ax.set_xticks(x_base)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Class F1 by Model (Held-out TEST set)")
    ax.legend(title="Model", ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "per_class_f1_comparison.png", dpi=150)
    plt.close(fig)

    print("\n  Classification summary:")
    print(summary_df.to_string(index=False))
    print("\n  Per-class summary (TEST set):")
    print(per_class_df[["model", "class", "test_precision", "test_recall", "test_f1", "test_support"]]
          .to_string(index=False))
    return summary_df, per_class_df


# ---------------------------------------------------------------------------
# Regression ablation
# ---------------------------------------------------------------------------
def run_regression_ablation(X, y_vsg, y_vsl, pool_idx, test_idx, folds):
    print("\n" + "=" * 70)
    print("  REGRESSION ABLATION  (Vsg, Vsl)")
    print("=" * 70)

    X_test = X[test_idx]
    y_test = np.stack([y_vsg[test_idx], y_vsl[test_idx]], axis=1)

    # Same Fold-3 training partition used by the final MTPINN checkpoint.
    final_train_idx = folds[FINAL_FOLD_IDX][0]
    X_final_train = X[final_train_idx]
    y_final_train = np.stack([y_vsg[final_train_idx], y_vsl[final_train_idx]], axis=1)

    cv_rows = []
    summary_rows = []

    for name in get_regressors():
        print(f"\n  -- {name} --")
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
            print(f"    fold {i + 1}: val_Vsg_MAE={vsg_mae:.4f}  val_Vsl_MAE={vsl_mae:.4f}")
            cv_rows.append({"model": name, "fold": i + 1, "val_vsg_mae": vsg_mae, "val_vsl_mae": vsl_mae})

        scaler = StandardScaler().fit(X_final_train)
        X_final_train_s = scaler.transform(X_final_train)
        X_test_s = scaler.transform(X_test)
        final_model = get_regressors()[name]
        final_model.fit(X_final_train_s, y_final_train)
        test_pred = final_model.predict(X_test_s)

        vsg_mae = mean_absolute_error(y_test[:, 0], test_pred[:, 0])
        vsg_rmse = float(np.sqrt(mean_squared_error(y_test[:, 0], test_pred[:, 0])))
        vsl_mae = mean_absolute_error(y_test[:, 1], test_pred[:, 1])
        vsl_rmse = float(np.sqrt(mean_squared_error(y_test[:, 1], test_pred[:, 1])))

        vsg_ci_mean, vsg_ci_half = confidence_interval(fold_vsg_mae)
        vsl_ci_mean, vsl_ci_half = confidence_interval(fold_vsl_mae)

        print(f"    TEST: Vsg MAE={vsg_mae:.4f} RMSE={vsg_rmse:.4f}  |  "
              f"Vsl MAE={vsl_mae:.4f} RMSE={vsl_rmse:.4f}")

        joblib.dump({"model": final_model, "scaler": scaler}, MODELS_DIR / f"{name}_regressor.joblib")

        summary_rows.append({
            "model": name,
            "cv_vsg_mae_mean": np.mean(fold_vsg_mae), "cv_vsg_mae_std": np.std(fold_vsg_mae),
            "cv_vsg_mae_ci95": vsg_ci_half,
            "cv_vsl_mae_mean": np.mean(fold_vsl_mae), "cv_vsl_mae_std": np.std(fold_vsl_mae),
            "cv_vsl_mae_ci95": vsl_ci_half,
            "test_vsg_mae": vsg_mae, "test_vsg_rmse": vsg_rmse,
            "test_vsl_mae": vsl_mae, "test_vsl_rmse": vsl_rmse,
        })

    cv_df = pd.DataFrame(cv_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values("test_vsg_mae")
    cv_df.to_csv(RESULTS_DIR / "regression_cv_folds.csv", index=False)
    summary_df.to_csv(RESULTS_DIR / "regression_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    x = np.arange(len(summary_df))

    vsg_cv_mean = summary_df["cv_vsg_mae_mean"].values
    vsg_cv_ci = summary_df["cv_vsg_mae_ci95"].values
    vsg_err_lower, vsg_err_upper = zip(*[
        clipped_asymmetric_error(m, h, lo=0.0, hi=np.inf) for m, h in zip(vsg_cv_mean, vsg_cv_ci)
    ])
    axes[0].bar(x - 0.2, vsg_cv_mean, width=0.4, yerr=[vsg_err_lower, vsg_err_upper],
                label="CV val MAE (95% CI)", color="tab:blue", capsize=4)
    axes[0].bar(x + 0.2, summary_df["test_vsg_mae"], width=0.4,
                label="Held-out test MAE", color="tab:orange")
    axes[0].set_xticks(x); axes[0].set_xticklabels(summary_df["model"])
    axes[0].set_ylabel("MAE (m/s)"); axes[0].set_title("Vsg -- MAE by Model")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis="y")

    vsl_cv_mean = summary_df["cv_vsl_mae_mean"].values
    vsl_cv_ci = summary_df["cv_vsl_mae_ci95"].values
    vsl_err_lower, vsl_err_upper = zip(*[
        clipped_asymmetric_error(m, h, lo=0.0, hi=np.inf) for m, h in zip(vsl_cv_mean, vsl_cv_ci)
    ])
    axes[1].bar(x - 0.2, vsl_cv_mean, width=0.4, yerr=[vsl_err_lower, vsl_err_upper],
                label="CV val MAE (95% CI)", color="tab:blue", capsize=4)
    axes[1].bar(x + 0.2, summary_df["test_vsl_mae"], width=0.4,
                label="Held-out test MAE", color="tab:orange")
    axes[1].set_xticks(x); axes[1].set_xticklabels(summary_df["model"])
    axes[1].set_ylabel("MAE (m/s)"); axes[1].set_title("Vsl -- MAE by Model")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "regression_error_comparison.png", dpi=150)
    plt.close(fig)

    print("\n  Regression summary:")
    print(summary_df.to_string(index=False))
    return summary_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 78)
    print("  CLASSICAL ML ABLATION STUDY  (RF / GBM / SVM / kNN / LR / DT)")
    print(f"  Reading Excel files directly from: {BASE_DIR}")
    print("=" * 78)

    X, y_cls, y_vsg, y_vsl, run_ids = load_data()
    pool_idx, test_idx, folds, file_manifest = make_master_split(run_ids)

    validate_master_split(y_cls, run_ids, pool_idx, test_idx, folds)
    split_manifest = save_split_manifest(run_ids, y_cls, y_vsg, y_vsl, test_idx, folds)

    print(f"Held-out test rows: {len(test_idx)}   CV pool rows: {len(pool_idx)}   folds: {len(folds)}")
    print(f"Final test training set: Fold {FINAL_FOLD_IDX + 1} train ({len(folds[FINAL_FOLD_IDX][0])} runs)")

    cls_summary, cls_per_class = run_classification_ablation(X, y_cls, pool_idx, test_idx, folds)
    reg_summary = run_regression_ablation(X, y_vsg, y_vsl, pool_idx, test_idx, folds)

    print("\n" + "=" * 78)
    print("  DONE. Results saved under:", RESULTS_DIR.resolve())
    print("  Models saved under:       ", MODELS_DIR.resolve())
    print("=" * 78)