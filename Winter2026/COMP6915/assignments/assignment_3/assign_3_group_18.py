#
#  Assignment 3
#
#  Group 18:
#  Isaac Adoboe                     ieadoboe@mun.ca
#  Blessing Ijeoma Benjamin-Igwe    bbenjaminigw@mun.ca
#  Promee Shankar Kundu             pskundu@mun.ca
#
##############################################################################
# Imports
##############################################################################

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def apply_mpl_config():
    print("Apply custom plotting configs...")
    mpl_config = {
        "figure.figsize": (12, 6),
        "savefig.dpi": 300,
        "figure.dpi": 150,
        "font.family": "serif",
        # "font.serif": ["Computer Modern"],
        # "text.usetex": True,
        "font.size": 11,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,
        "lines.linewidth": 2,
        "axes.linewidth": 0.8,
        "savefig.bbox": "tight",
    }
    mpl.rcParams.update(mpl_config)
    print("Done!")


###############################################################################
#  Implementation - helper functions and question wise result generators
###############################################################################

FEATURE_NAMES = [
    "ctx-lh-inferiorparietal",
    "ctx-lh-inferiortemporal",
    "ctx-lh-isthmuscingulate",
    "ctx-lh-middletemporal",
    "ctx-lh-posteriorcingulate",
    "ctx-lh-precuneus",
    "ctx-rh-isthmuscingulate",
    "ctx-rh-posteriorcingulate",
    "ctx-rh-inferiorparietal",
    "ctx-rh-middletemporal",
    "ctx-rh-precuneus",
    "ctx-rh-inferiortemporal",
    "ctx-lh-entorhinal",
    "ctx-lh-supramarginal",
]

performance_results = {}


def load_data(data_dir="."):
    train_sNC = pd.read_csv(
        os.path.join(data_dir, "train.fdg_pet.sNC.csv"), header=None,
        names=FEATURE_NAMES)
    train_sDAT = pd.read_csv(
        os.path.join(data_dir, "train.fdg_pet.sDAT.csv"), header=None,
        names=FEATURE_NAMES)
    test_sNC = pd.read_csv(
        os.path.join(data_dir, "test.fdg_pet.sNC.csv"), header=None,
        names=FEATURE_NAMES)
    test_sDAT = pd.read_csv(
        os.path.join(data_dir, "test.fdg_pet.sDAT.csv"), header=None,
        names=FEATURE_NAMES)

    # Combine into train / test with labels (sNC=0, sDAT=1)
    X_train = pd.concat([train_sNC, train_sDAT], ignore_index=True)
    y_train = np.array([0] * len(train_sNC) + [1] * len(train_sDAT))

    X_test = pd.concat([test_sNC, test_sDAT], ignore_index=True)
    y_test = np.array([0] * len(test_sNC) + [1] * len(test_sDAT))

    print(f"Training set: {len(train_sNC)} sNC + {len(train_sDAT)} sDAT"
          f" = {len(X_train)} total")
    print(f"Test set:     {len(test_sNC)} sNC + {len(test_sDAT)} sDAT"
          f" = {len(X_test)} total")
    print(f"Features:     {X_train.shape[1]}")

    return X_train, y_train, X_test, y_test


def calc_metrics(y_true, y_pred):
    """Calculate all required performance metrics.
    sNC=0 (negative), sDAT=1 (positive)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn)  # recall for positive class (sDAT)
    specificity = tn / (tn + fp)  # recall for negative class (sNC)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    return {
        "Accuracy": acc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "Recall": recall,
        "Balanced Accuracy": bal_acc,
    }


def print_metrics(metrics, title=""):
    if title:
        print(f"\n{title}")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.4f}")


# Load data
X_train, y_train, X_test, y_test = load_data()


###############################################################################
#  Question 1 - Linear SVM
###############################################################################

def Q1_results():
    print("\n" + "=" * 70)
    print("QUESTION 1: Linear SVM Classification")
    print("=" * 70)

    # Grid search over C
    C_range = np.logspace(-4, 4, 30)

    pipe_linear = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="linear"))
    ])

    param_grid = {"svm__C": C_range}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        pipe_linear,
        param_grid,
        cv=cv,
        scoring="balanced_accuracy",
        return_train_score=True,
        n_jobs=-1,
        verbose=1,
    )

    print("\nRunning grid search for C...")
    print(f"  C range: [{C_range[0]:.4e}, {C_range[-1]:.4e}]")
    print(f"  Number of C values: {len(C_range)}")
    print("  Cross-validation: 5-fold stratified")

    grid_search.fit(X_train, y_train)

    best_C = grid_search.best_params_["svm__C"]
    best_cv_score = grid_search.best_score_

    print(f"\nBest C: {best_C:.6f}")
    print(f"Best CV Balanced Accuracy: {best_cv_score:.4f}")

    # Plot CV performance vs C
    cv_results = grid_search.cv_results_
    C_values = cv_results["param_svm__C"].data
    mean_test_scores = cv_results["mean_test_score"]
    std_test_scores = cv_results["std_test_score"]
    mean_train_scores = cv_results["mean_train_score"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: CV balanced accuracy vs C
    axes[0].semilogx(C_values, mean_test_scores, "b-", linewidth=2,
                     label="CV Test Balanced Accuracy")
    axes[0].fill_between(C_values,
                         mean_test_scores - std_test_scores,
                         mean_test_scores + std_test_scores,
                         alpha=0.2, color="blue", label=r"$\pm$ 1 std")
    axes[0].axvline(best_C, color="red", linestyle="--", linewidth=1.5,
                    label=f"Best $C$ = {best_C:.4f}")
    axes[0].scatter([best_C], [best_cv_score], color="red", s=100, zorder=5)
    axes[0].set_xlabel("Regularization Parameter ($C$)")
    axes[0].set_ylabel("Balanced Accuracy")
    axes[0].set_title("Linear SVM: CV Performance vs $C$")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    # Right: Train vs Test
    axes[1].semilogx(C_values, mean_train_scores, "g-", linewidth=2,
                     label="Train Balanced Accuracy")
    axes[1].semilogx(C_values, mean_test_scores, "b-", linewidth=2,
                     label="Test Balanced Accuracy")
    axes[1].axvline(best_C, color="red", linestyle="--", linewidth=1.5,
                    label=f"Best $C$ = {best_C:.4f}")
    axes[1].set_xlabel("Regularization Parameter ($C$)")
    axes[1].set_ylabel("Balanced Accuracy")
    axes[1].set_title("Linear SVM: Train vs Test Performance")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    os.makedirs("q1", exist_ok=True)
    plt.savefig("q1/q1_linear_svm_tuning.pdf")
    plt.show()

    # Retrain on full training set with best C
    pipe_linear_final = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="linear", C=best_C))
    ])
    pipe_linear_final.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = pipe_linear_final.predict(X_test)
    metrics = calc_metrics(y_test, y_pred)
    print_metrics(metrics, "Linear SVM - Test Set Performance:")

    performance_results["Linear SVM"] = metrics

    return pipe_linear_final, best_C, metrics


###############################################################################
#  Question 2 - Polynomial Kernel SVM
###############################################################################

def Q2_results():
    print("\n" + "=" * 70)
    print("QUESTION 2: Polynomial Kernel SVM Classification")
    print("=" * 70)

    C_range = np.logspace(-2, 4, 15)
    d_range = [2, 3, 4, 5]

    pipe_poly = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="poly"))
    ])

    param_grid = {
        "svm__C": C_range,
        "svm__degree": d_range,
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        pipe_poly,
        param_grid,
        cv=cv,
        scoring="balanced_accuracy",
        return_train_score=True,
        n_jobs=-1,
        verbose=1,
    )

    print("\nRunning grid search for (C, d)...")
    print(f"  C range: [{C_range[0]:.4e}, {C_range[-1]:.4e}]"
          f" ({len(C_range)} values)")
    print(f"  d range: {d_range}")
    print("  Cross-validation: 5-fold stratified")

    grid_search.fit(X_train, y_train)

    best_C = grid_search.best_params_["svm__C"]
    best_d = grid_search.best_params_["svm__degree"]
    best_cv_score = grid_search.best_score_

    print(f"\nBest (C, d): ({best_C:.6f}, {best_d})")
    print(f"Best CV Balanced Accuracy: {best_cv_score:.4f}")

    # Plot: heatmap of CV scores for each (C, d) combination
    cv_results = grid_search.cv_results_

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Performance vs C for each degree
    for d in d_range:
        mask = cv_results["param_svm__degree"] == d
        c_vals = np.array(cv_results["param_svm__C"][mask], dtype=float)
        scores = cv_results["mean_test_score"][mask]
        sort_idx = np.argsort(c_vals)
        axes[0].semilogx(c_vals[sort_idx], scores[sort_idx],
                         marker="o", markersize=4, label=f"$d={d}$")

    axes[0].axvline(best_C, color="red", linestyle="--", linewidth=1,
                    label=f"Best $C$ = {best_C:.4f}")
    axes[0].set_xlabel("Regularization Parameter ($C$)")
    axes[0].set_ylabel("Balanced Accuracy")
    axes[0].set_title("Polynomial SVM: CV Performance vs $C$ by Degree")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    # Right: Heatmap
    score_matrix = np.zeros((len(d_range), len(C_range)))
    for i, d in enumerate(d_range):
        for j, c in enumerate(C_range):
            mask = ((cv_results["param_svm__degree"] == d) &
                    (np.isclose(
                        np.array(cv_results["param_svm__C"], dtype=float), c)))
            if np.any(mask):
                score_matrix[i, j] = cv_results["mean_test_score"][mask][0]

    im = axes[1].imshow(score_matrix, aspect="auto", cmap="viridis",
                        origin="lower")
    axes[1].set_xticks(range(len(C_range)))
    axes[1].set_xticklabels([f"{c:.1e}" for c in C_range],
                            rotation=45, ha="right", fontsize=6)
    axes[1].set_yticks(range(len(d_range)))
    axes[1].set_yticklabels([str(d) for d in d_range])
    axes[1].set_xlabel("$C$")
    axes[1].set_ylabel("Degree ($d$)")
    axes[1].set_title("Polynomial SVM: CV Balanced Accuracy Heatmap")
    plt.colorbar(im, ax=axes[1], label="Balanced Accuracy")

    plt.tight_layout()
    os.makedirs("q2", exist_ok=True)
    plt.savefig("q2/q2_poly_svm_tuning.pdf")
    plt.show()

    # Retrain on full training set with best (C, d)
    pipe_poly_final = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="poly", C=best_C, degree=best_d))
    ])
    pipe_poly_final.fit(X_train, y_train)

    y_pred = pipe_poly_final.predict(X_test)
    metrics = calc_metrics(y_test, y_pred)
    print_metrics(metrics, "Polynomial SVM - Test Set Performance:")

    performance_results["Polynomial SVM"] = metrics

    # Compare with Linear SVM
    print(f"\n{'='*70}")
    print("COMPARISON: Linear SVM vs Polynomial SVM")
    print(f"{'='*70}")
    for name in ["Linear SVM", "Polynomial SVM"]:
        print(f"\n{name}:")
        for k, v in performance_results[name].items():
            print(f"  {k:20s}: {v:.4f}")

    return pipe_poly_final, best_C, best_d, metrics


###############################################################################
#  Question 3 - RBF Kernel SVM
###############################################################################

def Q3_results():
    print("\n" + "=" * 70)
    print("QUESTION 3: RBF Kernel SVM Classification")
    print("=" * 70)

    C_range = np.logspace(-2, 4, 15)
    gamma_range = np.logspace(-5, 2, 15)

    pipe_rbf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf"))
    ])

    param_grid = {
        "svm__C": C_range,
        "svm__gamma": gamma_range,
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        pipe_rbf,
        param_grid,
        cv=cv,
        scoring="balanced_accuracy",
        return_train_score=True,
        n_jobs=-1,
        verbose=1,
    )

    print("\nRunning grid search for (C, gamma)...")
    print(f"  C range: [{C_range[0]:.4e}, {C_range[-1]:.4e}]"
          f" ({len(C_range)} values)")
    print(f"  gamma range: [{gamma_range[0]:.4e}, {gamma_range[-1]:.4e}]"
          f" ({len(gamma_range)} values)")
    print("  Cross-validation: 5-fold stratified")

    grid_search.fit(X_train, y_train)

    best_C = grid_search.best_params_["svm__C"]
    best_gamma = grid_search.best_params_["svm__gamma"]
    best_cv_score = grid_search.best_score_

    print(f"\nBest (C, gamma): ({best_C:.6f}, {best_gamma:.6f})")
    print(f"Best CV Balanced Accuracy: {best_cv_score:.4f}")

    # Plot
    cv_results = grid_search.cv_results_

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Performance vs C for selected gamma values
    gamma_show = gamma_range[::3]  # show subset for readability
    for g in gamma_show:
        mask = np.isclose(
            np.array(cv_results["param_svm__gamma"], dtype=float), g)
        c_vals = np.array(cv_results["param_svm__C"][mask], dtype=float)
        scores = cv_results["mean_test_score"][mask]
        sort_idx = np.argsort(c_vals)
        axes[0].semilogx(c_vals[sort_idx], scores[sort_idx],
                         marker="o", markersize=4,
                         label=f"$\\gamma={g:.1e}$")

    axes[0].axvline(best_C, color="red", linestyle="--", linewidth=1,
                    label=f"Best $C$ = {best_C:.4f}")
    axes[0].set_xlabel("Regularization Parameter ($C$)")
    axes[0].set_ylabel("Balanced Accuracy")
    axes[0].set_title("RBF SVM: CV Performance vs $C$ by $\\gamma$")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)

    # Right: Heatmap
    score_matrix = np.zeros((len(gamma_range), len(C_range)))
    for i, g in enumerate(gamma_range):
        for j, c in enumerate(C_range):
            mask = (np.isclose(
                np.array(cv_results["param_svm__gamma"],
                         dtype=float), g) &
                    np.isclose(
                        np.array(cv_results["param_svm__C"],
                                 dtype=float), c))
            if np.any(mask):
                score_matrix[i, j] = cv_results["mean_test_score"][mask][0]

    im = axes[1].imshow(score_matrix, aspect="auto", cmap="viridis",
                        origin="lower")
    axes[1].set_xticks(range(len(C_range)))
    axes[1].set_xticklabels([f"{c:.1e}" for c in C_range],
                            rotation=45, ha="right", fontsize=6)
    axes[1].set_yticks(range(len(gamma_range)))
    axes[1].set_yticklabels([f"{g:.1e}" for g in gamma_range], fontsize=6)
    axes[1].set_xlabel("$C$")
    axes[1].set_ylabel("$\\gamma$")
    axes[1].set_title("RBF SVM: CV Balanced Accuracy Heatmap")
    plt.colorbar(im, ax=axes[1], label="Balanced Accuracy")

    plt.tight_layout()
    os.makedirs("q3", exist_ok=True)
    plt.savefig("q3/q3_rbf_svm_tuning.pdf")
    plt.show()

    # Retrain on full training set
    pipe_rbf_final = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=best_C, gamma=best_gamma))
    ])
    pipe_rbf_final.fit(X_train, y_train)

    y_pred = pipe_rbf_final.predict(X_test)
    metrics = calc_metrics(y_test, y_pred)
    print_metrics(metrics, "RBF SVM - Test Set Performance:")

    performance_results["RBF SVM"] = metrics

    # Compare all three models
    print(f"\n{'='*70}")
    print("COMPARISON: Linear vs Polynomial vs RBF SVM")
    print(f"{'='*70}")
    for name in ["Linear SVM", "Polynomial SVM", "RBF SVM"]:
        print(f"\n{name}:")
        for k, v in performance_results[name].items():
            print(f"  {k:20s}: {v:.4f}")

    # Summary comparison bar chart
    model_names = ["Linear SVM", "Polynomial SVM", "RBF SVM"]
    metric_names = ["Accuracy", "Sensitivity", "Specificity",
                    "Precision", "Balanced Accuracy"]
    colors = ["steelblue", "coral", "seagreen"]

    fig, axes = plt.subplots(1, len(metric_names), figsize=(18, 4))

    for idx, metric in enumerate(metric_names):
        vals = [performance_results[m][metric] for m in model_names]
        bars = axes[idx].bar(model_names, vals, color=colors, alpha=0.8,
                             edgecolor="black")
        axes[idx].set_ylabel(metric)
        axes[idx].set_title(metric)
        axes[idx].set_ylim([min(0.5, min(vals) - 0.05), 1.0])
        axes[idx].grid(axis="y", alpha=0.3)
        axes[idx].tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, vals):
            axes[idx].text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                           f"{val:.4f}", ha="center", fontsize=7)

    plt.tight_layout()
    plt.savefig("q3/q3_comparison_all_svms.pdf")
    plt.show()

    return pipe_rbf_final, best_C, best_gamma, metrics


###############################################################################
#  Question 4 - Best SVM Classifier
###############################################################################

def diagnoseDAT(Xtest, data_dir):
    """Returns a vector of predictions with elements "0" for sNC and "1"
    for sDAT, corresponding to each of the N_test features vectors in Xtest

    Xtest       N_test x 14 matrix of test feature vectors
    data_dir    full path to the folder containing the following files:
                train.fdg_pet.sNC.csv, train.fdg_pet.sDAT.csv,
                test.fdg_pet.sNC.csv, test.fdg_pet.sDAT.csv
    """
    train_sNC = pd.read_csv(os.path.join(
        data_dir, "train.fdg_pet.sNC.csv"),  header=None)
    train_sDAT = pd.read_csv(os.path.join(
        data_dir, "train.fdg_pet.sDAT.csv"), header=None)
    test_sNC = pd.read_csv(os.path.join(
        data_dir, "test.fdg_pet.sNC.csv"),   header=None)
    test_sDAT = pd.read_csv(os.path.join(
        data_dir, "test.fdg_pet.sDAT.csv"),  header=None)

    X_all = pd.concat([train_sNC, train_sDAT, test_sNC,
                      test_sDAT], ignore_index=True)
    y_all = np.array(
        [0]*len(train_sNC) + [1]*len(train_sDAT) +
        [0]*len(test_sNC) + [1]*len(test_sDAT)
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="poly", C=1.3895, degree=3))
    ])
    pipe.fit(X_all, y_all)
    ytest = pipe.predict(Xtest)

    return ytest


###############################################################################
# Calls to generate the results
###############################################################################
if __name__ == "__main__":
    apply_mpl_config()
    pipe_linear, best_C_lin, metrics_lin = Q1_results()
    pipe_poly, best_C_poly, best_d, metrics_poly = Q2_results()
    pipe_rbf, best_C_rbf, best_gamma, metrics_rbf = Q3_results()
