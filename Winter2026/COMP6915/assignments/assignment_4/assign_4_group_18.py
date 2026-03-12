#
#  Assignment 4
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
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


mpl_config = {
    "figure.figsize": (12, 6),
    "savefig.dpi": 300,
    "figure.dpi": 150,
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.serif": ["Computer Modern"],
    "text.usetex": True,
    "legend.fontsize": 10,
    "lines.linewidth": 2,
    "axes.linewidth": 0.8,
    "savefig.bbox": "tight",
}
mpl.rcParams.update(mpl_config)


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
    train_sMCI = pd.read_csv(
        os.path.join(data_dir, "train.fdg_pet.sMCI.csv"), header=None,
        names=FEATURE_NAMES)
    train_pMCI = pd.read_csv(
        os.path.join(data_dir, "train.fdg_pet.pMCI.csv"), header=None,
        names=FEATURE_NAMES)
    test_sMCI = pd.read_csv(
        os.path.join(data_dir, "test.fdg_pet.sMCI.csv"), header=None,
        names=FEATURE_NAMES)
    test_pMCI = pd.read_csv(
        os.path.join(data_dir, "test.fdg_pet.pMCI.csv"), header=None,
        names=FEATURE_NAMES)

    # Combine into train / test with labels (sMCI=0, pMCI=1)
    X_train = pd.concat([train_sMCI, train_pMCI], ignore_index=True)
    y_train = np.array([0] * len(train_sMCI) + [1] * len(train_pMCI))

    X_test = pd.concat([test_sMCI, test_pMCI], ignore_index=True)
    y_test = np.array([0] * len(test_sMCI) + [1] * len(test_pMCI))

    print(f"Training set: {len(train_sMCI)} sMCI + {len(train_pMCI)} pMCI"
          f" = {len(X_train)} total")
    print(f"Test set:     {len(test_sMCI)} sMCI + {len(test_pMCI)} pMCI"
          f" = {len(X_test)} total")
    print(f"Features:     {X_train.shape[1]}")

    return X_train, y_train, X_test, y_test


def calc_metrics(y_true, y_pred):
    """Calculate all required performance metrics.
    sMCI=0 (negative), pMCI=1 (positive)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn)  # recall for positive class (pMCI)
    specificity = tn / (tn + fp)  # recall for negative class (sMCI)
    precision = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    return {
        "Accuracy": acc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "Recall": rec,
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
#  Question 1 - Single Decision Tree
###############################################################################

def Q1_results():
    print("\n" + "=" * 70)
    print("QUESTION 1: Single Decision Tree Classification")
    print("=" * 70)

    # Grid search over feature testing criterion
    param_grid = {"criterion": ["gini", "entropy", "log_loss"]}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    dt = DecisionTreeClassifier(random_state=42)

    grid_search = GridSearchCV(
        dt,
        param_grid,
        cv=cv,
        scoring="balanced_accuracy",
        return_train_score=True,
        n_jobs=-1,
        verbose=1,
    )

    print("\nRunning grid search for feature testing criterion...")
    print(f"  Criteria: {param_grid['criterion']}")
    print("  Cross-validation: 5-fold stratified")

    grid_search.fit(X_train, y_train)

    best_criterion = grid_search.best_params_["criterion"]
    best_cv_score = grid_search.best_score_

    print(f"\nBest criterion: {best_criterion}")
    print(f"Best CV Balanced Accuracy: {best_cv_score:.4f}")

    # Display CV results for all criteria
    cv_results = grid_search.cv_results_
    print("\nCV Results for all criteria:")
    for i, criterion in enumerate(cv_results["param_criterion"]):
        mean_score = cv_results["mean_test_score"][i]
        std_score = cv_results["std_test_score"][i]
        mean_train = cv_results["mean_train_score"][i]
        print(f"  {criterion:10s}: CV={mean_score:.4f} +/- {std_score:.4f}"
              f"  Train={mean_train:.4f}")

    # Plot CV performance for each criterion
    criteria = list(cv_results["param_criterion"])
    mean_test_scores = cv_results["mean_test_score"]
    mean_train_scores = cv_results["mean_train_score"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(criteria))
    width = 0.35
    bars_train = ax.bar(x_pos - width / 2, mean_train_scores, width,
                        label="Train", color="lightgreen",
                        edgecolor="black", alpha=0.8)
    bars_cv = ax.bar(x_pos + width / 2, mean_test_scores, width,
                     label="CV Test", color="steelblue",
                     edgecolor="black", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(criteria)
    ax.set_xlabel("Feature Testing Criterion")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Decision Tree: Train vs CV Balanced Accuracy")
    ax.set_ylim([0, 1.1])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars_train, mean_train_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.4f}", ha="center", fontsize=8)
    for bar, val in zip(bars_cv, mean_test_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.4f}", ha="center", fontsize=8)

    plt.tight_layout()
    os.makedirs("q1", exist_ok=True)
    plt.savefig("q1/q1_dt_criterion_tuning.pdf")
    plt.show()

    # Retrain on full training set with best criterion
    dt_final = DecisionTreeClassifier(
        criterion=best_criterion, random_state=42)
    dt_final.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = dt_final.predict(X_test)
    metrics = calc_metrics(y_test, y_pred)
    print_metrics(metrics, "Decision Tree - Test Set Performance:")

    performance_results["Decision Tree"] = metrics

    return dt_final, best_criterion, metrics


###############################################################################
#  Question 2 - Plot Decision Tree & Feature Importance
###############################################################################

def Q2_results(dt_final):
    print("\n" + "=" * 70)
    print("QUESTION 2: Decision Tree Visualization & Feature Importance")
    print("=" * 70)

    # Plot the decision tree
    fig, ax = plt.subplots(1, 1, figsize=(24, 12))
    plot_tree(
        dt_final,
        feature_names=FEATURE_NAMES,
        class_names=["sMCI", "pMCI"],
        filled=True,
        rounded=True,
        fontsize=7,
        ax=ax,
    )
    ax.set_title("Final Decision Tree (Best Criterion)", fontsize=14)
    plt.tight_layout()
    os.makedirs("q2", exist_ok=True)
    plt.savefig("q2/q2_decision_tree_plot.pdf")
    plt.show()

    # Feature importance
    importances = dt_final.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    print("\nFeature Importances (sorted):")
    for rank, idx in enumerate(sorted_idx, 1):
        print(f"  {rank:2d}. {FEATURE_NAMES[idx]:30s}: {importances[idx]:.4f}")

    # Plot feature importances
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.barh(range(len(FEATURE_NAMES)),
            importances[sorted_idx[::-1]],
            color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_yticks(range(len(FEATURE_NAMES)))
    ax.set_yticklabels([FEATURE_NAMES[i] for i in sorted_idx[::-1]])
    ax.set_xlabel("Feature Importance")
    ax.set_title("Decision Tree: Feature Importances")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("q2/q2_feature_importances.pdf")
    plt.show()

    # Tree structure summary
    n_nodes = dt_final.tree_.node_count
    max_depth = dt_final.get_depth()
    n_leaves = dt_final.get_n_leaves()
    print("\nTree Structure:")
    print(f"  Number of nodes:  {n_nodes}")
    print(f"  Maximum depth:    {max_depth}")
    print(f"  Number of leaves: {n_leaves}")

    return importances


###############################################################################
#  Question 3 - Random Forest
###############################################################################

def Q3_results():
    print("\n" + "=" * 70)
    print("QUESTION 3: Random Forest Classification")
    print("=" * 70)

    # Grid search over feature testing criterion
    param_grid = {"criterion": ["gini", "entropy", "log_loss"]}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=cv,
        scoring="balanced_accuracy",
        return_train_score=True,
        n_jobs=-1,
        verbose=1,
    )

    print("\nRunning grid search for feature testing criterion...")
    print(f"  Criteria: {param_grid['criterion']}")
    print("  n_estimators: 100")
    print("  Cross-validation: 5-fold stratified")

    grid_search.fit(X_train, y_train)

    best_criterion = grid_search.best_params_["criterion"]
    best_cv_score = grid_search.best_score_

    print(f"\nBest criterion: {best_criterion}")
    print(f"Best CV Balanced Accuracy: {best_cv_score:.4f}")

    # Display CV results for all criteria
    cv_results = grid_search.cv_results_
    print("\nCV Results for all criteria:")
    for i, criterion in enumerate(cv_results["param_criterion"]):
        mean_score = cv_results["mean_test_score"][i]
        std_score = cv_results["std_test_score"][i]
        mean_train = cv_results["mean_train_score"][i]
        print(f"  {criterion:10s}: CV={mean_score:.4f} +/- {std_score:.4f}"
              f"  Train={mean_train:.4f}")

    # Plot CV performance
    criteria = list(cv_results["param_criterion"])
    mean_test_scores = cv_results["mean_test_score"]
    mean_train_scores = cv_results["mean_train_score"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(criteria))
    width = 0.35
    bars_train = ax.bar(x_pos - width / 2, mean_train_scores, width,
                        label="Train", color="lightgreen",
                        edgecolor="black", alpha=0.8)
    bars_cv = ax.bar(x_pos + width / 2, mean_test_scores, width,
                     label="CV Test", color="steelblue",
                     edgecolor="black", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(criteria)
    ax.set_xlabel("Feature Testing Criterion")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Random Forest: Train vs CV Balanced Accuracy")
    ax.set_ylim([0, 1.1])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars_train, mean_train_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.4f}", ha="center", fontsize=8)
    for bar, val in zip(bars_cv, mean_test_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.4f}", ha="center", fontsize=8)

    plt.tight_layout()
    os.makedirs("q3", exist_ok=True)
    plt.savefig("q3/q3_rf_criterion_tuning.pdf")
    plt.show()

    # Retrain on full training set with best criterion
    rf_final = RandomForestClassifier(
        n_estimators=100, criterion=best_criterion, random_state=42)
    rf_final.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = rf_final.predict(X_test)
    metrics = calc_metrics(y_test, y_pred)
    print_metrics(metrics, "Random Forest - Test Set Performance:")

    performance_results["Random Forest"] = metrics

    # Compare Decision Tree vs Random Forest
    print(f"\n{'='*70}")
    print("COMPARISON: Decision Tree vs Random Forest")
    print(f"{'='*70}")
    for name in ["Decision Tree", "Random Forest"]:
        print(f"\n{name}:")
        for k, v in performance_results[name].items():
            print(f"  {k:20s}: {v:.4f}")

    # Summary comparison bar chart
    model_names = ["Decision Tree", "Random Forest"]
    metric_names = ["Accuracy", "Sensitivity", "Specificity",
                    "Precision", "Balanced Accuracy"]
    colors = ["steelblue", "coral"]

    fig, axes = plt.subplots(1, len(metric_names), figsize=(18, 4))

    for idx, metric in enumerate(metric_names):
        vals = [performance_results[m][metric] for m in model_names]
        bars = axes[idx].bar(model_names, vals, color=colors, alpha=0.8,
                             edgecolor="black")
        axes[idx].set_ylabel(metric)
        axes[idx].set_title(metric)
        axes[idx].set_ylim([min(0.3, min(vals) - 0.05), 1.0])
        axes[idx].grid(axis="y", alpha=0.3)
        axes[idx].tick_params(axis="x")
        for bar, val in zip(bars, vals):
            axes[idx].text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                           f"{val:.4f}", ha="center", fontsize=7)

    plt.tight_layout()
    plt.savefig("q3/q3_comparison_dt_vs_rf.pdf")
    plt.show()

    return rf_final, best_criterion, metrics


###############################################################################
#  Question 4 - Best Tree-Based Classifier
###############################################################################

def predictMCIconverters(Xtest, data_dir):
    """Returns a vector of predictions with elements "0" for sMCI and "1"
    for pMCI, corresponding to each of the N_test features vectors in Xtest

    Xtest       N_test x 14 matrix of test feature vectors
    data_dir    full path to the folder containing the following files:
                train.fdg_pet.sMCI.csv, train.fdg_pet.pMCI.csv,
                test.fdg_pet.sMCI.csv, test.fdg_pet.pMCI.csv
    """
    train_sMCI = pd.read_csv(os.path.join(
        data_dir, "train.fdg_pet.sMCI.csv"), header=None)
    train_pMCI = pd.read_csv(os.path.join(
        data_dir, "train.fdg_pet.pMCI.csv"), header=None)
    test_sMCI = pd.read_csv(os.path.join(
        data_dir, "test.fdg_pet.sMCI.csv"), header=None)
    test_pMCI = pd.read_csv(os.path.join(
        data_dir, "test.fdg_pet.pMCI.csv"), header=None)

    # Use all available data for training
    X_all = pd.concat([train_sMCI, train_pMCI, test_sMCI,
                       test_pMCI], ignore_index=True)
    y_all = np.array(
        [0]*len(train_sMCI) + [1]*len(train_pMCI) +
        [0]*len(test_sMCI) + [1]*len(test_pMCI)
    )

    # Random Forest with tuned hyperparameters
    model = RandomForestClassifier(
        n_estimators=500,
        criterion="entropy",
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_all, y_all)
    ytest = model.predict(Xtest)

    return ytest


###############################################################################
# Calls to generate the results
###############################################################################
if __name__ == "__main__":
    dt_final, best_criterion, metrics_dt = Q1_results()
    importances = Q2_results(dt_final)
    rf_final, best_rf_criterion, metrics_rf = Q3_results()
