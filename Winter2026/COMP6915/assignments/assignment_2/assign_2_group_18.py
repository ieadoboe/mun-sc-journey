#
#  Assignment 2
#
#  Group 18:
#  Isaac Adoboe                     ieadoboe@mun.ca
#  Blessing Ijeoma Benjamin-Igwe    bbenjaminigw@mun.ca
#  Promee Shankar Kundu             pskundu@mun.ca
#
####################################################################################
# Imports
####################################################################################

import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    KFold,
    train_test_split,
    cross_val_score,
    GridSearchCV,
)

#####################################################################################
#  Implementation - helper functions and question wise result generators
#####################################################################################


def apply_mpl_config():
    print("Apply custom plotting configs...")
    mpl_config = {
        "figure.figsize": (12, 6),
        "savefig.dpi": 300,
        "figure.dpi": 150,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "text.usetex": True,
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


def calc_rse(y_test, y_pred, p):
    """
    Calculate RSE - Residual Standard Error

    Args:
        y_test: test set
        y_pred: predicted set
        p: number of features

    Returns RSE
    """
    n = len(y_test)
    df = n - p - 1

    assert n == len(y_pred), "Prediction and target length mismatch"
    assert n > p + 1, f"Insufficient samples: need n > {p+1}, got {n}"

    residuals = y_test - y_pred
    rss = np.sum(residuals**2)

    return np.sqrt(rss / df)


def calc_r2(y_test, y_pred):
    """
    Calculate R^2 - Coefficient of determination

    Args:
        Args:
        y_test: test set
        y_pred: predicted set

    Returns R^2
    """
    assert len(y_test) == len(y_pred), "Prediction and target length mismatch"

    ss_res = np.sum((y_test - y_pred) ** 2)  # sum of squares residuals
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)  # sum of squares total

    return 1 - (ss_res / ss_tot)


original_cols = [
    "Cement_component1__kgInAM_3Mixture_",
    "BlastFurnaceSlag_component2__kgInAM_3Mixture_",
    "FlyAsh_component3__kgInAM_3Mixture_",
    "Water_component4__kgInAM_3Mixture_",
    "Superplasticizer_component5__kgInAM_3Mixture_",
    "CoarseAggregate_component6__kgInAM_3Mixture_",
    "FineAggregate_component7__kgInAM_3Mixture_",
    "Age_day_",
    "ConcreteCompressiveStrength_MPa_Megapascals_",
]


simple_cols = [
    "Cement",
    "BlastFurnaceSlag",
    "FlyAsh",
    "Water",
    "Superplasticizer",
    "CoarseAggregate",
    "FineAggregate",
    "Age_day",
    "ConcCompStrength_MPa",
]

performance_results = {}


def load_data():
    print("Loading data...")
    train = pd.read_csv("./train.csv")
    test = pd.read_csv("./test.csv")
    return train, test


# Load and split data
train, test = load_data()

# Rename columns in train and test
train.columns = simple_cols
test.columns = simple_cols

# Train features and target
X_train = train.drop("ConcCompStrength_MPa", axis=1)
y_train = train["ConcCompStrength_MPa"]

# Test features and target
X_test_holdout = test.drop("ConcCompStrength_MPa", axis=1)
y_test_holdout = test["ConcCompStrength_MPa"]

p = X_train.shape[1]  # n_features
print(f"Number of features: {p}")


# Split: train 60, val 20, test 20
# 80% temp, 20% test
X_temp, X_val_test, y_temp, y_val_test = train_test_split(
    X_train, y_train, test_size=0.20, random_state=42, shuffle=True
)

# 60% train, 20% val (from the 80% temp)
X_val_train, X_val_val, y_val_train, y_val_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=42,
    shuffle=True  # 0.25 * 0.8 = 0.2
)

print(
    f"Train size: {len(X_val_train)} ({len(X_val_train)/len(X_train)*100:.1f}%)")
print(
    f"Validation size: {len(X_val_val)} ({len(X_val_val)/len(X_train)*100:.1f}%)")
print(
    f"Test size: {len(X_val_test)} ({len(X_val_test)/len(X_train)*100:.1f}%)")
print(f"Total: {len(X_train):4d} samples")

# Combine train + val
X_train_full = pd.concat([X_val_train, X_val_val], axis=0)
y_train_full = pd.concat([y_val_train, y_val_val], axis=0)

#####################################################################################
#  Question 1
#####################################################################################


def Q1_results():
    # define scaler
    scaler = StandardScaler()

    # Define linear regression model (ols)
    reg = LinearRegression()

    # Define Pipeline
    pipe_lr = Pipeline([("scaler", scaler), ("reg", reg)])

    # Train on train + val (full)
    pipe_lr.fit(X_train_full, y_train_full)

    # 1.1 Validation Approach Results
    # ------------------------------------------------------------------------------------

    # Predict on held-out test set (20%)
    y_pred_val = pipe_lr.predict(X_val_test.values)

    # Calculate metrics
    p = X_train.shape[1]  # number of features
    rse_val = calc_rse(y_val_test, y_pred_val, p)
    r2_val = calc_r2(y_val_test, y_pred_val)
    rmse_val = np.sqrt(np.mean((y_val_test - y_pred_val)**2))

    print("\nValidation Approach Results (on 20% hold-out test):")
    print(f"  RMSE: {rmse_val:.4f}")
    print(f"  RSE:  {rse_val:.4f}")
    print(f"  R²:   {r2_val:.4f}")

    # 1.2 Cross-validation
    # ------------------------------------------------------------------------------------

    n_folds = [5, 10]  # testing diff folds
    cv_results = {}

    for n_folds in n_folds:
        print(f"\n{'='*60}")
        print(f"Cross-Validation with k={n_folds} folds")
        print(f"{'='*60}")

        # Create fresh pipeline for CV
        pipe_cv = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", LinearRegression())
        ])

        # Set up k-fold CV
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        # Perform cross-validation on FULL train.csv
        neg_mse_scores = cross_val_score(
            pipe_cv,
            X_train,  # ← Use FULL training data
            y_train,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1
        )

        # Convert to RMSE
        mse_scores = -neg_mse_scores
        rmse_scores = np.sqrt(mse_scores)

        # Calculate statistics
        rmse_mean = np.mean(rmse_scores)
        rmse_std = np.std(rmse_scores)
        rmse_se = rmse_std / np.sqrt(n_folds)  # Standard error of the mean

        # Store results
        cv_results[n_folds] = {
            'rmse_mean': rmse_mean,
            'rmse_std': rmse_std,
            'rmse_se': rmse_se,
            'rmse_scores': rmse_scores,
            'train_size_per_fold': len(X_train) * (n_folds - 1) / n_folds
        }

        print(
            f"  Training samples per fold: {cv_results[n_folds]['train_size_per_fold']:.0f}")  # noqa E501
        print(f"  Test samples per fold:     {len(X_train) / n_folds:.0f}")
        print(f"  RMSE: {rmse_mean:.4f} ± {rmse_std:.4f} (std)")
        print(f"        {rmse_mean:.4f} ± {rmse_se:.4f} (SE)")
        print(
            f"  RMSE range: [{np.min(rmse_scores):.4f}, {np.max(rmse_scores):.4f}]")
        print(
            f"  95% CI (approx): [{rmse_mean - 1.96*rmse_se:.4f}, {rmse_mean + 1.96*rmse_se:.4f}]")  # noqa E501

    print("\n" + "="*70)
    print("COMPARISON: Validation vs Cross-Validation Approaches")
    print("="*70)

    print("\n1. VALIDATION APPROACH (60% train + 20% val → test on 20%):")
    print(f"   RMSE = {rmse_val:.4f}")
    print(f"   RSE  = {rse_val:.4f}")
    print(f"   R²   = {r2_val:.4f}")
    print(f"   Test set size: {len(X_val_test)} samples")

    print("\n2. CROSS-VALIDATION (5-fold on 100% of data):")
    print(
        f"   RMSE = {cv_results[5]['rmse_mean']:.4f} ± {cv_results[5]['rmse_se']:.4f} (SE)")  # noqa E501
    print(
        f"   Training per fold: {cv_results[5]['train_size_per_fold']:.0f} samples (80%)")  # noqa E501
    print(f"   Test per fold: {len(X_train)/5:.0f} samples (20%)")

    print("\n3. CROSS-VALIDATION (10-fold on 100% of data):")
    print(
        f"   RMSE = {cv_results[10]['rmse_mean']:.4f} ± {cv_results[10]['rmse_se']:.4f} (SE)")  # noqa E501
    print(
        f"   Training per fold: {cv_results[10]['train_size_per_fold']:.0f} samples (90%)")  # noqa E501
    print(f"   Test per fold: {len(X_train)/10:.0f} samples (10%)")

    performance_results["OLS_validation"] = {
        "RMSE": rmse_val,
        "RSE": rse_val,
        "R2": r2_val,
    }
    performance_results["OLS_CV5"] = {
        "RMSE": cv_results[5]['rmse_mean'],
        "RMSE_std": cv_results[5]['rmse_std'],
    }
    performance_results["OLS_CV10"] = {
        "RMSE": cv_results[10]['rmse_mean'],
        "RMSE_std": cv_results[10]['rmse_std']
    }

    # 1.3 Train model on full train set
    # ------------------------------------------------------------------------------------

    # Train on full train.csv
    pipe_ols_final = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", LinearRegression())
    ])
    pipe_ols_final.fit(X_train, y_train)

    # Predict on test (train test subset)
    y_pred_ols = pipe_ols_final.predict(X_test_holdout)

    # Calc errors
    rse_ols = calc_rse(y_test_holdout, y_pred_ols, p)
    rmse_ols = np.sqrt(np.mean((y_test_holdout - y_pred_ols)**2))
    r2_ols = calc_r2(y_test_holdout, y_pred_ols)

    performance_results["OLS"] = {
        "RMSE": rmse_ols,
        "RSE": rse_ols,
        "R2": r2_ols,
    }

    print(f"RSE:  {rse_ols:.4f}")
    print(f"R^2:   {r2_ols:.4f}")

    return pipe_ols_final, rse_ols, r2_ols, rmse_ols

#####################################################################################
#  Question 2
#####################################################################################


def Q2_results(rmse_ols, rse_ols, r2_ols):
    # 2.1 Grid Search with Cross-Validation
    # ------------------------------------------------------------------------------------

    # Define Ridge pipeline
    pipe_ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge())
    ])

    # Define alpha grid (logarithmic spacing for better coverage)
    alphas = np.logspace(-3, 3, 50)  # 0.001 to 1000
    param_grid = {"ridge__alpha": alphas}

    print("\nGrid Search Configuration:")
    print(f"  Alpha range: [{alphas[0]:.4f}, {alphas[-1]:.2f}]")
    print(f"  Number of alpha values: {len(alphas)}")
    print("  Cross-validation: 10-fold")
    print(f"  Training data: Full train.csv ({len(X_train)} samples)")

    # Perform grid search with 10-fold CV
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        pipe_ridge,
        param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",
        return_train_score=True,
        n_jobs=-1,
        verbose=1,
    )

    print("\nRunning Grid Search...")
    grid_search.fit(X_train, y_train)

    # Extract results
    best_alpha_ridge = grid_search.best_params_["ridge__alpha"]
    best_cv_rmse = np.sqrt(-grid_search.best_score_)
    cv_results_grid = grid_search.cv_results_

    print(f"\n{'='*70}")
    print("BEST HYPERPARAMETER:")
    print(f"{'='*70}")
    print(f"Best alpha: {best_alpha_ridge:.6f}")
    print(f"Best CV RMSE: {np.sqrt(-grid_search.best_score_):.4f}")

    # Extract training and validation scores
    alphas_tested = cv_results_grid['param_ridge__alpha'].data
    mean_train_rmse = np.sqrt(-cv_results_grid['mean_train_score'])
    mean_test_rmse = np.sqrt(-cv_results_grid['mean_test_score'])
    std_test_rmse = np.sqrt(cv_results_grid['std_test_score'])

    # Find min/max performance
    idx_best = np.argmin(mean_test_rmse)
    idx_worst = np.argmax(mean_test_rmse)

    print("\nPerformance Range:")
    alpha_best = alphas_tested[idx_best]
    alpha_worst = alphas_tested[idx_worst]
    rmse_best = mean_test_rmse[idx_best]
    rmse_worst = mean_test_rmse[idx_worst]
    print(
        f"  Best  (alpha={alpha_best:.4f}: RMSE = {rmse_best}")
    print(
        f"  Worst (alpha={alpha_worst:.4f}): RMSE = {rmse_worst:.4f}")
    print("  Performance degradation:")
    print(
        f"  {((mean_test_rmse[idx_worst]/mean_test_rmse[idx_best] - 1)*100):.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: Test RMSE vs Alpha (log scale)
    axes[0].semilogx(alphas_tested, mean_test_rmse, 'b-',
                     linewidth=2, label='CV Test RMSE')
    axes[0].fill_between(
        alphas_tested,
        mean_test_rmse - std_test_rmse,
        mean_test_rmse + std_test_rmse,
        alpha=0.2,
        color='blue',
        label=r'$\pm$ 1 std'
    )
    axes[0].axvline(best_alpha_ridge, color='red', linestyle='--',
                    linewidth=1.5, label=f'Best $\\alpha$ = {best_alpha_ridge:.4f}')
    axes[0].scatter([best_alpha_ridge], [best_cv_rmse],
                    color='red', s=100, zorder=5)
    axes[0].set_xlabel('Regularization Parameter ($\\alpha$)', fontsize=11)
    axes[0].set_ylabel('RMSE', fontsize=11)
    axes[0].set_title(
        'Ridge: Cross-Validation Performance vs $\\alpha$', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    # Right plot: Train vs Test RMSE (overfitting analysis)
    axes[1].semilogx(alphas_tested, mean_train_rmse, 'g-',
                     linewidth=2, label='Train RMSE')
    axes[1].semilogx(alphas_tested, mean_test_rmse, 'b-',
                     linewidth=2, label='Test RMSE')
    axes[1].axvline(best_alpha_ridge, color='red', linestyle='--',
                    linewidth=1.5, label=f'Best $\\alpha$ = {best_alpha_ridge:.4f}')
    axes[1].set_xlabel('Regularization Parameter ($\\alpha$)', fontsize=11)
    axes[1].set_ylabel('RMSE', fontsize=11)
    axes[1].set_title('Ridge: Train vs Test Error', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    plt.tight_layout()

    # Save plot
    os.makedirs("q2", exist_ok=True)
    filename = "q2/q2_ridge_hyperparameter_tuning.pdf"
    plt.savefig(filename)
    plt.close()
    print(f"  Plot saved: {filename}")

    # 2.2 Train final model on best alpha value
    # ------------------------------------------------------------------------------------

    pipe_ridge_final = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=best_alpha_ridge)),
        ]
    )

    print("\nTraining final Ridge model:")
    print(f"  Alpha: {best_alpha_ridge:.6f}")
    print(f"  Training set: Full train.csv ({len(X_train)} samples)")
    print(f"  Features: {X_train.shape[1]}")

    pipe_ridge_final.fit(X_train, y_train)
    print("Model trained successfully!")

    # Predict on X_test_holdout
    y_pred_ridge = pipe_ridge_final.predict(X_test_holdout)

    # Calculate metrics
    rmse_ridge = np.sqrt(np.mean((y_test_holdout - y_pred_ridge)**2))
    rse_ridge = calc_rse(y_test_holdout, y_pred_ridge, p)
    r2_ridge = calc_r2(y_test_holdout, y_pred_ridge)

    print(f"\nRidge Performance on test.csv ({len(X_test_holdout)} samples):")
    print(f"  RMSE: {rmse_ridge:.4f}")
    print(f"  RSE:  {rse_ridge:.4f}")
    print(f"  R2:   {r2_ridge:.4f}")

    # Store results
    performance_results["Ridge"] = {
        "RMSE": rmse_ridge,
        "RSE": rse_ridge,
        "R2": r2_ridge,
        "alpha": best_alpha_ridge,
        "CV_RMSE": best_cv_rmse,
    }

    # 2.3 Compare: OLS vs Ridge
    # ------------------------------------------------------------------------------------

    print(f"\n{'='*70}")
    print("COMPARISON: OLS vs RIDGE REGRESSION")
    print(f"{'='*70}")

    print("\nOLS (no regularization):")
    print(f"  RMSE: {rmse_ols:.4f}")
    print(f"  RSE:  {rse_ols:.4f}")
    print(f"  R²:   {r2_ols:.4f}")

    print(f"\nRidge (alpha = {best_alpha_ridge:.6f}):")
    print(f"  RMSE: {rmse_ridge:.4f}")
    print(f"  RSE:  {rse_ridge:.4f}")
    print(f"  R²:   {r2_ridge:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    models = ['OLS', 'Ridge']
    rse_values = [performance_results['OLS']['RSE'],
                  performance_results['Ridge']['RSE']]
    r2_values = [performance_results['OLS']['R2'],
                 performance_results['Ridge']['R2']]
    rmse_values = [performance_results['OLS']['RMSE'],
                   performance_results['Ridge']['RMSE']]

    # RSE comparison
    axes[0].bar(models, rse_values, color=['steelblue',
                'coral'], alpha=0.8, edgecolor='black')
    axes[0].set_ylabel('RSE', fontsize=12)
    axes[0].set_title('RSE - Residual Squared Error (test.csv)', fontsize=12)
    axes[0].grid(axis='y', alpha=0.3)
    for i, (model, rse) in enumerate(zip(models, rse_values)):
        axes[0].text(i, rse + 0.1, f'{rse:.3f}', ha='center', fontsize=10)

    # R² comparison
    axes[1].bar(models, r2_values, color=['steelblue', 'coral'],
                alpha=0.8, edgecolor='black')
    axes[1].set_ylabel('$R^2$', fontsize=12)
    axes[1].set_title(
        r'$R^2$ - Coefficient of Determination (test.csv)', fontsize=12)
    axes[1].set_ylim([0, 1.0])
    axes[1].grid(axis='y', alpha=0.3)
    for i, (model, r2) in enumerate(zip(models, r2_values)):
        axes[1].text(i, r2 + 0.01, f'{r2:.4f}', ha='center', fontsize=10)

    # RMSE comparison
    axes[2].bar(models, rmse_values, color=['steelblue',
                'coral'], alpha=0.8, edgecolor='black')
    axes[2].set_ylabel('RMSE', fontsize=12)
    axes[2].set_title(
        'RMSE - Residual Mean Squared Error (test.csv)', fontsize=12)
    axes[2].grid(axis='y', alpha=0.3)
    for i, (model, rmse) in enumerate(zip(models, rmse_values)):
        axes[2].text(i, rmse + 0.1, f'{rmse:.3f}', ha='center', fontsize=10)
    plt.tight_layout()

    os.makedirs("q2", exist_ok=True)
    filename = "q2/q2_comparison_ols_ridge.pdf"
    plt.savefig(filename)
    plt.close()
    print(f" Plot saved: {filename}")

    return pipe_ridge_final, rse_ridge, r2_ridge, rmse_ridge, best_alpha_ridge


#####################################################################################
#  Question 3
#####################################################################################

# 3.1 Grid Search with Cross-Validation
# ------------------------------------------------------------------------------------

def Q3_results(rmse_ols, rse_ols, r2_ols,
               rse_ridge, r2_ridge, rmse_ridge, best_alpha_ridge):
    # Define Lasso pipeline
    pipe_lasso = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", Lasso())
    ])

    # Define alpha grid (logarithmic spacing for better coverage)
    alphas_lasso = np.logspace(-3, 3, 50)  # 0.001 to 1000
    param_grid_lasso = {"lasso__alpha": alphas_lasso}

    print("\nGrid Search Configuration:")
    print(f"  Alpha range: [{alphas_lasso[0]:.4f}, {alphas_lasso[-1]:.2f}]")
    print(f"  Number of alpha values: {len(alphas_lasso)}")
    print("  Cross-validation: 10-fold")
    print(f"  Training data: Full train.csv ({len(X_train)} samples)")

    # Perform grid search with 10-fold CV on FULL train.csv
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    grid_search_lasso = GridSearchCV(
        pipe_lasso,
        param_grid_lasso,
        cv=cv,
        scoring="neg_mean_squared_error",
        return_train_score=True,
        n_jobs=-1,
        verbose=1,
    )

    print("\nRunning Grid Search...")
    grid_search_lasso.fit(X_train, y_train)

    # Extract results
    best_alpha_lasso = grid_search_lasso.best_params_["lasso__alpha"]
    best_cv_rmse_lasso = np.sqrt(-grid_search_lasso.best_score_)
    cv_results_lasso = grid_search_lasso.cv_results_

    print(f"\n{'='*70}")
    print("BEST HYPERPARAMETER:")
    print(f"{'='*70}")
    print(f"  Alpha: {best_alpha_lasso:.6f}")
    print(f"  CV RMSE: {best_cv_rmse_lasso:.4f}")

    # Extract training and validation scores
    alphas_tested_lasso = cv_results_lasso['param_lasso__alpha'].data
    mean_train_rmse_lasso = np.sqrt(-cv_results_lasso['mean_train_score'])
    mean_test_rmse_lasso = np.sqrt(-cv_results_lasso['mean_test_score'])
    std_test_rmse_lasso = np.sqrt(cv_results_lasso['std_test_score'])

    # Find min/max performance
    idx_best_lasso = np.argmin(mean_test_rmse_lasso)
    idx_worst_lasso = np.argmax(mean_test_rmse_lasso)

    alpha_best_lasso = alphas_tested_lasso[idx_best_lasso]
    alpha_worst_lasso = alphas_tested_lasso[idx_worst_lasso]
    rmse_best_lasso = mean_test_rmse_lasso[idx_best_lasso]
    rmse_worst_lasso = mean_test_rmse_lasso[idx_worst_lasso]
    perf_degrad_lasso = (mean_test_rmse_lasso[idx_worst_lasso] /
                         mean_test_rmse_lasso[idx_best_lasso] - 1)*100

    print("\nPerformance Range:")
    print(
        f"  Best  (alpha={alpha_best_lasso:.4f}: RMSE = {rmse_best_lasso}")
    print(
        f"  Worst (alpha={alpha_worst_lasso:.4f}): RMSE = {rmse_worst_lasso:.4f}")
    print("  Performance degradation:")
    print(f"{perf_degrad_lasso:.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: Test RMSE vs Alpha (log scale)
    axes[0].semilogx(alphas_tested_lasso, mean_test_rmse_lasso,
                     'b-', linewidth=2, label='CV Test RMSE')
    axes[0].fill_between(
        alphas_tested_lasso,
        mean_test_rmse_lasso - std_test_rmse_lasso,
        mean_test_rmse_lasso + std_test_rmse_lasso,
        alpha=0.2,
        color='blue',
        label=r'$\pm$ 1 std'
    )
    axes[0].axvline(best_alpha_lasso, color='red', linestyle='--', linewidth=1.5,
                    label=f'Best $\\alpha$ = {best_alpha_lasso:.4f}')
    axes[0].scatter([best_alpha_lasso], [best_cv_rmse_lasso],
                    color='red', s=100, zorder=5)
    axes[0].set_xlabel('Regularization Parameter ($\\alpha$)', fontsize=11)
    axes[0].set_ylabel('RMSE', fontsize=11)
    axes[0].set_title(
        'Lasso: Cross-Validation Performance vs $\\alpha$', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    # Right plot: Train vs Test RMSE (overfitting analysis)
    axes[1].semilogx(alphas_tested_lasso, mean_train_rmse_lasso,
                     'g-', linewidth=2, label='Train RMSE')
    axes[1].semilogx(alphas_tested_lasso, mean_test_rmse_lasso,
                     'b-', linewidth=2, label='Test RMSE')
    axes[1].axvline(best_alpha_lasso, color='red', linestyle='--', linewidth=1.5,
                    label=f'Best $\\alpha$ = {best_alpha_lasso:.4f}')
    axes[1].set_xlabel('Regularization Parameter ($\\alpha$)', fontsize=11)
    axes[1].set_ylabel('RMSE', fontsize=11)
    axes[1].set_title('Lasso: Train vs Test Error', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    plt.tight_layout()

    # Save plot
    os.makedirs("q3", exist_ok=True)
    filename = "q3/q3_lasso_hyperparameter_tuning.pdf"
    plt.savefig(filename)
    plt.close()
    print(f"  Plot saved: {filename}")

    pipe_lasso_final = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", Lasso(alpha=best_alpha_lasso, max_iter=10000))
    ])

    print("\nTraining final Lasso model:")
    print(f"  Alpha: {best_alpha_lasso:.6f}")
    print(f"  Training set: Full train.csv ({len(X_train)} samples)")
    print(f"  Features: {X_train.shape[1]}")

    pipe_lasso_final.fit(X_train, y_train)
    print("Model trained successfully!")

    # Predict on X_test_holdout
    y_pred_lasso = pipe_lasso_final.predict(X_test_holdout)

    # Calculate metrics
    rmse_lasso = np.sqrt(np.mean((y_test_holdout - y_pred_lasso)**2))
    rse_lasso = calc_rse(y_test_holdout, y_pred_lasso, p)
    r2_lasso = calc_r2(y_test_holdout, y_pred_lasso)

    print(f"\nLasso Performance on test.csv ({len(X_test_holdout)} samples):")
    print(f"  RMSE: {rmse_lasso:.4f}")
    print(f"  RSE:  {rse_lasso:.4f}")
    print(f"  R²:   {r2_lasso:.4f}")

    # Store results
    performance_results["Lasso"] = {
        "RMSE": rmse_lasso,
        "RSE": rse_lasso,
        "R2": r2_lasso,
        "alpha": best_alpha_lasso,
        "CV_RMSE": best_cv_rmse_lasso
    }

    # 3.3 Compare: OLS vs Ridge vs Lasso
    # ------------------------------------------------------------------------------------

    print(f"\n{'='*70}")
    print("COMPARISON: OLS vs RIDGE vs LASSO REGRESSION")
    print(f"{'='*70}")

    print("\nOLS (no regularization):")
    print(f"  RMSE: {rmse_ols:.4f}")
    print(f"  RSE:  {rse_ols:.4f}")
    print(f"  R²:   {r2_ols:.4f}")

    print(f"\nRidge (alpha = {best_alpha_ridge:.6f}):")
    print(f"  RMSE: {rmse_ridge:.4f}")
    print(f"  RSE:  {rse_ridge:.4f}")
    print(f"  R²:   {r2_ridge:.4f}")

    print(f"\nLasso (alpha = {best_alpha_lasso:.6f}):")
    print(f"  RMSE: {rmse_lasso:.4f}")
    print(f"  RSE:  {rse_lasso:.4f}")
    print(f"  R²:   {r2_lasso:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    models = ['OLS', 'Ridge', 'Lasso']
    rse_values = [performance_results['OLS']['RSE'],
                  performance_results['Ridge']['RSE'],
                  performance_results['Lasso']['RSE']]
    r2_values = [performance_results['OLS']['R2'],
                 performance_results['Ridge']['R2'],
                 performance_results['Lasso']['R2']]
    rmse_values = [performance_results['OLS']['RMSE'],
                   performance_results['Ridge']['RMSE'],
                   performance_results['Lasso']['RMSE']]

    # RSE comparison
    axes[0].bar(models, rmse_values, color=['steelblue',
                'coral', 'seagreen'], alpha=0.8, edgecolor='black')
    axes[0].set_ylabel('RSE', fontsize=12)
    axes[0].set_title('RSE - Residual Squared Error (test.csv)', fontsize=12)
    axes[0].grid(axis='y', alpha=0.3)
    for i, (model, rse) in enumerate(zip(models, rse_values)):
        axes[0].text(i, rse - 0.3, f'{rse:.3f}', ha='center', fontsize=10)

    # R2 comparison
    axes[1].bar(models, r2_values, color=['steelblue', 'coral', 'seagreen'],
                alpha=0.8, edgecolor='black')
    axes[1].set_ylabel('$R^2$', fontsize=12)
    axes[1].set_title(
        r'$R^2$ - Coefficient of Determination (test.csv)', fontsize=12)
    axes[1].set_ylim([0, 1.0])
    axes[1].grid(axis='y', alpha=0.3)
    for i, (model, r2) in enumerate(zip(models, r2_values)):
        axes[1].text(i, r2 + 0.01, f'{r2:.4f}', ha='center', fontsize=10)

    # RMSE comparison
    axes[2].bar(models, rmse_values, color=['steelblue',
                'coral', 'seagreen'], alpha=0.8, edgecolor='black')
    axes[2].set_ylabel('RMSE', fontsize=12)
    axes[2].set_title(
        'RMSE - Residual Mean Squared Error (test.csv)', fontsize=12)
    axes[2].grid(axis='y', alpha=0.3)
    for i, (model, rmse) in enumerate(zip(models, rmse_values)):
        axes[2].text(i, rmse + 0.1, f'{rmse:.3f}', ha='center', fontsize=10)
    plt.tight_layout()

    # Save plot
    os.makedirs("q3", exist_ok=True)
    filename = "q3/q3_comparison_ols_ridge_lasso.pdf"
    plt.savefig(filename)
    plt.close()
    print(f" Plot saved: {filename}")

    return pipe_lasso_final, rse_lasso, r2_lasso, rmse_lasso


def predictCompressiveStrength(Xtest, data_dir):
    """
    Returns a vector of predictions of real number values,
    corresponding to each of the N_test features vectors in Xtest

    Xtest       N_test x 8 matrix of test feature vectors
    data_dir    full path to the folder containing the following files:
                train.csv, test.csv
    """

    # Load data
    train = pd.read_csv(
        os.path.join(data_dir, "train.csv"),
    )
    test = pd.read_csv(
        os.path.join(data_dir, "test.csv"),
    )

    # Combine train and test for full model capacity
    all_data = pd.concat([train, test], ignore_index=True)

    # Train features and target
    X_train_all = all_data.drop(
        "ConcCompStrength_MPa", axis=1)
    y_train_all = all_data["ConcCompStrength_MPa"]

    pipe_strength = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", LinearRegression())
    ])
    pipe_strength.fit(X_train_all, y_train_all)

    # Predict on test (train test subset)
    ypred = pipe_strength.predict(Xtest)

    return ypred


#########################################################################################
# Calls to generate the results
#########################################################################################
if __name__ == "__main__":
    apply_mpl_config()
    pipe_ols_final, rse_ols, r2_ols, rmse_ols = Q1_results()
    pipe_ridge_final, rse_ridge, r2_ridge, rmse_ridge, best_alpha_ridge = Q2_results(  # noqa
        rmse_ols, rse_ols, r2_ols)
    pipe_lasso_final, rse_lasso, r2_lasso, rmse_lasso = Q3_results(
        rmse_ols, rse_ols, r2_ols, rse_ridge, r2_ridge, rmse_ridge, best_alpha_ridge)  # noqa
