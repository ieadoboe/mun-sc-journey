#
#  Assignment 1
#
#  Group 18:
#  Isaac Adoboe                     ieadoboe@mun.ca
#  Blessing Ijeoma Benjamin-Igwe    bbenjaminigw@mun.ca
#  Promee Shankar Kundu             pskundu@mun.ca
####################################################################################
# Imports
####################################################################################
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

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


def load_data():
    print("Loading data...")
    # feature_names=["isthmuscingulate", "precuneus"] # original names
    feature_names = ["IC", "PC"]  # short hands

    # train sets
    train_sNC = pd.read_csv("train.sNC.csv", names=feature_names)
    train_sDAT = pd.read_csv("train.sDAT.csv", names=feature_names)

    # test sets
    test_sNC = pd.read_csv("test.sNC.csv", names=feature_names)
    test_sDAT = pd.read_csv("test.sDAT.csv", names=feature_names)

    # 2D grid points
    # axis_names = ["x", "y"]
    grid = pd.read_csv("2D_grid_points.csv", names=feature_names)

    print("Data loaded successfully!")
    return train_sNC, train_sDAT, test_sNC, test_sDAT, grid


def create_labels(train_sNC, train_sDAT, test_sNC, test_sDAT):
    print("Creating target labels...")
    ytrain_sNC = pd.DataFrame({"y": [0] * len(train_sNC)})
    ytrain_sDAT = pd.DataFrame({"y": [1] * len(train_sDAT)})
    ytest_sNC = pd.DataFrame({"y": [0] * len(test_sNC)})
    ytest_sDAT = pd.DataFrame({"y": [1] * len(test_sDAT)})

    print("Labels created")
    return ytrain_sNC, ytrain_sDAT, ytest_sNC, ytest_sDAT


def concat_data(
    train_sDAT,
    train_sNC,
    test_sDAT,
    test_sNC,
    ytrain_sDAT,
    ytrain_sNC,
    ytest_sDAT,
    ytest_sNC,
):
    # Concatenate datasets
    print("Concatenating datasets...")

    X_train = pd.concat([train_sDAT, train_sNC], ignore_index=True)
    X_test = pd.concat([test_sDAT, test_sNC], ignore_index=True)
    y_train = pd.concat([ytrain_sDAT, ytrain_sNC], ignore_index=True)
    y_test = pd.concat([ytest_sDAT, ytest_sNC], ignore_index=True)

    print(f"Concate complete!  X_train:{len(X_train)}, X_test: {len(X_test)}")

    return X_train, X_test, y_train, y_test


def train_knn(
    k, X_train, y_train, X_test, y_test, grid, metric="euclidean", save_loc="q1"
):
    # Train kNN classifier
    print(f"Training kNN (k={k}, metric={metric})...")
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    knn.fit(X_train, y_train.values.flatten())
    print("    Training complete!")

    # Predictions
    print("    Making predictions...")
    train_pred = knn.predict(X_train)
    test_pred = knn.predict(X_test)
    grid_pred = knn.predict(grid)

    # Calculate errors
    train_error = 1 - accuracy_score(y_train, train_pred)
    test_error = 1 - accuracy_score(y_test, test_pred)
    print(f"  Train Error: {train_error:.4f}, Test Error: {test_error:.4f}")

    # Visualization
    print("    Generating visualization...")
    plt.figure(figsize=(6, 6))
    colors = np.array(["green", "blue"])

    # Decision boundary
    plt.scatter(grid["IC"], grid["PC"], c=colors[grid_pred], marker=".", s=1, alpha=0.3)

    # Training data
    y_train_flat = y_train.values.flatten()
    plt.scatter(
        X_train[y_train_flat == 0]["IC"],
        X_train[y_train_flat == 0]["PC"],
        c="green",
        marker="o",
        s=50,
        edgecolors="black",
        linewidths=0.5,
        label="sNC Train",
    )
    plt.scatter(
        X_train[y_train_flat == 1]["IC"],
        X_train[y_train_flat == 1]["PC"],
        c="blue",
        marker="o",
        s=50,
        edgecolors="black",
        linewidths=0.5,
        label="sDAT Train",
    )

    # Test data
    y_test_flat = y_test.values.flatten()
    plt.scatter(
        X_test[y_test_flat == 0]["IC"],
        X_test[y_test_flat == 0]["PC"],
        c="green",
        marker="+",
        s=100,
        linewidths=2,
        label="sNC Test",
    )
    plt.scatter(
        X_test[y_test_flat == 1]["IC"],
        X_test[y_test_flat == 1]["PC"],
        c="blue",
        marker="+",
        s=100,
        linewidths=2,
        label="sDAT Test",
    )

    plt.xlabel("Isthmuscingulate")
    plt.ylabel("Precuneus")
    plt.legend(loc="lower right")
    plt.title(
        f"k={k}: \
                Train Error={train_error:.4f}, \
                Test Error={test_error:.4f}"
    )
    plt.tight_layout()

    # Create output directory
    os.makedirs(save_loc, exist_ok=True)
    filename = f"{save_loc}/{save_loc}_{metric}_k{k}.pdf"
    plt.savefig(filename)
    # plt.close()

    print(f"Plot saved: {filename}")
    return knn, train_error, test_error


def train_k_vals(
    X_train,
    y_train,
    X_test,
    y_test,
    grid,
    ks=[1, 3, 5, 10, 20, 30, 50, 100, 150, 200],
    metric="euclidean",
    save_loc="q1",
):
    """Train kNN for multiple k values"""
    print(f"\nTraining kNN for k values: {ks}")
    results = []

    for k in ks:
        knn, train_err, test_err = train_knn(
            k, X_train, y_train, X_test, y_test, grid, metric=metric, save_loc=save_loc
        )
        results.append(
            {"k": k, "train_error": train_err, "test_error": test_err, "model": knn}
        )
    return results


def Q1_results():
    print(f"\nQUESTION 1")
    print("Generating results for Q1...")

    # Load and prepare data
    train_sNC, train_sDAT, test_sNC, test_sDAT, grid = load_data()
    ytrain_sNC, ytrain_sDAT, ytest_sNC, ytest_sDAT = create_labels(
        train_sNC, train_sDAT, test_sNC, test_sDAT
    )
    X_train, X_test, y_train, y_test = concat_data(
        train_sDAT,
        train_sNC,
        test_sDAT,
        test_sNC,
        ytrain_sDAT,
        ytrain_sNC,
        ytest_sDAT,
        ytest_sNC,
    )

    # Train for all k values
    ks = [1, 3, 5, 10, 20, 30, 50, 100, 150, 200]
    results = train_k_vals(
        X_train, y_train, X_test, y_test, grid, ks=ks, metric="euclidean", save_loc="q1"
    )

    # Display results
    results_df = pd.DataFrame(results)[["k", "train_error", "test_error"]]
    print(results_df.to_string(index=False))

    # Find optimal k
    optimal_idx = results_df["test_error"].idxmin()
    optimal_k = results_df.loc[optimal_idx, "k"]
    best_test_error = results_df.loc[optimal_idx, "test_error"]
    print(f"\nOptimal k: {optimal_k} (Test Error: {best_test_error:.4f})")

    return results, X_train, y_train, X_test, y_test, grid


def Q2_results(q1_results, X_train, y_train, X_test, y_test, grid):
    print(f"\nQUESTION 2")
    print("Generating results for Q2...")

    # Find optimal k from Q1
    results_df = pd.DataFrame(q1_results)[["k", "train_error", "test_error"]]
    optimal_idx = results_df["test_error"].idxmin()
    optimal_k = int(results_df.loc[optimal_idx, "k"])
    print(f"\nUsing optimal k={optimal_k} from Q1")

    q1_train_error = results_df.loc[optimal_idx, "train_error"]
    q1_test_error = results_df.loc[optimal_idx, "test_error"]

    print(
        f"Q1 (euclidean): Train error={q1_train_error:.4f}, Test_error={q1_test_error:.4f}"
    )

    # Train with manhattan distance
    print(f"\nTraining kNN with Manhattan distance (k={optimal_k})...")
    knn_manhattan, train_error, test_error = train_knn(
        optimal_k,
        X_train,
        y_train,
        X_test,
        y_test,
        grid,
        metric="manhattan",
        save_loc="q2",
    )

    return {
        "k": optimal_k,
        "euclidean_train": q1_train_error,
        "euclidean_test": q1_test_error,
        "manhattan_train": train_error,
        "manhattan_test": test_error,
    }


def Q3_results(q2_results, X_train, y_train, X_test, y_test, grid):
    print(f"\nQUESTION 3")
    print("Generating results for Q3...")

    # Q1 vs Q2
    print("\nComparing Q1 and Q2 distance metric performance...")
    euclidean_test_error = q2_results["euclidean_test"]
    manhattan_test_error = q2_results["manhattan_test"]

    if manhattan_test_error < euclidean_test_error:
        best_metric = "manhattan"
        best_error = manhattan_test_error
        print(f"  Manhattan distance selected (Test Error: {best_error:.4f})")
    else:
        best_metric = "euclidean"
        best_error = euclidean_test_error
        print(f"  Euclidean distance selected (Test Error: {best_error:.4f})")

    # Generate k values for model capacity range [0.01, 1.00]
    # Model capacity = 1/k, so k ranges from 1 to 100
    print("\nGenerating k values for model capacity range [0.01, 1.00]...")

    # Sampling k values #
    # Create a dense sampling of k values
    k_values = []

    # Fine-grained sampling for small k (high capacity, overfitting region)
    k_values.extend(range(1, 11))  # k = 1 to 10

    # Medium sampling for moderate k
    k_values.extend(range(12, 31, 2))  # k = 12, 14, 16, ..., 30

    # Coarser sampling for large k (low capacity, underfitting region)
    k_values.extend(range(35, 101, 5))  # k = 35, 40, 45, ..., 100

    print(f"  Training models for {len(k_values)} different k values")
    # print(f'  k range: [{min(k_values)}, {max(k_values)}]')
    # print(f'  Model capacity (1/k) range: [{1/max(k_values):.4f}, {1/min(k_values):.4f}]')

    # Train models for all k values
    print("\nTraining kNN models...")
    results = []

    for i, k in enumerate(k_values, 1):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(k_values)} models trained")

        # Train model
        knn = KNeighborsClassifier(n_neighbors=k, metric=best_metric)
        knn.fit(X_train, y_train.values.flatten())

        # Predictions
        train_pred = knn.predict(X_train)
        test_pred = knn.predict(X_test)

        # Errors
        train_error = 1 - accuracy_score(y_train, train_pred)
        test_error = 1 - accuracy_score(y_test, test_pred)

        # Model capacity
        model_capacity = 1.0 / k

        results.append(
            {
                "k": k,
                "model_capacity": model_capacity,
                "train_error": train_error,
                "test_error": test_error,
            }
        )

    print(f"  Complete! {len(results)} models trained")

    # Convert to DataFrame for easier manipulation
    results_df = pd.DataFrame(results)

    # Find optimal model capacity (minimum test error)
    optimal_idx = results_df["test_error"].idxmin()
    optimal_k = results_df.loc[optimal_idx, "k"]
    optimal_capacity = results_df.loc[optimal_idx, "model_capacity"]
    optimal_train_error = results_df.loc[optimal_idx, "train_error"]
    optimal_test_error = results_df.loc[optimal_idx, "test_error"]

    print(f"\n====== OPTIMAL MODEL =======")
    print(f"Optimal k: {optimal_k}")
    print(f"Model Capacity (1/k): {optimal_capacity:.4f}")
    print(f"Train Error: {optimal_train_error:.4f}")
    print(f"Test Error: {optimal_test_error:.4f}")

    # Generate plot
    print("\nGenerating Error Rate vs Model Capacity plot...")

    plt.figure(figsize=(10, 6))

    # Plot training and test error curves
    plt.plot(
        results_df["model_capacity"],
        results_df["train_error"],
        "o-",
        label="Training Error",
        linewidth=2,
        markersize=4,
        color="blue",
        alpha=0.7,
    )
    plt.plot(
        results_df["model_capacity"],
        results_df["test_error"],
        "s-",
        label="Test Error",
        linewidth=2,
        markersize=4,
        color="red",
        alpha=0.7,
    )

    # Mark optimal point
    plt.plot(
        optimal_capacity,
        optimal_test_error,
        "*",
        markersize=15,
        color="gold",
        markeredgecolor="black",
        markeredgewidth=1.5,
        label=f"Optimal (k={optimal_k})",
        zorder=5,
    )

    # Add vertical line at optimal capacity
    plt.axvline(
        x=optimal_capacity, color="gold", linestyle="--", linewidth=1, alpha=0.5
    )

    # Annotations for overfitting and underfitting regions
    # Overfitting region (high capacity, small k)
    plt.text(
        0.7,
        0.02,
        "Overfitting\n(High Variance)",
        fontsize=10,
        ha="center",
        style="italic",
        bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.3),
    )

    # Underfitting region (low capacity, large k)
    plt.text(
        0.02,
        0.20,
        "Underfitting\n(High Bias)",
        fontsize=10,
        ha="center",
        style="italic",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3),
    )

    # Optimal region
    plt.text(
        optimal_capacity,
        0.25,
        "Sweet Spot",
        fontsize=9,
        ha="center",
        style="italic",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.3),
    )

    plt.xlabel("Model Capacity (1/k)", fontsize=12)
    plt.ylabel("Error Rate", fontsize=12)
    plt.title(
        f"Error Rate vs Model Capacity ({best_metric.capitalize()} Distance)",
        fontsize=13,
    )
    plt.xscale("log")
    plt.grid(True, alpha=0.3, which="both", linestyle="--")
    plt.legend(loc="best", fontsize=10)
    plt.tight_layout()

    # Save plot
    os.makedirs("q3", exist_ok=True)
    filename = f"q3/q3_capacity_vs_error_{best_metric}.pdf"
    plt.savefig(filename)
    plt.close()
    print(f"  Plot saved: {filename}")


def diagnoseDAT(Xtest, data_dir):
    """
    Returns a vector of predictions with elements "0" for sNC and "1" for sDAT,
    corresponding to each of the N_test features vectors in Xtest

    Xtest       N_test x 2 matrix of test feature vectors
    data_dir    full path to the folder containing the following files:
                train.sNC.csv, train.sDAT.csv, test.sNC.csv, test.sDAT.csv
    """

    # Load data
    feature_names = ["IC", "PC"]
    train_sNC = pd.read_csv(
        os.path.join(data_dir, "train.sNC.csv"), names=feature_names
    )
    train_sDAT = pd.read_csv(
        os.path.join(data_dir, "train.sDAT.csv"), names=feature_names
    )
    test_sNC = pd.read_csv(os.path.join(data_dir, "test.sNC.csv"), names=feature_names)
    test_sDAT = pd.read_csv(
        os.path.join(data_dir, "test.sDAT.csv"), names=feature_names
    )

    # Combine train and test for full model capacity
    all_sNC = pd.concat([train_sNC, test_sNC], ignore_index=True)
    all_sDAT = pd.concat([train_sDAT, test_sDAT], ignore_index=True)

    # target labels
    y_sNC = np.zeros(len(all_sNC), dtype=int)
    y_sDAT = np.ones(len(all_sDAT), dtype=int)

    # Concatenate
    X_train_all = pd.concat([all_sDAT, all_sNC], ignore_index=True).values
    y_train_all = np.concatenate([y_sDAT, y_sNC])

    # Convert Xtest to values (if dataframe)
    if isinstance(Xtest, pd.DataFrame):
        Xtest = Xtest.values

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_all)
    Xtest_scaled = scaler.transform(Xtest)

    # Train k-NN with optimal parameters
    knn = KNeighborsClassifier(n_neighbors=30, metric="euclidean", weights="distance")

    knn.fit(X_train_scaled, y_train_all)
    ypred = knn.predict(Xtest_scaled)

    return ypred


#########################################################################################
# Calls to generate the results
#########################################################################################
if __name__ == "__main__":
    apply_mpl_config()
    q1_results, X_train, y_train, X_test, y_test, grid = Q1_results()
    q2_results = Q2_results(q1_results, X_train, y_train, X_test, y_test, grid)
    Q3_results(q2_results, X_train, y_train, X_test, y_test, grid)
