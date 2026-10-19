import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fontTools.voltLib.voltToFea import TABLES
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import root_mean_squared_error


from src.constants import *

def calculate_average(values):
    return sum(values) / len(values)


def print_header(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)

def load_dataset():
    df = pd.read_csv(FILE_DATASET)

    # The target is the first column and the features are the remaining columns
    target = df.columns[0] # target is specific growth rate
    features = df.columns[1:] # Features are lactate, glucose, temp, and pH

    print(f"Number of samples: {df.shape[0]}")
    print(f"Target:   {target}")
    print(f"Features: {list(features)}")
    print()
    print(df.describe().round(3).to_string())

    return df, features, target

# ============================================
#           Figures and Tables
# ===========================================
def plot_poly_validation_curve(poly):
    df = pd.DataFrame(poly.model_gscv.cv_results_)
    degrees=df.loc[:,"param_polynomialfeatures__degree"].to_numpy()
    rmse_train = -df.loc[:, "mean_train_score"].to_numpy()
    rmse_test = -df.loc[:, "mean_test_score"].to_numpy()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.5,4), dpi=200, layout="constrained")
    ax.scatter(degrees, rmse_train, label="Train (Polynomial)", color="tab:blue", marker="o", s=16, alpha=0.7)
    ax.scatter(degrees, rmse_test, label="Test (Polynomial)", color="tab:red", marker="X", s=16, alpha=0.7)
    ax.set_xlabel("Polynomial degree", fontsize=10)
    ax.set_ylabel("RMSE", fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0,0.3)
    ax.xaxis.set_major_locator(MultipleLocator(1))

    fig.savefig(os.path.join("figures", "figure_01_poly_validation_curve.png"))
    plt.close(fig)


def plot_validation_curve_knn(knn_models):
    colors = ["tab:orange", "tab:red"]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.5, 4), dpi=200, layout="constrained")
    for knn, color in zip(knn_models, colors):
        df = pd.DataFrame(knn.model_gscv.cv_results_)

        k = df.loc[:, "param_kneighborsregressor__n_neighbors"].to_numpy()
        rmse_train = - df.loc[:, "mean_train_score"].to_numpy()
        rmse_valid = - df.loc[:, "mean_test_score"].to_numpy()

        # Label the distance-weighted model as "weighted" in the legend
        name = knn.name.replace("distance", "weighted")
        ax.plot(k, rmse_train, label=f"Train ({name})", color=color, linestyle="-")
        ax.plot(k, rmse_valid, label=f"Test ({name})", color=color, linestyle="--")

    ax.set_xlabel("k (number of neighbors)", fontsize=10)
    ax.set_ylabel("RMSE", fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)

    fig.savefig(os.path.join("figures", "figure_02_knn_validation_curve.png"))
    plt.close(fig)


def plot_learning_curve(learning_curve):
    df = learning_curve.df_results
    n = df["Training Samples"]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.5, 4), dpi=200, layout="constrained")
    ax.plot(n, df["Poly RMSE Train"], label="RMSE Train (Poly)", color="tab:blue", linestyle="-")
    ax.plot(n, df["Poly RMSE Test"], label="RMSE Test (Poly)", color="tab:blue", linestyle="--")
    ax.plot(n, df["kNN RMSE Train"], label="RMSE Train (kNN)", color="tab:orange", linestyle="-")
    ax.plot(n, df["kNN RMSE Test"], label="RMSE Test (kNN)", color="tab:orange", linestyle="--")
    ax.set_xlabel("Number of training samples", fontsize=10)
    ax.set_ylabel("RMSE", fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)

    fig.savefig(os.path.join("figures", "figure_03_learning_curve.png"))
    plt.close(fig)


def plot_response_curves(models, df, features, target, x_train):
    # Every feature except the one being varied is held at its mean value in the training set
    x_mean = np.mean(x_train, axis=0)
    colors = ["tab:blue", "tab:orange", "tab:red"]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 7), dpi=200, layout="constrained")

    for j, (ax, feature) in enumerate(zip(axes.flatten(), features)):
        # 100 copies of the "average" sample, then overwrite column j with evenly spaced values
        x_plot = np.tile(x_mean, (100, 1))
        x_plot[:, j] = np.linspace(df[feature].min(), df[feature].max(), num=100)

        ax.scatter(df[feature], df[target], s=10, color="grey", alpha=0.3, edgecolors="none", label="Samples")
        for model, color in zip(models, colors):
            ax.plot(x_plot[:, j], model.predict(x_plot), color=color, linewidth=2, label=model.name)

        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel(target, fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=10)

    axes[0, 0].legend(fontsize=10)

    fig.savefig(os.path.join("figures", "figure_04_response_curves.png"))
    plt.close(fig)


def export_dataset_summary(df):
    # One row for each column of the dataset (the target and the 4 features)
    records = []
    for column in df.columns:
        record = (column, df[column].mean(), df[column].std(), df[column].min(), df[column].max())
        records.append(record)

    column_names = ["Variable", "Mean", "Standard Deviation", "Minimum", "Maximum"]
    df_export = pd.DataFrame(records, columns=column_names).round(4)
    df_export.to_csv(os.path.join("tables", "table_01_dataset_summary.csv"), index=False)


def export_poly_coefficients(poly, features):
    model = poly.model_gscv.best_estimator_
    degree = model['polynomialfeatures'].degree
    powers = model['polynomialfeatures'].powers_
    coefs = model['linearregression'].coef_
    intercept = model['linearregression'].intercept_

    records = [("Intercept", degree, intercept)]
    for coef, power in zip(coefs, powers):
        partial_terms = []
        for j in range(len(power)):
            feature_name = features[j].split(" [")[0]
            if power[j] == 1:
                partial_terms.append(f"[{feature_name}]")
            elif power[j] > 1:
                partial_terms.append(f"[{feature_name}]^{power[j]}")
        record = ("*".join(partial_terms), degree, coef)
        records.append(record)

    column_names = ["Term", "Degree", "Coefficient"]
    df_export = pd.DataFrame(records, columns=column_names).round(4)
    df_export.to_csv(os.path.join("tables", "table_02_poly_coefficients.csv"), index=False)


def export_rmse(models, x_train, y_train, x_test, y_test):
    records = []
    for model in models:
        rmse_train = root_mean_squared_error(y_train, model.predict(x_train))
        rmse_valid = -model.model_gscv.best_score_  # Mean validation RMSE of the best hyperparameters
        rmse_test = root_mean_squared_error(y_test, model.predict(x_test))
        record = (model.name, rmse_train, rmse_valid, rmse_test)
        records.append(record)

    column_names = ["Model", "RMSE Train [d^-1]", "RMSE Valid [d^-1]", "RMSE Test [d^-1]"]
    df_export = pd.DataFrame(records, columns=column_names).round(4)
    df_export.to_csv(os.path.join("tables", "table_03_rmse.csv"), index=False)


def export_batch_predictions(batches):
    records = []
    for batch in batches:
        record = (batch.batch_id, batch.ph_opt, batch.temperature_opt, batch.glucose, batch.lactate,
                  batch.in_optimal_range(), batch.growth_rate)
        records.append(record)

    column_names = ["Batch ID", "pH", "Temperature [C]", "Glucose [mM]", "Lactate [mM]",
                    "In Optimal Range", "Predicted Growth Rate [d^-1]"]
    df_export = pd.DataFrame(records, columns=column_names).round(4)
    df_export.to_csv(os.path.join("tables", "table_04_batch_predictions.csv"), index=False)



