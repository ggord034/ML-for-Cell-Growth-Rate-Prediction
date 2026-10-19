# =======================================================================
# CLASSES
# Define custom classes in this file.
# =======================================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from src.constants import *


class Batch:
    def __init__(self, batch_id, ph_opt, temperature_opt, glucose=GLUCOSE_DEFAULT, lactate=LACTATE_DEFAULT):
        self.batch_id = batch_id
        self.ph_opt = ph_opt
        self.temperature_opt = temperature_opt
        self.glucose = glucose
        self.lactate = lactate
        self.growth_rate = None

    def in_optimal_range(self):
        ph_ok = (PH_LIMITS[0] <= self.ph_opt <= PH_LIMITS[1])
        temp_ok = (TEMPERATURE_LIMITS[0] <= self.temperature_opt <= TEMPERATURE_LIMITS[1])
        return ph_ok and temp_ok

    def predict_growth_rate(self, model):
        x = np.array([[self.temperature_opt, self.ph_opt, self.glucose, self.lactate]])
        self.growth_rate = model.predict(x)[0]
        return self.growth_rate

    def display_info(self):
        print(f"Batch ID: {self.batch_id:4d} | "
              f"Optimal pH: {self.ph_opt:4.1f} | "
              f"Optimal Temperature: {self.temperature_opt:4.1f} C | "
              f"Glucose: {self.glucose:4.1f} mM | "
              f"Lactate: {self.lactate:4.1f} mM | "
              f"In optimal range: {self.in_optimal_range()} | "
              f"Predicted growth rate: {self.growth_rate:.3f} d^-1")

class GrowthRateModel:
    def __init__(self, name, pipeline, param_grid):
        self.name = name
        self.pipeline = pipeline
        self.param_grid = param_grid
        self.model_gscv = None  # Will hold the GridSearchCV object once grid_search() is called

    def grid_search(self, x_train, y_train):
        # Try every combination of hyperparameters in param_grid using cross-validation on the training set.
        # GridSearchCV then refits the best combination on the whole training set.
        self.model_gscv = GridSearchCV(self.pipeline, param_grid=self.param_grid,
                                       cv=CV, scoring="neg_root_mean_squared_error", return_train_score=True)
        self.model_gscv.fit(x_train, y_train)

    def predict(self, x):
        return self.model_gscv.predict(x)

    def display_rmse(self, x_train, y_train, x_test, y_test):
        rmse_train = root_mean_squared_error(y_train, self.predict(x_train))
        rmse_valid = -self.model_gscv.best_score_  # Mean validation RMSE of the best hyperparameters
        rmse_test = root_mean_squared_error(y_test, self.predict(x_test))
        print(f"{self.name:<16} | RMSE Train: {rmse_train:.4f} | Valid: {rmse_valid:.4f} | Test: {rmse_test:.4f} d^-1")

    def display_polynomial(self, features):
        # Only works for the polynomial pipeline
        model = self.model_gscv.best_estimator_
        degree = model['polynomialfeatures'].degree
        powers = model['polynomialfeatures'].powers_
        coefs = model['linearregression'].coef_
        intercept = model['linearregression'].intercept_

        print(f"Degree:    {degree}")
        print(f"Intercept: {intercept:.4f}")
        print("Coefficients (standardized features):")
        for coef, power in zip(coefs, powers):
            partial_terms = []
            for j in range(len(power)):
                feature_name = features[j].split(" [")[0]
                if power[j] == 1:
                    partial_terms.append(f"[{feature_name}]")
                elif power[j] > 1:
                    partial_terms.append(f"[{feature_name}]^{power[j]}")
            print(f"  {coef:+.4f}  {'*'.join(partial_terms)}")

    def display_knn(self):
        knn = self.model_gscv.best_estimator_['kneighborsregressor']
        print(f"{self.name:<16} | k (n_neighbors): {knn.n_neighbors} | Weights: {knn.weights}")


class LearningCurve:
    def __init__(self, df, features, target, sample_sizes):
        self.x_all = df.loc[:, features].to_numpy()
        self.y_all = df.loc[:, target].to_numpy()
        self.sample_sizes = sample_sizes
        self.df_results = None

        # Initialize a random number generator to select random subsets from dataset
        self.rng = np.random.default_rng(seed=1)

    def run(self):
        rows = []
        for n_samples in self.sample_sizes:
            # Select a random subset of n_samples from the dataset
            subset = self.rng.choice(self.x_all.shape[0], size=n_samples, replace=False)
            x = self.x_all[subset]
            y = self.y_all[subset]

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE,
                                                                random_state=RANDOM_STATE)

            # New models are created for every subset so that no information carries over between subsets
            poly = GrowthRateModel(
                "Polynomial",
                make_pipeline(StandardScaler(), PolynomialFeatures(include_bias=False), LinearRegression()),
                dict(polynomialfeatures__degree=POLY_DEGREES))
            knn = GrowthRateModel(
                "kNN",
                make_pipeline(StandardScaler(), KNeighborsRegressor()),
                dict(kneighborsregressor__n_neighbors=LC_KNN_NEIGHBORS, kneighborsregressor__weights=KNN_WEIGHTS))
            poly.grid_search(x_train, y_train)
            knn.grid_search(x_train, y_train)

            rows.append({
                "Training Samples": x_train.shape[0],
                "Poly Degree": poly.model_gscv.best_params_["polynomialfeatures__degree"],
                "Poly RMSE Train": root_mean_squared_error(y_train, poly.predict(x_train)),
                "Poly RMSE Test": root_mean_squared_error(y_test, poly.predict(x_test)),
                "kNN k": knn.model_gscv.best_params_["kneighborsregressor__n_neighbors"],
                "kNN Weights": knn.model_gscv.best_params_["kneighborsregressor__weights"],
                "kNN RMSE Train": root_mean_squared_error(y_train, knn.predict(x_train)),
                "kNN RMSE Test": root_mean_squared_error(y_test, knn.predict(x_test)),
            })

        self.df_results = pd.DataFrame(rows)

    def display_results(self):
        print(self.df_results.round(4).to_string())


