# =====================================================================
# MAIN PROGRAM
# Predicts the specific growth rate of cells from the temperature, pH,
# glucose concentration, and lactate concentration of a cell culture.
# =====================================================================

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt

import src.functions as sf
from src.classes import Batch, GrowthRateModel, LearningCurve
from src.constants import *

# 1. Load the dataset
sf.print_header("1. Dataset")
df, features, target = sf.load_dataset()

# 2. Split the data into training and testing sets
x = df.loc[:, features].to_numpy()
y = df.loc[:, target].to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# 3. Polynomial regression: find the best degree with GridSearchCV
sf.print_header("2. Polynomial regression")
poly = GrowthRateModel(
    name="Polynomial",
    pipeline=make_pipeline(StandardScaler(), PolynomialFeatures(include_bias=False), LinearRegression()),
    param_grid=dict(polynomialfeatures__degree=POLY_DEGREES),
)
poly.grid_search(x_train, y_train)
poly.display_polynomial(features)

# 4. kNN regression: find the best k with GridSearchCV, once for each weighting method
sf.print_header("3. kNN regression")
knn_uniform = GrowthRateModel(
    name="kNN (uniform)",
    pipeline=make_pipeline(StandardScaler(), KNeighborsRegressor()),
    param_grid=dict(kneighborsregressor__n_neighbors=KNN_NEIGHBORS, kneighborsregressor__weights=["uniform"]),
)
knn_distance = GrowthRateModel(
    name="kNN (distance)",
    pipeline=make_pipeline(StandardScaler(), KNeighborsRegressor()),
    param_grid=dict(kneighborsregressor__n_neighbors=KNN_NEIGHBORS, kneighborsregressor__weights=["distance"]),
)
knn_uniform.grid_search(x_train, y_train)
knn_distance.grid_search(x_train, y_train)
knn_uniform.display_knn()
knn_distance.display_knn()

# 5. Compare the RMSE of the three models
sf.print_header("4. RMSE comparison")
poly.display_rmse(x_train, y_train, x_test, y_test)
knn_uniform.display_rmse(x_train, y_train, x_test, y_test)
knn_distance.display_rmse(x_train, y_train, x_test, y_test)

# 6. Learning curve: does adding more training samples improve the models?
sf.print_header("5. Learning curve")
learning_curve = LearningCurve(df, features, target, LC_SAMPLE_SIZES)
learning_curve.run()
learning_curve.display_results()

# 7. Predict the growth rate of a few batches with the polynomial model
sf.print_header("6. Batch predictions")
ph_optimal = sf.calculate_average(PH_LIMITS)
temperature_optimal = sf.calculate_average(TEMPERATURE_LIMITS)

batches = [
    Batch(batch_id=1, ph_opt=ph_optimal, temperature_opt=temperature_optimal),
    Batch(batch_id=2, ph_opt=ph_optimal, temperature_opt=temperature_optimal, glucose=30.0, lactate=3.0),
    Batch(batch_id=3, ph_opt=ph_optimal, temperature_opt=temperature_optimal, glucose=3.0, lactate=30.0),
    Batch(batch_id=4, ph_opt=6.4, temperature_opt=40.0),
]

for batch in batches:
    batch.predict_growth_rate(poly)
    batch.display_info()

# 8. Plots and tables
sf.plot_poly_validation_curve(poly)
sf.plot_validation_curve_knn([knn_uniform, knn_distance])
sf.plot_learning_curve(learning_curve)
sf.plot_response_curves([poly, knn_uniform, knn_distance], df, features, target, x_train)

sf.export_dataset_summary(df)
sf.export_poly_coefficients(poly, features)
sf.export_rmse([poly, knn_uniform, knn_distance], x_train, y_train, x_test, y_test)
sf.export_batch_predictions(batches)