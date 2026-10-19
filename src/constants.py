import os

# File locations
FILE_DATASET = os.path.join("datasets", "dataset_cell_growth_4factor.csv")

# Batch setpoints
PH_LIMITS = (6.8, 7.2)
TEMPERATURE_LIMITS = (35.0, 37.0)
GLUCOSE_DEFAULT = 15.0  # [mM]
LACTATE_DEFAULT = 15.0  # [mM]

# Model development
TEST_SIZE = 0.2
RANDOM_STATE = 0
CV = 5
POLY_DEGREES = [i for i in range(1, 7)]
KNN_NEIGHBORS = [i for i in range(1, 51)]
KNN_WEIGHTS = ["uniform", "distance"]

# Learning curve
LC_SAMPLE_SIZES = [i for i in range(40, 320, 20)] + [318]
LC_KNN_NEIGHBORS = [i for i in range(1, 11)]
