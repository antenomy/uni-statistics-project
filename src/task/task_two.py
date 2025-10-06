import pandas as pd
import numpy as np
from matplotlib import pyplot

from config import BASE_DIR

def inverse(x: float) -> float: 
    return np.log(x / (1 - x))

def calculate_sample_covariance(X: np.array)-> np.array:
    
    sample_count = X.shape[0]
    dim = X.shape[1]

    result = np.zeros([dim, dim])
    mean = np.mean(X, axis=0)
    
    for i in range(sample_count):
        result += np.outer(X[i,:] - mean, X[i,:] - mean)

    result /= sample_count

    return result



def task_two(part: str = None) -> None:
    # Import data from csv
    df = pd.read_csv(BASE_DIR + "data/sls22_cleaned.csv")

    ### Part A ###

    # Column names to modify
    columns_to_change = ["run 1", "run 2", "trick 1", "trick 2", "trick 3", "trick 4"]

    df[columns_to_change] = df[columns_to_change].applymap(lambda x: inverse(x) if x != 0 else x)

    if part == "a":
        print(df)


    ### Part B ###
    samples = df[["run 1", "run 2"]].to_numpy()

    sample_covariance_matrix = calculate_sample_covariance(X=samples)
    
    if part == "b":
        print(sample_covariance_matrix)
    

    ### Part C ###