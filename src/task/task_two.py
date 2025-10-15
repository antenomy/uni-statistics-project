import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

def g_func(x: float) -> float: 
    return np.log(x / (1 - x))

def g_inverse(x: float) -> float: 
    return np.exp(x) / (1 + np.exp(x))

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
    input_df = pd.read_csv("data/sls22_cleaned.csv")

    ### Part A ###

    # Column names to modify
    columns_to_change = ["run 1", "run 2", "trick 1", "trick 2", "trick 3", "trick 4"]

    input_df[columns_to_change] = input_df[columns_to_change].map(lambda x: g_func(x) if x != 0 else x)

    if part == "a":
        print(input_df)


    ### Part B ###
    samples = input_df[["run 1", "run 2"]].to_numpy()

    sample_covariance_matrix = calculate_sample_covariance(X=samples)
    
    if part == "b":
        print(sample_covariance_matrix)
    

    ### Part C ###
    participants = pd.DataFrame(columns=["id", "runs", "mean_1", "mean_2", "lam", "var_1", "var_2"])

    for name in input_df["id"].unique().tolist():
        mask = input_df["id"] == name

        runs = input_df[mask].iloc[:, 6:8].values.flatten().tolist()
        r_1 = input_df[mask].iloc[:, 6:7].values.flatten()
        r_2 = input_df[mask].iloc[:, 7:8].values.flatten()

        mean_1 = np.mean(r_1)

        # Create input matrix for covariance calculation function, then calculate it
        R = np.column_stack((r_1, r_2))
        sample_covariance_matrix = calculate_sample_covariance(R)

        var_1 = sample_covariance_matrix[0, 0] 
        cov_1_2 = sample_covariance_matrix[0, 1]

        lam = cov_1_2 / var_1

        var_2 = sample_covariance_matrix[1, 1] - (var_1 * lam ** 2)

        mean_2 = np.mean(r_2) - (lam * mean_1)

        participants.loc[len(participants)] = [name, runs, mean_1, mean_2, lam, var_1, var_2]
        
    if part == "c":
        for _, row in participants.iterrows():
            print(row["id"], row["mean_1"], row["mean_2"], row["lam"], row["var_1"], row["var_2"])


    ### PART D ###
    participants["samples"] = 10
    num_samples = 100
    names = participants["id"].unique().tolist()
    simulated_values = np.zeros([num_samples, len(names)])

    inverse = np.vectorize(g_inverse)
    iter = 0

    for _, row in participants.iterrows():
        r_1_samples = inverse(norm.rvs(loc=row["mean_1"], scale=row["var_1"], size=num_samples))
        r_2_samples = inverse(row["lam"] * r_1_samples + norm.rvs(loc=row["mean_2"], scale=abs(row["var_2"]), size=num_samples))

        new_values = np.max(np.column_stack((r_1_samples, r_2_samples)), axis=1)

        simulated_values[:, iter] = new_values
        iter += 1
    
    if part == "d":
        print(simulated_values.shape)
        print(names)
        plt.boxplot(
            x=simulated_values,
            labels=names
        )

        plt.xticks(rotation=45)
        plt.title("Simulated Value Distributions")
        plt.ylabel("Value")
        plt.show()