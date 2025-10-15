import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pandas.core.common import T
from scipy.stats import norm

from config import BASE_DIR

def g_inverse(x: float) -> float: 
    return np.exp(x) / (1 + np.exp(x))


def task_three(part: str = None) -> None:
    input_df = pd.read_csv(BASE_DIR + "data/sls22_cleaned.csv")

    ### Part A ###
    participants = pd.DataFrame(columns=["id", "tricks_results", "successes", "success_probability", "mean", "var"])

    for id in input_df["id"].unique().tolist():
        mask = input_df["id"] == id

        tricks = input_df[mask].iloc[:, 8:12].values.flatten().tolist()
        tricks_nonzero = [t for t in tricks if t != 0]

        successes = len(tricks_nonzero)
        success_probability = successes / len(tricks)

        mean = sum(tricks) / successes

        var = 0
        for trick in tricks_nonzero:
            var += (trick - mean)**2

        var /= successes - 1

        participants.loc[len(participants)] = [id, tricks, successes, success_probability, mean, var]
    
    



    if part == "a":
        for _, row in participants.iterrows():
            print(row["id"], row["mean"], row["var"])


    ### Part B ###
    num_samples = 100
    names = participants["id"].unique().tolist()
    simulated_values = np.zeros([num_samples, len(names), 4])

    inverse = np.vectorize(g_inverse)

    iter = 0
    for id in names:
        row = input_df[input_df["id"] == id].iloc[0]
        simulated_values[0] = inverse(norm.rvs(loc=row["mean"], scale=row["var"], size=num_samples))

        iter += 1

    simulated_values = np.sort(simulated_values, axis=1)

    if part == "b":
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
    
    # ### Part D ###

    # for _, row in participants.iterrows():
    #     name = row["id"]
    #     tricks = [x for x in row["tricks"] if x != 0]

    #     m1 = np.mean(tricks)

    #     tricks_squared = [trick ** 2 for trick in tricks] 
    #     m2 = np.mean(tricks_squared)

    #     alpha = m1 * (m2 - m1) / (m1 ** 2 - m2)
    #     beta = (1 - m1) * (m2 - m1) / (m1 ** 2 - m2)

    #     participants.loc[participants["id"] == name, "alpha"] = alpha
    #     participants.loc[participants["id"] == name, "beta"] = beta
    
    # if part == "d":
    #     for _, row in participants.iterrows():
    #         print(row["id"], row["alpha"], row["beta"])
    
    
    # ### Part E ###

    # if part == "e":
    #     for _, row in participants.iterrows():
    #         success_probability = row["success_probability"]
    #         alpha = row["alpha"]
    #         beta = row["beta"]
    #         expected_value = success_probability * alpha / (alpha + beta)
    #         print(row["id"], expected_value)
    