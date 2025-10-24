import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import bernoulli, norm

def g_func(x: float) -> float: 
    return np.log(x / (1 - x))

def g_inverse(x: float) -> float: 
    return np.exp(x) / (1 + np.exp(x))


def main(part: str = None, return_participants: bool = False) -> pd.DataFrame:
    input_df = pd.read_csv("data/sls22_cleaned.csv")

    ### Part A ###
    print("running 3.a")
    participants = pd.DataFrame(columns=["id", "tricks_nonzero", "successes", "success_probability", "mean", "var"])
    
    for id in input_df["id"].unique().tolist():
        mask = input_df["id"] == id

        tricks = input_df[mask].iloc[:, 8:12].values.flatten().tolist()
        tricks_nonzero = [g_func(t) for t in tricks if t != 0]

        successes = len(tricks_nonzero)
        success_probability = successes / len(tricks)

        mean = sum(tricks_nonzero) / successes

        var = 0
        for trick in tricks_nonzero:
            var += (trick - mean)**2

        var /= successes - 1

        participants.loc[len(participants)] = [id, tricks_nonzero, successes, success_probability, mean, var]
    
    if return_participants:
        return participants

    if part == "a":
        for _, row in participants.iterrows():
            print(row["id"], row["mean"], row["var"])


    ### Part B ###
    print("running 3.b")
    num_samples = 500000
    names = participants["id"].unique().tolist()
    num_participants = len(names)
    simulated_values = np.zeros([num_samples, num_participants, 4])

    inverse = np.vectorize(g_inverse)

    names = []
    iter = 0
    for _, row in participants.iterrows():
        names.append(row["id"])

        trick_success = bernoulli.rvs(row["success_probability"], size = (num_samples, 1, 4))
        # print(trick_success)
        trick_results = norm.rvs(loc=row["mean"], scale=row["var"], size=(num_samples, 1, 4))
        # print(trick_results)
        # print(trick_success * inverse(trick_results))
        simulated_values[:, iter:iter+1, :] = trick_success * inverse(trick_results)
        
        iter += 1

    simulated_values = np.sort(simulated_values, axis=2)
    totals = simulated_values[:, :, 2:4].sum(axis=2)

    print(totals[:, 0])
    

    if part == "b":
        # print(participants["success_probability"])
        print(simulated_values[:, 0:1, :])
        # print(names)
        plt.boxplot(
            x=totals,
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
    
if __name__ == "__main__":
    main(sys.argv[1])    