from typing import Callable
import pandas as pd
import numpy as np
from matplotlib import pyplot
from scipy.special import polygamma
from scipy.optimize import fsolve
import sys

def main(part: str = "") -> None:
    print(f"running taks one, { f'part {part}' if part != '' else 'all parts'}")
    input_df = pd.read_csv("data/sls22_cleaned.csv")

    ### Part A ###
    print("running part A")
    participant_names = input_df.iloc[:, 0].unique()
    
    tricks_flattened = input_df.iloc[:, 8:12].values.flatten().tolist()

    if part == "a" or part == "":
        pyplot.hist(tricks_flattened, bins=20)
        pyplot.show()


    ### Part B ###
    print("running part B")
    participants = pd.DataFrame(columns=["id", "tricks", "successes", "success_probability", "alpha", "beta"])



    for name in input_df["id"].unique().tolist():
        mask = input_df["id"] == name

        tricks = input_df[mask].iloc[:, 8:12].values.flatten().tolist()
        successes = np.count_nonzero(tricks)
        success_probability = successes / len(tricks)

        participants.loc[len(participants)] = [name, tricks, successes, success_probability, 0, 0]

    if part == "b" or part == "":
        for _, row in participants.iterrows():
            print(row["id"], row["success_probability"])

    
    ### Part D ###
    print("running part D")
    for _, row in participants.iterrows():
        name = row["id"]
        tricks = [x for x in row["tricks"] if x != 0]

        m1 = np.mean(tricks)

        tricks_squared = [trick ** 2 for trick in tricks] 
        m2 = np.mean(tricks_squared)

        alpha = m1 * (m2 - m1) / (m1 ** 2 - m2)
        beta = (1 - m1) * (m2 - m1) / (m1 ** 2 - m2)

        participants.loc[participants["id"] == name, "alpha"] = alpha
        participants.loc[participants["id"] == name, "beta"] = beta
    
    if part == "d" or part == "":
        for _, row in participants.iterrows():
            print(row["id"], row["alpha"], row["beta"])
    
    
    ### Part E ###
    print("running part E")
    if part == "e" or part == "":
        for _, row in participants.iterrows():
            success_probability = row["success_probability"]
            alpha = row["alpha"]
            beta = row["beta"]
            expected_value = success_probability * alpha / (alpha + beta)
            print(row["id"], expected_value)

if __name__ == "__main__":
    main(sys.argv[1])