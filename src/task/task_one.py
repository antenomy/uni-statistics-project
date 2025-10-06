from typing import Callable
import pandas as pd
import numpy as np
from matplotlib import pyplot
from scipy.special import polygamma
from scipy.optimize import fsolve

from config import BASE_DIR


def task_one(part: str = None) -> None:
    input_df = pd.read_csv(BASE_DIR + "data/sls22_cleaned.csv")

    ### Part A ###
    participant_names = input_df.iloc[:, 0].unique()
    
    tricks_flattened = input_df.iloc[:, 8:12].values.flatten().tolist()

    if part == "a":
        pyplot.hist(tricks_flattened, bins=20)
        pyplot.show()


    ### Part B ###
    participants = pd.DataFrame(columns=["id", "tricks", "successes", "success_probability", "alpha", "beta"])



    for name in input_df["id"].unique().tolist():
        mask = input_df["id"] == name

        tricks = input_df[mask].iloc[:, 8:12].values.flatten().tolist()
        successes = np.count_nonzero(tricks)
        success_probability = successes / len(tricks)

        participants.loc[len(participants)] = [name, tricks, successes, success_probability, 0, 0]

    if part == "b":
        for _, row in participants.iterrows():
            print(row["id"], row["success_probability"])

    
    ### Part D ###

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
    
    if part == "d":
        for _, row in participants.iterrows():
            print(row["id"], row["alpha"], row["beta"])
    
    
    ### Part E ###

    if part == "e":
        for _, row in participants.iterrows():
            success_probability = row["success_probability"]
            alpha = row["alpha"]
            beta = row["beta"]
            expected_value = success_probability * alpha / (alpha + beta)
            print(row["id"], expected_value)
    