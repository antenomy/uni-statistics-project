from typing import Callable
import pandas as pd
import numpy as np
from matplotlib import pyplot
from scipy.special import polygamma
from scipy.optimize import fsolve

from config import BASE_DIR

# print(df.describe())

class Participant():
    def __init__(self, name:str):
        self.name = name

        self.total_tricks: int = 0
        self.successes: int = 0
        self.scores: list = []
        


def task_one(part: str = None) -> None:
    df = pd.read_csv(BASE_DIR + "data/sls22_cleaned.csv")

    # Part A
    participant_names = df.iloc[:, 0].unique()
    # participant_names = participant_names.unique()
    participants = {participant: Participant(participant) for participant in participant_names}

    trick_scores = df.iloc[:, 8:11]

    sublist_of_tricks = trick_scores.values.tolist()
    tricks_flattened = [item for sublist in sublist_of_tricks for item in sublist]

    if part == "a":
        pyplot.hist(tricks_flattened, bins=20)

    # Part B
    for _, row in df.iterrows():
        participant_name = row["id"]
        participant_tricks = row[8:12]

        participants[participant_name].successes += np.count_nonzero(participant_tricks)
        participants[participant_name].scores.extend(participant_tricks)

    participant_probs = []

    for p in participants.values():
        successes = np.count_nonzero(np.array(p.scores))
        prob = successes / len(p.scores)

        participant_probs.append(prob)

    if part == "b":
        pyplot.bar(participant_names , participant_probs)
        pyplot.xticks(rotation=90)

    
    # Part D
    def digam(x: float)-> float: return polygamma(1, x)
    def trigam(x: float)-> float: return polygamma(2, x)

    def jacobian(prim, sec, n, x_sum)-> float: 
        
        return n * digam(prim + sec) - n * digam(prim) + (prim -1) * x_sum

    def optimize_bloop(x: np.array, jacobian: Callable) -> np.array:
       
        x = x[x != 0]
        n = len(x)

        ln_x_sum = np.sum(np.log(x))
        sub_ln_x_sum = np.sum(np.log(1 - x))

        sol = fsolve(
            func=lambda a: np.array([
                jacobian(a[0], a[1], n, ln_x_sum),
                jacobian(a[1], a[0], n, sub_ln_x_sum)
            ]),
            x0=[0.5, 0.5]
        )

        print(sol)
    
    for p in participants.values():
        optimize_bloop(np.array(p.scores), jacobian)
        
    if part == "d":
        pass
        
        

    pyplot.show()