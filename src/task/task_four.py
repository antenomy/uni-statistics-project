import pandas as pd
import numpy as np
from tqdm import tqdm

from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy.stats import cauchy
import random

from task_three import main as task_three

def sample_and_apriori(theta: float, var: float, data_array: np.array) -> float:
    exponent = -np.sum((data_array - theta)**2)/(2*var)
    return (((1 + (theta - 2)**2)/0.36)**(5/2)) * np.exp(exponent)

def aposteori_theta(theta: float, var: float, data_array: np.array) -> float:
    apriori_predictive, _ = quad(lambda x: sample_and_apriori(x, var, data_array), -np.inf, np.inf)
    return sample_and_apriori(theta, var, data_array) / apriori_predictive

def metropolis(var: float, data_array: np.array) -> np.array:
    

    markov_chain = []
   #markov_chain_values = []

    old_x = cauchy.rvs()
    old_value = aposteori_theta(old_x, var, data_array)
   
    # markov_chain_values.append(old_value)
    markov_chain.append(old_x)

    # iter = 0
    for i in tqdm(range(10000-1)):
        new_x = (random.random() * 0.2) - 0.1 + old_x
        
        u = random.random()

        new_value = aposteori_theta(new_x, var, data_array)

        R = new_value/old_value

        markov_chain.append(old_x)
        #markov_chain_values.append(new_value)
        
        if u < min(1, R):
            old_x = new_x
            old_value = new_value

            # iter += 1

    return np.array(markov_chain) #np.vstack((markov_chain, markov_chain_values))
    
    
        


def task_four(part: str = None) -> None:
    
    participants = task_three()
    chain_count = 4

    ### Part A ###
    for _, row in participants.iterrows():
        # row = participants["id"=="Papa"]

        var = row["var"]
        tricks_results = np.array(row["tricks_results"])

        result_array = np.zeros([10000, chain_count])

        for i in range(chain_count):
            result_array[:, i] = metropolis(var, tricks_results)

        # print(result_array)

        plt.plot(result_array, marker='o', linestyle='-', markersize=1)
        plt.show()

        element_count = len(result_array[0])
        rolling_average = np.zeros([element_count])
        for iter in range(1, element_count):
            # if iter < 100:
            #     print(result_array[0, 0:iter])
            rolling_average[iter] = np.mean(result_array[0, 0:iter])
        
        # for in i range(chain_count)
        plt.plot(rolling_average, linestyle='-')
        plt.show()

        plt.plot(result_array[0, 20000:], marker='o', linestyle='-', markersize=1)
        plt.show()

        plt.plot(rolling_average[20000:], linestyle='-')
        plt.show()

        plt.plot(result_array[0, 5000:], marker='o', linestyle='-', markersize=1)
        plt.show()

        plt.plot(rolling_average[5000:], linestyle='-')
        plt.show()