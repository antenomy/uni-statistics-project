import pandas as pd

from config import BASE_DIR

df = pd.read_csv(BASE_DIR + "data/sls22_cleaned.csv")

print(df.describe())