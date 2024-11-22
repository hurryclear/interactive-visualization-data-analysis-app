import pandas as pd
import numpy as np

pd.read_csv('wein.csv', delimiter=',')
df = pd.read_csv('wein.csv', delimiter=',')
print(df.describe())