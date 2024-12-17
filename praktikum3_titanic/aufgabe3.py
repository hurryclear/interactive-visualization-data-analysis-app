import pandas as pd
import numpy as np
#from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('titanic.csv')
df.head()
df.tail()

df.info()

df.isnull().sum()

df.isnull().sum()/df.shape[0]*100