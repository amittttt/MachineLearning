import numpy as np  # mathamatical library
import matplotlib.pyplot as plt  ## Plot chart
import pandas as pd   # import data set and manage dataset

# Importing the dataset

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values
