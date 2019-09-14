# Data Preprocessing ::::::::

### Importing the Libraries ::::

import numpy as np  # mathamatical library
import matplotlib.pyplot as plt  ## Plot chart
import pandas as pd   # import data set and manage dataset

# Importing the dataset

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

# Taking Care of Missing 

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean', verbose=0, copy=True)
imputer = imputer.fit(x[:, 1:3])
x[:,1:3] = imputer.transform(x[:, 1:3])

# Encoding Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x  = LabelEncoder();
x[:,0]=labelencoder_x.fit_transform(x[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()
labelencoder_y  = LabelEncoder();
y=labelencoder_x.fit_transform(y)


# Training and Test Set Data need to divide ::::
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

# Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)





