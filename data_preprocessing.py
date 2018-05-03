# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("Mall_Customers.csv")
df['Gen'] = np.where(df.Genre == "Male",0,1)
df = pd.get_dummies(df,columns =['Gen'],drop_first=True)
df.drop(['Genre','CustomerID'],axis=1,inplace=True)
df.to_csv('ProcessedData.csv')