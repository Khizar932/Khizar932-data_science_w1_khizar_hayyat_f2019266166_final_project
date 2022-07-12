import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("wfp_food_prices_pakistan.csv")

from sklearn.linear_model import LinearRegressions
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

from sklearn.preprocessing import LabelEncoder
cols = ['cmname', 'unit', 'category', 'currency', 'country', 'admname', 'adm1id', 'mktname', 'cmid', 'catid', 'sn']
le = LabelEncoder()
for col in cols:
    df[col] = le.fit_transform(df[col])

from sklearn.model_selection import train_test_split
x = df[['cmname', 'unit', 'category', 'price', 'currency', 'country',
       'admname', 'adm1id', 'mktname', 'mktid', 'cmid', 'ptid', 'umid',
       'catid', 'sn']]
y = df['price']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 42)

lin_reg = LinearRegression()
sv = lin_reg.fit(x_train,y_train)

pickle.dump(sv, open('iri.pkl', 'wb'))
