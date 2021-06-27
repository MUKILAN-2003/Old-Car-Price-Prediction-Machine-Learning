import numpy as np
import pandas as pd
import datetime
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

dataset = pd.read_csv("data\Old_Car_Price.csv")

x_train, x_test, y_train, y_test = train_test_split(dataset.iloc[:, :-1], 
                                                    dataset.iloc[:, -1],
                                                    test_size = 0.3)

x_train = pd.get_dummies(x_train,
                         columns = ['brand','body-style','drive-wheels','engine-type','fuel-system'],
                         drop_first = True)

x_test = pd.get_dummies(x_test,
                         columns = ['brand','body-style','drive-wheels','engine-type','fuel-system'],
                         drop_first = True)

missing_cols = set(x_train.columns) - set(x_test.columns)
for col in missing_cols:
    x_test[col] = 0
x_test = x_test[x_train.columns]



standardScaler = StandardScaler()
standardScaler.fit(x_train)
print(x_train.columns)
print(x_test)
x_train = standardScaler.transform(x_train)
x_test = standardScaler.transform(x_test)

#Linear Regression Model
model = LinearRegression()
model.fit(x_train, y_train)
y_pred_l = model.predict(x_test)
print(r2_score(y_test, y_pred_l))

'''
86%
m_name = 'old_car_price_LR.ml'
pickle.dump(model,open(m_name,'wb'))
'''

#RandomForest Model
model = RandomForestRegressor(n_estimators = 100)
model.fit(x_train, y_train)
y_pred_r = model.predict(x_test)
print(r2_score(y_test, y_pred_r))

'''
93%
m_name = 'old_car_price_RF.ml'
pickle.dump(model,open(m_name,'wb'))
'''


