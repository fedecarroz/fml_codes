import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, learning_curve

seed = 42

# read the dataset of houses prices
df = pd.read_csv('../datasets/petrol_consumption.csv')

# select feature value
x = df[['Petrol_tax', 'Average_income', 'Paved_Highways', 'Population_Driver_licence(%)']].values
# select target value
y = df['Petrol_Consumption'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=seed)

lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

print('R2 Score: {}'.format(lr.score(x_test, y_test)))
print('MSE: {}'.format(mean_squared_error(y_test, y_pred)))
print('MAE: {}'.format(mean_absolute_error(y_test, y_pred)))
# print('R2: {}'.format(r2_score(y_test, y_pred)))
