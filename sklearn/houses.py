import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, learning_curve

seed = 42

# read the dataset of houses prices
df = pd.read_csv('../datasets/houses.csv')

# replace missing values with mean of all the other values of that feature
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean())
df = df.dropna(axis=1)

# select only some features
x = df[['GrLivArea', 'LotArea', 'GarageArea', 'FullBath']].values
# select target value
y = df['SalePrice'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=seed)

lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

print('Intercept: {}'.format(lr.intercept_))  # Theta0
print('Thetas: {}'.format(lr.coef_))  # Other thetas
print('Score: {}'.format(lr.score(x_test, y_test)))
print('MAE: {}'.format(mean_absolute_error(y_test, y_pred)))
print('MSE: {}'.format(mean_squared_error(y_test, y_pred)))
print('R2: {}'.format(r2_score(y_test, y_pred)))
