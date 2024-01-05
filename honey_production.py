# Honey Production
# Intro to Machine Learning Course from Codeacademy

# Importing necessary libraries
import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# Reading the honey production data from Codecademy's URL into a DataFrame
df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")

# Displaying the first few rows of the DataFrame
print(df.head())

# Grouping the data by 'year' and calculating the mean of 'totalprod' for each year
prod_per_year = df.groupby('year').totalprod.mean().reset_index()
print(prod_per_year)

# Extracting 'year' column and reshaping it into a column vector
X = prod_per_year['year']
X = X.values.reshape(-1, 1)

# Extracting 'totalprod' column
y = prod_per_year['totalprod']

# Creating a scatter plot of 'year' against 'totalprod'
plt.scatter(X, y)

# Creating a linear regression model
regr = linear_model.LinearRegression()
regr.fit(X, y)

# Displaying the slope and intercept of the linear regression line
print('Slope of the line (m): ', regr.coef_[0])
print('Intercept of the line (b): ', regr.intercept_)

# Predicting 'totalprod' using the linear regression model
y_predict = regr.predict(X)

# Plotting the linear regression line
plt.plot(X, y_predict)
plt.show()

# Generating future years for prediction (2013 to 2050)
X_future = np.array(range(2013, 2051))

# Reshaping the future years array into a column vector
X_future = X_future.reshape(-1, 1)

# Predicting 'totalprod' for future years using the linear regression model
future_predict = regr.predict(X_future)

# Plotting the predicted values for future years
plt.plot(X_future, future_predict)
plt.show()