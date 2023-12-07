# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

data = pd.read_csv('../input/nfl-fantasy-football-dataset/fantasy-football-data-train.csv', header = None, skiprows=1)
test_data = pd.read_csv('../input/nfl-fantasy-football-dataset/fantasy-football-data-test.csv', header = None, skiprows=1)
data.columns =(['Player','Bye','Total Fantasy Points','Passing Attempts','Passing Completions','Passing Yards','Passing Touchdowns','Interceptions','Passing 2pt Conversion','Rushing Attempts','Rushing Yards','Rushing Touchdowns', 'Rushing 2pt Conversions', 'Receptions', 'Receiving Yards', 'Receiving Touchdowns', 'Receiving 2pt Conversions', 'Fumbles Lost', 'Fumble Touchdowns'])
test_data.columns = (['Player','Bye','Total Fantasy Points','Passing Attempts','Passing Completions','Passing Yards','Passing Touchdowns','Interceptions','Passing 2pt Conversion','Rushing Attempts','Rushing Yards','Rushing Touchdowns', 'Rushing 2pt Conversions', 'Receptions', 'Receiving Yards', 'Receiving Touchdowns', 'Receiving 2pt Conversions', 'Fumbles Lost', 'Fumble Touchdowns'])
data.head()

X = data.iloc[:,6] # Passing TDs
y = data.iloc[:,2] # Total Fantasy Points
m = len(y) # number of training example

#Plot Data
plt.scatter(X, y)
plt.xlabel('Passing TDs')
plt.ylabel('Total Fantasy Points')
plt.show()

X = data.iloc[:,5] # Passing Yards
y = data.iloc[:,2] # Total Fantasy Points
m = len(y) # number of training example

#Plot Data
plt.scatter(X, y)
plt.xlabel('Passing Yards')
plt.ylabel('Total Fantasy Points')
plt.show()

X = data.iloc[:,4] # Passing Completions
y = data.iloc[:,2] # Total Fantasy Points
m = len(y) # number of training example

#Plot Data
plt.scatter(X, y)
plt.xlabel('Passing Completions')
plt.ylabel('Total Fantasy Points')
plt.show()

X = data.iloc[:,3] # Passing Attempts
y = data.iloc[:,2] # Total Fantasy Points
m = len(y) # number of training example

#Plot Data
plt.scatter(X, y)
plt.xlabel('Passing Attempts')
plt.ylabel('Total Fantasy Points')
plt.show()

X = data.iloc[:,7] # Interceptions
y = data.iloc[:,2] # Total Fantasy Points
m = len(y) # number of training example

#Plot Data
plt.scatter(X, y)
plt.xlabel('Interceptions')
plt.ylabel('Total Fantasy Points')
plt.show()

X = data.iloc[:,8] # Interceptions
y = data.iloc[:,2] # Total Fantasy Points
m = len(y) # number of training example

#Plot Data
plt.scatter(X, y)
plt.xlabel('2pt Passing Conversions')
plt.ylabel('Total Fantasy Points')
plt.show()

X = data.iloc[:,9] # Interceptions
y = data.iloc[:,2] # Total Fantasy Points
m = len(y) # number of training example

#Plot Data
plt.scatter(X, y)
plt.xlabel('Rushing Attempts')
plt.ylabel('Total Fantasy Points')
plt.show()

X = data.iloc[:,10] # Interceptions
y = data.iloc[:,2] # Total Fantasy Points
m = len(y) # number of training example

#Plot Data
plt.scatter(X, y)
plt.xlabel('Rushing Yards')
plt.ylabel('Total Fantasy Points')
plt.show()

X = data.iloc[:,11] # Interceptions
y = data.iloc[:,2] # Total Fantasy Points
m = len(y) # number of training example

#Plot Data
plt.scatter(X, y)
plt.xlabel('Rushing TDs')
plt.ylabel('Total Fantasy Points')
plt.show()

X = data.iloc[:,12] # Interceptions
y = data.iloc[:,2] # Total Fantasy Points
m = len(y) # number of training example

#Plot Data
plt.scatter(X, y)
plt.xlabel('Rushing 2pt Conversions')
plt.ylabel('Total Fantasy Points')
plt.show()

X = data.iloc[:,13] # Interceptions
y = data.iloc[:,2] # Total Fantasy Points
m = len(y) # number of training example

#Plot Data
plt.scatter(X, y)
plt.xlabel('Receptions')
plt.ylabel('Total Fantasy Points')
plt.show()

X = data.iloc[:,14] # Interceptions
y = data.iloc[:,2] # Total Fantasy Points
m = len(y) # number of training example

#Plot Data
plt.scatter(X, y)
plt.xlabel('Receiving Yards')
plt.ylabel('Total Fantasy Points')
plt.show()

X = data.iloc[:,15] # Interceptions
y = data.iloc[:,2] # Total Fantasy Points
m = len(y) # number of training example

#Plot Data
plt.scatter(X, y)
plt.xlabel('Receiving TDs')
plt.ylabel('Total Fantasy Points')
plt.show()

X = data.iloc[:,16] # Interceptions
y = data.iloc[:,2] # Total Fantasy Points
m = len(y) # number of training example

#Plot Data
plt.scatter(X, y)
plt.xlabel('Receiving 2pt Conversions')
plt.ylabel('Total Fantasy Points')
plt.show()

X = data.iloc[:,17] # Interceptions
y = data.iloc[:,2] # Total Fantasy Points
m = len(y) # number of training example

#Plot Data
plt.scatter(X, y)
plt.xlabel('Fumbles Lost')
plt.ylabel('Total Fantasy Points')
plt.show()

X = data.iloc[:,18] # Interceptions
y = data.iloc[:,2] # Total Fantasy Points
m = len(y) # number of training example

#Plot Data
plt.scatter(X, y)
plt.xlabel('Fumble TDs')
plt.ylabel('Total Fantasy Points')
plt.show()

# # Helper Functions

X = data.iloc[:, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]  # y is the same as before # read first two columns into X
y = np.array(data.iloc[:,2])[:,np.newaxis] # y is the same as before

X_scaled = (X - X.mean())/X.std()
print(X_scaled)

ones = np.ones((m,1))
X_scaled = np.hstack((ones, X_scaled))
alpha = 0.001
num_iters = 1000
theta = np.zeros((17,1))

def computeCost(X, y, theta):
    error = np.dot(X, theta) - y # f(x) - y
    return np.sum(np.power(error, 2)) / (2*m)
J = computeCost(X_scaled, y, theta)

def gradientDescent(X, y, theta, alpha, iterations):
    J_hist = [] # to keep track of the history of J
    theta_hist = [] # to keep track of the history of theta
    for _ in range(iterations):
        error = np.dot(X, theta) - y # f(x) -y
        error_x_input = np.dot(X.T, error) # sum(error*xj)
        theta = theta - (alpha/m) * error_x_input # theta = theta - alpha * d J/ d theta
        J_hist.append(computeCost(X, y, theta))
        theta_hist.append(theta)
    return theta_hist, J_hist
theta_hist, J_hist = gradientDescent(X_scaled, y, theta, alpha, num_iters)

# # Linear Regression With Multiple Variables

J = computeCost(X_scaled, y, theta)

theta_hist, J_hist = gradientDescent(X_scaled, y, theta, alpha, num_iters)

plt.scatter(range(len(J_hist)),J_hist)
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.savefig('graph.png')
plt.show()

# # Linear Regression Predictions

X_test = test_data.iloc[:, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
print(X_test)
X_test_scaled = (X_test - X.mean())/X.std()
print(X_test_scaled)
ones = np.ones((len(X_test), 1))
X_test_scaled = np.hstack((ones, X_test_scaled))

predictions = np.dot(X_test_scaled, theta_hist[-1])
print(predictions)
actual_values = test_data.iloc[:, 2]  # Assuming the target variable is in the third column
print(actual_values)

r2 = r2_score(actual_values, predictions)

print(f"R-squared: {r2:.2%}")

# Assuming predictions and actual_values are 1D arrays
mse = mean_squared_error(actual_values, predictions)

print(f"Mean Squared Error: {mse:.2f}")