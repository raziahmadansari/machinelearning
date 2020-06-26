from sklearn import linear_model
import matplotlib.pyplot as plt
import sklearn.metrics as sm
import numpy as np


filename = 'data_singlevar.txt'
X = []
Y = []
with open(filename, 'r') as f:
    for line in f.readlines():
        xt, yt = [float(i) for i in line.split(',')]
        X.append(xt)
        Y.append(yt)

# print(X)
# print(Y)
# DATA HAS BEEN LOADED

num_training = int(0.8 * len(X))
num_test = len(X) - num_training

#TRAINING DATA
X_train = np.array(X[:num_training]).reshape((num_training, 1))
Y_train = np.array(Y[:num_training])

#TEST DATA
X_test = np.array(X[num_training:]).reshape((num_test, 1))
Y_test = np.array(Y[num_training:])


#CREATE LINEAR REGRESSION OBJECT
linear_regressor = linear_model.LinearRegression()

#TRAIN THE MODEL USING THE TRAINING SETS
linear_regressor.fit(X_train, Y_train)

y_train_pred = linear_regressor.predict(X_train)
plt.figure()
plt.scatter(X_train, Y_train, color='green')
plt.plot(X_train, y_train_pred, color='black', linewidth=4)
plt.title('Training data')
#plt.show()

plt.figure()
y1_test_pred = linear_regressor.predict(X_test)
plt.scatter(X_test, Y_test, color='green')
plt.plot(X_test, y1_test_pred, color='black', linewidth=4)
plt.title('Test data')
plt.show()

# ACCURACY
print('Mean absolute error =', round(sm.mean_absolute_error(Y_test, y1_test_pred), 2))
print('Mean squared error =', round(sm.mean_squared_error(Y_test, y1_test_pred), 2))
print('Median absolute error =', round(sm.median_absolute_error(Y_test, y1_test_pred), 2))
print('Explained variance score =', round(sm.explained_variance_score(Y_test, y1_test_pred), 2))
