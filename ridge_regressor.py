from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics as sm


filename = 'data_multivar.txt'
X = []
Y = []
with open(filename, 'r') as f:
    for line in f.readlines():
        data = [float(i) for i in line.split(',')]
        xt, yt = data[:-1], data[-1]
        X.append(xt)
        Y.append(yt)


print(X)
print(Y)


num_training = int(0.8 * len(X))
num_test = len(X) - num_training

#TRAINING DATA
#X_train = np.array(X[:num_training]).reshape((num_training, 1))
X_train = np.array(X[:num_training])
Y_train = np.array(Y[:num_training])

print(X_train)
print(Y_train)

#TEST DATA
#X_test = np.array(X[num_training:]).reshape((num_test, 1))
X_test = np.array(X[num_training:])
Y_test = np.array(Y[num_training:])

#CREATE RIDGE REGRESSOR OBJECT
ridge_regressor = linear_model.Ridge(alpha=0.01, fit_intercept=True, max_iter=10000)

#TRAIN THE MODEL USING THE TRAINING SETS
ridge_regressor.fit(X_train, Y_train)

#TEST
y_test_pred_ridge = ridge_regressor.predict(X_test)

#ERROR
print('Mean absolute error =', round(sm.mean_absolute_error(Y_test, y_test_pred_ridge), 2))
print('Mean squared error =', round(sm.mean_squared_error(Y_test, y_test_pred_ridge), 2))
print('Median absolute error =', round(sm.median_absolute_error(Y_test, y_test_pred_ridge), 2))
print('Explain variance score =', round(sm.explained_variance_score(Y_test, y_test_pred_ridge), 2))
print('R2 score =', round(sm.r2_score(Y_test, y_test_pred_ridge), 2))

x = []
for i in range(len(X_train)):
    sum = 0
    i = 0
    for a in X_train[i]:
        sum += a
        i += 1
    sum = sum / i
    x.append(sum)

x_test = []
for i in range(len(X_test)):
    sum = 0
    i = 0
    for a in X_test[i]:
        sum += a
        i += 1
    sum = sum / i
    x_test.append(sum)


plt.figure()
plt.scatter(x, Y_train, color='green')
plt.plot(x_test, y_test_pred_ridge, color='black', linewidth=4)
plt.title('Training data')
plt.show()
