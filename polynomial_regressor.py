from sklearn import linear_model
import numpy as np
from sklearn import metrics as sm
from sklearn.preprocessing import PolynomialFeatures


filename = 'data_multivar.txt'
X = []
Y = []
with open(filename, 'r') as f:
    for line in f.readlines():
        data = [float(i) for i in line.split(',')]
        xt, yt = data[:-1], data[-1]
        X.append(xt)
        Y.append(yt)

num_training = int(0.8 * len(X))
num_test = len(X) - num_training

#TRAINING DATA
X_train = np.array(X[:num_training])
Y_train = np.array(Y[:num_training])

#TEST DATA
X_test = np.array(X[num_training:])
Y_test = np.array(Y[num_training:])

linear_regressor = linear_model.LinearRegression()
polynomial = PolynomialFeatures(degree=3)

linear_regressor.fit(X_train, Y_train)
X_train_transformed = polynomial.fit_transform(X_train)

datapoint = [[0.39, 2.78, 7.11]]
poly_datapoint = polynomial.fit_transform(datapoint)

poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, Y_train)

print('Linear regression:', linear_regressor.predict(datapoint))
print('Polynomial regressor:', poly_linear_model.predict(poly_datapoint))
