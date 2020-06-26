import csv
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, explained_variance_score

def plot_feature_importance(feature_importance, title, feature_names):
    #Normalize the importance
    feature_importance = 100.0 * (feature_importance / max(feature_importance))

    #sort the index values and flip them so that they are arranged in decreasing
    #order of importance
    index_sorted = np.flipud(np.argsort(feature_importance))

    #center the location of the labels on the x-axis for display purpose only
    pos = np.arange(index_sorted.shape[0]) + 0.5

    #plot the bar graph
    plt.figure()
    plt.bar(pos, feature_importance[index_sorted], align='center')
    plt.xticks(pos, feature_names[index_sorted])
    plt.ylabel('Relative importance')
    plt.title(title)
    plt.show()


def load_dataset(filename):
    file_reader = csv.reader(open(filename, 'r'), delimiter=',')
    x, y = [], []
    for row in file_reader:
        x.append(row[2:13])
        y.append(row[-1])

    #EXTRACT FEATURE NAMES
    feature_names = np.array(x[0])

    #REMOVE THE FIRST ROW BECAUSE THEY ARE FEATURE NAMES
    return np.array(x[1:]).astype(np.float32), np.array(y[1:]).astype(np.float32), feature_names


filename = 'bike_day.csv'
x, y, feature_names = load_dataset(filename)
'''print(x)
print(y)
print(feature_names)'''
x, y = shuffle(x, y, random_state=7)

num_training = int(0.9 * len(x))
x_train, y_train = x[:num_training], y[:num_training]
x_test, y_test = x[num_training:], y[num_training:]

rf_regressor = RandomForestRegressor(n_estimators=1000, max_depth=10, min_samples_split=0.1)
rf_regressor.fit(x_train, y_train)

#LET'S EVALUAATE
y_pred = rf_regressor.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
print('### Randome forest regressor performance ###')
print('mean squared error:', round(mse, 2))
print('explained variance score:', round(evs, 2))

plot_feature_importance(rf_regressor.feature_importances_, 'Random forest regressor',
                        feature_names)
