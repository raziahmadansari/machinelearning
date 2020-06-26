import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


housing_data = datasets.load_boston()
#print(housing_data.data)
#print(housing_data.target)

x, y = shuffle(housing_data.data, housing_data.target, random_state=7)

num_training = int(0.8 * len(x))
x_train, y_train = x[:num_training], y[:num_training]
x_test, y_test = x[num_training:], y[num_training:]

#print(x_test)
#print(y_test)

dt_regressor = DecisionTreeRegressor(max_depth=4)
dt_regressor.fit(x_train, y_train)

ab_regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                                 n_estimators=400, random_state=7)
ab_regressor.fit(x_train, y_train)


#LET'S EVALUATE THE PERFORMANCE
y_pred_dt = dt_regressor.predict(x_test)
mse = mean_squared_error(y_test, y_pred_dt)
evs = explained_variance_score(y_test, y_pred_dt)
print('### Decision Tree Performance ###')
print('Mean squared error:', round(mse, 2))
print('Explained variance score:', round(evs, 2))

y_pred_ab = ab_regressor.predict(x_test)
mse = mean_squared_error(y_test, y_pred_ab)
evs = explained_variance_score(y_test, y_pred_ab)
print('### AdaBoost performance ###')
print('Mean squared error:', round(mse, 2))
print('Explained variance score:', round(evs, 2))



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


#LET'S PLOT RELATIVE IMPORTANCE OF THE FEATURES
print(dt_regressor.feature_importances_)
plot_feature_importance(dt_regressor.feature_importances_, 'Decision Tree regressor',
                        housing_data.feature_names)
plot_feature_importance(ab_regressor.feature_importances_, 'AdaBoost regressor',
                        housing_data.feature_names)

