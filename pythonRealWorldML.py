import numpy as np
from sklearn import preprocessing


data = np.array([[3, -1, 2, -5.4], [0, 4, -0.3, 2.1], [1, 3.3, -1.9, -4.3]])

data_standardized = preprocessing.scale(data)
print('Mean = ', data_standardized.mean(axis=0))
print('std deviation = ', data_standardized.std(axis=0))

data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled = data_scaler.fit_transform(data)
print('Min Max scaled data =\n', data_scaled)

data_normalized = preprocessing.normalize(data, norm='l1')
print('L1 normalized data=\n', data_normalized)

data_binarized = preprocessing.Binarizer(threshold=1.4).transform(data)
print('Binarized data=\n', data_binarized)

#ONE HOT ENCODING
encoder = preprocessing.OneHotEncoder()
encoder.fit([[0, 2, 1, 12], [1, 3, 5, 3], [2, 3, 2, 12], [1, 2, 4, 3]])
encoded_vector = encoder.transform([[2, 3, 5, 3]]).toarray()
print('Encoded Vector=', encoded_vector)

#LABEL ENCODING
label_encoder = preprocessing.LabelEncoder()
input_classes = ['audi', 'ford', 'audi', 'toyota', 'ford', 'bmw']

label_encoder.fit(input_classes)
print('class mapping:')
for i, item in enumerate(label_encoder.classes_):
    print(item, '-->', i)


labels = ['toyota', 'ford', 'audi']
encoded_labels = label_encoder.transform(labels)
print('Labels =', labels)
print('Encoded labels =', list(encoded_labels))

encoded_labels = [2, 1, 0, 3, 1]
decoded_labels = label_encoder.inverse_transform(encoded_labels)
print('Encoded labels =', encoded_labels)
print('Decoded labels =', list(decoded_labels))
