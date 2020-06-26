import matplotlib.pyplot as plt
import numpy as np
import cv2
# import os

from glob import glob
from keras import preprocessing

from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam

import time


class_name = ['APPLE', 'BANANA', 'PAPAYA', 'KIWI']

'''# get the reference to the webcam
camera = cv2.VideoCapture(0)
camera_height = 500
raw_frames_type_1 = []
raw_frames_type_2 = []
raw_frames_type_3 = []
raw_frames_type_4 = []

while(True):
    # read a new frame
    _, frame = camera.read()

    # flip the frame
    frame = cv2.flip(frame, 1)

    # rescaling camera output
    aspect = frame.shape[1] / float(frame.shape[0])
    res = int(aspect * camera_height)   #landscape orientation - wide image
    frame = cv2.resize(frame, (res, camera_height))

    # add rectangle
    cv2.rectangle(frame, (300, 75), (650, 425), (0, 255, 0), 2)

    #show the frame
    cv2.imshow('Capturing frames', frame)

    key = cv2.waitKey(1)

    #quit camera if 'q' is pressed
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('1'):
        #save the frame
        raw_frames_type_1.append(frame)
        print('1 key pressed - saved TYPE_1 frame')
    elif key & 0xFF == ord('2'):
        #save the frame
        raw_frames_type_2.append(frame)
        print('2 key pressed - saved TYPE_2 frame')
    elif key & 0xFF == ord('3'):
        #save the frame
        raw_frames_type_3.append(frame)
        print('3 key pressed - saved TYPE_3 frame')
    elif key & 0xFF == ord('4'):
        #save the frame
        raw_frames_type_4.append(frame)
        print('4 key pressed - saved TYPE_4 frame')

camera.release()
cv2.destroyAllWindows()

save_width = 399
save_height = 399

for i, frame in enumerate(raw_frames_type_1):
    roi = frame[75+2:425-2, 300+2:650-2]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (save_width, save_height))
    cv2.imwrite('images_type_1/{}.png'.format(i), cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

for i, frame in enumerate(raw_frames_type_2):
    roi = frame[75+2:425-2, 300+2:650-2]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (save_width, save_height))
    cv2.imwrite('images_type_2/{}.png'.format(i), cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

for i, frame in enumerate(raw_frames_type_3):
    roi = frame[75+2:425-2, 300+2:650-2]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (save_width, save_height))
    cv2.imwrite('images_type_3/{}.png'.format(i), cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

for i, frame in enumerate(raw_frames_type_4):
    roi = frame[75+2:425-2, 300+2:650-2]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (save_width, save_height))
    cv2.imwrite('images_type_4/{}.png'.format(i), cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))'''

# IMAGES ARE SAVED


# now let's load again the images
width = 96
height = 96


# load images type_1
images_type_1 = []
for image_path in glob('images_type_1/*.*'):
    image = preprocessing.image.load_img(image_path,
                                         target_size=(width, height))
    x = preprocessing.image.img_to_array(image)
    images_type_1.append(x)


# load images type_2
images_type_2 = []
for image_path in glob('images_type_2/*.*'):
    image = preprocessing.image.load_img(image_path,
                                         target_size=(width, height))
    x = preprocessing.image.img_to_array(image)
    images_type_2.append(x)


# load images type_3
images_type_3 = []
for image_path in glob('images_type_3/*.*'):
    image = preprocessing.image.load_img(image_path,
                                         target_size=(width, height))
    x = preprocessing.image.img_to_array(image)
    images_type_3.append(x)


# load images type_4
images_type_4 = []
for image_path in glob('images_type_4/*.*'):
    image = preprocessing.image.load_img(image_path,
                                         target_size=(width, height))
    x = preprocessing.image.img_to_array(image)
    images_type_4.append(x)



# displaying images of type 1
'''plt.figure(figsize=(12, 8))

for i, x in enumerate(images_type_1[:5]):
    plt.subplot(1, 5, i+1)
    image = preprocessing.image.array_to_img(x)
    plt.imshow(image)

    plt.axis('off')
    plt.title('{} image'.format(class_name[0]))

# show the plot
#plt.show()


# displaying images of type 2
plt.figure(figsize=(12, 8))

for i, x in enumerate(images_type_2[:5]):
    plt.subplot(1, 5, i+1)
    image = preprocessing.image.array_to_img(x)
    plt.imshow(image)

    plt.axis('off')
    plt.title('{} image'.format(class_name[1]))

# show the plot
plt.show()'''


# PREPARE IMAGES AS TENSORS
x_type_1 = np.array(images_type_1)
x_type_2 = np.array(images_type_2)
x_type_3 = np.array(images_type_3)
x_type_4 = np.array(images_type_4)

'''print(x_type_1.shape)
print(x_type_2.shape)
print(x_type_3.shape)
print(x_type_4.shape)'''

x = np.concatenate((x_type_1, x_type_2), axis=0)

if len(x_type_3):
    x = np.concatenate((x, x_type_3), axis=0)

if len(x_type_4):
    x = np.concatenate((x, x_type_4), axis=0)


# scale the data to [0, 1] values
x = x / 255
#print(x.shape)


#we need to create a y_train
#so we'll use 0 to indicate TYPE_1 and 1 to indicate TYPE_2

y_type_1 = [0 for item in enumerate(x_type_1)]
y_type_2 = [1 for item in enumerate(x_type_2)]
y_type_3 = [2 for item in enumerate(x_type_3)]
y_type_4 = [3 for item in enumerate(x_type_4)]


y = np.concatenate((y_type_1, y_type_2), axis=0)

if len(y_type_3):
    y = np.concatenate((y, y_type_3), axis=0)

if len(y_type_4):
    y = np.concatenate((y, y_type_4), axis=0)

y = to_categorical(y, num_classes=len(class_name))

print(y.shape)
print(y)
#print(x)



#==================================================================
# convolutional network configuration
# let's create a deep network which will learn our emotions and
# then will try to predict them

# default parameters
conv_1 = 16
conv_1_drop = 0.2
conv_2 = 32
conv_2_drop = 0.2
dense_1_n = 1024
dense_1_drop = 0.2
dense_2_n = 512
dense_2_drop = 0.2
lr = 0.001

epochs = 10
batch_size = 32
colors_channels = 3

def build_model(conv_1_drop=conv_1_drop, conv_2_drop=conv_2_drop,
                dense_1_n=dense_1_n, dense_1_drop=dense_1_drop,
                dense_2_n=dense_2_n, dense_2_drop=dense_2_drop,
                lr=lr):
    model = Sequential()

    model.add(Convolution2D(conv_1, (3, 3),
                            input_shape=(width, height, colors_channels),
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conv_1_drop))

    model.add(Convolution2D(conv_2, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conv_2_drop))

    model.add(Flatten())

    model.add(Dense(dense_1_n, activation='relu'))
    model.add(Dropout(dense_1_drop))

    model.add(Dense(dense_2_n, activation='relu'))
    model.add(Dropout(dense_2_drop))

    model.add(Dense(len(class_name), activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr),
                  metrics=['accuracy'])
    return model


#model with base parameters
model = build_model()

model.summary()

model.fit(x, y, epochs=epochs)


# get the reference to the webcame
camera = cv2.VideoCapture(0)
camera_height = 500

while(True):
    #read a new frame
    _, frame = camera.read()

    #flip the frame
    frame = cv2.flip(frame, 1)

    #rescaling camera output
    aspect = frame.shape[1] / float(frame.shape[0])
    res = int(aspect * camera_height)   #landscape orientation - wide image
    frame = cv2.resize(frame, (res, camera_height))

    #add rectangle
    cv2.rectangle(frame, (300, 75), (650, 425), (240, 100, 0), 2)

    #get ROI
    roi = frame[75+2:425-2, 300+2:650-2]

    #parse BRG TO RGB
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    #resize
    roi = cv2.resize(roi, (width, height))

    #predict!!
    roi_X = np.expand_dims(roi, axis=0)
    predictions = model.predict(roi_X)

    type_1_pred, type_2_pred, type_3_pred, type_4_pred = predictions[0]

    #add text 1
    type_1_text = '{}: {}%'.format(class_name[0], int(type_1_pred*100))
    cv2.putText(frame, type_1_text, (70, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)

    # add text 2
    type_2_text = '{}: {}%'.format(class_name[1], int(type_2_pred * 100))
    cv2.putText(frame, type_2_text, (70, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)

    # add text 3
    type_3_text = '{}: {}%'.format(class_name[2], int(type_3_pred * 100))
    cv2.putText(frame, type_3_text, (70, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)

    # add text 4
    type_4_text = '{}: {}%'.format(class_name[3], int(type_4_pred * 100))
    cv2.putText(frame, type_4_text, (70, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)


    # show the frame
    cv2.imshow('Test Out', frame)
    key = cv2.waitKey(1)

    #quit camera if 'q' is pressed
    if key & 0xFF == ord('q'):
        break


camera.release()
cv2.destroyAllWindows()


