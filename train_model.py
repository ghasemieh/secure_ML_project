import cv2
import matplotlib.image as mpimg
import numpy as np
import os
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 160, 320, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def preprocessing(data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
    choice = np.random.choice(3)
    if choice == 0:
        image = mpimg.imread(os.path.join(data_dir, left))
        steering_angle = steering_angle + 0.2
    elif choice == 1:
        image = mpimg.imread(os.path.join(data_dir, right))
        steering_angle = steering_angle - 0.2
    else:
        image = mpimg.imread(os.path.join(data_dir, center)), steering_angle

    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle

    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))

    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    image = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:, :, 2] = hsv[:, :, 2] * ratio
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return image, steering_angle


def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            if index == 1:
                index = 2
            center, left, right = image_paths[index]
            steering_angle = float(steering_angles[index])
            if is_training and np.random.rand() < 0.6:
                image, steering_angle = preprocessing(data_dir, center, left, right, steering_angle)
            else:
                image = mpimg.imread(os.path.join(data_dir, center))
            image = image[60:-25, :, :]
            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
            images[i] = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers

np.random.seed(0)
data_df = pd.read_csv(os.path.join(os.getcwd(), "model/", 'driving_log.csv'),
                      names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
X = data_df[['center', 'left', 'right']]
X = X.drop(X.index[0])
X = X.values
y = data_df['steering']
y = y.drop(y.index[0])
y = y.values
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=64, random_state=0)
model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))
model.add(Conv2D(24, 5, activation='elu', strides=(2, 2)))
model.add(Conv2D(36, 5, activation='elu', strides=(2, 2)))
model.add(Conv2D(48, 5, activation='elu', strides=(2, 2)))
model.add(Conv2D(64, 3, activation='elu', strides=(1, 1)))
model.add(Conv2D(64, 3, activation='elu', strides=(1, 1)))
model.add(Dropout(0.02))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))
print(model.summary())
checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))
model.fit(batch_generator("model/", X_train, y_train, 64, True), steps_per_epoch=64, epochs=200,
          max_queue_size=1, validation_data=batch_generator("model/", X_valid, y_valid, 64, False),
          validation_steps=len(X_valid), callbacks=[checkpoint], verbose=1)
