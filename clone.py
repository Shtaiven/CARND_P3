import csv
import cv2
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from math import ceil

lines = []
lines_udacity = []
with open('data/driving_log.csv') as csvfile:
    next(csvfile)
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
with open('data_udacity/driving_log.csv') as csvfile:
    next(csvfile)
    reader = csv.reader(csvfile)
    for line in reader:
        lines_udacity.append(line)

# Create tuples of (filepath, measurement, flip?) to feed into batch_gen
data_set = []
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = 'data/IMG/' + filename
        measurement = float(line[3])
        data_set.append((current_path, measurement, True))
        data_set.append((current_path, measurement, False))
        
for line in lines_udacity:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = 'data_udacity/IMG/' + filename
        measurement = float(line[3])
        data_set.append((current_path, measurement, True))
        data_set.append((current_path, measurement, False))

# Split the data set into training and validation
train_set, valdation_set = train_test_split(data_set, test_size=0.2)

bin_angles = [0, 0, 0]
for data in train_set:
    if data[1] > 0:
        bin_angles[0] += 1
    elif data[1] < 0:
        bin_angles[2] += 1
    else:
        bin_angles[1] += 1
        
print(bin_angles)

# Generate a batch of images and measurements (tuple with image, measurement)
def batch_gen(samples, batch_size):
    n_samples = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0, n_samples, batch_size):
            images = []
            measurements = []
            batch = samples[offset:offset + batch_size]
            for sample in batch:
                # get image path from line
                image = cv2.imread(sample[0])
                # image = cv2.resize(image, (224, 112))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                measurement = sample[1]
                if sample[2]:
                    image = cv2.flip(image, 1)
                    measurement = measurement*-1.0
                # append image to images
                images.append(image)
                # append measurement to measurements
                measurements.append(measurement)
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(X_train, y_train)

batch_size = 32
train_gen = batch_gen(train_set, batch_size)
valid_gen = batch_gen(valdation_set, batch_size)
train_steps = len(train_set)
valid_steps = len(valdation_set)

train = True

if train:
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Lambda, Activation, Dropout
    from keras.layers.convolutional import Convolution2D, ZeroPadding2D
    from keras.layers.pooling import MaxPooling2D
    from keras.layers import Cropping2D

    # TODO: Try a different architecture
    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    model.add(Convolution2D(8,5,5,activation='relu',border_mode='valid',subsample=(2, 2)))
    model.add(Convolution2D(16,5,5,activation='relu',border_mode='valid',subsample=(2, 2)))
    model.add(Convolution2D(32,5,5,activation='relu',border_mode='valid',subsample=(2, 2)))
    model.add(Convolution2D(64,3,3,activation='relu',border_mode='valid'))
    model.add(Convolution2D(64,3,3,activation='relu',border_mode='valid'))
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(60))
    model.add(Dense(30))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    # Debug
    for layer in model.layers:
        print('Layer %s: %s -> %s' % (str(type(layer)), str(layer.input_shape), str(layer.output_shape)))

    model.fit_generator(train_gen, samples_per_epoch=train_steps, validation_data=valid_gen, nb_val_samples=valid_steps, nb_epoch=5, max_q_size=1, nb_worker=1)

    model.save('model.h5')
sys.exit()
