import csv
import cv2
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

lines = []
with open('data/driving_log.csv') as csvfile:
    next(csvfile)
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

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

# Split the data set into training and validation
train_set, valdation_set = train_test_split(data_set, test_size=0.2)

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

batch_size = 128
train_gen = batch_gen(train_set, batch_size)
valid_gen = batch_gen(valdation_set, batch_size)
train_steps = len(train_set)
valid_steps = len(valdation_set)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

# TODO: Try a different architecture
model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Convolution2D(6,5,5,activation='elu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation='elu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_gen, samples_per_epoch=train_steps, validation_data=valid_gen, nb_val_samples=valid_steps, nb_epoch=5)

model.save('model.h5')
sys.exit()
