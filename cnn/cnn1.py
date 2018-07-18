"""
 Convolutional Neural Networks for Diabetic Retinopathy Detection

 Will only run on floydhub.com on my diabret project

 ConvNet 1 Results:
 loss: 4.1806 - acc: 0.7406 - val_loss: 4.2411 - val_acc: 0.7369

 Log file: cnn1.txt

 Time taken: 41:28 Mins

 GPU: NVidia Tesla K80 12GB

 Total Data: 35125 images -> 5002 test, rest train

 Attempts: 11 :)

 Date: April 3, 2018
"""

from keras.layers import (Conv2D, Dense, Dropout, Flatten, MaxPooling2D)
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

classifier = Sequential()

# Architecture
classifier.add(
    Conv2D(32, (5, 5), input_shape=(512, 512, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(64, (5, 5), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Conv2D(128, (5, 5), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Conv2D(256, (5, 5), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Flatten())
classifier.add(Dense(256, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(5, activation='softmax'))

classifier.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_set = train_datagen.flow_from_directory(
    '/data/train',
    target_size=(512, 512),
    color_mode="grayscale",
    batch_size=32,
    class_mode='categorical')
test_set = test_datagen.flow_from_directory(
    '/data/test',
    target_size=(512, 512),
    color_mode="grayscale",
    batch_size=32,
    class_mode='categorical')

classifier.fit_generator(
    train_set,
    steps_per_epoch=100,
    epochs=20,
    validation_data=test_set,
    validation_steps=50)
