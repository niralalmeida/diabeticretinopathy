"""
 Convolutional Neural Networks for Diabetic Retinopathy Detection

 Date: April 3, 2018
"""

from tensorflow.python.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                            MaxPooling2D)
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

classifier = Sequential()

# Architecture
classifier.add(Conv2D(32, (5, 5), input_shape=(512, 512), activation='relu'))
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
    optimizer='adam', loss='catergorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(horizontal_flip=True)
test_datagen = ImageDataGenerator()

train_set = train_datagen.flow_from_directory(
    './sample/train', target_size=(512, 512), batch_size=32)
test_set = test_datagen.flow_from_directory(
    './sample/test', target_size=(512, 512), batch_size=32)

classifier.fit_generator(
    train_set,
    steps_per_epoch=100,
    epochs=3,
    validation_data=test_set,
    validation_steps=50)