import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.models import load_model
from keras.callbacks import History

classifier=Sequential()

classifier.add(Conv2D(64, (3,3), input_shape=(224,224,3), strides=1, activation='relu'))
classifier.add(Conv2D(64, (3,3), strides=1, activation='relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2), strides=2))

classifier.add(Conv2D(128, (3,3), strides=1, activation='relu'))
classifier.add(Conv2D(128, (3,3), strides=1, activation='relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2), strides=2))

classifier.add(Conv2D(256, (3,3), strides=1, activation='relu'))
classifier.add(Conv2D(256, (3,3), strides=1, activation='relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2), strides=2))

classifier.add(Conv2D(512, (3,3), strides=1, activation='relu'))
classifier.add(Conv2D(512, (3,3), strides=1, activation='relu'))
classifier.add(Conv2D(512, (3,3), strides=1, activation='relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2), strides=2))

classifier.add(Flatten())

classifier.add(Dense(units = 4096, activation = 'relu'))
classifier.add(Dense(units = 4096, activation = 'relu'))
classifier.add(Dense(units = 7, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
	                                   shear_range = 0.2,
	                                   zoom_range = 0.2,
	                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('Dataset/Training Set',
	                                                 target_size = (224, 224),
	                                                 batch_size = 32,
	                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('Dataset/Test Set',
	                                            target_size = (224, 224),
	                                            batch_size = 32,
	                                            class_mode = 'categorical')
	                                            

    #History=History()
classifier.fit_generator(training_set,
	                         steps_per_epoch = 8633,
	                         epochs = 50,
	                         validation_data = test_set,
	                         validation_steps = 1380, shuffle=True)