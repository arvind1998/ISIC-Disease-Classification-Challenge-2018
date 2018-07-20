# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
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
	# Initialising the CNN
    classifier = Sequential()

	# Step 1 - Convolution
    classifier.add(Conv2D(96, (11, 11), input_shape = (227, 227, 3), strides=4, activation = 'relu'))

	# Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size = (3, 3), strides=2))
    classifier.add(BatchNormalization())

	# Adding a second convolutional layer
    classifier.add(Conv2D(256, (5, 5), strides=1, padding='valid', activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (3, 3), strides=2))
    classifier.add(BatchNormalization())

    classifier.add(Conv2D(384, (3, 3), strides=1, padding='valid', activation = 'relu'))
    #classifier.add(MaxPooling2D(pool_size = (2, 2)))
    #classifier.add(BatchNormalization(axis=1, momentum=0.7, epsilon=0.001))
    classifier.add(Conv2D(384, (3, 3), strides=1, padding='valid', activation = 'relu'))
    classifier.add(Conv2D(256, (3, 3), strides=1, padding='valid', activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (3, 3), strides=2))
    classifier.add(Dropout(rate=0.5))

	# Step 3 - Flattening
	classifier.add(Flatten())
	print('flatenning done')

	# Step 4 - Full connection
	classifier.add(Dense(units = 4096, activation = 'relu'))
	classifier.add(Dropout(rate=0.5))
    	classifier.add(Dense(units = 4096, activation = 'relu'))

	classifier.add(Dense(units = 7, activation = 'softmax'))

	#classifier.add(Dropout(p=0.2))
	print('full connectiong done')
	# Compiling the CNN
	classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    classifier.summary()
	# Part 2 - Fitting the CNN to the images
    
	from keras.preprocessing.image import ImageDataGenerator

	train_datagen = ImageDataGenerator(rescale = 1./255,
	                                   shear_range = 0.2,
	                                   zoom_range = 0.2,
	                                   horizontal_flip = True)

	test_datagen = ImageDataGenerator(rescale = 1./255)

	training_set = train_datagen.flow_from_directory('Dataset/Training Set',
	                                                 target_size = (227, 227),
	                                                 batch_size = 32,
	                                                 class_mode = 'categorical')

	test_set = test_datagen.flow_from_directory('Dataset/Test Set',
	                                            target_size = (227, 227),
	                                            batch_size = 32,
	                                            class_mode = 'categorical')
	                                            

    #History=History()
	classifier.fit_generator(training_set,
	                         steps_per_epoch = 8633,
	                         epochs = 50,
	                         validation_data = test_set,
	                         validation_steps = 1380, shuffle=True)
    print(history.history.keys())
    classifier.save('./model_3.h5')
    
	# Part 3 - Making new predictions

	import numpy as np
	from keras.preprocessing import image
	test_image = image.load_img('Dataset/Single Prediction/melanoma.jpg', target_size = (224, 224))
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis = 0)
	result = classifier.predict(test_image)
	print(result)

	    
	import numpy as np
	from keras.preprocessing import image
	test_image = image.load_img('Dataset/Single Prediction/basal_cell.jpg', target_size = (224, 224))
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis = 0)
	result = classifier.predict(test_image)
	print(result)
    
    
	from keras.preprocessing import image
	test_image_1 = image.load_img('Dataset/Single Prediction/dermatofibroma1.jpg', target_size = (224, 224))
	test_image_1 = image.img_to_array(test_image_1)
	test_image_1 = np.expand_dims(test_image_1, axis = 0)
	a = classifier.predict(test_image_1)
	print(a)
    
    scores = classifier.evaluate(training_set, test_set)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    classifier.fit
    
    classifier.fit(x=training_set, y=test_set, batch_size=32, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
                classifier.evaluate_generator(training_set, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False)
                classifier.evaluate_generator(test_set, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False)
        print(classifier.metrics_names[1][3])

    model=load_model('./model_2.h5')
    classifier.summary()
    
    import matplotlib.pyplot as plt
    plt.plot(History.history['acc'])
    plt.plot(History.history['val_acc'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train', 'test'], loc='upper_left')
    plt.show()