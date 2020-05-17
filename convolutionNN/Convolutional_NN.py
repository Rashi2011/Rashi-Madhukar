

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initializing the CNN Model
Classifier = Sequential()
#Convolution Layer
Classifier.add(Convolution2D(32,3,3,input_shape = (64,64,3),activation = 'relu'))

#MaxPooling2D
Classifier.add(MaxPooling2D(pool_size = (2,2)))

#Flatten
Classifier.add(Flatten())

#Dense(Full Connections in Hidden Layer)
Classifier.add(Dense(output_dim = 128, activation = 'relu'))

#OutPut Layer
Classifier.add(Dense(output_dim = 1,activation  = 'sigmoid'))

#Compile
Classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Image Preprocessing(DATA AUGMENTATION)
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)


#Fitting to the Dataset
training_set =train_datagen.flow_from_directory('DATASET/training_set',
                                                target_size = (64,64),
                                                batch_size = 32,
                                                class_mode = 'binary')
 
test_set = test_datagen.flow_from_directory('DATASET/test_set',
                                            target_size = (64,64),
                                            batch_size = 32,
                                            class_mode = 'binary')
Classifier.fit_generator(training_set,
                         samples_per_epoch = 6000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 400)