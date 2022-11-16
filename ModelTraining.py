# -*- coding: utf-8 -*-
__author__ = '翁飞龙'
import cv2 as cv
import random
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_modelvscode
from keras import backend as K
from FaceDateSet import load_dataset, resize_image, IMAGE_SIZE
import warnings
warnings.filterwarnings('ignore')
 
 
class Dataset:
    def __init__(self, path_name, person_name):
        # Traning set
        self.train_images = None
        self.train_labels = None
        # validating set
        # self.valid_images = None
        # self.valid_labels = None
        # test set
        self.test_images = None
        self.test_labels = None
        # image saving path
        self.path_name = path_name

        # current image dimension order
        self.input_shape = None
        # Current person name
        self.person_name = person_name
 
    def load(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE, img_channels=3, nb_classes=2):
        # load the image into RAM
        print("The path is:", self.path_name)
        images, labels = load_dataset(self.path_name, self.person_name)
        
        train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.3,
                                                                                random_state=random.randint(0, 10))
        
        if K.image_data_format() == 'channels_first':
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)
        else:
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)
 
            # print out the size of test set and train set
            print(train_images.shape[0], 'train samples')
            print(test_images.shape[0], 'test samples')

            # Our model uses categorical_crossentropy as the loss function, so it needs to be transformed according to the number of categories nb_classes 
            # The category labels are one-hot encoded to make them vectorized, where we have only two categories, and after transformation the label data becomes two-dimensional
            train_labels = np_utils.to_categorical(train_labels, nb_classes)
            test_labels = np_utils.to_categorical(test_labels, nb_classes)
            
            # convert the pixel into floating number
            train_images = train_images.astype('float32')
            test_images = test_images.astype('float32')

            # Normalize the image by normalizing the pixel values to the interval 0~1
            train_images /= 255.0
            test_images /= 255.0
            self.train_images = train_images
            self.test_images = test_images
            self.train_labels = train_labels
            self.test_labels = test_labels
 
 
# CNN model Class
class Model:
    def __init__(self):
        self.model = None
        self.person_name = None
        # self.predicting_models = []
 
    # 建立模型
    def build_model(self, dataset, nb_classes=2):
        # Construct an empty network model, which is a linear stacking model 
        # where each neural network layer will be added sequentially, professionally 
        # known as a sequential model or linear stacking model
       
        self.model = Sequential()

        # The following code will add the layers needed for the 
        # CNN network in order, an add is a network layer
        self.model.add(Conv2D(32, 3, 3, padding='same',
                                     input_shape=dataset.input_shape))  # 1st 2D convolution layer 
        self.model.add(Activation('relu'))  # 2nd Activation layer
 
        self.model.add(Conv2D(32, 3, 3, padding='same'))  # 3rd convolution layer
        self.model.add(Activation('relu'))  # 4th Activation layer
 
        self.model.add(MaxPool2D(pool_size=(2, 2)))  # 5th Max pooling layer
        self.model.add(Dropout(0.25))  # 6th Dropout layer
 
        self.model.add(Conv2D(64, 3, 3, padding='same'))  # 7th   2D convolution layer
        self.model.add(Activation('relu'))  # 8th  Activation layer
 
        self.model.add(Conv2D(64, 3, 3, padding='same'))  # 9  2D convolution layer
        self.model.add(Activation('relu'))  # 10th 激活函数层
 
        self.model.add(MaxPool2D(pool_size=(2, 2), padding='same'))  # 11th 池化层
        self.model.add(Dropout(0.25))  # 12th Dropout layer
 
        self.model.add(Flatten())  # 13th Flatten layer
        self.model.add(Dense(512))  # 14th Dense layer
        self.model.add(Activation('relu'))  # 15th Activation layer
        self.model.add(Dropout(0.5))  # 16th Dropout layer
        self.model.add(Dense(nb_classes))  # 17th Dense layer
        self.model.add(Activation('softmax'))  # 18th classification output the result

        # the summary of the model
        self.model.summary()
 
    # Model training
    def train(self, dataset, batch_size=20, nb_epoch=64, data_augmentation=False):
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)  # An optimizer with SGD+momentum is used for training, and an optimizer object is first generated
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])  #Complete the actual model configuration work


 
        # Without using data boosting, so-called boosting is the creation of 
        # new from the training data we provide using methods such as rotation, 
        # flipping, adding noise, etc.
        # Training data, consciously increase the size of 
        # training data and increase the amount of model training
        if data_augmentation:
            self.model.fit(dataset.train_images,
                           dataset.train_labels,
                           batch_size=batch_size,
                           epochs=nb_epoch,
                           validation_data=(dataset.test_images, dataset.test_labels),
                           shuffle=True)
        # use real-time data augmentation
        else:
            print("Data Augmentation has been used")

            # Define a data generator for data lifting, which returns a generator object 
            # datagen, which generates a set of data (sequential generation) every time 
            # it is called, saving memory, which is actually python's data generator
            datagen = ImageDataGenerator(
                featurewise_center=False,  # whether to decenter the input data (with a mean of 0).
                samplewise_center=False,  # Whether to make the mean value of each sample of the input data 0
                featurewise_std_normalization=False,  # Whether the data is normalized (input data divided by the standard deviation of the data set)
                samplewise_std_normalization=False,  # Whether to divide each sample data by its own standard deviation
                zca_whitening=False,  # Whether to apply ZCA whitening to the input data
                rotation_range=20,  # The angle of random rotation of the picture during data lifting (range 0 to 180)
                width_shift_range=0.2,  # The magnitude of the horizontal offset of the picture when 
                                        #the data is lifted (in units of the percentage of the picture width, a floating point number between 0 and 1)
                height_shift_range=0.2,  # Same as above, except here it is vertical
                horizontal_flip=True,  # Whether to perform random horizontal flipping
                vertical_flip=False)  # Whether to perform random vertical flipping
 
            # Calculate the number of the entire training sample set for eigenvalue normalization, ZCA whitening, etc.
            datagen.fit(dataset.train_images)
 
            # Start training the model with the generator
            self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels,
                                                  batch_size=batch_size),
                                                epochs = nb_epoch,
                                                validation_data=(dataset.test_images, dataset.test_labels))

        return 'Finished'
 
    MODEL_PATH = './Model/face.model.h5'
 
    def save_model(self, file_path=MODEL_PATH):
        self.model.save(file_path)
 
    def load_model(self, file_path=MODEL_PATH):
        self.model = load_model(file_path)
    
    def load_person_name(self, person_name):
        self.person_name =  person_name  

    def get_person_name(self):
        return self.person_name

    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose=1)
        # print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))
        print(f'{self.model.metrics_names[1]}:{score[1] * 100}%')
 

    # Face Recognition
    def face_predict(self, image):
        # Still determine the dimension order based on the back-end system
        #if K.image_dim_ordering() == 'th' 
        if K.image_data_format() == 'channels_first'and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            image = resize_image(image)  # The size must be consistent with the training set all should be IMAGE_SIZE x IMAGE_SIZE
            image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))  # Unlike the model training, this time it only predicts for 1 image
        #elif K.image_dim_ordering() == 'tf' 
        elif K.image_data_format() == 'channels_last'and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            image = resize_image(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
 
        # Normalized the floating number
        image = image.astype('float32')
        image /= 255.0

        result = self.model.predict(image)

        print("This is the model of: ", self.get_person_name())
        print("The data type of results:", result)

        # return the final result
        if result[0][0] > result[0][1]:
            return 'T', result[0][0]
        else:
            return 'F', result[0][1]
 
if __name__ == '__main__':
    # print("The program has reached in here!!!-----")
    dataset = Dataset('FaceImageDate', 'mingyang')
    dataset.load()
 
    # train the model
    model = Model()
    model.build_model(dataset)
    
    # test the model
    model.train(dataset)
    
    model.save_model(file_path='./Model/mingyang.face.model.h5')

    # # # evaluate the model
    # model = Model()
    # model.load_model(file_path='./Model/mingyang.face.model.h5')

    # model_2 = Model()
    # model_2.load_model(file_path='./Model/mingyang.face.model.h5')
    # # # model_3 = Model()
    # # # model_3.load_model(file_path='./Model/jinbo.face.model.h5')

    # image2 = cv.imread("1_79.jpg")
    # result = model_2.face_predict(image2)
    # print(result)
    # # # # result = model_2.face_predict(image2)
    # # # # result = model_3.face_predict(image2)

    # # # print("The final result is:", result)
    # # # model.evaluate(dataset)
    # # del dataset