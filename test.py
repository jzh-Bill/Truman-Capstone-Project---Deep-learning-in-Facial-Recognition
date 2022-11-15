import cv2 as cv
import random
import numpy as np
import os
from  ModelTraining import Model
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
from FaceDateSet import load_dataset, resize_image, IMAGE_SIZE

predicting_models = []

def load_models(path = "./Model"):
    # load up all the models in the model directory 
    for item in os.listdir(path):
        full_path = os.path.abspath(os.path.join(path, item))
        model = Model()
        # Initial a temporary model
        model.load_model(full_path)

        #Using partial of string to name the model with person's name.
        person_name = item[:-14]
        model.load_person_name(person_name)
        print("The name is:", person_name)
        print("The item is:", item)
        predicting_models.append(model)

def final_recognition_check(name_results_pairs):
    face_found = False
    for entry in name_results_pairs:
        if entry[0] == 'T':
            face_found = True # When at least one model recognized the face, set the flag to true.

    if face_found == True: # In case multiple models recognizes the face, find the one with highest possibility
        name_results_pairs = dict(sorted(name_results_pairs.items(), key=lambda item: item[1][1], reverse=True))

        for key in list(name_results_pairs.keys()):
            if name_results_pairs[key][0] == 'F':
                del name_results_pairs[key]

        highest_possible_entry_name = list(name_results_pairs.items())[0][0] # get the 1st element's name of the key-value pair list
        highest_possibility = list(name_results_pairs.items())[0][1][1] # get the 1st element's value of the key-value pair list
        return highest_possible_entry_name, highest_possibility
    else:
        return 'Other', 1



load_models()
print(predicting_models)

name_results_pairs = {}
# name_results_pairs['jinbo'] = ['T', 0.9]
# print(name_results_pairs)

for model in predicting_models:
    image = cv.imread("333_65.jpg") # test the model prediction
    result, possibility = model.face_predict(image)
    result_entry = [result, possibility]
    name_results_pairs[model.get_person_name()]=result_entry

name_results_pairs["SB"]=['T', 1.0]
name_results_pairs["T2"]=['T', 0.4]
name_results_pairs["T3"]=['F', 0.98]


print(final_recognition_check(name_results_pairs))








