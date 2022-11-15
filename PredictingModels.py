import cv2 as cv

import os

from ModelTraining import *
warnings.filterwarnings('ignore')

class PredictingModels:
    def __init__(self):
        self.predicting_models = []
        self.load_models() # load up all the models that already exists in the folder

    def load_models(self, path = "./Model"):
        # load up all the models in the model directory 
        for item in os.listdir(path):
            full_path = os.path.abspath(os.path.join(path, item))
            model = Model()
            # Initial a temporary model
            model.load_model(full_path)

            # Using partial of string to name the model with person's name.
            person_name = item[:-14]
            model.load_person_name(person_name)
            print("The person is:", person_name)
            print("The  model name is:", item)
            self.predicting_models.append(model)

    def final_recognition_check(self, image):
        print("One final check starts")
        name_results_pairs = {} # Initial the dictionary (key-value pair)

        for model in self.predicting_models:
            result, possibility = model.face_predict(image)
            result_entry = [result, possibility]
            name_results_pairs[model.get_person_name()]=result_entry

        # print("The maps contains the value and pairs:", name_results_pairs)

        face_found = False
        for key in name_results_pairs.keys():
            if name_results_pairs[key][0] == 'T':
                face_found = True # When at least one model recognized the face, set the flag to true.

        if face_found == True: # In case multiple models recognizes the face, find the one with highest possibility
            name_results_pairs = dict(sorted(name_results_pairs.items(), key=lambda item: item[1][1], reverse=True))

            for key in list(name_results_pairs.keys()):
                if name_results_pairs[key][0] == 'F':
                    del name_results_pairs[key]

            # print("The maps contains the value and pairs after deleting:", name_results_pairs)
            highest_possible_entry_name = list(name_results_pairs.items())[0][0] # get the 1st element's name of the key-value pair list
            highest_possibility = list(name_results_pairs.items())[0][1][1] # get the 1st element's value of the key-value pair list
            return highest_possible_entry_name, highest_possibility
        else:
            return 'Other', 1

if __name__ == '__main__':
    models = PredictingModels()
    image2 = cv.imread("1_182.jpg")
    result = models.final_recognition_check(image2)
    print(result)
