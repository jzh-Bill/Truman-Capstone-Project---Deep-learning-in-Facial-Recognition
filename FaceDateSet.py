import os
import numpy as np
import cv2
# Define image size
IMAGE_SIZE = 64
 
 
# Scaling to defined image size
def resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    top, bottom, left, right = 0, 0, 0, 0
    # Get image size
    h, w, _ = image.shape
    # Find the longest side of the picture
    longest_edge = max(h, w)
    # Calculate how much of the short side needs to be filled to make it equal to the long side
    if h < longest_edge:
        d = longest_edge - h
        top = d // 2
        bottom = d // 2
    elif w < longest_edge:
        d = longest_edge - w
        left = d // 2
        right = d // 2
    else:
        pass
 
    # Set the fill color
    BLACK = [0, 0, 0]
    # Fill operation for the original image
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    # Resize the image and return
    return cv2.resize(constant, (height, width))
 
images, labels = list(), list()

# Read training data
def read_path(path):
    for dir_item in os.listdir(path):
        
        # Merge into recognizable operation paths
        full_path = os.path.abspath(os.path.join(path, dir_item))
        
        # If it is a folder, the recursive call continues
        if os.path.isdir(full_path):
            read_path(full_path)
        else:
            if dir_item.endswith('.jpg'):
                # print(dir_item)
                image = cv2.imread(full_path)
                image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
                images.append(image)
                
                labels.append(path)
    # print(labels)
    return images, labels
 
 
# Read training data from specified path
def load_dataset(path, name):
    images, labels = read_path(path)
    # Since the image is calculated based on a matrix, converting it to a matrix
    images = np.array(images)
    # print(images.shape)
    labels = np.array([0 if label.endswith(name) else 1 for label in labels])
    return images, labels
 
if __name__ == '__main__':
    images, labels = load_dataset("FaceImageData", "jinbo")
    # for img in images:
    #     cv2.imshow('123', img)
    #     cv2.waitKey(0)


