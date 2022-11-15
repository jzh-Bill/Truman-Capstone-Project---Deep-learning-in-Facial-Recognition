# -*- coding: utf-8 -*-
__author__ = '翁飞龙'
import cv2
import numpy as np
import os

def path(person_name):
    path="./FaceImageDate/" + str(person_name)
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)  
        print ( '\n ' + path + 'Folder Created successfully')
    else:
        print ( '\n ' + path+ ' Folder name already exists')

    return path

def CatchPICFromVideo(path, catch_num):
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml")

    # Initiative the camera, laptop built-in camera is 0 and the other could be 1 or 2 
    # The input image has properties: 720 * 1280 * 3
    camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


    face_id = "1"
    print('\n Please remove your glasses, mask and other coverings before collecting data, please keep the light good')

    seconds_remaining = 7.0
    while True:
        ret, img = camera.read()

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f'Ready in: {str(seconds_remaining)}'[0:13], (60, 30), font, 1, (255, 0, 0), 4)

        center_coordinate = (640, 360)
        axesLength = (200, 300)
        angle = 0
        startAngle = 0
        endAngle = 360
        color = (0, 0, 255)
        thickness = 5

        img = cv2.ellipse(img, center_coordinate, 
                                axesLength,
                                angle, 
                                startAngle, 
                                endAngle, 
                                color, 
                                thickness
                            )


        cv2.imshow('image', img)
        seconds_remaining = seconds_remaining - 0.01 * 10
        k = cv2.waitKey(10)
        if seconds_remaining <= 0.0:
            break    

    print('\n Face data is being collected, please wait ...')

    count = 0

    while True:

        # Read in the image from the camera

        ret, img = camera.read()
        
        # convert the image to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # face detections
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(64, 64)
        )
        
        # Detect the eyes in base on the detected faces
        result = []
        for (x, y, w, h) in faces:
            fac_gray = gray[y: (y+h), x: (x+w)]
            eyes = eyeCascade.detectMultiScale(fac_gray, 1.3, 2)

            # change the eyes coordinates
            for (ex, ey, ew, eh) in eyes:
                result.append((x+ex, y+ey, ew, eh))

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+w), (255, 0, 0),2)
            
        for (ex, ey, ew, eh) in result:

            center_coordinate = (640, 360)
            axesLength = (240, 340)
            angle = 0
            startAngle = 0
            endAngle = 360
            color = (0, 0, 255)
            thickness = 5

            img = cv2.ellipse(img, center_coordinate, 
                                    axesLength,
                                    angle, 
                                    startAngle, 
                                    endAngle, 
                                    color, 
                                    thickness
                                )

            cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            # count how many faces were captured
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, f'Collected:{str(count)}', (x + 30, y + 30), font, 1, (255, 0, 255), 4)
            
            # save the images in a particular folder
            img_name = path + '/' + str(face_id) + '_' + str(count) + '.jpg'

            count += 1 # increase the count, counting how many photos we have recorded.

            # save the images
            cv2.imencode('.jpg',  gray[y: y + h, x: x + w])[1].tofile(img_name)

            cv2.imshow('image', img)

        # Keep the loop and 1ms for each iteration

        k = cv2.waitKey(100)

        if k == 27:   # press "Esc" to quit the program
            break
        elif count >= catch_num:
            break
    
    print("\n Human Faces Collecting Complete")
    # Turn off the camera
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    path=path("333")
    CatchPICFromVideo(path,200)