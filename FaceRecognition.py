import cv2
from ModelTraining import Model
from PredictingModels import *

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt,QObject




global global_image

def face_recognize(suspicious_image_display):

    # load up the models
    models = PredictingModels()

 
    # The color of the rectangular border framing the face
    color = (0, 255, 0)
 
    # Capture a live video stream from a specified camera
    camera = cv2.VideoCapture(1 , cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cascade_path = "haarcascade_frontalface_default.xml"

    count = 0
    suspicious_level = 0
    # Recurrent detection to recognize faces
    while True:
        ret, img = camera.read()  # Read a frame of video

        # Image graying to reduce computational complexity
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Using face recognition classifier, read in classifier
        cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        # Identify which area is the face using the classifier
        fac_gray = cascade.detectMultiScale(
                                            gray,
                                            scaleFactor=1.1,
                                            minNeighbors=5,
                                            minSize=(64, 64)
                                        )
        print(len(fac_gray)," faces has or have been detected.")
        if len(fac_gray) > 0:
            for (x, y, w, h) in fac_gray:
                # The face image is intercepted and submitted to the model to determine who this is.
                image = img[y: y + h, x: x + w]
                faceID, possibility= models.final_recognition_check(image)
                print(faceID, "  ", str(possibility))

                if possibility < 0.9 and faceID != "Other":
                    faceID = "Uncertain"
                
                cv2.rectangle(img, (x, y), (x + w, y + h +  50), color, thickness=2)

                # Words Prompt
                cv2.putText(img, faceID + " confidence: " + str(round(possibility, 2)),
                            (x + 30, y + 30),  # Coordinates
                            cv2.FONT_HERSHEY_SIMPLEX,  # Font
                            1,  # font-size
                            (255, 0, 255),  # Color
                            2)  # Thickness

                if faceID == "Other" and count < 5:
                    suspicious_level += 1
                    if suspicious_level == 10:
                        count += 1
                        img_name = './SuspectImages/' + 'suspect_image'+ '_' + str(count) + '.jpg'
                        cv2.imencode('.jpg',  img)[1].tofile(img_name)     
                        temp = convert_cv_qt(img)
                        suspicious_image_display.setPixmap(temp)
                        suspicious_level = 0
            cv2.imshow("camera", img)

        # Wait 200 milliseconds to see if there is a key input
        k = cv2.waitKey(40)
        # Exit loop if q is entered
        if k & 0xFF == ord('q'):
            break
 
    # Release the camera and destroy all windows
    camera.release()
    cv2.destroyAllWindows()

def convert_cv_qt(cv_img):
    """Convert from an opencv image to QPixmap"""
    disply_width = 251
    display_height = 151
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    p = convert_to_Qt_format.scaled(disply_width, display_height, Qt.KeepAspectRatio)
    return QPixmap.fromImage(p)



if __name__ == '__main__':
    face_recognize()