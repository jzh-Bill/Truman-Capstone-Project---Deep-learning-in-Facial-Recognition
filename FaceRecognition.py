import cv2
from ModelTraining import Model
from PredictingModels import *

global global_image

def face_recognize():

    # load up the models
    models = PredictingModels()

 
    # The color of the rectangular border framing the face
    color = (0, 255, 0)
 
    # Capture a live video stream from a specified camera
    camera = cv2.VideoCapture(1 , cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cascade_path = "haarcascade_frontalface_default.xml"

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
                cv2.putText(img, faceID + " Possibility: " + str(possibility),
                            (x + 30, y + 30),  # Coordinates
                            cv2.FONT_HERSHEY_SIMPLEX,  # Font
                            1,  # 字号
                            (255, 0, 255),  # Color
                            2)  # Thickness

            global_image = img
            cv2.imshow("camera", img)

        # Wait 200 milliseconds to see if there is a key input
        k = cv2.waitKey(50)
        # Exit loop if q is entered
        if k & 0xFF == ord('q'):
            break
 
    # Release the camera and destroy all windows
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    face_recognize()