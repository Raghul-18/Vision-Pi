import cv2
import os, time
from obj import detect
from Detector import *
modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz"
classFile = "coco.names"
detector = Detector()
detector.readClasses(classFile)
detector.downloadModel(modelURL)
detector.loadModel()

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 2)
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "detect.png"
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        detector.predictImage(img_name)
        #detect(img_name)
        os.remove(img_name)
cam.release()
cv2.destroyAllWindows()