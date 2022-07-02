import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BOARD)
buttonPin = 16
GPIO.setup(buttonPin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

import cv2
import os, time
from obj import detect
cam = cv2.VideoCapture(0)

while True:
    buttonState = GPIO.input(buttonPin)
    ret, frame = cam.read()
    frame = cv2.flip(frame, 2)
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    if buttonState == False:
        # SPACE pressed
        img_name = "detect.png"
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        detect(img_name)
        time.sleep(3)
        os.remove(img_name)
    else:
        # ESC pressed
        print("Escape hit, closing...")
        break
cam.release()
cv2.destroyAllWindows()