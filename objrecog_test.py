import numpy as np
import cv2, time

cap = cv2.VideoCapture(0)

car_cascade = cv2.CascadeClassifier('cars_backup.xml')

while True:
    # uncomment next two lines to use the video feed
    check, frame = cap.read()
    cv2.imshow('Frame', frame)

    # uncomment next line to use example image
    # eximage = cv2.imread('freewaycars.jpg')

    # switch between "eximage" for example image, or "frame" for live video feed
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to grayscale

    
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors = 4, minSize = (100, 100))

    for (x, y, w, h) in cars:
        if(w < 40 and h < 40): # eliminate detections that are too small
            continue
        cv2.rectangle(eximage,(x,y),(x+w, y+h), (0,255,255), 2)

    # uncomment appropriate video footage
    cv2.imshow('Video', frame)
    # cv2.imshow('Example Image', eximage)
# quits when q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
