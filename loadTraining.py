# load training
import numpy as np
import cv2, time
import os, shutil, random
import matplotlib.pyplot as plt
import random
import logging, threading

from binaryclassifier import BinaryClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from skimage import data, exposure

from joblib import dump, load

from slider import Slider

# logging configuration
format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt = "%H:%M:%S")

(winW, winH) = (64, 64)
boxes = []

pos_test = "./test/pos"
neg_test = "./test/neg"

svc = load('svc.joblib')
scaler = load('scaler.joblib')

bc = BinaryClassifier(svc, scaler)

def sliderThread(name, frame, boxes):
    logging.info("Thread %s : starting...", name)
    for (x, y, window) in Slider(frame, name, stepSize=16, windowSize = (64, 64)):# name is a numerical value that determines what row this thread should handle
        ex_features, ex_hog_image = hog(window, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel = True)
        arr = ex_features.reshape(1, -1)
        if(arr.shape == (1, 128)):
            if bc.predict(arr):
                boxes.append((x, y, window))

### Signal that program is ready to receive images

cap = cv2.VideoCapture(0) ## if using live image stream

print("entering while loop...")
    
while True:
    # wait for program to send image
    # once image is received:
    output = []
    check, frame = cap.read()
    # use the slider to feed the classifier the window
    # define window width and height

    print("Starting detection...")
    timeDetect = time.time()
    x = threading.Thread(target = sliderThread, args=(0, frame, boxes,), daemon=True)
    x1 = threading.Thread(target = sliderThread, args=(1, frame, boxes,), daemon=True)
    x2 = threading.Thread(target = sliderThread, args=(2, frame, boxes,), daemon=True)
    x3 = threading.Thread(target = sliderThread, args=(3, frame, boxes,), daemon=True)
    x4 = threading.Thread(target = sliderThread, args=(4, frame, boxes,), daemon=True)
    x5 = threading.Thread(target = sliderThread, args=(5, frame, boxes,), daemon=True)
    x6 = threading.Thread(target = sliderThread, args=(6, frame, boxes,), daemon=True)
    x7 = threading.Thread(target = sliderThread, args=(7, frame, boxes,), daemon=True)
    x8 = threading.Thread(target = sliderThread, args=(8, frame, boxes,), daemon=True)
    x9 = threading.Thread(target = sliderThread, args=(9, frame, boxes,), daemon=True)
    x10 = threading.Thread(target = sliderThread, args=(10, frame, boxes,), daemon=True)
    x11 = threading.Thread(target = sliderThread, args=(11, frame, boxes,), daemon=True)
    logging.info("Main: before starting threads...")
    x.start()
    x1.start()
    x2.start()
    x3.start()
    x4.start()
    x5.start()
    x6.start()
    x7.start()
    x8.start()
    x9.start()
    x10.start()
    x11.start()
    logging.info("Main: wait for threads to finish...")
    x.join()
    x1.join()
    x2.join()
    x3.join()
    x4.join()
    x5.join()
    x6.join()
    x7.join()
    x8.join()
    x9.join()
    x10.join()
    x11.join()
    logging.info("Main: threads are done")
    
    
##    # all instances of "image" should be the image passed to this function
##    for (x, y, window) in Slider(frame, stepSize=16, windowSize = (winW, winH)): 
##        ex_features, ex_hog_image = hog(window, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel = True)
##        arr = ex_features.reshape(1, -1)
##        if(arr.shape == (1, 128)):
##            if bc.predict(arr):
##                conf = bc.conf(arr)
##                boxes.append((x, y, window, conf))
                
    print("size of boxes: ", len(boxes))
    print("Time to detection: ", np.round(time.time()-timeDetect, 2))
    # draw boxes from the array over the image
    for (x, y, window) in boxes:  
        #label = "{:.2f}%".format(conf*100)
        cv2.rectangle(frame, (x,y),(x+winW, y+winH), (0,255,255), 2)
        #cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
    cv2.destroyAllWindows()
    cv2.imshow("Image", frame)
    
    # send 'image' to server
    key=cv2.waitKey(0)
    boxes.clear()
    if (key==27): # Esc key
        break
    
cap.release()
cv2.destroyAllWindows()


