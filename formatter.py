import numpy as np
import cv2
import random
import os
from slider import Formatter

outlier_found = 0 # assume all images are properly formatted
imgdir = "./n"
allFiles = os.listdir(imgdir)
for img in allFiles:
    image = cv2.imread(imgdir + "/" + img, cv2.COLOR_BGR2GRAY)
    if image.shape != (64, 64, 3):
        outlier_found = 1
        print("found an outlier: ", image.shape)
        new = 0
        for (x, y, window) in Formatter(image, stepSize = 64, windowSize = (64, 64)):
            if window.shape == (64, 64, 3):
                cv2.imwrite(imgdir + "/" + img + "_" + str(new) + ".jpg", window)
                new = new + 1
        os.remove(imgdir + "/" + img)

imgdir = "./p"
#if os.path.exists(posImgs):
allFiles = os.listdir(imgdir)   
for img in allFiles:
    image = cv2.imread(imgdir + "/" + img, cv2.COLOR_BGR2GRAY)
    if image.shape != (64, 64, 3):
        outlier_found = 1
        print("found an outlier: ", image.shape)
        new = 0
        for (x, y, window) in Formatter(image, stepSize = 64, windowSize = (64, 64)):
            if window.shape == (64, 64, 3):
                cv2.imwrite(imgdir + "/" + img + "_" + str(new) + ".jpg", window)
                new = new + 1
        os.remove(imgdir + "/" + img)

if outlier_found == 0:
    print("all images were properly formatted.")



