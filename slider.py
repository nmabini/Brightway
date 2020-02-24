import numpy as np
import cv2


# changed slider to be a slider across one row
def Slider(image, yIncrement, stepSize, windowSize):
    xIncrement = 0
    if yIncrement%2 == 0:
        xIncrement = int(image.shape[1]/2)
    yIncrement = int(round((yIncrement) / 2))
    for x in range(int(xIncrement), int(xIncrement + image.shape[1]/2), stepSize):
        yield(x, yIncrement*64, image[yIncrement*64:yIncrement*64 + windowSize[1], x:x + windowSize[0]])
            
def Formatter(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield(x, y, image[y:y+windowSize[1], x:x + windowSize[0]])
