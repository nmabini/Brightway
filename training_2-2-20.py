import numpy as np
import cv2, time
import os, shutil
import matplotlib.pyplot as plt
import random
from slider import Slider

import dask
from dask import delayed
#from dask.distributed import Client
#client = Client() # start local client
#from dask import delayed, compute
dask.config.set(scheduler='threads')

from binaryclassifier import BinaryClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from skimage import data, exposure

# feature arrays
posFeatures_arr=[]
negFeatures_arr=[]
exFeatures_arr=[]

once = 0
# array to hold boxes to draw on image

# function to apply to each chunk
def predictFunc(x, y, window, bc, boxes):
    ex_features, ex_hog_image = hog(window, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel = True)
    arr = ex_features.reshape(1, -1)
    if(arr.shape == (1, 128)):
        if bc.predict(arr):
            conf = bc.conf(arr)
            boxes.append((x, y, window, conf))
    return 1

posImgs = "./p"
negImgs = "./n"
timeSplit = time.time()
if os.path.exists(posImgs):
    allFiles = os.listdir(posImgs)
    for img in allFiles:
        image = cv2.imread(posImgs + "/" + img, cv2.COLOR_BGR2GRAY)
        
        # hog_image grabs the gradients from each cell
        features, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel = True)
        posFeatures_arr.append(features)


    posFeatures_arr = np.asarray(posFeatures_arr)
    totalvehicles = posFeatures_arr.shape[0] # number of images with vehicles
    
if os.path.exists(negImgs):
    allFiles = os.listdir(negImgs)
    for img in allFiles:    
        image = cv2.imread(negImgs + "/" + img, cv2.COLOR_BGR2GRAY)
       
        # hog_image grabs the gradients from each cell
        features, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel = True)
        negFeatures_arr.append(features)
    
    negFeatures_arr = np.asarray(negFeatures_arr)
    totalnonvehicles = negFeatures_arr.shape[0] # number of images without vehicles
    
print()
print("Feature Extraction time: ", np.round(time.time() - timeSplit, 2))
timeScale = time.time()
# normalize features before training
unscaled_x = np.vstack((posFeatures_arr, negFeatures_arr)).astype(np.float64)
scaler = StandardScaler().fit(unscaled_x) # normalize positive and negative features
x = scaler.transform(unscaled_x) # x is the normalized values
y = np.hstack((np.ones(totalvehicles), np.zeros(totalnonvehicles))) # y is an array of a tuple [1, 0]

print("Feature Scaling time: ", np.round(time.time()-timeScale, 2))
timeTrain = time.time()
# train the classifier

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = random.randint(1, 100))

svc = LinearSVC(max_iter=10000)
svc.fit(x_train, y_train)
accuracy = svc.score(x_test, y_test)
print("Training time: ", np.round(time.time()-timeTrain, 2))
print("Accuracy: ", np.round(accuracy, 4))


cap = cv2.VideoCapture(0)
while True:
    boxes = []
    output = []
    check, frame = cap.read()
    bc = BinaryClassifier(svc, scaler)
    # use the slider to feed the classifier the window
    # define window width and height
    (winW, winH) = (64, 64)

    timeDetect = time.time()

### USE THREADING TO OPTIMIZE TIME
    for (x, y, window) in Slider(frame, stepSize=16, windowSize = (winW, winH)):
        task = delayed(predictFunc)(x, y, window, bc, boxes)
        output.append(task)
        #task = client.submit(predictFunc, (x, y, window, bc, boxes))

    print('size of output: ', len(output)) # show that tasks are added
    total = delayed(sum)(output)

    total.compute()
    print("size of boxes: ", len(boxes))
    print("Time to detection: ", np.round(time.time()-timeDetect, 2))
    # draw boxes from the array over the image
    for (x, y, window, conf) in boxes:
        label = "{:.2f}%".format(conf*100)
        cv2.rectangle(frame, (x,y),(x+winW, y+winH), (0,255,255), 2)
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
            
    cv2.imshow('Camera Frame', frame)
    key=cv2.waitKey(0)
    if (key==27): # Esc key
        break
    
cap.release()
cv2.destroyAllWindows()

