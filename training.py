import numpy as np
import cv2, time
import os, shutil
import matplotlib.pyplot as plt
import random

from binaryclassifier import BinaryClassifier

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from skimage import data, exposure

# feature arrays
posFeatures_arr=[]
negFeatures_arr=[]

posImgs = "./train/pos"
negImgs = "./train/neg"

if os.path.exists(posImgs):
    allFiles = os.listdir(posImgs)
    for img in allFiles:
        image = cv2.imread(posImgs + "/" + img)
        
        # hog_image grabs the gradients from each cell
        features, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel = True)

        posFeatures_arr.append(features)


    posFeatures_arr = np.asarray(posFeatures_arr)
    totalvehicles = posFeatures_arr.shape[0]
    
if os.path.exists(negImgs):
    allFiles = os.listdir(negImgs)
    for img in allFiles:    
        image = cv2.imread(negImgs + "/" + img)
       
        # hog_image grabs the gradients from each cell
        features, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel = True)

        negFeatures_arr.append(features)

    negFeatures_arr = np.asarray(negFeatures_arr)
    totalnonvehicles = negFeatures_arr.shape[0]

print("Total Vehicles Shape: " + str(totalvehicles))
print("Total Non Vehicles Shape: " + str(totalnonvehicles))

# scale the features, allow for easier tracking
unscaled_x = np.vstack((posFeatures_arr, negFeatures_arr)).astype(np.float64)
scaler = StandardScaler().fit(unscaled_x)
x = scaler.transform(unscaled_x)
y = np.hstack((np.ones(totalvehicles), np.zeros(totalnonvehicles)))

# train the classifier
t_start = time.time()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = random.randint(1, 100))

svc = LinearSVC(max_iter=10000)
svc.fit(x_train, y_train)
accuracy = svc.score(x_test, y_test)

print("Time taken: ", np.round(time.time() - t_start, 2))
print("Accuracy: ", np.round(accuracy, 4))

