import numpy as np
import cv2, time
import os, shutil, random
import matplotlib.pyplot as plt
import random

from joblib import dump, load

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

#### implement train/test split ####
print("train test split start...")

posImgs = "./p"
negImgs = "./n"
timeSplit = time.time()
if os.path.exists("./train"):
    shutil.rmtree("./train")
    os.mkdir("./train/")
    os.mkdir("./train/pos")
    os.mkdir("./train/neg")
else:
    os.mkdir("./train/")
    os.mkdir("./train/pos")
    os.mkdir("./train/neg")

if os.path.exists("./test"):
    shutil.rmtree("./test")
    os.mkdir("./test/")
    os.mkdir("./test/pos")
    os.mkdir("./test/neg")
else:
    os.mkdir("./test/")
    os.mkdir("./test/pos")
    os.mkdir("./test/neg")

if os.path.exists("./val"):
    shutil.rmtree("./val")
    os.mkdir("./val/")
    os.mkdir("./val/pos")
    os.mkdir("./val/neg")
else:
    os.mkdir("./val/")
    os.mkdir("./val/pos")
    os.mkdir("./val/neg")
    
### add training and testing data ###
src = posImgs
allFileNames = os.listdir(src)
np.random.shuffle(allFileNames)
train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames), [int(len(allFileNames)*0.8), int(len(allFileNames)*0.85)])

train_FileNames = [src+'/'+name for name in train_FileNames.tolist()]
val_FileNames = [src+'/'+name for name in val_FileNames.tolist()]
test_FileNames = [src+'/'+name for name in test_FileNames.tolist()]

for name in train_FileNames:
    shutil.copy(name, "./train/pos")

for name in test_FileNames:
    shutil.copy(name, "./test/pos")

for name in val_FileNames:
    shutil.copy(name, "./val/pos")

src = negImgs
allFileNames = os.listdir(src)
np.random.shuffle(allFileNames)
train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames), [int(len(allFileNames)*0.8), int(len(allFileNames)*0.85)])

train_FileNames = [src+'/'+name for name in train_FileNames.tolist()]
val_FileNames = [src+'/'+name for name in val_FileNames.tolist()]
test_FileNames = [src+'/'+name for name in test_FileNames.tolist()]

for name in train_FileNames:
    shutil.copy(name, "./train/neg")

for name in test_FileNames:
    shutil.copy(name, "./test/neg")

for name in val_FileNames:
    shutil.copy(name, "./val/neg")

pos_train = "./train/pos"
neg_train = "./train/neg"
pos_test = "./test/pos"
neg_test = "./test/neg"

### TRAINING STARTS ###
if os.path.exists(pos_train):
    print("posimg feature starts...")
    allFiles = os.listdir(posImgs)
    for img in allFiles:
        image = cv2.imread(posImgs + "/" + img, cv2.COLOR_BGR2GRAY)
        features, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel = True)
        posFeatures_arr.append(features)


    posFeatures_arr = np.asarray(posFeatures_arr)
    totalvehicles = posFeatures_arr.shape[0] # number of images with vehicles
    
if os.path.exists(neg_train):
    print("negimg feature starts...")
    allFiles = os.listdir(negImgs)
    for img in allFiles:
        image = cv2.imread(negImgs + "/" + img, cv2.COLOR_BGR2GRAY)
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

### find accuracy with test data ###
tp_test = time.time()

dump(svc, 'svc.joblib')
dump(scaler, 'scaler.joblib')
