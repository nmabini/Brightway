import os, shutil
import numpy as np

# one time use, separates images into training and testing sets

# shutil modifies folders
posImgs = ("./p")
negImgs = ("./n")

# make training and testing directory, delete if already exist
if os.path.exists("./train"):
    shutil.rmtree("./train")
    os.mkdir("./train") 
    os.mkdir("./train/pos")
    os.mkdir("./train/neg")
    
else:
    os.mkdir("./train")
    os.mkdir("./train/pos")
    os.mkdir("./train/neg")

if os.path.exists("./test"):
    shutil.rmtree("./test")
    os.mkdir("./test")
    os.mkdir("./test/pos")
    os.mkdir("./test/neg")
else:
    os.mkdir("./test")
    os.mkdir("./test/pos")
    os.mkdir("./test/neg")

if os.path.exists("./val"):
    shutil.rmtree("./val")
    os.mkdir("./val")
    os.mkdir("./val/pos")
    os.mkdir("./val/neg")
else:
    os.mkdir("./val")
    os.mkdir("./val/pos")
    os.mkdir("./val/neg")

# create training and testing folders
# positive Images
src = posImgs

allFileNames = os.listdir(src)
np.random.shuffle(allFileNames)
train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames), [int(len(allFileNames)*0.8), int(len(allFileNames)*0.85)]) # 80/20 split between train and test

train_FileNames = [src+'/'+name for name in train_FileNames.tolist()]
val_FileNames = [src+'/'+name for name in val_FileNames.tolist()]
test_FileNames = [src+'/'+name for name in test_FileNames.tolist()]

print('Total images: ', len(allFileNames))
print('Training: ', len(train_FileNames))
print('Validation: ', len(val_FileNames))
print('Testing: ', len(test_FileNames))

for name in train_FileNames:
    shutil.copy(name, "./train/pos")

for name in val_FileNames:
    shutil.copy(name, "./val/pos")

for name in test_FileNames:
    shutil.copy(name, "./test/pos")

# negative Images
src = negImgs

allFileNames = os.listdir(src)
np.random.shuffle(allFileNames)
train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames), [int(len(allFileNames)*0.8), int(len(allFileNames)*0.85)])

train_FileNames = [src+'/'+name for name in train_FileNames.tolist()]
val_FileNames = [src+'/'+name for name in val_FileNames.tolist()]
test_FileNames = [src+'/'+name for name in test_FileNames.tolist()]

print('Total images: ', len(allFileNames))
print('Training: ', len(train_FileNames))
print('Validation: ', len(val_FileNames))
print('Testing: ', len(test_FileNames))

for name in train_FileNames:
    shutil.copy(name, "./train/neg")

for name in val_FileNames:
    shutil.copy(name, "./val/neg")

for name in test_FileNames:
    shutil.copy(name, "./test/neg")

