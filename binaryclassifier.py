# Binary classifier
# mithi on github

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import numpy as np 

class BinaryClassifier:

  def __init__(self, svc, scaler): # initializes the svc
    self.svc = svc
    self.scaler = scaler 

  def predict(self, f): # predictor function
    f = self.scaler.transform([f])
    r = self.svc.predict(f)
    return np.int(r[0]) 
