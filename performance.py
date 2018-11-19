# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 17:45:06 2018

@author: AE
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from numpy import linalg as LA
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.preprocessing import normalize
import time
import sys
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import os,psutil
from facerec import *
from sys import getsizeof

### Load data
x_train,y_train,x_test,y_test = load_data()
D, N = x_train.shape# Calculate mean face
meanface = x_train.mean(axis=1).reshape((D,1))
A = x_train - meanface #normalised training data D*N


# D*D Covariance matrix
tDs = np.zeros((3,))
for i in range(0,3):
	ts = time.time()
	S = (1 / N) * np.dot(A, A.T) # D*D matrix
	wn, Ue = np.linalg.eig(S)
	tDs[i]=time.time() - ts
tD = tDs.mean(axis=0)	
mD = getsizeof(S)/1000000
# N*N Covariance matrix
tNs = np.zeros((10,))
for i in range(0,10):
	ts = time.time()
	Se = (1 / N) * np.dot(A.T, A) # N*N matrix
	le, Ue = np.linalg.eig(Se)
	tNs[i]=time.time() - ts	
tN = tNs.mean(axis=0)
mN = getsizeof(Se)/1000000