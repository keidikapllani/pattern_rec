#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 16:58:59 2018

@author: keidi
"""

#import sys
## append tinyfacerec to module search path
#sys.path.append("..")

from facerec import lda, pca, load_data, fisherfaces, project
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import numpy as np

x_train, y_train, x_test, y_test = load_data()
[d,n] = x_train.shape
#[eigenvalues_lda, W] = lda(x_train.T, y_train.T, 0)
#[D, W, mu] = fisherfaces(x_train.T, y_train.T)
[eigenvalues_pca, W, mu_pca] = pca(x_train, y_train, (n-52))
x_train_proj = project(x_train, W, mu_pca)
x_test_proj = project(x_test, W, mu_pca)

k=1

y_train = np.ravel(y_train)
knn = KNeighborsClassifier(n_neighbors = k)
knn.fit(x_train_proj.T, y_train)
y_knn = knn.predict(x_test_proj.T)
accuracy = 100*accuracy_score(y_test.T, y_knn)

