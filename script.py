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
[D, W, mu] = fisherfaces(x_train, y_train)
#[W, mu_pca] = pca_ae(x_train, y_train, (n-52))
x_train_proj = project(x_train, W, mu)
x_test_proj = project(x_test, W, mu)

plt.imshow(W[:,50].reshape(46,56).T, cmap = 'gist_gray')
#

#x_train_pca = np.dot(x_train.T,Ue[:,:100])
#x_test_pca = np.dot(x_test.T,Ue[:,:100])
## 2. Generate and train KNN classifier
#knn_classifier = KNeighborsClassifier(n_neighbors = 1)
#knn_classifier.fit(x_train_pca, y_train.T)

# 3. Classify the test data
#y_pred = knn_classifier.predict(x_test_pca)  
#accuracy = 100*accuracy_score(y_test.T, y_pred)