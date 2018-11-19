#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 21:38:13 2018

@author: keidi
"""

from facerec import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as ldah
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


x_train, y_train, x_test, y_test = load_data()
[d,n] = x_train.shape
[W_pca, mu_pca] = pca(x_train, y_train, 415)


accuracy_fda = np.zeros((n,c-1))
y_fda = np.zeros((n,c-1,nts))
knn = KNeighborsClassifier(n_neighbors = 1)
accuracy_keidi
A_train = x_train - mu_pca
A_test = x_test - mu_pca

# Project train and test onto the PCA eigenspace
x_train_pca = np.dot(A_train.T,W_pca[:,:102])
x_test_pca = np.dot(A_test.T,W_pca[:,:102])
	
	# Project train and test onto the LDA space
	
		# Create LDA model
LDA_model = ldah(priors=None, shrinkage=None, solver='svd', store_covariance=True)
LDA_model.fit(x_train_pca,(y_train.T).ravel())		
W_lda = LDA_model.scalings_[:,:43]
W_fda = np.dot(W_pca[:,:102],W_lda)
		# Project onto the FDA
plt.figure()

for i in range(0,4):
    plt.subplot(1, 4, i+1)
    plt.imshow(np.reshape(W_fda[:,i],(46,56)).T,cmap = 'gist_gray')
