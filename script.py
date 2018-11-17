#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 16:58:59 2018

@author: keidi
"""

#import sys
## append tinyfacerec to module search path
#sys.path.append("..")

#from facerec import lda, pca, load_data, fisherfaces, project, resample_w_pca, resample_faces
from facerec import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier  
from sklearn.utils import resample
import numpy as np
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from statistics import mode
x_train, y_train, x_test, y_test = load_data()
[d,n] = x_train.shape
#[eigenvalues_lda, W] = lda(x_train.T, y_train.T, 0)
#[D, W, mu] = fisherfaces(x_train, y_train)
[W, mu_pca] = pca(x_train, y_train, (n-52))
#x_train_proj = project(x_train, W, mu_pca)
#x_test_proj = project(x_test, W, mu)

#plt.imshow(W[:,50].reshape(46,56).T, cmap = 'gist_gray')
#

y_alt = pca_classifier(x_train,y_train,x_test,7)

#x_train_pca = np.dot(x_train.T,Ue[:,:100])
#x_test_pca = np.dot(x_test.T,Ue[:,:100])
## 2. Generate and train KNN classifier
#knn_classifier = KNeighborsClassifier(n_neighbors = 1)
#knn_classifier.fit(x_train_pca, y_train.T)

# 3. Classify the test data
#y_pred = knn_classifier.predict(x_test_pca)  
#accuracy = 100*accuracy_score(y_test.T, y_pred)

# PCA-LDA Ensemble 
#bootstrap_size = int(0.5*len(x_train_proj))
##bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=1.0, max_features=1.0, bootstrap_features = True)
##bagging.fit(x_train.T,y_train.T)
##samples = bagging.estimators_features_
##samples = bagging.estimators_samples_
#x_bag, y_bag = resample(x_train_proj, y_train.T, n_samples = bootstrap_size , replace = True)
#y_bag = y_bag.T
#[eigenvalues_lda, W] = lda(x_bag,y_bag, 0)


#Resampling
k = 40
accuracy = np.zeros((k,))
y_knn = np.zeros((104,k*2))
subspaces, rn = resample_w_pca(W,10,k)
for i in range(0,k):
	Wi= subspaces[i,:,:rn[i]]
	x_pca = project(Wi,x_train,mu_pca).T
	
	w_lda = lda(x_pca,y_train,0)
	eigen = np.dot(Wi,w_lda)
	x_final = np.dot((x_train-mu_pca).T,eigen)
	x_tst_proj = np.dot((x_test-mu_pca).T,eigen)
	knn = KNeighborsClassifier(n_neighbors = 1)
	knn.fit(x_final, y_train.T)
	y_knn[:,i] = knn.predict(x_tst_proj)
	accuracy[i] = 100*accuracy_score(y_test.T, y_knn[:,i])

#rec = np.dot(np.real(w_lda),x_pca) 
#plt.imshow(np.reshape(np.real(eigen[:,2]),(46,56)).T,cmap = 'gist_gray')
	
#Bootstrapping
k_boot = 40
accuracy = np.zeros((k_boot,))
#y_knn = np.zeros((104,k_boot))
x_train_pca = np.dot((x_train-mu_pca).T,W).T
subfaces, y_sub, rn = resample_faces(x_train_pca,y_train,k_boot)
for i in range(0,k_boot):
	subface = subfaces[i,:,:rn[i]]
	w_lda = lda_gen(subface,y_sub[i,:rn[i]],0)
	
	x_final = np.dot(subface.T,w_lda)
	
	x_tst_pca = np.dot((x_test-mu_pca).T,W)
	x_tst_proj = np.dot(x_tst_pca,w_lda)
	
	knn = KNeighborsClassifier(n_neighbors = 1)
	knn.fit(x_final, y_sub[i,:rn[i]].T)
	y_knn[:,k+i] = knn.predict(x_tst_proj)
	accuracy[i] = 100*accuracy_score(y_test.T, y_knn[:,i])
	
y_final = np.zeros((104,))
for i in range(0,104):	
	y_final[i] = mode(y_knn[i,:])
	
accuracy_final = 100*accuracy_score(y_test.T, y_final)
