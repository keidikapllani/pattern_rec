#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 01:44:14 2018

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


k_range = range(5,50)
accuracy_av = np.zeros((60,3))
accuracy_ens = np.zeros((60,3))
for j in range(0,3):
    cntr = 0
    for k in range(2,100,5):
        #Resampling
       
        accuracy = np.zeros((k*2,))
        y_knn = np.zeros((104,k*2))
        [W, mu_pca] = pca(x_train, y_train, None)
        subspaces, rn = resample_w_pca(W,10,k)
        for i in range(0,k):
            Wi= subspaces[i,:,:rn[i]]
            x_pca = project(Wi,x_train,mu_pca).T
        	
            #w_lda = lda(x_pca,y_train,0)
            kd = ldah(n_components=0, priors=None, shrinkage=None, solver='svd', store_covariance=True)
            kd.fit(x_pca.T,y_train.T)
            w_lda = kd.scalings_
            eigen = np.dot(Wi,w_lda)
            x_final = np.dot((x_train-mu_pca).T,eigen)
            x_tst_proj = np.dot((x_test-mu_pca).T,eigen)
            knn = KNeighborsClassifier(n_neighbors = 1)
            knn.fit(x_final, y_train.T)
            y_knn[:,i] = knn.predict(x_tst_proj)
            accuracy[i] = 100*accuracy_score(y_test.T, y_knn[:,i])
        
        #rec = np.dot(np.real(w_lda),x_pca) 
        #plt.imshow(np.reshape(np.real(eigen[:,2]),(46,56)).T,cmap = 'gist_gray')
        	
        ##Bootstrapping
        
        
        #y_knn = np.zeros((104,k_boot))
        
        x_train_pca = np.dot((x_train-mu_pca).T,W).T
        x_tst_pca = np.dot((x_test-mu_pca).T,W)
        subfaces, y_sub, rn = resample_faces(x_train_pca,y_train,k)
        for i in range(0,k):
            subface = subfaces[i,:,:rn[i]]
            kd = ldah(n_components=0, priors=None, shrinkage=None, solver='svd', store_covariance=True)
            kd.fit(subface.T,y_sub[i,:rn[i]].T)   
            w_lda = kd.scalings_
            x_final = np.dot(subface.T,w_lda)
            
            
            x_tst_proj = np.dot(x_tst_pca,w_lda)
            knn = KNeighborsClassifier(n_neighbors = 1)
            knn.fit(x_final, y_sub[i,:rn[i]].T)
            y_knn[:,k+i] = knn.predict(x_tst_proj)
            accuracy[i+k] = 100*accuracy_score(y_test.T, y_knn[:,i])
            
        y_ensemble,accuracy_ens[cntr,j] = maj_voting(y_knn,y_train,y_test)
        accuracy_av[cntr,j]= accuracy.mean(axis=0)
        cntr+=1
            
