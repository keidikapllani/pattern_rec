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
from facerec import fisherfaces
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier  
from sklearn.utils import resample
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as ldah
import numpy as np
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from statistics import mode
import random as rnd
x_train, y_train, x_test, y_test = load_data()
[d,n] = x_train.shape
#[eigenvalues_lda, W] = lda(x_train.T, y_train.T, 0)
#[D, W, mu] = fisherfaces(x_train, y_train)

#x_train_proj = project(x_train, W, mu_pca)
#x_test_proj = project(x_test, W, mu)
#plt.imshow(W[:,50].reshape(46,56).T, cmap = 'gist_gray')
#

#y_alt = pca_classifier(x_train,y_train,x_test,7)

#x_train_pca = np.dot(x_train.T,Ue[:,:100])
#x_test_pca = np.dot(x_test.T,Ue[:,:100])
## 2. Generate and train KNN classifier
#knn_classifier = KNeighborsClassifier(n_neighbors = 1)
#knn_classifier.fit(x_train_pca, y_train.T)

# 3. Classify the test data
#y_pred = knn_classifier.predict(x_test_pca)  
#accuracy = 100*accuracy_score(y_test.T, y_pred)

#Fisherfaces 
#x_train, y_train, x_test, y_test = split_load(0.8)
my_list = iter([50,100,150,200,250,300])


cntr = 0
#rnd.shuffle(x_test)
m_pca = [i for i in range(52,417,30)]
m_lda = [i for i in range(11,53,2)]
accuracy = np.zeros((len(m_pca)*len(m_lda),3))
y_knn = np.zeros((104,(len(m_pca)*len(m_lda))))

for i in m_pca:
    
    for j in m_lda:
       
        [  W, mu] = fisherfaces(x_train, y_train, i,j)
        x_final = np.dot((x_train-mu_pca).T,W)
        x_tst_proj = np.dot((x_test-mu).T,W)
        knn = KNeighborsClassifier(n_neighbors = 1)
        knn.fit(x_final, y_train.T)
        y_knn[:,cntr]= knn.predict(x_tst_proj)
        accuracy[cntr,0] = 100*accuracy_score(y_test.T, y_knn[:,cntr])
        accuracy[cntr,1] = i
        accuracy[cntr,2] = j
        cntr += 1

#Resampling
k = 5
accuracy = np.zeros((2*k,))
y_knn = np.zeros((104,k*2))
[W, mu_pca] = pca(x_train, y_train, None)
subspaces, rn = resample_w_pca(W,5,k)
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
	
#Bootstrapping
k_boot = 5

#y_knn = np.zeros((104,k_boot))
[W, mu_pca] = pca(x_train, y_train, None)
x_train_pca = np.dot((x_train-mu_pca).T,W).T
subfaces, y_sub, rn = resample_faces(x_train_pca,y_train,k_boot)
for i in range(0,k_boot):
    subface = subfaces[i,:,:rn[i]]

    kd = ldah(n_components=0, priors=None, shrinkage=None, solver='svd', store_covariance=True)
    kd.fit(subface.T,y_sub)   
    w_lda = kd.scalings_
    eigen = np.dot(subface,w_lda)
   
    
    x_tst_proj = np.dot(x_tst_pca,eigen)
	
    knn = KNeighborsClassifier(n_neighbors = 1)
    knn.fit(x_final, y_sub[i,:rn[i]].T)
    y_knn[:,k+i] = knn.predict(x_tst_proj)
    accuracy[i+k] = 100*accuracy_score(y_test.T, y_knn[:,i])
	
y_final = np.zeros((104,))
for i in range(0,104):	
	y_final[i] = mode(y_knn[i,:])
	
accuracy_final = 100*accuracy_score(y_test.T, y_final)





# Print 52 faces ______________________________________________________________
for i in range(0,416,8):
	plt.figure()
	plt.imshow(np.reshape(x_train[:,i],(46,56)).T,cmap = 'gist_gray')

