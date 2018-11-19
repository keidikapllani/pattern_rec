#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 16:58:59 2018

@author: Keidi Kapllani, Antonio Enas
"""
from facerec import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as ldah
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


x_train, y_train, x_test, y_test = load_data()
[d,n] = x_train.shape

#Fisherface
'''
Vary M_pca and M_lda for fisherface and store predication accuracies
'''
cntr = 0
m_pca = [i for i in range(52,417,30)]
m_lda = [i for i in range(11,53,2)]
accuracy = np.zeros((len(m_pca)*len(m_lda),3))
y_knn = np.zeros((104,(len(m_pca)*len(m_lda))))

for i in m_pca:
    
    for j in m_lda:
        
        [W, mu] = fisherfaces(x_train, y_train, i,j)
        x_final = np.dot((x_train-mu_pca).T,W)
        x_tst_proj = np.dot((x_test-mu).T,W)
        knn = KNeighborsClassifier(n_neighbors = 1)
        knn.fit(x_final, y_train.T)
        y_knn[:,cntr]= knn.predict(x_tst_proj)
        accuracy[cntr,0] = 100*accuracy_score(y_test.T, y_knn[:,cntr])
        accuracy[cntr,1] = i
        accuracy[cntr,2] = j
        cntr += 1

#PCA-LDA Ensemble
'''
Apply feature resampling and bootstrapping and create a commitee machine with majority voting
'''

#Resampling
k = 10
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
k_boot = 10

#y_knn = np.zeros((104,k_boot))

x_train_pca = np.dot((x_train-mu_pca).T,W).T
x_tst_pca = np.dot((x_test-mu_pca).T,W)
subfaces, y_sub, rn = resample_faces(x_train_pca,y_train,k_boot)
for i in range(0,k_boot):
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
	

"""
Maj_voting doesnt currently work.
"""
final_score = maj_voting	(y_knn,y_test)




	


# Plot 52 faces ______________________________________________________________
c = 1
for i in range(0,416,8):
	plt.figure()
	plt.imshow(np.reshape(x_train[:,i],(46,56)).T,cmap = 'gist_gray')
	plt.title(f'Class {c}')
	c += 1

# Plot confusion matrix ______________________________________________________
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test.T, y_test.T)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[i for i in range(1,53)], normalize=True,
                      title='Normalized confusion matrix')