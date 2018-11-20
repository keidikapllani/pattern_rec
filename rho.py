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
knn = KNeighborsClassifier(n_neighbors = 1)
[W, mu_pca] = pca(x_train, y_train, None)


#Shape of accuracy is ro_bagging,ro_feature,k
accuracy_av = np.zeros((5,6,2))
accuracy_ens = np.zeros((5,6,2))
k = 60
#For averaging results
for j in range(0,2):
	cntr_ro_bagging = 0
	cntr_ro_feature = 0
	
	accuracy = np.zeros((k*2,))
	y_knn = np.zeros((104,k*2))
	#Vary randomness of bagging (minimum test set)
	for ro_bagging in range(10,416,50):
		cntr_ro_feature = 0
		#Vary randomness of feature resample (minimum constant set)
		for ro_feature in range(5,100,20):
			
					
			#Resampling
			subspaces, rn = resample_w_pca(W,ro_feature,k)
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
				
				knn.fit(x_final, (y_train.T).ravel())
				y_knn[:,i] = knn.predict(x_tst_proj)
				accuracy[i] = 100*accuracy_score(y_test.T, y_knn[:,i])
	        
	    	
	        ##Bootstrapping
			x_train_pca = np.dot((x_train-mu_pca).T,W).T
			x_tst_pca = np.dot((x_test-mu_pca).T,W)       
			
			subfaces, y_sub, rn = resample_faces(x_train_pca,y_train,ro_bagging,k)
			for i in range(0,k):
				subface = subfaces[i,:,:rn[i]]
				kd = ldah(n_components=0, priors=None, shrinkage=None, solver='svd', store_covariance=True)
				kd.fit(subface.T,(y_sub[i,:rn[i]].T).ravel())   
				w_lda = kd.scalings_
				x_final = np.dot(subface.T,w_lda)
            
            
				x_tst_proj = np.dot(x_tst_pca,w_lda)
				
				knn.fit(x_final, (y_sub[i,:rn[i]].T).ravel())
				y_knn[:,k+i] = knn.predict(x_tst_proj)
				accuracy[i+k] = 100*accuracy_score(y_test.T, y_knn[:,i])
			
			cntr_ro_feature += 1
	            
		y_ensemble,accuracy_ens[cntr_ro_bagging,cntr_ro_feature,j] = maj_voting(y_knn,y_train,y_test)
		accuracy_av[cntr_ro_bagging,cntr_ro_feature,j]= accuracy.mean(axis=0)
				       
			
			
		cntr_ro_bagging += 1
	            
# Plot the heatmap of ensamble accuracy________________________________________
plt.imshow(accuracy_keidi.T,aspect = 'auto',cmap = 'YlOrRd')
cb = plt.colorbar()
cb.set_label('% Accuracy',fontsize=14)
plt.xlabel('$\rho_{bagging}$', fontsize = 14)
plt.ylabel('$\rho_{features}$', fontsize = 14)
plt.title('Ensemble Classifier accuracy\nas function of the randomisation parameters'
		  , fontsize = 16)