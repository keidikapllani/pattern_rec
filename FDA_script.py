# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 2018

@author: Antonio Enas
"""

from facerec import *
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier  
from sklearn.utils import resample
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as ldah

from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from statistics import mode
import random as rnd

# Load and split dataset
x_train, y_train, x_test, y_test = load_data()
d,n = x_train.shape
c = len(np.unique(y_train))
dts,nts = x_test.shape
# Calculate eigenfaces
[W_pca, mu_pca] = pca(x_train, y_train, 415)

# Determine accuracy as function of the hyperparameters M_pca, M_lda____________
accuracy_fda = np.zeros((n,c-1))
y_fda = np.zeros((n,c-1,nts))
knn = KNeighborsClassifier(n_neighbors = 1)

A_train = x_train - mu_pca
A_test = x_test - mu_pca
for m_pca in range(1,n):
    # Project train and test onto the PCA eigenspace
	x_train_pca = np.dot(A_train.T,W_pca[:,:m_pca])
	x_test_pca = np.dot(A_test.T,W_pca[:,:m_pca])
	
	# Project train and test onto the LDA space
	for m_lda in range(1,c-1):
		# Create LDA model
		LDA_model = ldah(priors=None, shrinkage=None, solver='svd', store_covariance=True)
		LDA_model.fit(x_train_pca,(y_train.T).ravel())		
		W_lda = LDA_model.scalings_[:,:m_lda]
		W_fda = np.dot(W_pca[:,:m_pca],W_lda)
		# Project onto the FDA
		x_train_fda = np.dot((x_train - mu_pca).T, W_fda)
		x_test_fda = np.dot((x_test - mu_pca).T, W_fda)       
        # Train the kNN on the projected data
		knn.fit(x_train_fda, y_train.T)
		# Classify the projected test data
		y_fda[m_pca,m_lda,:]= knn.predict(x_test_fda)
		accuracy_fda[m_pca,m_lda] = 100*accuracy_score(y_test.T, y_fda[m_pca,m_lda,:])

# Plot the heatmap of FDA accuracy_____________________________________________

plt.imshow(accuracy_fda.T,aspect = 'auto',cmap = 'RdYlGn')
cb = plt.colorbar()
cb.set_label('$\%$ Accuracy',fontsize=14)
plt.xlabel('$M_{PCA}$', fontsize = 14)
plt.ylabel('$M_{LDA}$', fontsize = 14)
plt.title('FDA-NN Classifier accuracy\nas function of the hyperparameters'
		  , fontsize = 16)