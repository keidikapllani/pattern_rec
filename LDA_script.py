# -*- coding: utf-8 -*-
"""
LDA for face recognition script
Pattern recognition EE4_64 coursework 1

Created on Mon Nov 12 14:02:12 2018

@author: Antonio Enas, Keidi Kapllani
"""

# Import libraries
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from numpy import linalg as LA
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from facerec import *
# Import data
x_train,y_train,x_test,y_test = load_data()
	
###_____________________________ START LDA ____________________________________
N = 416
D = 2576
c = 52	
M = c - 1
#1. Compute the global mean
m = x_train.mean(axis = 1).reshape((2576,1))

#2. Compute the mean of each class mi
#3. Compute Sw = sum over c{(x - mi)*(x - mi).T}, rank(Sw) = N - c
#	Sw is the within class scatter matrix
#4. Compute Sb = sum over c{(mi - m)*(mi - m).T}, it has rank(c-1)
#	Sb is the between class scatter matrix
mi = np.zeros((2576,52))
Sw = np.zeros((D,D))
Sb = np.zeros((D,D))
_ix = 0
for c in range(0,52):
	xi = x_train[:,_ix:_ix+8]
	#2
	mi[:,c] = xi.mean(axis = 1)
	_mi = mi[:,c].reshape((D,1))
	#3
	Sw = Sw + np.dot((xi-_mi),(xi-_mi).T)
	#4
	Sb = Sb + np.dot((_mi - m),(_mi - m).T)
	_ix += 8
print(f'rank(Sw) = {np.linalg.matrix_rank(Sw)}') #Sanity check, should be N -c 
print(f'rank(Sb) = {np.linalg.matrix_rank(Sb)}') #Sanity check, should be c-1

#5. Perform PCA for dimensionality reduction using M = c - 1
A_pca = x_train - m

#Which S to use???
S_pca = (1 / N) * np.dot(A_pca.T, A_pca) #Returns a N*N matrix
#S_pca = Sw + Sb

print('dim S_pca = ', S_pca.shape)
l_pca, v_pca = np.linalg.eig(S_pca)
_U = np.dot(A_pca, v_pca)
U_pca = _U / np.apply_along_axis(np.linalg.norm, 0, _U) #normalise each eigenvector
print('dim U_pca = ',U_pca.shape)

#What is W_pca?????
#W_pca = np.dot(A_pca.T, np.real(U_pca[:,:M+1])).T
W_pca = U_pca[:,:M+1]

#6. Find generalised eigenvals of (W_pca.T*Sw*W_pca)**-1(W_pca.T*Sb*W_pca)

b = np.dot(np.dot(W_pca.T,Sb),W_pca)
a = np.dot(np.dot(W_pca.T,Sw),W_pca)

generalised_eigenvecs = np.dot(np.linalg.inv(a),b) #Should be like eigenfaces
_Wopt = np.dot(A_pca, generalised_eigenvecs)
plt.imshow(np.reshape(np.real(lol[:,1]),(46,56)).T,cmap = 'gist_gray')	


