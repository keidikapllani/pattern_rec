#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 16:53:18 2018

@author: keidi
"""
import numpy as np
import scipy.io as sio
import random as rnd
import matplotlib.pyplot as plt


def load_data():
    
    mat_content = sio.loadmat('assests/face.mat')
    face_data = mat_content['X']
    out_data = mat_content['l']
    
    ### Split train and test data
   
    pt_test = 2
    n_people = 52
    
    #Initialise train and test matrices
    
    n_test  = int(face_data.shape[1]*pt_test/10)
    x_test  = np.zeros((len(face_data),n_test), dtype = int)
    y_test  = np.zeros((len(out_data),n_test), dtype = int)
    x_train = face_data
    y_train = out_data
    
    #Initialise counter to build output matrices
    ix = 0
    
    #for each person split data
    for ix_splitter in range(n_people):
        #generate random indexes within the face range per person
        rng = list(range(0,10,1))
        indx = rnd.sample(rng,pt_test)
        r = [indx[i] + ix_splitter*10 for i in range(len(indx))]
        x_test[:,[ix, ix+1]] = face_data[:,r]
        y_test[:,[ix, ix+1]] = out_data[:,r]   
        x_train = np.delete(x_train,r[0]-ix,1)
        x_train = np.delete(x_train,r[1]-ix,1)
        y_train = np.delete(y_train,r[0]-ix,1)
        y_train = np.delete(y_train,r[1]-ix,1)
        
        ix = ix + 2
	
    return(x_train, y_train, x_test, y_test)

def project(W, X, mu=None):
	if mu is None:
		return np.dot(X.T,W)
	return np.dot((X - mu), W)

def reconstruct(W, Y, mu=None):
    if mu is None:
        return np.dot(Y,W.T)
    return np.dot(Y, W.T) + mu

<<<<<<< HEAD
def pca(X, y, num_components=0):
    [d, n] = X.shape
    if (num_components <= 0) or (num_components>n):
        num_components = n
    mu = X.mean(axis=1).reshape((d,1))
    
    X = X - mu
   
    C = np.dot(X.T,X)
    [eigenvalues,eigenvectors] = np.linalg.eigh(C)
    eigenvectors = np.dot(X,eigenvectors)
    for i in range(n):
        eigenvectors[:,i] = eigenvectors[:,i]/np.linalg.norm(eigenvectors[:,i])
	# or simply perform an economy size decomposition
	# eigenvectors, eigenvalues, variance = np.linalg.svd(X.T, full_matrices=False)
	# sort eigenvectors descending by their eigenvalue
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
	# select only num_components
    eigenvalues = eigenvalues[0:num_components].copy()
    eigenvectors = eigenvectors[:,0:num_components].copy()
    return [eigenvalues, eigenvectors, mu]
		
def lda(X, y, num_components=0):
    y = np.asarray(y)
    [d,n] = X.shape
    c = np.unique(y)
    if (num_components <= 0) or (num_components>(len(c)-1)):
        num_components = (len(c)-1)
    meanTotal = X.mean(axis=1).reshape((2576,1))
    Sw = np.zeros((d, d), dtype=np.float32)
    Sb = np.zeros((d, d), dtype=np.float32)
    for i in c:
        Xi = X[:,np.where(y==i)[0]]
        meanClass = Xi.mean(axis=1)
        Sw = Sw + np.dot((Xi-meanClass), (Xi-meanClass).T)
        Sb = Sb + n * np.dot((meanClass - meanTotal).T, (meanClass - meanTotal).T)
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(Sw)*Sb)
    idx = np.argsort(-eigenvalues.real)
    eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:,idx]
    eigenvalues = np.array(eigenvalues[0:num_components].real, dtype=np.float32, copy=True)
    eigenvectors = np.array(eigenvectors[0:,0:num_components].real, dtype=np.float32, copy=True)
    return [eigenvalues, eigenvectors]
=======
#def pca(X, y, num_components=0):
#    [d, n] = X.shape
#    if (num_components <= 0) or (num_components>n):
#        num_components = n
#    mu = X.mean(axis=1).reshape((d,1))
#    
#    X = X - mu
#   
#    C = np.dot(X.T,X)
#    [eigenvalues,eigenvectors] = np.linalg.eigh(C)
#    eigenvectors = np.dot(X,eigenvectors)
#    for i in range(n):
#        eigenvectors[:,i] = eigenvectors[:,i]/np.linalg.norm(eigenvectors[:,i])
#	# or simply perform an economy size decomposition
#	# eigenvectors, eigenvalues, variance = np.linalg.svd(X.T, full_matrices=False)
#	# sort eigenvectors descending by their eigenvalue
#    idx = np.argsort(-eigenvalues)
#    eigenvalues = eigenvalues[idx]
#    eigenvectors = eigenvectors[:,idx]
#	# select only num_components
#    eigenvalues = eigenvalues[0:num_components].copy()
#    eigenvectors = eigenvectors[:,0:num_components].copy()
#    return [eigenvalues, eigenvectors, mu]

def pca(X_train, y_train, M):
	[d,n] = X_train.shape
	mu = X_train.mean(axis = 1).reshape(d,1)
	A = X_train - mu
	Se = (1 / n) * np.dot(A.T, A) #Returns a N*N matrix
	# Calculate eigenvalues `l` and eigenvectors `v`
	l, V = np.linalg.eig(Se)
	# Sort eigenvectors according to decreasing magnitude of eigenvalues
	idx = l.real.argsort()[::-1]   
	l = l[idx]
	V = V[:,idx]
	# Rescale eigenvectors
	_W = np.dot(A, V)
	# Normalise eigenvectors
	W = _W / np.apply_along_axis(np.linalg.norm, 0, _W)
	return [W, mu]
			
#def lda(X, y, num_components=0):
#	y = np.asarray(y)
#	[d,n] = X.shape
#	c = np.unique(y)
#	if (num_components <= 0) or (num_components>(len(c)-1)):
#		num_components = (len(c)-1)
#	meanTotal = X.mean(axis=1)
#	Sw = np.zeros((d, d), dtype=np.float32)
#	Sb = np.zeros((d, d), dtype=np.float32)
#	for i in c:
#		Xi = X[:,np.where(y==i)[0]]
#		meanClass = Xi.mean(axis=1).reshape((2576,1))
#		Sw = Sw + np.dot((Xi-meanClass), (Xi-meanClass).T)
#		Sb = Sb + n * np.dot((meanClass - meanTotal), (meanClass - meanTotal).T)
#	eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(Sw)*Sb)
#	idx = np.argsort(-eigenvalues.real)
#	eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:,idx]
#	eigenvalues = np.array(eigenvalues[0:num_components].real, dtype=np.float32, copy=True)
#	eigenvectors = np.array(eigenvectors[0:,0:num_components].real, dtype=np.float32, copy=True)
#	return [eigenvalues, eigenvectors]


def lda(x_train, y, num_components=0):
	d,n = x_train.shape
	mi = np.zeros((d,52))
	y = np.asarray(y)
	[d,n] = x_train.shape
	c = np.unique(y)
	if (num_components <= 0) or (num_components>(len(c)-1)):
		num_components = (len(c)-1)
	m = x_train.mean(axis=1).reshape((d,1))
	Sw = np.zeros((d, d), dtype=np.float32)
	Sb = np.zeros((d, d), dtype=np.float32)
	_ix = 0
	for c in range(0,52):
		xi = x_train[:,_ix:_ix+8]
		#2
		mi[:,c] = xi.mean(axis = 1)
		_mi = mi[:,c].reshape((d,1))
		#3
		Sw = Sw + np.dot((xi-_mi),(xi-_mi).T)
		#4
		Sb = Sb + np.dot((_mi - m),(_mi - m).T)
		_ix += 8
	print(f'rank(Sw) = {np.linalg.matrix_rank(Sw)}') #Sanity check, should be N -c 
	print(f'rank(Sb) = {np.linalg.matrix_rank(Sb)}') #Sanity check, should be c-1
	eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(Sw)*Sb)
	idx = np.argsort(-eigenvalues.real)
	eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:,idx]
	eigenvalues = np.array(eigenvalues[0:num_components].real, dtype=np.float32, copy=True)
	eigenvectors = np.array(eigenvectors[0:,0:num_components].real, dtype=np.float32, copy=True)
	return [eigenvalues, eigenvectors]
>>>>>>> e7e8eb998d94cae918f2c6dfa94ef038c0e014a0

def fisherfaces(X,y,num_components=0):
	y = np.asarray(y)
	[d,n] = X.shape
	c = len(np.unique(y))
<<<<<<< HEAD
	[eigenvalues_pca, eigenvectors_pca, mu_pca] = pca(X, y, (n-c))
	[eigenvalues_lda, eigenvectors_lda] = lda(project(eigenvectors_pca.T, X, mu_pca), y, num_components)
=======
	[ eigenvectors_pca, mu_pca] = pca(X, y, (n-c))
	w_proj = np.dot((X-mu_pca).T,eigenvectors_pca)
	[eigenvalues_lda, eigenvectors_lda] = lda(w_proj, y, num_components)
>>>>>>> e7e8eb998d94cae918f2c6dfa94ef038c0e014a0
	eigenvectors = np.dot(eigenvectors_pca,eigenvectors_lda)
	return [eigenvalues_lda, eigenvectors, mu_pca]

def asRowMatrix(X):
	if len(X) == 0:
		return np.array([])
	mat = np.empty((0, X[0].size), dtype=X[0].dtype)
	for row in X:
		mat = np.vstack((mat, np.asarray(row).reshape(1,-1)))
	return mat

def asColumnMatrix(X):
	if len(X) == 0:
		return np.array([])
	mat = np.empty((X[0].size, 0), dtype=X[0].dtype)
	for col in X:
		mat = np.hstack((mat, np.asarray(col).reshape(-1,1)))
	return mat


def scale_linear_bycolumn(rawpoints, high=100.0, low=0.0):
    mins = np.min(rawpoints, axis=0)
    maxs = np.max(rawpoints, axis=0)
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be removed by setting `normalize=False`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
