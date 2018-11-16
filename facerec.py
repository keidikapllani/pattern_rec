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

def split_load(ratio):
    '''
    Function to generate test and train sets keeping in class ratios
    '''
    data = io.loadmat('face.mat')
    data['X']
    # Images
    # N: number of images
    # D: number of pixels
    X = data['X']  # shape: [D x N]
    y = data['l']  # shape: [1 x N]
    
    test_id = []
    train_id = []
    pool = [i for i in range(0,10)]
    for c in range(0,52):
        _inclass_id = rnd.sample(pool,int(ratio*10))
        
        _train_id = [x + c*10 for x in _inclass_id]
  
        _test_id =[]
        for i in range(0,10):
            if i in set(_inclass_id):
                None
            else:
                _test_id.append(i+ c*10)
            
        
        test_id += _test_id
        train_id += _train_id
   
    print(test_id)
    x_train = X[:,train_id]
    y_train = y[:,train_id]
    x_test = X[:,test_id]
    y_test = y[:,test_id]
    return (x_train,y_train,x_test,y_test)

def project(W, X, mu=None):
	if mu is None:
		return np.dot(X.T,W)
	return np.dot((X - mu).T, W)

def reconstruct(W, Y, mu=None):
    if mu is None:
        return np.dot(Y,W.T)
    return np.dot(Y, W.T) + mu





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
			



def lda(x_train, y, num_components=0):
	d,n = x_train.shape
	mi = np.zeros((d,52))
	y = np.asarray(y)
	[d,n] = x_train.shape
	c = np.unique(y)
	if (num_components <= 0) or (num_components>(len(c)-1)):
		num_components = (len(c)-1)
	m = x_train.mean(axis=1).reshape((d,1))
	Sw = np.zeros((d, d), dtype=np.float64)
	Sb = np.zeros((d, d), dtype=np.float64)
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
	return  eigenvectors

def fisherfaces(X,y,num_components=0):
	y = np.asarray(y)
	[d,n] = X.shape
	c = len(np.unique(y))
	[ eigenvectors_pca, mu_pca] = pca(X, y, (n-c))
	w_proj = np.dot((X-mu_pca).T,eigenvectors_pca)
	[eigenvalues_lda, eigenvectors_lda] = lda(w_proj, y, num_components)
	eigenvectors = np.dot(eigenvectors_pca,eigenvectors_lda)
	return [eigenvalues_lda, eigenvectors, mu_pca]

def resample_w_pca(W,n0,k):
	d,n = W.shape
#	W0 = W[:,:n0]
	nrange = range(50,300,1)
	rn = rnd.sample(nrange,k)
	
	Wk= np.zeros((k,d,n0+max(rn)))
	for i in range(0,k):
		#Generate subspaces
		
		Wk[:,:,:n0] = W[:,:n0]
		rng = list(range(n0,n,1))
		indx = rnd.sample(rng,rn[i])
		Wk[i,:,n0:n0+rn[i]] = W[:,indx]
		
#		Wk[i,:,:] = np.append(W[:,indx])
	return Wk,[x + n0 for x in rn]

def resample_faces(X,Y,k):
	d,n = X.shape
	nrange = range(150,416,1)
	rn = rnd.sample(nrange,k)
	X_out= np.zeros((k,d,max(rn)))
	Y_out = np.zeros((k,max(rn)))
	for i in range(0,k):
		rng = list(range(0,n,1))
		indx = rnd.sample(rng,rn[i])
		indx.sort()
		X_out[i,:,:rn[i]] = X[:,indx]
		Y_out[i,:rn[i]] = Y[0,indx]
	return X_out,Y_out, rn

def lda_gen(x_train, y, num_components=0):
	d,n = x_train.shape
	label = np.unique(y)
	c = len(label)
	
	mi = np.zeros((d,c))
	y = np.asarray(y)
	
	if (num_components <= 0) or (num_components>(c-1)):
		num_components = (c-1)
		
	m = x_train.mean(axis=1).reshape((d,1))
	Sw = np.zeros((d, d), dtype=np.float64)
	Sb = np.zeros((d, d), dtype=np.float64)
	_ix = 0
	for i in range(0,c):
		xi = x_train[:,y == label[i]]
		#2
		mi[:,i] = xi.mean(axis = 1)
		_mi = mi[:,i].reshape((d,1))
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
	return  eigenvectors