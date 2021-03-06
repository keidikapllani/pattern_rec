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
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as ldah

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
    data = sio.loadmat('face.mat')
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
    return [W[:,0:M], mu]
			



def lda(x_train, y, num_components):
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

def fisherfaces(X,y,num_comp_pca, num_com_lda):
    y = np.asarray(y)
    [d,n] = X.shape
    c = len(np.unique(y))
    [ eigenvectors_pca, mu_pca] = pca(X, y, num_comp_pca)
    w_proj = np.dot((X-mu_pca).T,eigenvectors_pca)
    kd = ldah(n_components=num_com_lda, priors=None, shrinkage=None, solver='svd', store_covariance=True)
    kd.fit(w_proj,y.T)
    eigenvectors_lda = kd.scalings_[:,:num_com_lda]
    eigenvectors = np.dot(eigenvectors_pca,eigenvectors_lda)
    return [ eigenvectors, mu_pca]

def resample_w_pca(W,n0,k):
	d,n = W.shape
#	W0 = W[:,:n0]
	nrange = range(n0,n-n0,1)
	rn = np.random.choice(nrange,k)
	
	Wk= np.zeros((k,d,n0+max(rn)))
	for i in range(0,k):
		#Generate subspaces
		
		Wk[:,:,:n0] = W[:,:n0]
		rng = list(range(n0,n,1))
		while rn[i]>len(rng):
			rn[i]-=1
		indx = rnd.sample(rng,rn[i])
		Wk[i,:,n0:n0+rn[i]] = W[:,indx]
		
#		Wk[i,:,:] = np.append(W[:,indx])
	return Wk,[x + n0 for x in rn]

def resample_faces(X,Y,n0,k):
	d,n = X.shape
	nrange = range(n0,n,1)
	rn = np.random.choice(nrange,k)
	X_out= np.zeros((k,d,max(rn)+1))
	Y_out = np.zeros((k,max(rn)+1))
	for i in range(0,k):
		rng = list(range(0,n,1))
		#Avoid singularity of kNN, when n_samples==n_features
		indx = rnd.sample(rng,rn[i])
		while len(indx)==len(np.unique(Y[0,indx])):
			rn[i] += 1
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


def pca_classifier(x_train,y_train,x_test,M):
	'''
	Alternative PCA classifier based on minimum reconstruction error
	'''
	
	d_tst,n_tst = x_test.shape 	#Dimensions of the test set
	label = np.unique(y_train)	#Set of classes
	cln = len(label)			#Number of classes
	
	#TRAIN SUBSPACE PCA
	Wsub = np.zeros((d_tst,8,cln),float) 	   #Eigenvector matrices for each class
	meanface_s = np.zeros((d_tst,cln),float) #Meanfaces for each class
	ls = np.zeros((8,cln),float)

	ix = 0
	for c in range(0,cln):
		_As = x_train[:,ix:ix+8] #Class subspace training set
		ix += 8
		
		meanface_s[:,c] = _As.mean(axis = 1) #Class mean
		As = _As - np.reshape(meanface_s[:,c],(d_tst,1))
		
		#Find subspace eigenvector matrix
		Ss = (1 / 8) * np.dot(As.T, As) #Returns a Nc*Nc matrix, Nc = 8
		_ls, _vs = np.linalg.eig(Ss)
		#Sort the eigenvalues and eigenvectors
		idx = _ls.real.argsort()[::-1]   
		ls[:,c] = _ls[idx]
		vs = _vs[:,idx]
		_Wsub = np.dot(As, vs)
		Wsub[:,:,c] = _Wsub / np.apply_along_axis(np.linalg.norm, 0, _Wsub)
	
	#FIT
	Js_test = np.zeros((cln,n_tst))
	for c in range(0,cln):
		#Remove the meanface
		Phi_s = x_test - np.reshape(meanface_s[:,c],(d_tst,1))
		#Create the projection vectors
		ws_test = np.dot(Phi_s.T, np.real(Wsub[:,:M,c])).T
		#Reconstruct test set using m = 8 PCs
		recon_test_s = np.dot(np.real(Wsub[:,:M,c]),ws_test[:,:]) + np.reshape(meanface_s[:,c],(d_tst,1))
		#Test reconstruction error for each face
		for i in range(0,n_tst):
			Js_test[c,i] = np.linalg.norm(x_test[:,i] - recon_test_s[:,i])	
	
	#Classifier to minimise the reconstruction error
	y_predict = np.argmin(Js_test,axis = 0) +1
		
	return y_predict


def knn(x_train,y_train,x_test,y_test):
	d,n = x_train.shape
	dt,nt = x_test.shape
	distance = np.zeros((n,))
	y_knn = np.zeros((1,nt))
	# for each test face
	for i in range(0,nt):
		# measure euclidean distance with each train vector
		for j in range(0,n):
			distance[j] = np.linalg.norm(x_test[:,i]-x_train[:,j])
		idx = np.argmin(distance)
		
		y_knn[0,i] = y_train[0,idx]
	accuracy = accuracy_score(y_knn.T, y_test.T)
	return y_knn,accuracy

def maj_voting(Y_models,y_train,y_test):
	'''
	Majority voting fusion technique. For tie breaks choose random.
	Y_ensamble.shape = (n_votes,n_components)
	'''
	N,T = Y_models.shape
	# Identify classes
	label = np.unique(y_train)
	C = len(label)
    
	y_ensemble = np.zeros((N,))
	#for each test
	for i in range(0,N):
		sum = np.zeros((C,))
		#for each class
		for c in range(1,C+1):
			#for each model
			for t in range(0, T):
				if Y_models[i,t] == c:
					sum[c-1] += 1  
		y_ensemble[i] = np.argmax(sum)+1     
    
	accuracy_ens = 100 * np.sum(y_test.ravel() == y_ensemble) / 104

	return y_ensemble,accuracy_ens


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
    plt.title(title, fontsize = 14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize = 2)
    plt.yticks(tick_marks, classes, fontsize = 2)
	
    plt.ylabel('True label', fontsize = 14)
    plt.xlabel('Predicted label', fontsize = 14)
	