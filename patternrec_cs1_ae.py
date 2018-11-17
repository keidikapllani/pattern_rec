# EE4_68 Pattern recognition coursework 1
# Antonio Enas, Keidi Kapllani
#
# 30/10/2018
#______________________________________________________________________________

# Import libraries
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from numpy import linalg as LA
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.preprocessing import normalize
import time
import sys
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import os,psutil
from facerec import *


### Load data
x_train,y_train,x_test,y_test = load_data()

### BASELINE with KNN
# 1. Generate and train KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors = 1,algorithm = 'brute')
knn_classifier.fit(x_train.T, y_train.T)
# 2. Classify the test data
y_pred = knn_classifier.predict(x_test.T)  
accuracy_knn = 100*accuracy_score(y_test.T, y_pred)
alfa = knn_classifier.score(x_test.T,y_test.T)

 
#______________________________ Start PCA _____________________________________
#______________________________________________________________________________
    
### MEAN FACE__________________________________________________________________
    
#D ~ dimensionality, N ~ number of entries
D, N = x_train.shape
print(f'The dimensionality D of the data is {D} , while the datapoints N are {N}')

# Calculate mean face
meanface = x_train.mean(axis=1).reshape((D,1))

#Plot the mean face
plt.imshow(np.reshape(meanface,(46,56)).T,cmap = 'gist_gray')
plt.title('Mean Face\n')

# Remove mean face from the train data
A = x_train - meanface #normalised training data D*N

### NAIVE PCA__________________________________________________________________
#Covariance matrix
S = (1 / N) * np.dot(A, A.T) # D*D matrix
print('dim S = ',S.shape)

#Eigenvalues and eigenvectors
wn, U = np.linalg.eig(S)
U = np.real(U)
print('dim u = ', U.shape)
# ->   Here we find D eigenval and eigenvectors, however only N are non zero 

### PCA WITH SVD OPTIMISATION__________________________________________________
Se = (1 / N) * np.dot(A.T, A) #Returns a N*N matrix
print('dim Se = ', Se.shape)

# Calculate eigenvalues `w` and eigenvectors `v`
we, V = np.linalg.eig(Se)
# ->   Here we find only the N eigenval and eigenvectors that are non zero
Unot = np.dot(A, V)
Ue = Unot/ np.apply_along_axis(np.linalg.norm, 0, Unot) #normalise each eigenvector
#Ue = normalize(Usss,axis = 0, norm = 'l2')
print('dim ue = ',Ue.shape)

#Sort the eigenvalues based on their magnitude
w_n = sorted(np.real(wn), reverse=True)     #naive
w_e = sorted(np.real(we), reverse=True)    #efficient

# Plot the first 10 eigenfaces from the naive PCA
for i in range(0,10):
    plt.subplot(2, 5, i+1)
    plt.imshow(np.reshape(np.real(U[:,i]),(46,56)).T,cmap = 'gist_gray')
    
# Plot the first 10 eigenfaces from the efficient PCA
plt.figure()
for i in range(0,10):
    plt.subplot(2, 5, i+1)
    plt.imshow(np.reshape(np.real(Ue[:,i]),(46,56)).T,cmap = 'gist_gray')
    
#Apply SVD to eigenvectors of Se to find the eigenvectors of S
#Ue = np.dot(A,V)

# Plot the first 10 eigenfaces from the efficient PCA
#for i in range(0,10):
#    plt.subplot(2, 5, i+1)
#    plt.imshow(np.reshape(np.real(V[:,i]),(46,56)).T,cmap = 'gist_gray')    

#Reconstruct first face of training
#rec_face = np.zeros((len(face_data),1), dtype = int) # initialise
Wm = np.dot(A[:,1].T, np.real(U)) #This shoud be a vector N*1
Wm = np.reshape(Wm,(2576,1))

#Find the weighted combination of eigenfaces
partial_face = np.dot(np.real(U),Wm)
# Add back the mean	
rec_face  = partial_face + meanface
plt.imshow(np.reshape(np.real(rec_face),(46,56)).T,cmap = 'gist_gray')


### TRAIN SET RECONSTRUCTION __________________________________________________
#Determine face projection in the eigenspace
W_train = np.dot(A.T, np.real(Ue)).T
#Reconstruct
rec_train_face  = np.dot(np.real(Ue),W_train) + meanface

### TEST FACE RECONSTRUCTION __________________________________________________
#1.Remove training mean from test image
#2.Project onto the egenspace a_i = x_n.T u_i
#3.Represent the projection vector as w = [a1,a2,...aM].T
#4.Group projections in the matrix W

Phi = x_test - meanface
W_test = np.dot(Phi.T, np.real(Ue)).T #This shoud be a vector N*1
rec_test_face = np.dot(np.real(Ue),W_test) + meanface

### RECONSTRUCTION ERROR J AS FUNCTION OF M ___________________________________
#Initialise variables
J_theor = np.zeros((416,),float)
J_train = np.zeros((416,416),float)
J_test  = np.zeros((416,104),float)
eigsum 	= sum(w_n) #Total sum of the eigenvalues

#Vary M from 0 to N
for m in range(0,416):
	#Reconstruct train set using m PCs
	recon_train = np.dot(np.real(Ue[:,:m]),W_train[:m,:]) + meanface
	#Reconstruct test set using m PCs
	recon_test = np.dot(np.real(Ue[:,:m]),W_test[:m,:]) + meanface
	#Theoretical reconstruction error
	J_theor[m] = (eigsum - sum(w_n[:m]))**0.5
	#Train reconstruction error for each face
	for i in range(0,416):
		J_train[m,i] = LA.norm(x_train[:,i] - recon_train[:,i])
	#Test reconstruction error for each face
	for i in range(0,104):
		J_test[m,i] = LA.norm(x_test[:,i] - recon_test[:,i])
#Average test and train errors	
J_train = J_train.mean(axis = 1)	
J_test = J_test.mean(axis = 1)

#Plot the results	
x_m = np.arange(1,417)
plt.plot(x_m,J_theor, linewidth=6, color= '#00ff00ff')
plt.plot(x_m,J_train, linewidth=3, color= '#0055ff')
plt.plot(x_m,J_test, linewidth=3, color= '#ff5500')
plt.xlabel('$M$ ~ Number of principal components', fontsize = 14)
plt.ylabel('$J$ ~ Reconstruction error', fontsize = 14)
plt.title('Reconstruction error \nas function of number of principal components'
		  , fontsize = 16)
plt.legend(['Theoretical', 'Training set', 'Test set'], fontsize = 14)
plt.tight_layout()

### RECOGNITION WITH KNN ###___________________________________________________

# 1. Projection onto subspace
x_train_pca = np.dot((x_train-meanface).T,Ue[:,:312])
x_test_pca = np.dot((x_test-meanface).T,Ue[:,:312])
# 2. Generate and train KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors = 1)
knn_classifier.fit(x_train_pca, y_train.T)

# 3. Classify the test data
y_pred = knn_classifier.predict(x_test_pca)  
accuracy = 100*accuracy_score(y_test.T, y_pred)


### KNN Classifier accuracy as function of hyperparameters M and K ____________
accuracy = np.zeros((3,416),float)
t_compression = []
t_train = np.zeros((3,416),float)
t_test = np.zeros((3,416),float)

# Increment M
for m in range(1,416):
	#Project onto the eigenspace and measure the projection time
	_t_comp = time.time()
	x_train_pca = np.dot(x_train.T,Ue[:,:m])
	x_test_pca = np.dot(x_test.T,Ue[:,:m])
	t_compression.append([time.time() - _t_comp])
	
	# Increment K
	for k in range(1, 4):
		# Train classifier
		_t_train = time.time()
		knn = KNeighborsClassifier(n_neighbors = k)
		knn.fit(x_train_pca, y_train.T)
		# Measure training time
		t_train[k-1,m] = time.time() - _t_train
		
		# Classification of test data
		_t_test = time.time()
		y_knn = knn.predict(x_test_pca)
		accuracy[k-1, m] = 100*accuracy_score(y_test.T, y_knn)
		# Measure classification time
		t_test[k-1,m] = time.time() - _t_test


#Plot accuracy vs hyperparameters
plt.imshow(accuracy,aspect = 'auto',cmap = 'RdYlGn')
cb = plt.colorbar()
cb.set_label('$\%$ Accuracy',fontsize=14)
plt.xlabel('$M$ ~ Number of principal components', fontsize = 14)
plt.ylabel('$k$ neighbours', fontsize = 14)
plt.title('KNN Classifier accuracy\nas function of the hyperparameters'
		  , fontsize = 16)


# pct memory usage
#memory.append(psutil.Process(os.getpid()).memory_percent())


### RECOGNITION WITH MINIMUM SUBSPACE RECONSTRUCTION ERROR ###_________________
#Initialise variables
Wsub = np.zeros((2576,8,52),float) 	   #Eigenvector matrices for each class
meanface_s = np.zeros((2576,52),float) #Meanfaces for each class
ls = np.zeros((8,52),float)

ix = 0
# Training time start
_trt_subs = time.time()
#For each class
for c in range(0,52):
	_As = x_train[:,ix:ix+8] #Class subspace training set
	ix += 8
	
	meanface_s[:,c] = _As.mean(axis = 1) #Class mean
	As = _As - np.reshape(meanface_s[:,c],(2576,1))
	
	#Find subspace eigenvector matrix
	Ss = (1 / 8) * np.dot(As.T, As) #Returns a Nc*Nc matrix, Nc = 8
	_ls, _vs = np.linalg.eig(Ss)
	#Sort the eigenvalues and eigenvectors
	idx = _ls.real.argsort()[::-1]   
	ls[:,c] = _ls[idx]
	vs = _vs[:,idx]
	_Wsub = np.dot(As, vs)
	Wsub[:,:,c] = _Wsub / np.apply_along_axis(np.linalg.norm, 0, _Wsub)
# Training time end
time_train_sub = time.time() - _trt_subs
	
#Plot the theoretical subspace reconstruction error____________________________
sum_ls = ls.sum(axis = 0)
pct_ls = 100* ls/sum_ls
mean_ls  = pct_ls.mean(axis = 1)
min_ls = pct_ls[:,pct_ls[0,:] == np.min(pct_ls[0,:])]
max_ls = pct_ls[:,pct_ls[0,:] == np.max(pct_ls[0,:])]
J_subs = np.zeros((9,),float)
J_s_min = np.zeros((9,),float)
J_s_max = np.zeros((9,),float)
for m in range(0,9):
	J_subs[m] = 100 - sum(mean_ls[:m])
	J_s_min[m]= 100 - sum(min_ls[:m])
	J_s_max[m]= 100 - sum(max_ls[:m])

plt.figure()	
plt.plot(J_subs, linewidth=3, color= '#0055ff')
plt.plot(J_s_max, linewidth=3, color= '#00ff00ff')
plt.plot(J_s_min, linewidth=3, color= '#ff5500')
plt.xlabel('$M_{c}$ ~ Principal components per class', fontsize = 14)
plt.ylabel('$J_{\%}$ ~ Reconstruction error', fontsize = 14)
plt.title('Theoretical subspace reconstruction error \nas function of number of principal components'
		  , fontsize = 16)
plt.legend(['Mean', 'Good class','Bad class'], fontsize = 14)
plt.tight_layout()
#______________________________________________________________________________


# Reconstruct a train face to check subspace generation
_ws = np.dot((x_train[:,1] - meanface_s[:,1]).T, np.real(Wsub[:,:,1])) #This shoud be a vector N*1
ws = np.reshape(_ws,(8,1))

#Find the weighted combination of eigenfaces
rec_face_s = np.dot(np.real(Wsub[:,:,1]),ws) + np.reshape(meanface_s[:,1],(2576,1))

plt.imshow(np.reshape(np.real(rec_face_s),(46,56)).T,cmap = 'gist_gray')	

# MINIMUM RECONSTRUCTION ERROR CLASSIFIER______________________________________
N_t = 104
accuracy_s = np.zeros((8,1),float)
time_test_sub = np.zeros((8,100))
mem = []
for time_test in range(0,10):
	
	for mc in range(0,8):
		memory = []
		# Test time start
		_tst_sub = time.time()
		Js_test = np.zeros((52,N_t))
		for c in range(0,52):
			#Remove the meanface
			Phi_s = x_test - np.reshape(meanface_s[:,c],(2576,1))
			#Create the projection vectors
			ws_test = np.dot(Phi_s.T, np.real(Wsub[:,:mc,c])).T
			#Reconstruct test set using m = 8 PCs
			recon_test_s = np.dot(np.real(Wsub[:,:mc,c]),ws_test[:,:]) + np.reshape(meanface_s[:,c],(2576,1))
			#Test reconstruction error for each face
			for i in range(0,N_t):
				Js_test[c,i] = LA.norm(x_test[:,i] - recon_test_s[:,i])	
		#Classifier to minimise the reconstruction error
		y_subs = np.argmin(Js_test,axis = 0) +1
		# Test time end
		time_test_sub[mc,time_test] = time.time() - _tst_sub
		#Overall accuracy
		accuracy_s[mc] = 100*accuracy_score(y_test.T, y_subs)
		memory.append(psutil.Process(os.getpid()).memory_percent())
	mem += memory
		
#Plot the mean accuracy vs Mc
ind = np.arange(len(accuracy_s)) + 1
plt.figure()
plt.plot(ind,accuracy_s, linewidth=3, color= '#0055ff')
plt.xlabel('$M_{c}$ ~ Principal components per class', fontsize = 14)
plt.ylabel('$\%$ Accuracy', fontsize = 14)
plt.title('Mean accuracy \nas function of number of principal components'
		  , fontsize = 16)
plt.tight_layout()

#Plot the average test time vs Mc relationship ________________________________
mean_test_time = time_test_sub.mean(axis = 1)
mean_memory = mem/10

fig, ax1 = plt.subplots()
ax1.plot(ind, mean_test_time, 'b-')
ax1.set_xlabel('$M_{c}$ ~ In-class principal components used', fontsize = 14)
ax1.set_ylabel('Mean test time', color='b', fontsize = 14)
ax1.tick_params('y', colors='b')
ax2 = ax1.twinx()
ax2.plot(ind, mem[:8], 'r.')
ax2.set_ylabel('Memory usage', color='r', fontsize = 14)
ax2.tick_params('y', colors='r')
fig.tight_layout()
#______________________________________________________________________________


#Confusion matrix
	
plt.imshow(Js_test,aspect = 'auto')
cb = plt.colorbar()

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test.T, y_subs)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[i for i in range(1,53)], normalize=True,
                      title='Normalized confusion matrix')
