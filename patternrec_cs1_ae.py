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
import time

### Load data
mat_content = sio.loadmat('face.mat')
face_data = mat_content['X']
out_data = mat_content['l']

### Split train and test data
pt_train = 8
pt_test = 2

n_people = 52
face_range = list(range(0,10,1))

#Initialise train and test matrices
n_train = int(face_data.shape[1]*pt_train/10)
n_test  = int(face_data.shape[1]*pt_test/10)
x_test  = np.zeros((len(face_data),n_test), dtype = int)
#x_train = np.zeros((len(face_data),n_train), dtype = int)
y_test  = np.zeros((len(out_data),n_test), dtype = int)
#y_train = np.zeros((len(out_data),n_train), dtype = int)
x_train = face_data
y_train = out_data
#Initialise counter to build output matrices
ix = 0
ix2= 0
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
    
#______________________________ Start PCA _____________________________________
#______________________________________________________________________________
    
### MEAN FACE__________________________________________________________________
    
#D ~ dimensionality, N ~ number of entries
D, N = x_train.shape
print(f'The dimensionality D of the data is {D} , while the datapoints N are {N}')

# Calculate mean face
meanface = face_data.mean(axis=1)
meanface = np.reshape(meanface,(D,1)) #To correct array shape
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
print('dim u = ', U.shape)
# ->   Here we find D eigenval and eigenvectors, however only N are non zero 

### PCA WITH SVD OPTIMISATION__________________________________________________
Se = (1 / N) * np.dot(A.T, A) #Returns a N*N matrix
print('dim Se = ', Se.shape)

# Calculate eigenvalues `w` and eigenvectors `v`
we, V = np.linalg.eig(Se)
# ->   Here we find only the N eigenval and eigenvectors that are non zero
_U = np.dot(A, V)
Ue = _U / np.apply_along_axis(np.linalg.norm, 0, _U) #normalise each eigenvector
print('dim ue = ',Ue.shape)

#Sort the eigenvalues based on their magnitude
w_n = sorted(np.real(wn), reverse=True)     #naive
w_e = sorted(np.real(we), reverse=True)    #efficient

# Plot the first 10 eigenfaces from the naive PCA
for i in range(0,10):
    plt.subplot(2, 5, i+1)
    plt.imshow(np.reshape(np.real(U[:,i]),(46,56)).T,cmap = 'gist_gray')
    
# Plot the first 10 eigenfaces from the efficient PCA
for i in range(0,10):
    plt.subplot(2, 5, i+1)
    plt.imshow(np.reshape(np.real(Ue[:,i]),(46,56)).T,cmap = 'gist_gray')
    
#Apply SVD to eigenvectors of Se to find the eigenvectors of S
Ue = np.dot(A,V)

# Plot the first 10 eigenfaces from the efficient PCA
for i in range(0,10):
    plt.subplot(2, 5, i+1)
    plt.imshow(np.reshape(np.real(V[:,i]),(46,56)).T,cmap = 'gist_gray')    

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
W_train = np.dot(A.T, np.real(U)).T
#Reconstruct
rec_train_face  = np.dot(np.real(U),W_train) + meanface

### TEST FACE RECONSTRUCTION __________________________________________________
#1.Remove training mean from test image
#2.Project onto the egenspace a_i = x_n.T u_i
#3.Represent the projection vector as w = [a1,a2,...aM].T
#4.Group projections in the matrix W

Phi = x_test - meanface
W_test = np.dot(Phi.T, np.real(U)).T #This shoud be a vector N*1
rec_test_face = np.dot(np.real(U),W_test) + meanface

### RECONSTRUCTION ERROR J AS FUNCTION OF M ___________________________________
#Initialise variables
J_theor = np.zeros((416,),float)
J_train = np.zeros((416,416),float)
J_test  = np.zeros((416,104),float)
eigsum 	= sum(w_n) #Total sum of the eigenvalues

#Vary M from 0 to N
for m in range(0,416):
	#Reconstruct train set using m PCs
	recon_train = np.dot(np.real(U[:,:m]),W_train[:m,:]) + meanface
	#Reconstruct test set using m PCs
	recon_test = np.dot(np.real(U[:,:m]),W_test[:m,:]) + meanface
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
		
### RECOGNITION WITH NN ###____________________________________________________
k = 1 # number of neighbors (hyperparameter)
# Generate KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors = k)
# Train the classifier
knn_classifier.fit(x_train, y_train)

# Classify the test data
y_pred = classifier.predict(X_test)  

# Classification metrics
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))


### KNN Classifier accuracy as function of hyperparameters M and K
accuracy = np.zeros((5,416),float)
t_compression = []
t_train = np.zeros((5,416),float)
t_test = np.zeros((5,416),float)
# Increment M
for m in range(0,416):
	#Measure the data compression time
	_t_comp = time.time()
	#Reconstruct test set using m PCs
	recon_test = np.dot(np.real(U[:,:m]),W_test[:m,:]) + meanface
	t_compression.append([time.time() - _t_comp])
	# Increment K
	for k in range(1, 6):
		# Train classifier
		_t_train = time.time()
		knn = KNeighborsClassifier(n_neighbors = k)
		knn.fit(x_train.T, y_train.T)
		# Measure training time
		t_train[k-1,m] = time.time() - _t_train
		
		# Classification of test data
		_t_test = time.time()
		y_knn = knn.predict(recon_test.T)
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
Us = np.zeros((2576,8,52),float)
meanface_s = np.zeros((2576,52),float)
ix = 0
#For each class
for c in range(1,52):
	_As = x_train[:,ix:ix+8]
	ix += 8
	meanface_s[:,c] = _As.mean(axis = 1)
	As = _As - np.reshape(meanface_s[:,c],(2576,1))
	
	Ss = (1 / 8) * np.dot(As.T, As) #Returns a N*N matrix
	ls, _vs = np.linalg.eig(Ss)
	_Us = np.dot(As, _vs)
	Us[:,:,c] = _Us / np.apply_along_axis(np.linalg.norm, 0, _Us)
	
	
# Reconstruct a train face to check subspace generation
_ws = np.dot((x_train[:,5] - meanface_s[:,1]).T, np.real(Us[:,:,1])) #This shoud be a vector N*1
ws = np.reshape(_ws,(8,1))

#Find the weighted combination of eigenfaces
rec_face_s = np.dot(np.real(Us[:,:,1]),ws) + np.reshape(meanface_s[:,1],(2576,1))

plt.imshow(np.reshape(np.real(rec_face_s),(46,56)).T,cmap = 'gist_gray')	

#!!! Reconstruction error due to mistake in code somewhere