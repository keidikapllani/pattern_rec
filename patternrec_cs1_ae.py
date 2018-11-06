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
    
#____________________________ Start PCA _____________________________________
#____________________________________________________________________________
    
### 1. MEAN FACE ###
    
#D ~ dimensionality, N ~ number of entries
D, N = x_train.shape
print(f'The dimensionality D of the data is {D} , while the datapoints N are {N}')

# Calculate mean face
meanface = face_data.mean(axis=1)
meanface = np.reshape(meanface,(D,1)) #To correct array shape
#Plot the mean face
plt.figure()
plt.imshow(np.reshape(meanface,(46,56)).T,cmap = 'gist_gray')
plt.title('Mean Face\n')

# Remove mean face from the train data
A = x_train - meanface #normalised training data D*N


#    plt.savefig('data/out/mean_face_eig_a.pdf',
#                format='pdf', dpi=1000, transparent=True)

### 2. NAIVE PCA ###

#Covariance matrix
S = (1 / N) * np.dot(A, A.T) # D*D matrix
print('dim S = ',S.shape)

#Eigenvalues and eigenvectors
wn, U = np.linalg.eig(S)
print('dim u = ', U.shape)
# ->   Here we find D eigenval and eigenvectors, however only N are non zero 


### 3. Efficient PCA ###
Se = (1 / N) * np.dot(A.T, A) #Returns a N*N matrix
print('dim Se = ', Se.shape)

# Calculate eigenvalues `w` and eigenvectors `v`
we, V = np.linalg.eig(Se)
# ->   Here we find only the N eigenval and eigenvectors that are non zero
_U = np.dot(A, V)
Ue = _U / np.apply_along_axis(np.linalg.norm, 0, _U) #normalise each eigenvector
print('dim ue = ',Ue.shape)

#Sort the eigenvalues based on their magnitude
w_n = sorted(abs(wn), reverse=True)     #naive
w_e = sorted(abs(we), reverse=True)    #efficient

# Plot eigenvalues for naive and efficient PCA
x_naive = np.arange(1,len(w_n)+1)
plt.figure()
plt.plot(x_naive,w_n)
x_effct = np.arange(1,len(w_e)+1)
plt.plot(x_effct,w_e)
plt.xlabel('$w_{m}$ ~ $m^{th}$ eigenvalue')
plt.ylabel('Re{$w_{m}$}')
plt.title('Ordered eigenvalues - naive and efficient PCA')
plt.legend(['Naive PCA', 'Efficient PCA'])

# Plot the first 10 eigenfaces from the naive PCA
plt.figure()
for i in range(0,10):
    plt.subplot(2, 5, i+1)
    plt.imshow(np.reshape(abs(U[:,i]),(46,56)).T,cmap = 'gist_gray')
    
# Plot the first 10 eigenfaces from the efficient PCA
plt.figure()
for i in range(0,10):
    plt.subplot(2, 5, i+1)
    plt.imshow(np.reshape(abs(Ue[:,i]),(46,56)).T,cmap = 'gist_gray')
    
#Apply SVD to eigenvectors of Se to find the eigenvectors of S
Ue = np.dot(A,V)


# Plot the first 10 eigenfaces from the efficient PCA
plt.figure()
for i in range(0,10):
    plt.subplot(2, 5, i+1)
    plt.imshow(np.reshape(abs(V[:,i]),(46,56)).T,cmap = 'gist_gray')

### Reconstruction error as function of number of eigenvalues
eigsum = abs(sum(wn[:,]))
csum = 0
tv = np.zeros((416,),float)
for m in range(0,416):
    csum = csum + wn[m]
    tv[m] = 100 - 100*csum/eigsum
    
# Plot reconstruction error as a function of the number of PCs
x_m = np.arange(1,416+1)
plt.figure()
plt.plot(x_m,tv)
plt.xlabel('Number of principal components $m$')
plt.ylabel('% reconstruction error')
plt.title('Reconstruction error \nas function of number of principal components')

#Reconstruct first face of training
#rec_face = np.zeros((len(face_data),1), dtype = int) # initialise
Wm = np.dot(A[:,1].T, abs(U)) #This shoud be a vector N*1
Wm = np.reshape(Wm,(2576,1))

wx = np.zeros((U.shape), dtype = int)
for k in range(0,416):
    wx[:,k] = abs(Wm[k]*abs(U[:,k]))
    #rec_face = np.sum(rec_face, abs(Wm[k]*abs(U[:,k])))
wx = wx.sum(axis = 1)    
rec_face  = sum(wx[:,], meanface[:,0])
plt.imshow(np.reshape(abs(rec_face),(46,56)).T,cmap = 'gist_gray')