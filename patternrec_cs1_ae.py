# EE4_68 Pattern recognition coursework 1
# Antonio Enas, Keidi Kapllani
#
# 30/10/2018
#______________________________________________________________________________

# Import libraries
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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
    
#D ~ dimensionality, N ~ number of entries
D, N = x_train.shape
print('The dimensionality of the data is {} , while the datapoint N are =%d', D,N)

# Calculate mean face
meanface = face_data.mean(axis=1)
#Plot the mean face
plt.imshow(np.reshape(meanface,(46,56)).T,cmap = 'gist_gray')
plt.title('Mean Face\n')

##Remove mean face
A = x_train.T - meanface
y = out_data.T 
#    plt.savefig('data/out/mean_face_eig_a.pdf',
#                format='pdf', dpi=1000, transparent=True)


S = (1 / N) * np.dot(A.T, A)
print('dim S = ',S.shape)
#Eigenvalues and eigenvectors
_w, _v = np.linalg.eig(S)
_u = np.dot(A, _v)
print('dim u = ',_u.shape)

### Efficient PCA, Se ~ N*N matrix
Se = (1 / N) * np.dot(A, A.T)
print('dim Se = ', Se.shape)

# Calculate eigenvalues `w` and eigenvectors `v`
_we, _ve = np.linalg.eig(Se)
_ue = np.dot(A.T, _ve)
print('dim ue = ',_ue.shape)
#Sort the eigenvalues based on their magnitude
ws = sorted(abs(_w), reverse=True)

#print(ws)

### PCA LIBRARY TEST ###
#pca = PCA()
#pca.fit(A)
#print(pca.explained_variance_ratio_)  
#print(pca.singular_values_)  


#Perform PCA
#pca = PCA(n_components=2)
#pca.fit(face_data)
#PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
#svd_solver='auto', tol=0.0, whiten=False)
#print(pca.explained_variance_ratio_)  
