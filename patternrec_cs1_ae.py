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
    print(r)    
    x_train = np.delete(x_train,r[0]-ix,1)
    x_train = np.delete(x_train,r[1]-ix,1)
    y_train = np.delete(y_train,r[0]-ix,1)
    y_train = np.delete(y_train,r[1]-ix,1)
#   
#        
        
    ix = ix + 2
    
#    
#
#
#
#print(face_data) # Each column represents one face image, each row a pixel value for a particular coordinate of the image
#print(face_data.shape)
#face_157 = face_data[:,157]
#
#print(face_157.shape)
#print(face_157)
## face data is in 46x56 format
#
#face_157 = np.reshape(face_157,(46,56))
#plt.imshow(face_157, cmap = 'gist_gray')
#face_157 = face_157.T
#plt.imshow(face_157,cmap = 'gist_gray')
#
##Find mean face
meanface1 = face_data.mean(axis=1)
meanface=np.reshape(meanface1,(46,56))
meanface = meanface.T
plt.imshow(meanface,cmap = 'gist_gray')
#
##Remove mean face
X = face_data.T - meanface1
y = out_data.T 

pca = PCA()
pca.fit(X)
var_ration = pca.explained_variance_ratio_
singular_val = pca.singular_values_  

plt.plot(singular_val.T)
plt.show()
#Perform PCA
#pca = PCA(n_components=2)
#pca.fit(face_data)
#PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
#svd_solver='auto', tol=0.0, whiten=False)
#print(pca.explained_variance_ratio_)  
