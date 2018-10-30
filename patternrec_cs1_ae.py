# EE4_68 Pattern recognition coursework 1

# Import libraries
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random as rnd

# Load 
mat_content = sio.loadmat('face.mat')

mat_content # Let's see the content...
face_data = mat_content['X']
out_data = mat_content['l']

#Split train and test data
pt_train = 8
pt_test = 2
n_people = 52
population = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
x_train = face_data
#for each person split data
for test_ix in range(n_people):
    #generate random indexes
    r = rnd.sample(population,2);
    y_test = y_test + face_data


    test_ix = [test_ix (i-1)*10+r];

    



print(face_data) # Each column represents one face image, each row a pixel value for a particular coordinate of the image
print(face_data.shape)
face_157 = face_data[:,157]

print(face_157.shape)
print(face_157)
# face data is in 46x56 format

face_157 = np.reshape(face_157,(46,56))
plt.imshow(face_157, cmap = 'gist_gray')
face_157 = face_157.T
plt.imshow(face_157,cmap = 'gist_gray')

#Find mean face
meanface1 = face_data.mean(axis=1)
meanface=np.reshape(meanface1,(46,56))
meanface = meanface.T
plt.imshow(meanface,cmap = 'gist_gray')

#Remove mean face
X = face_data.T
y = out_data.T
# Split into a training set and a test set using a stratified k fold
# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

pca = PCA(n_components = 100)
pca.fit(X_train)
print(pca.explained_variance_ratio_)  
print(pca.singular_values_)  


#Perform PCA
#pca = PCA(n_components=2)
#pca.fit(face_data)
#PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
#svd_solver='auto', tol=0.0, whiten=False)
#print(pca.explained_variance_ratio_)  
