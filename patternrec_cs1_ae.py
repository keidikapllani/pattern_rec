# EE4_68 Pattern recognition coursework 1

# Import libraries
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# Load 
mat_content = sio.loadmat('face.mat')

mat_content # Let's see the content...
face_data = mat_content['X']

print(face_data) # Each column represents one face image, each row a pixel value for a particular coordinate of the image
print(face_data.shape)
face_157 = face_data[:,157]

print(face_157.shape)
print(face_157)
# face data is in 46x56 format

face_157 = np.reshape(face_157,(46,56))
#keidikeidi
plt.imshow(face_157, cmap = 'gist_gray')
face_157 = face_157.T
plt.imshow(face_157,cmap = 'gist_gray')

meanface1 = face_data.mean(axis=1)
meanface=np.reshape(meanface1,(46,56))
meanface = meanface.T
plt.imshow(meanface,cmap = 'gist_gray')
