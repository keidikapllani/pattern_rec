import numpy as np
from sklearn.decomposition import PCA
import scipy.io as sio
import matplotlib.pyplot as plt

mat_content = sio.loadmat('assests/face.mat')

mat_content # Let's see the content...
face_data = mat_content['X']
face_157 = face_data[:,100]
face_157 = np.reshape(face_157,(46,56))

plt.imshow(face_157.T, cmap = 'gist_gray')

