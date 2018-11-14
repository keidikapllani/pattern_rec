# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 16:12:28 2018

@author: AE
"""


# Import libraries
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from numpy import linalg as LA
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Import data
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
y_test  = np.zeros((len(out_data),n_test), dtype = int)
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
	


#### LDA WITH SKLEARN	

sklearn_lda = LDA(n_components=51)
X_lda_sklearn = sklearn_lda.fit_transform(x_train.T, y_train.T)
Wlda = sklearn_lda.scalings_
X_test_lda = np.dot(Wlda.T,x_test)
X_train_lda = np.dot(x_train.T,Wlda)

X_test_lda = sklearn_lda.transform(x_test.T)
knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(X_lda_sklearn, y_train.T)
y_knn = knn.predict(X_test_lda)
accuracy = 100*accuracy_score(y_test.T, y_knn)


#def plot_scikit_lda(X, title):
#
#    ax = plt.subplot(111)
#    for label,marker,color in zip(
#        range(1,4),('^', 's', 'o'),('blue', 'red', 'green')):
#
#        plt.scatter(x=X[:,0][y == label],
#                    y=X[:,1][y == label] * -1, # flip the figure
#                    marker=marker,
#                    color=color,
#                    alpha=0.5,
#                    label=label_dict[label])
#
#    plt.xlabel('LD1')
#    plt.ylabel('LD2')
#
#    leg = plt.legend(loc='upper right', fancybox=True)
#    leg.get_frame().set_alpha(0.5)
#    plt.title(title)
#
#    # hide axis ticks
#    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
#            labelbottom="on", left="off", right="off", labelleft="on")
#
#    # remove axis spines
#    ax.spines["top"].set_visible(False)  
#    ax.spines["right"].set_visible(False)
#    ax.spines["bottom"].set_visible(False)
#    ax.spines["left"].set_visible(False)    
#
#    plt.grid()
#    plt.tight_layout
#    plt.show()
	
#def plot_step_lda():
#
#    ax = plt.subplot(111)
#    for label,marker,color in zip(
#        range(1,4),('^', 's', 'o'),('blue', 'red', 'green')):
#
#        plt.scatter(x=X_lda[:,0].real[y == label],
#                y=X_lda[:,1].real[y == label],
#                marker=marker,
#                color=color,
#                alpha=0.5,
#                label=label_dict[label]
#                )
#
#    plt.xlabel('LD1')
#    plt.ylabel('LD2')
#
#    leg = plt.legend(loc='upper right', fancybox=True)
#    leg.get_frame().set_alpha(0.5)
#    plt.title('LDA: Iris projection onto the first 2 linear discriminants')
#
#    # hide axis ticks
#    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
#            labelbottom="on", left="off", right="off", labelleft="on")
#
#    # remove axis spines
#    ax.spines["top"].set_visible(False)  
#    ax.spines["right"].set_visible(False)
#    ax.spines["bottom"].set_visible(False)
#    ax.spines["left"].set_visible(False)    
#
#    plt.grid()
#    plt.tight_layout
#    plt.show()
#
#plot_step_lda()
#plot_scikit_lda(X_lda_sklearn, title='Default LDA via scikit-learn')

label = 1
plt.scatter(x=X_lda_sklearn[:,0],y=X_lda_sklearn[:,1])

plt.figure()
ix = 0
for i in range(0,51):
	plt.scatter(x=X_lda_sklearn[:,0][ix:ix+8],y=X_lda_sklearn[:,1][ix:ix+8])
	ix += 8

### KNN CLASSIFIER ____________________________________________________________
Wm = np.dot(x_train.T, W_lda)

accuracy = np.zeros((3,416),float)
t_compression = []
t_train = np.zeros((3,416),float)
t_test = np.zeros((3,416),float)
# Increment M
for m in range(0,416):
	#Measure the data compression time
	_t_comp = time.time()
	#Reconstruct test set using m PCs
	recon_test = np.dot(np.real(U[:,:m]),W_test[:m,:]) + meanface
	t_compression.append([time.time() - _t_comp])
	# Increment K
	for k in range(1, 3):
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