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
#plt.imshow(np.reshape(meanface,(46,56)).T,cmap = 'gist_gray')
#plt.title('Mean Face\n')

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


#______________________________________________________________________________
### FACE RECONSTRUCTION VARYING M

### Theoretical reconstruction error as function of number of eigenvalues
eigsum = np.real(sum(wn[:,]))
csum = 0
J = np.zeros((416,),float)
for m in range(0,416):
    csum = csum + wn[m]
    J[m] = 100 - 100*csum/eigsum
	
# Plot reconstruction error as a function of the number of PCs
x_m = np.arange(1,416+1)
plt.plot(x_m,J)
plt.xlabel('Number of principal components $M$', fontsize = 14)
plt.ylabel('$J_{\%}$ ~ % error', fontsize = 14)
plt.title('Reconstruction error \nas function of number of principal components'
		  , fontsize = 16)
plt.legend(['Theoretical', 'Training data', 'Test data'], fontsize = 14)
plt.tight_layout()

im_r = 0
_none, axarr = plt.subplots(2, 5)
#For two different faces
for f in range(0,2):
	# Plot the original face first
	axarr[im_r,0].imshow(np.reshape(np.real(x_train[:,7+f]),(46,56)).T,cmap = 'gist_gray')
	axarr[im_r,0].axis('off')
	#Plot title only for the firsy row of the subplot
	if im_r == 0:
		axarr[im_r,0].set_title("Original\nface", fontsize = 14)
	
	#Find the weights for each eigenfaces
	Wm = np.dot(x_train[:,7+f].T, np.real(U)) #This shoud be a vector N*1
	Wm = np.reshape(Wm,(2576,1))
	
	#Vary M
	M = ['#',300,150,50,6]
	print(f'Iteration {f}, row = {im_r}, col = {figix}')
	for im_n in range(1,5):
		m = M[im_n]
		#Reconstruct face
		partial_face = np.dot(np.real(U[:,:m]),Wm[:m,])
		rec_face  = partial_face + meanface
		
		#Plot the reconstructed face in the subplot
		axarr[im_r,im_n].imshow(np.reshape(np.real(rec_face),(46,56)).T,cmap = 'gist_gray')
		axarr[im_r,im_n].axis('off')
		#Print titles only for the first row
		if im_r == 0:
			err = np.round_(J[m-1],1)
			axarr[im_r,im_n].set_title(f"M = {m}\n$J \simeq {err}\%$", fontsize = 14)
	
	im_r += 1 #subplot row
plt.tight_layout()


### Test image reconstruction__________________________________________________
#1.Remove training mean from test image
#2.Project onto the egenspace a_i = x_n.T u_i
#3.Represent the projection vector as w = [a1,a2,...aM].T

FI = x_test - meanface
w_test = np.dot(FI[:,2].T, np.real(U)) #This shoud be a vector N*1
W_test = np.reshape(w_test,(2576,1))
m = 300
partial_face = np.dot(np.real(U[:,:m]),W_test[:m,])
rec_test_face  = partial_face + meanface

_none, axarr = plt.subplots(1, 4)

axarr[0].imshow(np.reshape(np.real(x_test[:,2]),(46,56)).T,cmap = 'gist_gray')
axarr[0].axis('off')
axarr[0].set_title("$\mathbf{x}_{test}$", fontsize = 20)

axarr[1].imshow(np.reshape(np.real(rec_test_face),(46,56)).T,cmap = 'gist_gray')
axarr[1].axis('off')
axarr[1].set_title("$\mathbf{\widetilde{x}}_{test}, M = 300$", fontsize = 20)

axarr[2].imshow(np.reshape(np.real(x_train[:,12]),(46,56)).T,cmap = 'gist_gray')
axarr[2].axis('off')
axarr[2].set_title("$\mathbf{x}_{train}$", fontsize = 20)

w_tr = np.dot(x_train[:,12].T, np.real(U)) #This shoud be a vector N*1
W_tr = np.reshape(w_tr,(2576,1))
partial_tr_face = np.dot(np.real(U[:,:m]),w_tr[:m,])
rec_train_face  = partial_tr_face + meanface

axarr[3].imshow(np.reshape(np.real(partial_tr_face),(46,56)).T,cmap = 'gist_gray')
axarr[3].axis('off')
axarr[3].set_title("$\mathbf{\widetilde{x}}_{train}, M = 300$", fontsize = 20)

plt.tight_layout()
