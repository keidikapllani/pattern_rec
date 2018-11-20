# -*- coding: utf-8 -*-
"""
Script to generate plots for coursework EE4_68 CW1

Created on Tue Nov  6 12:55:05 2018
@author: Antonio Enas, Keidi Kapllani
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

# Plot eigenvalues for naive and efficient PCA_________________________________
x_naive = np.arange(1,len(w_n)+1)
x_effct = np.arange(1,len(w_e)+1)
lines = plt.plot(x_naive,w_n, x_effct,w_e)
l1, l2 = lines
plt.setp(l1, linewidth=4, color= '#0055ff')
plt.setp(l2, linewidth=4, color='#00aa00')
plt.xlabel('$\lambda_{m}$ ~ $m^{th}$ eigenvalue', fontsize = 14)
plt.ylabel('Re{$\lambda_{m}$}', fontsize = 14)
plt.title('Ordered eigenvalues from PCA', fontsize = 16)
plt.legend(['Simple PCA', 'Optimised PCA'], fontsize = 14)
plt.tight_layout()
#______________________________________________________________________________



# Generate plot with mean face and first 6 eigenfaces__________________________
plt.figure()
grid = gridspec.GridSpec(2, 5, wspace=0.2, hspace=0.1)

plt.subplot(grid[0:2, 0:2])
plt.imshow(np.reshape(meanface,(46,56)).T,cmap = 'gist_gray')
plt.axis('off')
plt.title('Mean face')

plt.subplot(grid[0, 2])
plt.imshow(np.reshape(abs(U[:,0]),(46,56)).T,cmap = 'gist_gray')
plt.title(r'$\mathbf{u}_{1}$')
plt.axis('off')

plt.subplot(grid[0, 3])
plt.imshow(np.reshape(abs(U[:,1]),(46,56)).T,cmap = 'gist_gray')
plt.title(r'$\mathbf{u}_{2}$')
plt.axis('off')

plt.subplot(grid[0, 4])
plt.imshow(np.reshape(abs(U[:,2]),(46,56)).T,cmap = 'gist_gray')
plt.title(r'$\mathbf{u}_{3}$')
plt.axis('off')

plt.subplot(grid[1, 2])
plt.imshow(np.reshape(abs(U[:,3]),(46,56)).T,cmap = 'gist_gray')
plt.title(r'$\mathbf{u}_{4}$')
plt.axis('off')

plt.subplot(grid[1, 3])
plt.imshow(np.reshape(abs(U[:,4]),(46,56)).T,cmap = 'gist_gray')
plt.title(r'$\mathbf{u}_{5}$')
plt.axis('off')

plt.subplot(grid[1, 4])
plt.imshow(np.reshape(abs(U[:,5]),(46,56)).T,cmap = 'gist_gray')
plt.title(r'$\mathbf{u}_{6}$')
plt.axis('off')
#______________________________________________________________________________	
#______________________________________________________________________________
### FACE RECONSTRUCTION VARYING M

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

### TEST FACE RECONSTRUCTION EXAMPLE___________________________________________	
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
#______________________________________________________________________________

### CLASSIFICATION SUCCESS FAILURES EXAMPLES __________________________________	
tst = [2,3,12,13]
trt = [14,136,48,0]
_none, axarr = plt.subplots(2, 4)
for i in range(0,4):
	#Plot test faces
	axarr[0,i].imshow(np.reshape(x_test[:,tst[i]],(46,56)).T,cmap = 'gist_gray')
	axarr[0,i].axis('off')
	if i == 0 or i == 1:
		axarr[0,i].set_title("$kNN$ Input", fontsize = 14)
	else:
		axarr[0,i].set_title("$Min(J_{rec})$ Input", fontsize = 14)
	#Plot class face
	axarr[1,i].imshow(np.reshape(x_train[:,trt[i]],(46,56)).T,cmap = 'gist_gray')
	axarr[1,i].axis('off')
	if i == 0 or i == 2:
		axarr[1,i].set_title("Successfully\nclassified as", fontsize = 14)
	else:
		axarr[1,i].set_title("Unsuccessfully\nclassified as", fontsize = 14)
plt.tight_layout()

### PLOT ACCURACY VS T ___________________________________________________
T_range = np.arange(2,100,5)
avg_ensamble = accuracy_ens.mean(axis=1)
avg_models = accuracy_av.mean(axis = 1)
lines = plt.plot(T_range,avg_ensamble, T_range,avg_models)
l1, l2 = lines
plt.setp(l1, linewidth=4, color= '#0055ff')
plt.setp(l2, linewidth=4, color='#00aa00')
plt.xlabel('$T$ ~ Number of models', fontsize = 14)
plt.ylabel('% Accuracy', fontsize = 14)
plt.title('Ensamble model accuracy\n as a function of the number of models', fontsize = 16)
plt.legend(['Ensamble', 'Individual models'], fontsize = 14)
plt.tight_layout()

### PLOT ACCURACY VS T ___________________________________________________
plt.figure()
T_range = 2*np.arange(2,100,5)
avg_ensamble = accuracy_ens[:20].max(axis=1)
avg_models = accuracy_av[:20].mean(axis = 1)
lines = plt.plot(T_range,avg_ensamble, T_range,avg_models)
l1, l2 = lines
plt.setp(l1, linewidth=4, color= '#0055ff')
plt.setp(l2, linewidth=4, color='#00aa00')
plt.xlabel('$T$ ~ Number of models', fontsize = 14)
plt.ylabel('% Accuracy', fontsize = 14)
plt.title('Ensemble model accuracy\n as a function of the number of models', fontsize = 16)
plt.legend(['Ensemble', 'Individual models'], fontsize = 14)
plt.tight_layout()