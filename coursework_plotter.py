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

# Generate plot with mean face and first 6 eigenfaces
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
	
	