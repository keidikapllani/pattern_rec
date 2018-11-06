# -*- coding: utf-8 -*-
"""
Script to generate plots for coursework EE4_68 CW1

Created on Tue Nov  6 12:55:05 2018
@author: Antonio Enas, Keidi Kapllani
"""
import matplotlib as plt

# Generate plot with mean face and first 6 eigenfaces
plt.figure()
grid = plt.GridSpec(2, 5, wspace=0.4, hspace=0.3)
plt.subplot(grid[0:2, 0:2])
plt.imshow(np.reshape(meanface,(46,56)).T,cmap = 'gist_gray')
plt.xlabel('Mean face')

for i in range(0,3):
    plt.subplot(grid[0, i+2])
    plt.imshow(np.reshape(abs(U[:,i]),(46,56)).T,cmap = 'gist_gray')
	plt.xlabel('Mean face')
	
for i in range(3,6):
    plt.subplot(grid[1, i-1])
    plt.imshow(np.reshape(abs(U[:,i]),(46,56)).T,cmap = 'gist_gray')
	plt.xlabel('Mean face')	
	