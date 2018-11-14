# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 10:26:49 2018

@author: AE
"""

X = x_train
y = y_train

N = 416
D = 2576
c = 52	
M = c - 1
#1. Compute the global mean
m = x_train.mean(axis = 1).reshape((2576,1))

#2. Compute the mean of each class mi
#3. Compute Sw = sum over c{(x - mi)*(x - mi).T}, rank(Sw) = N - c
#	Sw is the within class scatter matrix
#4. Compute Sb = sum over c{(mi - m)*(mi - m).T}, it has rank(c-1)
#	Sb is the between class scatter matrix
mi = np.zeros((2576,52))
Sw = np.zeros((D,D))
Sb = np.zeros((D,D))
_ix = 0
for c in range(0,52):
	xi = x_train[:,_ix:_ix+8]
	#2
	mi[:,c] = xi.mean(axis = 1)
	_mi = mi[:,c].reshape((D,1))
	#3
	Sw = Sw + np.dot((xi-_mi),(xi-_mi).T)
	#4
	Sb = Sb + np.dot((_mi - m),(_mi - m).T)
	_ix += 8
	
	
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

# Plot the sorted eigenvalues
lol = [np.abs(eig_vals[i]) for i in range(len(eig_vals))]
lol = sorted(lol, reverse=True)
x_eig = np.arange(1,len(lol)+1)
lines = plt.plot(x_eig,lol)
l1 = lines
plt.setp(l1, linewidth=4, color= '#0055ff')
plt.xlabel('$\lambda_{m}$ ~ $m^{th}$ eigenvalue', fontsize = 14)
plt.ylabel('Re{$\lambda_{m}$}', fontsize = 14)
plt.title('Ordered eigenvalues from PCA', fontsize = 16)
plt.legend(['Simple PCA', 'Optimised PCA'], fontsize = 14)
plt.tight_layout()

W = eig_vecs.real[:,:51]
X_lda = np.dot(X.T,W)

X_test_lda = np.dot(x_test.T,W)

k=1
knn = KNeighborsClassifier(n_neighbors = k)
knn.fit(X_lda, y_train.T)
y_knn = knn.predict(X_test_lda)
accuracy = 100*accuracy_score(y_test.T, y_knn)
