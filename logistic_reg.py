import pandas as pd
import sklearn.datasets
import numpy as np
from numpy import cov

# load train data
TrainDF = pd.read_csv(
    "/Users/adebisiafolalu/Desktop/Onedrive/Documents/MSc_Assignments/Other/Kaggle_Santander_Customer_Satisfaction/dataset/train.csv")
# print(TrainDF)

# split train data into X and class labels y
X = TrainDF.iloc[:, 0:370]
y = TrainDF.iloc[:, 370]

# data scaling
X_std = sklearn.preprocessing.StandardScaler().fit_transform(X)

# covariance matrix
print('Covariance matrix: \n%s' % cov(X_std.T))

# eigendecomposition on covariance matrix
cov_mat = np.cov(X_std.T)

eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eigen_vecs)
print('\nEigenvalues \n%s' %eigen_vals)

# singular vector decomposition
u,s,v = np.linalg.svd(X_std.T)

# make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]

# sort the (eigenvalue, eigenvector) tuples in descending order
eigen_pairs.sort()
eigen_pairs.reverse()

# visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eigen_pairs:
    print(i[0] )
