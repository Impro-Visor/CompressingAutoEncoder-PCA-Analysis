import numpy as np
import scipy as sp
from scipy import special
from matplotlib.mlab import PCA
import itertools

featureMatrix = np.loadtxt("featureMatrix.m", delimiter=",")
referenceQueueMatrix = np.loadtxt("refQueueFeatureMatrix.m", delimiter=",")
featureMatrix = sp.special.logit(featureMatrix)
referenceQueueMatrix = sp.special.logit(referenceQueueMatrix)
results = PCA(featureMatrix)
print(results.Wt)
print(len(results.Wt[:,0]))


pca_mean = results.mu #the mean vector
pca_components = results.Wt #matrix of the component vectors
pca_component_strengths = results.s #vector of the eigenvalues for each component

#the principal components (added to the eigenvalue for each feature) added to the mean
final_components = results.mu + (results.Wt * results.s[:,np.newaxis])

start = 0

graph_data = results.Y[:,start:start+3]

#############################################

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#############################################

def generate_sample_matrix(n=0, d=8):
	'''generates a matrix with d rows of a vector that is a copy of the nth component'''
	desiredVector = final_components[n]
	desiredMatrix = [desiredVector]*d
	np.savetxt('sample_matrix' + str(n) + '_' + str(d) + '.mat',desiredMatrix, delimiter=',')

generate_sample_matrix(1,100)