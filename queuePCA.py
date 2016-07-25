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

final_components = results.mu + (results.Wt * results.s[:,np.newaxis])

graph_data = results.Y[:,:3]

############################
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm





#############################################
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Percentage of resting time steps")
rest_values = np.loadtxt("features_property_values_rest.v", delimiter=",")
colors = [element for element in rest_values]
boop = ax.scatter(graph_data[:,0], graph_data[:,1], graph_data[:,2], c=colors, cmap='viridis', marker='o', edgecolors='none')
ax.set_xlabel('PCA[0] magnitude')
ax.set_ylabel('PCA[1] magnitude')
ax.set_zlabel('PCA[2] magnitude')

plt.colorbar(boop)
#############################################
fig2 = plt.figure()
ax1 = fig2.add_subplot(111, projection='3d')
ax1.set_title("Percentage of sustaining time steps")
sustain_values = np.loadtxt("features_property_values_sustain.v", delimiter=",")
colors = [element for element in sustain_values]
boop = ax1.scatter(graph_data[:,0], graph_data[:,1], graph_data[:,2], c=colors, cmap='viridis', marker='o', edgecolors='none')
ax1.set_xlabel('PCA[0] magnitude')
ax1.set_ylabel('PCA[1] magnitude')
ax1.set_zlabel('PCA[2] magnitude')
plt.colorbar(boop)
##########################################
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.set_title("Percentage of articulating time steps")
articulate_values = np.loadtxt("features_property_values_articulate.v", delimiter=",")
colors = [element for element in articulate_values]
boop = ax2.scatter(graph_data[:,0], graph_data[:,1], graph_data[:,2], c=colors, cmap='viridis', marker='o', edgecolors='none')
ax2.set_xlabel('PCA[0] magnitude')
ax2.set_ylabel('PCA[1] magnitude')
ax2.set_zlabel('PCA[2] magnitude')
plt.colorbar(boop)
#######################################
plt.show()
#################################

projectedQueueMatrix = results.project(referenceQueueMatrix)
projectedQueueMatrix = sp.special.expit(projectedQueueMatrix)
sigmoidedResults = sp.special.expit(final_components)
np.savetxt("sValues.m", results.s, delimiter=",")
np.savetxt("projectedQueueMatrix.m", projectedQueueMatrix, delimiter=",")
np.savetxt("pcaResults.m", sigmoidedResults, delimiter=",")
