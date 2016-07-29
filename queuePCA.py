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

start = 0

graph_data = results.Y[:,start:start+3]

############################
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm


point_color = 'Set3'
background = 'white'
border = 'none'

axes = []

#############################################
interpFeatureMatrix = np.loadtxt('interp_feature_matrix', delimiter=',')
interpFeatureMatrix_PCA = results.project(sp.special.logit(interpFeatureMatrix))
interpGraphData_PCA = np.array(np.split(interpFeatureMatrix_PCA,8))

#print interpGraphData_PCA.ndim

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d', axisbg = background)
ax1.set_title('Interpolation Feature Matrix with PCA')
#colors = [element for element in interpGraphData[:,2]]
colors = [i % 8 for i in range(len(interpGraphData_PCA[:,2]))]
branno = ax1.scatter(interpGraphData_PCA[:,0], interpGraphData_PCA[:,1], interpGraphData_PCA[:,2], cmap=point_color, marker='o', edgecolors='none')
ax1.set_xlabel('axis 0')
ax1.set_ylabel('axis 1')
ax1.set_zlabel('axis 2')
#plt.colorbar(branno)

plt.show()

#############################################
'''fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', axisbg = background)
ax.set_title("Percentage of sustaining time steps")
rest_values = np.loadtxt("features_property_values_sustain.v", delimiter=",")
colors = [element for element in rest_values[:fourth]]

boop = ax.scatter(graph_data[:,0], graph_data[:,1], graph_data[:,2], c=colors, cmap=point_color, marker='o', edgecolors='none')
ax.set_xlabel('PCA[0] magnitude')
ax.set_ylabel('PCA[1] magnitude')
ax.set_zlabel('PCA[2] magnitude')

plt.colorbar(boop)
#############################################
fig2 = plt.figure()
ax1 = fig2.add_subplot(111, projection='3d', axisbg = background)
ax1.set_title("Percentage of sustaining time steps")
sustain_values = np.loadtxt("features_property_values_sustain.v", delimiter=",")
colors = [element for element in sustain_values]
boop = ax1.scatter(graph_data[:,0], graph_data[:,1], graph_data[:,2], c=colors, cmap=point_color, marker='o', edgecolors='none')
ax1.set_xlabel('PCA[0] magnitude')
ax1.set_ylabel('PCA[1] magnitude')
ax1.set_zlabel('PCA[2] magnitude')
plt.colorbar(boop)
##########################################
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d', axisbg = background)
ax2.set_title("Percentage of articulating time steps")
articulate_values = np.loadtxt("features_property_values_articulate.v", delimiter=",")
colors = [element for element in articulate_values]
boop = ax2.scatter(graph_data[:,0], graph_data[:,1], graph_data[:,2], c=colors, cmap=point_color, marker='o', edgecolors='none')
ax2.set_xlabel('PCA[0] magnitude')
ax2.set_ylabel('PCA[1] magnitude')
ax2.set_zlabel('PCA[2] magnitude')
plt.colorbar(boop)'''
#######################################
def plot_figure_colors(title, data_list, input_file, xlabel = 'PCA[0] magnitude', ylabel = 'PCA[1] magnitude', zlabel = 'PCA[2] magnitude'):
	fig = plt.figure()
	ax = fig2.add_subplot(111, projection='3d', axisbg = background)
	ax.set_title(title)
	plotted_values = np.loadtxt(input_file, delimiter=",")
	colors = [element for element in plotted_values]
	boop = ax2.scatter(data_list[:,0], data_list[:,1], data_list[:,2], c=colors, cmap=point_color, marker='o', edgecolors=border)
	ax2.set_xlabel(xlabel)
	ax2.set_ylabel(ylabel)
	ax2.set_zlabel(zlabel)
	plt.colorbar(boop)
	plt.show()

def plot_figure(title, data_list, input_file, xlabel = 'PCA[0] magnitude', ylabel = 'PCA[1] magnitude', zlabel = 'PCA[2] magnitude'):
	fig = plt.figure()
	ax = fig2.add_subplot(111, projection='3d', axisbg = background)
	ax.set_title(title)
	boop = ax2.scatter(data_list[:,0], data_list[:,1], data_list[:,2], cmap=point_color, marker='o', edgecolors=border)
	ax2.set_xlabel(xlabel)
	ax2.set_ylabel(ylabel)
	ax2.set_zlabel(zlabel)
	plt.colorbar(boop)
	plt.show()

#################################

projectedQueueMatrix = results.project(referenceQueueMatrix)
projectedQueueMatrix = sp.special.expit(projectedQueueMatrix)
sigmoidedResults = sp.special.expit(final_components)
np.savetxt("sValues.m", results.s, delimiter=",")
np.savetxt("projectedQueueMatrix.m", projectedQueueMatrix, delimiter=",")
np.savetxt("pcaResults.m", sigmoidedResults, delimiter=",")