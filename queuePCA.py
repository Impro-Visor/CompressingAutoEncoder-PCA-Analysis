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


point_color = 'viridis'
background = 'white'
border = 'none'

axes = []

#############################################
interpFeatureMatrix = np.loadtxt('interp_feature_matrix', delimiter=',')
interpFeatureMatrix_PCA = results.project(sp.special.logit(interpFeatureMatrix))

numbers = enumerate(interpFeatureMatrix)

interpGraphData_PCA = np.array(np.split(interpFeatureMatrix_PCA,5)) #dimensions = interpolationStep, queueFeatureIndex, principal component mapping

def plot_interp_state(xVals, yVals, zVals, i):
	fig1 = plt.figure()
	ax = fig1.add_subplot(111, projection='3d', axisbg = background)
	ax.set_title('Interpolation Feature Matrix with PCA (' + str(i) + ')')

	sustain_values = np.loadtxt("features_property_values_sustain.v", delimiter=",")
	colors = [element for element in sustain_values]
	boop = ax.scatter(graph_data[:,0], graph_data[:,1], graph_data[:,2], c=colors, cmap=point_color, marker='o', alpha=0.04, edgecolors=border)
	branno = ax.scatter(xVals, yVals, zVals, c='red', marker='o', edgecolors='none')
	# i, data in numbers:
		#print(interpFeatureMatrix_PCA[i,:3])
		#ax1.text(interpFeatureMatrix_PCA[i,0], interpFeatureMatrix_PCA[i,1], interpFeatureMatrix_PCA[i,2], i) #interpFeatureMatrix_PCA[i,:3], i
	ax.set_xlabel('axis 0')
	ax.set_ylabel('axis 1')
	ax.set_zlabel('axis 2')
	plt.colorbar(boop)

perInterp = True

if perInterp:
	for i in range(8):
		plot_interp_state(interpGraphData_PCA[:,i,0], interpGraphData_PCA[:,i,1], interpGraphData_PCA[:,i,2], i)
else:
	for i in range(1):
		plot_interp_state(interpGraphData_PCA[i,:,0], interpGraphData_PCA[i,:,1], interpGraphData_PCA[i,:,2], i)

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
	ax = fig.add_subplot(111, projection='3d', axisbg = background)
	ax.set_title(title)
	plotted_values = np.loadtxt(input_file, delimiter=",")
	colors = [element for element in plotted_values]
	boop = ax.scatter(data_list[:,0], data_list[:,1], data_list[:,2], c=colors, cmap=point_color, marker='o', edgecolors=border)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_zlabel(zlabel)
	plt.colorbar(boop)
	plt.show()

def plot_figure(title, data_list, input_file, xlabel = 'PCA[0] magnitude', ylabel = 'PCA[1] magnitude', zlabel = 'PCA[2] magnitude'):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d', axisbg = background)
	ax.set_title(title)
	boop = ax.scatter(data_list[:,0], data_list[:,1], data_list[:,2], cmap=point_color, marker='o', edgecolors=border)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_zlabel(zlabel)
	plt.colorbar(boop)
	plt.show()

#################################

projectedQueueMatrix = results.project(referenceQueueMatrix)
projectedQueueMatrix = sp.special.expit(projectedQueueMatrix)
sigmoidedResults = sp.special.expit(final_components)
np.savetxt("sValues.m", results.s, delimiter=",")
np.savetxt("projectedQueueMatrix.m", projectedQueueMatrix, delimiter=",")
np.savetxt("pcaResults.m", sigmoidedResults, delimiter=",")
