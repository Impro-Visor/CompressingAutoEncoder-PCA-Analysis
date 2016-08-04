import numpy as np
import scipy as sp
from scipy import special
from matplotlib.mlab import PCA
import itertools

featureMatrix = np.loadtxt("featureMatrix.mat", delimiter=",")
referenceQueueMatrix = np.loadtxt("refQueueFeatureMatrix.mat", delimiter=",")
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

############################
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm


point_color = 'viridis'
background = 'white'
border = 'none'

#############################################

def stacked_1D_graphs(n=len(results.Wt[:,0])):
	'''displays 1D graphs for each of the principal components stacked on top of each other'''
	if n>len(results.Wt[:,0]):
		n = len(results.Wt[:,0])
	elif n<1:
		n = 1
	x_graph_data = []
	y_graph_data = []
	for i in range(n):
		x_data = list(results.Y[:,i])
		y_data = [i]*len(x_data)
		x_graph_data += x_data
		y_graph_data += y_data
	##########################################
	fig = plt.figure()
	art = fig.add_subplot(111)
	art.set_title('Principal Component Distribution: Articulate')
	art.set_xlabel('Values')
	art.set_ylabel('Component')
	articulate_values = np.loadtxt("features_property_values_articulate.vec", delimiter=",")
	art_colors = [element for element in articulate_values]*n
	beep = art.scatter(x_graph_data, y_graph_data, c=art_colors, cmap=point_color, marker='o', edgecolors=border)
	plt.colorbar(beep)
	plt.savefig('PCA_distribution_articulate.png')
	##########################################
	fig = plt.figure()
	rest = fig.add_subplot(111)
	rest.set_title('Principal Component Distribution: Rest')
	rest.set_xlabel('Values')
	rest.set_ylabel('Component')
	rest_values = np.loadtxt("features_property_values_rest.vec", delimiter=",")
	rest_colors = [element for element in rest_values]*n
	boop = rest.scatter(x_graph_data, y_graph_data, c=rest_colors, cmap=point_color, marker='o', edgecolors=border)
	plt.colorbar(boop)
	plt.savefig('PCA_distribution_rest.png')
	##########################################
	fig = plt.figure()
	sust = fig.add_subplot(111)
	sust.set_title('Principal Component Distribution: Sustain')
	sust.set_xlabel('Values')
	sust.set_ylabel('Component')
	sustain_values = np.loadtxt("features_property_values_sustain.vec", delimiter=",")
	sust_colors = [element for element in sustain_values]*n
	lolo = sust.scatter(x_graph_data, y_graph_data, c=sust_colors, cmap=point_color, marker='o', edgecolors=border)
	plt.colorbar(lolo)
	plt.savefig('PCA_distribution_sustain.png')

	plt.show()
	
#############################################

'''
Interpolation Graph Functions
'''
def load_interp_files():
	'''loads all the files and such needed for the interpolation graph functions'''
	interpFeatureMatrix = np.loadtxt('interp_feature_matrix', delimiter=',')
	interpFeatureMatrix_PCA = results.project(sp.special.logit(interpFeatureMatrix))

	numbers = enumerate(interpFeatureMatrix)

	interpGraphData_PCA = np.array(np.split(interpFeatureMatrix_PCA,5)) #dimensions = interpolationStep, queueFeatureIndex, principal component mapping

def plot_interp_state(xVals, yVals, zVals, i, showGraph = True):
	'''plots the interpolation state on top of the original graph'''
	load_interp_files()
	fig1 = plt.figure()
	ax = fig1.add_subplot(111, projection='3d', axisbg = background)
	ax.set_title('Interpolation Feature Matrix with PCA (' + str(i) + ')')

	sustain_values = np.loadtxt("features_property_values_sustain.vec", delimiter=",")
	colors = [element for element in sustain_values]
	boop = ax.scatter(graph_data[:,0], graph_data[:,1], graph_data[:,2], c=colors, cmap=point_color, marker='o', alpha=0.04, edgecolors=border)
	branno = ax.scatter(xVals, yVals, zVals, c='red', marker='o', edgecolors='none')
	# i, data in numbers:
		#print(interpFeatureMatrix_PCA[i,:3])
		#ax1.text(interpFeatureMatrix_PCA[i,0], interpFeatureMatrix_PCA[i,1], interpFeatureMatrix_PCA[i,2], i) #interpFeatureMatrix_PCA[i,:3], i
	ax.set_xlabel('Interpolation Step')
	ax.set_ylabel('Queue Feature Index')
	ax.set_zlabel('Principal Component Mapping')
	plt.colorbar(boop)
	if showGraph:
		plt.show()
	else:
		plt.savefig('interp_state.png')

def whole_interp_graph(showGraph = True):
	'''plots all the points in the interpolation feature matrix'''
	load_interp_files()
	fig0 = plt.figure()
	ax = fig0.add_subplot(111, projection='3d', axisbg=background)
	ax.set_title('Interpolation Feature Matrix with PCA')
	branno = ax.scatter(interpGraphData_PCA[:,0], interpGraphData_PCA[:,1], interpGraphData_PCA[:,2])
	ax.set_xlabel('Interpolation Step')
	ax.set_ylabel('Queue Feature Index')
	ax.set_zlabel('Principal Component Mapping')
	if showGraph:
		plt.show()
	else:
		plt.savefig('interp_feature_matrix.png')

def interp_graphs(perInterp = True, showGraph = True):
	'''calls plot_interp_state for each feature'''
	if perInterp:
		for i in range(8):
			plot_interp_state(interpGraphData_PCA[:,i,0], interpGraphData_PCA[:,i,1], interpGraphData_PCA[:,i,2], i)
			if showGraph:
				plt.show()
			else:
				plt.savefig('interp_state' + str(i) + '.png')
	else:
		for i in range(1):
			plot_interp_state(interpGraphData_PCA[i,:,0], interpGraphData_PCA[i,:,1], interpGraphData_PCA[i,:,2], i)
			if showGraph:
				plt.show()
			else:
				plt.savefig('interp_state' + str(i) + '.png')

#############################################

def absurdly_many_graphs():
	'''In the interest of being thorough, this generates 3D graphs for all the components'''
	for i in range(98):
		data = results.Y[:,i:i+3]
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d', axisbg = background)
		ax.set_title("Percentage of resting time steps")
		rest_values = np.loadtxt("features_property_values_rest.vec", delimiter=",")
		colors = [element for element in rest_values]
		boop = ax.scatter(data[:,0], data[:,1], data[:,2], c=colors, cmap=point_color, marker='o', edgecolors='none')
		ax.set_xlabel('PCA[' +str(i)+ '] magnitude')
		ax.set_ylabel('PCA['+str(i+1)+'] magnitude')
		ax.set_zlabel('PCA['+str(i+2)+'] magnitude')
		plt.colorbar(boop)
		plt.savefig('PCArest_dim' + str(i) + 'through' + str(i+2) + '.png')
		#############################################
		fig2 = plt.figure()
		ax1 = fig2.add_subplot(111, projection='3d', axisbg = background)
		ax1.set_title("Percentage of sustaining time steps")
		sustain_values = np.loadtxt("features_property_values_sustain.vec", delimiter=",")
		colors = [element for element in sustain_values]
		boop = ax1.scatter(data[:,0], data[:,1], data[:,2], c=colors, cmap=point_color, marker='o', edgecolors='none')
		ax.set_xlabel('PCA['+ str(i) +'] magnitude')
		ax.set_ylabel('PCA['+ str(i+1) + '] magnitude')
		ax.set_zlabel('PCA['+ str(i+2) + '] magnitude')
		plt.colorbar(boop)
		plt.savefig('PCAsust_dim' + str(i) + 'through' + str(i+2) + '.png')
		##########################################
		fig2 = plt.figure()
		ax2 = fig2.add_subplot(111, projection='3d', axisbg = background)
		ax2.set_title("Percentage of articulating time steps")
		articulate_values = np.loadtxt("features_property_values_articulate.vec", delimiter=",")
		colors = [element for element in articulate_values]
		boop = ax2.scatter(data[:,0], data[:,1], data[:,2], c=colors, cmap=point_color, marker='o', edgecolors='none')
		ax.set_xlabel('PCA['+ str(i) +'] magnitude')
		ax.set_ylabel('PCA['+ str(i+1) + '] magnitude')
		ax.set_zlabel('PCA['+ str(i+2) + '] magnitude')
		plt.colorbar(boop)
		plt.savefig('PCAart_dim' + str(i) + 'through' + str(i+2) + '.png')

#############################################

def original():
	'''generates the original three graphs of the first three principal components'''
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d', axisbg = background)
	ax.set_title("Percentage of resting time steps")
	rest_values = np.loadtxt("features_property_values_rest.vec", delimiter=",")
	colors = [element for element in rest_values]
	boop = ax.scatter(graph_data[:,0], graph_data[:,1], graph_data[:,2], c=colors, cmap=point_color, marker='o', edgecolors='none')
	ax.set_xlabel('PCA[0] magnitude')
	ax.set_ylabel('PCA[1] magnitude')
	ax.set_zlabel('PCA[2] magnitude')
	plt.colorbar(boop)
	#############################################
	fig2 = plt.figure()
	ax1 = fig2.add_subplot(111, projection='3d', axisbg = background)
	ax1.set_title("Percentage of sustaining time steps")
	sustain_values = np.loadtxt("features_property_values_sustain.vec", delimiter=",")
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
	articulate_values = np.loadtxt("features_property_values_articulate.vec", delimiter=",")
	colors = [element for element in articulate_values]
	boop = ax2.scatter(graph_data[:,0], graph_data[:,1], graph_data[:,2], c=colors, cmap=point_color, marker='o', edgecolors='none')
	ax2.set_xlabel('PCA[0] magnitude')
	ax2.set_ylabel('PCA[1] magnitude')
	ax2.set_zlabel('PCA[2] magnitude')
	plt.colorbar(boop)
	plt.show()
	#################################
	projectedQueueMatrix = results.project(referenceQueueMatrix)
	projectedQueueMatrix = sp.special.expit(projectedQueueMatrix)
	sigmoidedResults = sp.special.expit(final_components)
	np.savetxt("sValues.mat", results.s, delimiter=",")
	np.savetxt("projectedQueueMatrix.mat", projectedQueueMatrix, delimiter=",")
	np.savetxt("pcaResults.mat", sigmoidedResults, delimiter=",")

#######################################

'''
General Graph Functions
'''
def plot_figure_colors(title, data_list, input_file, xlabel = 'PCA[0] magnitude', ylabel = 'PCA[1] magnitude', zlabel = 'PCA[2] magnitude'):
	'''plots a color-coded, three-dimensional numpy graph'''
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
	'''plots a  three-dimensional numpy graph with no particular color-coding scheme'''
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d', axisbg = background)
	ax.set_title(title)
	boop = ax.scatter(data_list[:,0], data_list[:,1], data_list[:,2], cmap=point_color, marker='o', edgecolors=border)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_zlabel(zlabel)
	plt.colorbar(boop)
	plt.show()