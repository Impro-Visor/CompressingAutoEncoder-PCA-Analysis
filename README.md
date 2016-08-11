# Compressing Autoencoder Principal Component Analysis

Principal Component Analysis and visualization code for musical feature vectors. This project uses the numpy, scipy, and matplotlib libraries and is written in Python 3.5.

queuePCA.py takes the principal component analysis of some matrices whose rows are feature vectors and creates graphs of them, most of which are color-coded according to percentage of articulating, resting, or sustaining time steps.

PCA_distribution_articulate.png, PCA_distribution_rest.png, and PCA_distribution_sustain.png are graphs generated using queuePCA.py.

reconstructFromPCA.py takes the principal component analysis of a matrix whose rows are feature vectors without adding in the mean, re-applies the sigmoid function to the result, and produces a matrix of linear combinations of the resulting vectors and some interpolation feature vectors. When the resulting matrices have been put through the second half of the autoencoder from lstmprovisor-java, it produces something resembling a melody.

linear_combination.mat, sample_matrix0, and sample_matrix1 are matrices generated using reconstructFromPCA.py.

All other files in this project were created using lstmprovisor.

Written for the Intelligent Music Software project at Harvey Mudd College, Summer 2016.