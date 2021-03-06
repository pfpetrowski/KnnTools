#! usr/bin/env python3

#Author: Paul F. Petrowski
#Contact: pfpetrowski@gmail.com
#You are free to use and distribute under the terms of the MIT License.

import numpy as np


def dist(vec1, vec2):
	assert len(vec1) == len(vec2), "Observations must have the same number of predictors."
	sumsq = 0
	for i in range(len(vec1)):
		sumsq += (vec1[i] - vec2[i])**2
	euclidean_distance = sumsq**(1/2)
	return euclidean_distance


def ladist(matrix, vector):
	assert matrix.shape[1] == len(vector)
	distances = np.sum(((matrix - vector)**2), axis = 1)**(1/2)
	return(distances)


def predict(train, sample, targets, k):
	assert train.shape[0] == len(targets),"Must have equal numbers of samples and targets."
	distances = ladist(train, sample)
	nearest_neighbors = np.argsort(distances)[0:k]
	prediction = np.mean([targets[i] for i in nearest_neighbors])
	return prediction


def classify(train, sample, targets, k):
	assert train.shape[0] == len(targets),"Must have equal numbers of samples and targets."
	from scipy import stats
	distances = ladist(train, sample)
	nearest_neighbors = np.argsort(distances)[0:k]
	classification = stats.mode([targets[i] for i in nearest_neighbors])
	return classification


def knn(train, test, targets, k, mode = 'r'):
	assert (mode == 'r' or mode == 'c'), 'Mode must be "r" for regression or "c" for classification.'
	if mode == 'r':
		predictions = [predict(train, test_sample, targets, k) for test_sample in test]
		return predictions
	elif mode == 'c':
		classifications = [classify(train, test_sample, targets, k) for test_sample in test]
		return classifications
	else:
		print('Please use a valid mode')


def evaluate(data, targets, k):
	residuals = []
	for i, observation in enumerate(data):
		other = np.delete(data,i,0)
		other_targets = np.delete(targets, i)
		distances = ladist(other, observation)
		nearest_neighbors = np.argsort(distances)[0:k]
		yhat = np.mean([other_targets[k] for k in nearest_neighbors])
		residuals.append(yhat - targets[i])
	rse = [i**2 for i in residuals]
	return sum(rse)

