#! usr/bin/env python3

#Author: Paul F. Petrowski
#Contact: pfpetrowski@gmail.com
#You are free to use and distribute under the terms of the MIT License.

import numpy as np

def dist(vec1, vec2):
	assert len(vec1) == len(vec2), "Samples must be of equal length."
	sumsq = 0
	for i in range(len(vec1)):
		sumsq += (vec1[i] - vec2[i])**2
	result = sumsq**(1/2)
	return result


def knnreg(data, targets, k):
	assert data.shape[0] == len(targets), "Must have equal numbers of samples and targets."
	residuals = []
	for i, observation in enumerate(data):
		other = np.delete(data,i,0)
		other_targets = np.delete(targets, i)
		distances = [dist(observation, other[j,:]) for j in range(other.shape[0])]
		leastdist = np.argpartition(distances, -k)[-k:]
		yhat = np.mean([other_targets[k] for k in leastdist])
		residuals.append(yhat - targets[i])
	rse = [i**2 for i in residuals]
	return(sum(rse))



#def predict(train, test, k):
#	for i, test_sample in test.shape[0]:
#		for j, train_sample in enumerate(train.shape[0]):
#			dist(test_sample, train_sample)

