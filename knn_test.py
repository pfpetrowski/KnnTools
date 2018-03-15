#! usr/bin/env python3

#Author: Paul F. Petrowski
#Contact: pfpetrowski@gmail.com
#You are free to use and distribute under the terms of the MIT License.


from knnreg import *
import numpy as np



training_data = np.loadtxt('/home/paul/Distributions/KNNRegression/Data/train_data.csv', delimiter = ',')
training_data = training_data.astype(int)

training_response = training_data[:,3]
training_predictors = np.delete(training_data,3,1)


test_data = np.loadtxt('/home/paul/Distributions/KNNRegression/Data/test_data.csv', delimiter = ',')
test_data = test_data.astype(int)

test_response = test_data[:,3]
test_predictors = np.delete(test_data,3,1)



result = knn(training_predictors,test_predictors,training_response, k = 20)
result = np.array(result)

error = test_response - result
sse = np.mean([i**2 for i in error])
print(sse)


#pcterror = (error / result) * 100
#print(pcterror)


