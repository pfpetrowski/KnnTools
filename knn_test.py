#! usr/bin/env python3

#Author: Paul F. Petrowski
#Contact: pfpetrowski@gmail.com
#You are free to use and distribute under the terms of the MIT License.



from knnreg import *
import numpy as np

size = 1000



x1 = np.random.randint(1,20,size) + np.random.randn(size)
x2 = np.random.randint(1,20,size) + np.random.randn(size)
x3 = np.random.randint(1,20,size) + np.random.randn(size)
y = 2*x1 + 0.5*x2 + 1.1*x3 + np.random.randn(size) * 1.5

data = np.vstack((x1,x2,x3,y))
data = np.round(data,2)
data = data.T

targets = data[:,3]
data = np.delete(data,3,1)


print(knnreg(data,targets,10))