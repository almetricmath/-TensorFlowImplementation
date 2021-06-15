# set up custom layers for input 
# and processing to show how keras works

__author__ = 'Al Bernstein'
__license__ = 'MIT License'

import numpy as np
import tensorflow as tf 
from keras import backend as K
import keras
import neuroFit as nf
import sys

# take a numpy array input and output a stacked array dim = output_dim. - of the input
# this fans the input out so multiple neurons can act on it

class ReLUFanOut(keras.layers.Layer):   # takes a single input vector, fans it out
                                        # and applies an ReLU activation function.
    
    def __init__(self, _weights, _biases, _dim=1):
        super(ReLUFanOut, self).__init__()  # instantiate Layer base class
        self._dim = _dim
        self._weights = _weights
        self._biases = _biases
        
    def call(self, _input):
        
        ret = []
         
        # Add input to each bias term to create 
        # a list of tensors
        
        for i in range(self._dim):
            # add bias term to input
            tmp = _input + self._biases[i]
            #tmp = tf.add(_input, self._biases[i])
            ret.append(tmp)
            
        ret = tf.stack(ret) # convert tensor list to a single tensor
        # apply ReLU activation function
        ret = K.relu(ret)
        
        return ret
    

class  sumLayer(keras.layers.Layer):
    
    def __init__(self, _weights, _biases, _dim = 1):
        super(sumLayer, self).__init__()  # instantiate Layer base class
        self._weights = _weights
        self._biases = _biases
        self._dim = _dim
    
    def call(self, _input):
        
        if _input.shape[0] != self._weights.shape[0]:
            return 'shape mismatch error'
        
        ret = tf.einsum('i,ij', self._weights, _input) + self._biases[0]
        return ret
        
        
# function to be approximated
#func = lambda x: x**2
func = lambda x: (1/2)*(5*x**3 - 3*x)

# function string to be used in plots
#func_str = '$x^2$'
func_str = '(1/2)(5$x^3$ - 3x)'
    
fit = nf.neuroFit()

m = 20  # number of neurons
n = 200 # number of points

# generate model using m neurons between x = -2 to x = 2

a = -1
b = 1

x_fit = np.linspace(a, b, m)
y_fit = func(x_fit)

# compute weights and biases analytically for relu activation functions

coeffs = fit.reluFit(m, x_fit, y_fit)
if coeffs == []:
    sys.exit()    
    
weights = [i[1] for i in coeffs]
biases0 = [-i[0] for i in coeffs]
weights = tf.constant(weights, tf.float32)
biases0 = tf.constant(biases0, tf.float32)


_input0 = ReLUFanOut(weights, biases0, m)

biases1 = [i[2] for i in coeffs]
biases1 = tf.constant(biases1, tf.float32)
_sum = sumLayer(weights, biases1, m)

x = np.linspace(-1, 1, n)
y = _input0(x)
yr = _sum(y)
 
y = func(x)

# compute mean squared error in the interval [a, b]
 
MSE = (1/n)*np.sum((y - yr)**2)

x1 = np.linspace(2*a, 2*b, 2*n)
yt = func(x1)

y1 = _input0(x1)
y1r = _sum(y1)

y_min = tf.math.reduce_min(y1r)
y_max = tf.math.reduce_max(y1r)

fplot = nf.fitPlot((2*a, 2*b), (-3 + y_min - 1, 3 + y_max + 1), func_str, m, n, MSE, 30, 40)
fplot.plot(x1, yt, y1r)
