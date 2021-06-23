# set up custom layers for input 
# and processing to show how tensorflow and keras work

__author__ = 'Al Bernstein'
__license__ = 'Apache  License Version 2.0'

import numpy as np
import tensorflow as tf 
from keras import backend as K
import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import neuroFit as nf
import sys

# take a numpy array input and output a stacked array dim = output_dim. - of the input
# this fans the input out so multiple neurons can act on it

class FanOut(keras.layers.Layer):   # takes a single input vector, fans it out
                                        # and applies an ReLU activation function.
    
    def __init__(self, _dim = 1):
        super(FanOut, self).__init__()  # instantiate Layer base class
        self._dim = _dim
        
    def call(self, _input):
        
        ret = []
         
        _mul = np.ones(self._dim)
        
        ret = tf.einsum('i, j', _mul, _input)
        return ret
    
class ReLU(keras.layers.Layer):
    
    def __init__(self, _biases, _dim = 1):
        super(ReLU, self).__init__()  # instantiate Layer base class
        self._biases = _biases
        
    def call(self, _input):
        # _input is a matrix where each row is an input vector
         
        n = _input.shape[0] # number of rows
        tmp = [0]*n
         
        # add biases to data
         
        for i in range(n):
            tmp[i] = _input[i] + self._biases[i]
            
        # apply ReLU function
        ret = K.relu(tmp)
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


_input0 = FanOut(m)
x = np.linspace(a, b, n)
X = _input0(x)

_ReLU = ReLU(biases0, m)
Y = _ReLU(X)


biases1 = [i[2] for i in coeffs]
biases1 = tf.constant(biases1, tf.float32)
_sum = sumLayer(weights, biases1, m)

yr = _sum(Y)
 
y = func(x)

# compute mean squared error in the interval [a, b]
 
MSE = (1/n)*np.sum((yr - y)**2)

y_min = tf.math.reduce_min(yr)
y_max = tf.math.reduce_max(yr)


# compute function and reconstructed function
# between [2a, 2b] with twice as many points

x1 = np.linspace(2*a, 2*b, 2*n)
y1 = func(x1)

X1 = _input0(x1)
Y1 = _ReLU(X1)
y1r = _sum(Y1)

fplot = nf.fitPlot((2*a, 2*b), (-3 + y_min - 1, 3 + y_max + 1), func_str, m, n, MSE, 30, 40)
fplot.plot(x1, y1, y1r)



