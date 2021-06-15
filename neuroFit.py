__author__ = 'Al Bernstein'
__license__ = 'MIT License'

import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from keras import backend as K


# Plotting class to make plots

class fitPlot:

    def __init__(self, _xlim, _ylim, _func_str, _num_neurons, _num_pts, _MSE, _tick_fontsize, _title_fontsize):
        
        
        self._MSE = "{:e}".format(_MSE)
        self._xlim = _xlim
        self._ylim = _ylim
        self._func_str = _func_str
        self._num_neurons = _num_neurons
        self._num_pts = _num_pts
        self._tick_fontsize = _tick_fontsize
        self._title_fontsize = _title_fontsize
    
        fig = plt.figure()
        self._ax1 = fig.add_subplot(111)# initial vector

        self.setLimits(_xlim, self._ylim)
        plt.xlabel('x', fontsize = self._tick_fontsize)
        plt.ylabel(_func_str, fontsize = self._tick_fontsize, rotation = 0, labelpad = 52, y = 0.55)
        plt.xticks(fontsize = self._tick_fontsize, ha = 'left')
        plt.yticks(fontsize = self._tick_fontsize)


    def setLimits(self, _xlim, _ylim):
        plt.xlim(_xlim[0], _xlim[1])
        plt.ylim(_ylim[0], _ylim[1])
        


    # plot function only

    def plotFunc(self, _x, _y):
        
        if len(_x) != len(_y):
            logging.error('_x and _y lengths are different')
            return
        
        plt.title(self._func_str + ' vs x -- ' + str(self._num_pts) + ' pts in x', 
                  fontsize = self._title_fontsize)
        self._ax1.plot(_x, _y, color = 'green', label = self._func_str)
        plt.show()
        

    # plot function and fit

    def plot(self, _x, _y, _func_r):

        if len(_x) != len(_y):
            logging.error('_x and _y lengths are different')
            return
        
        plt.title('Fit of ' + self._func_str + ' vs x -- ' + str(self._num_neurons) + ' neurons -- '  
                  + str(self._num_pts) + ' pts in x -- ' + 'mean squared error = ' + str(self._MSE), 
                  fontsize = self._title_fontsize, y = 1.08)

        curve = mpatches.Patch(color='green', label= self._func_str)
        curve_r = mpatches.Patch(color='black', label='reconstructed ' + self._func_str)
        plt.legend(handles=[curve, curve_r])

        self._ax1.plot(_x, _func_r, color = 'black')
        self._ax1.plot(_x, _y, color = 'green', label = self._func_str)
        plt.show()


# class to fit and reconstruct a function using relu activation functions
# within a given interval

class neuroFit:

    # given input points (x, y), compute initial y intercept and slopes 
    # over each x interval

    def reluFit(self, _num_pts, _x, _y):
        
        # create a grid 
        
        x = _x
        y = _y
        
        if len(x) != len(y):
            logging.error('_x and _y lengths are different')
            return []
        
        
        b0 = y[0]
        den = x[1] - x[0]
        if den == 0.0:
            logging.error('division by zero x[' + str(1) + '] - x[' + str(0) +'] = 0')
            return []
            
        m0 = (y[1] - y[0])/den
        x0 = x[0]
        
        ret = [(x0, m0, b0)] # in form (x, slope, y intercept)
        
        for i in range(2, _num_pts):
            den = (x[i] - x[i-1])
            if den == 0.0:
                logging.error('division by zero x[' + str(i-1) + '] - x[' + str(i) +'] = 0')
                return []

            tmp = (y[i] - y[i-1])/den
            m = tmp - m0
            m0 = tmp
            ret.append((x[i-1], m, 0))
    
        ret.append((x[i], -m0, 0))
        return ret


    def reluReconst(self, _inpt, _x):
    
        # _inpt is a list of (starting x value, slope, y intercept)
        # _xlist is a list of linear regions
        # _x is the variable
        
        ret = 0
        
        for item in _inpt:
            ret += item[1] * K.relu(_x - item[0]) + item[2]
        
        return ret


    def changeInterval(self, _x, _a, _b, _c, _d):
       ratio = (_d - _c)/(_b - _a)
       ret = ratio*(_x - _a) + _c
       return ret

    # normalize data in x and y

    def normalize(self, _x, _y, _a, _b, _c, _d):
    
        mag = max(_y)
        _yn = _y/mag
        _xn = self.changeInterval(_x, _a, _b, _c, _d)
        return _xn, _yn, mag


