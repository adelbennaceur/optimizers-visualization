import numpy as np
import matplotlib.plt as plt

from optimizers import *

__all__ = ['GradientDescent' , 'RMSprop', 'Adagrad' ,'Adadelta','Adagrad']


gd_params = {'lr' :0 }
adam_params = {'lr' :0 }
n_epoch = 100

def fucntion():
    pass

def grad_func():
    pass

def visualize(data_hist):

    for x_hist , y_hist in data_hist
    ax.plot()
    
def main():
    #initial values
    x , y  , z = 10 , -5 , function(x,y)

    #change to **kwargs
    gd = GradientDescent()
    adam = Adam(**adam_params)

    for i in range(n_epoch):
        gd.optimize()
        adam.optimize()

    data_hist = None
    visualize(data_hist)

if __name__ =='__main__':
    main()
