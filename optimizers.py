import numpy as np

from utils import function, grad_func


class GradientDescent(object):
    """
    class implementation of gradient descent algorithm.
    """

    def __init__(self, x, y, params):
        # initiale values
        assert x <= 1.5 and x >= -3, "x values mus be in range [-1,1]"

        self.x = x
        self.y = y
        self.lr = params["lr"]

        self.x_hist = [x]
        self.y_hist = [y]
        self.z_hist = [function(x, y)]

    def minimize(self):

        grad_x, grad_y = grad_func(self.x, self.y)

        self.x = self.x - self.lr * grad_x
        self.y = self.y - self.lr * grad_y
        # for visualization purpose
        z = function(self.x, self.y)
        self.x_hist.append(self.x)
        self.y_hist.append(self.y)
        self.z_hist.append(z)


class Momentum(object):
    """class implmentation of gradient descent with momentum optmization algorithm"""

    def __init__(self, x, y, params):
        assert x <= 1.5 and x >= -3, "x values mus be in range [-1,1]"

        self.x = x
        self.y = y

        self.v_x = 0
        self.v_y = 0

        self.lr = params["lr"]
        self.gamma = params["gamma"]

        self.x_hist = [x]
        self.y_hist = [y]
        self.z_hist = [function(x, y)]

    def minimize(self):

        grad_x, grad_y = grad_func(self.x, self.y)

        self.v_x = self.gamma * self.v_x + self.lr * grad_x
        self.v_y = self.gamma * self.v_y + self.lr * grad_y

        self.x = self.x - self.v_x
        self.y = self.y - self.v_y

        # for visualization purposes
        z = function(self.x, self.y)

        self.x_hist.append(self.x)
        self.y_hist.append(self.y)
        self.z_hist.append(z)


class Adagrad(object):
    """class implmentation of adagrad optmization algorithm"""

    def __init__(self, x, y, params):
        self.x = x
        self.y = y

        self.grad_sqr_x = 0
        self.grad_sqr_y = 0

        self.lr = params["lr"]
        self.eps = params["eps"]

        self.x_hist = [x]
        self.y_hist = [y]
        self.z_hist = [function(x, y)]

    def minmize(self):

        grad_x, grad_y = grad_func(self.x, self.y)

        self.grad_sqr_x += np.square(grad_x)
        self.grad_sqr_y += np.square(grad_y)

        new_grad_x = self.lr * (1 / np.sqrt(self.eps + self.grad_sqr_x)) * grad_x
        new_grad_y = self.lr * (1 / np.sqrt(self.eps + self.grad_sqr_y)) * grad_y

        self.x = self.x - new_grad_x
        self.y = self.y - new_grad_y

        # for visualization purposes
        z = function(self.x, self.y)

        self.x_hist.append(self.x)
        self.y_hist.append(self.y)
        self.z_hist.append(z)


class Rmsprop(object):
    """class implmentation of rmsprop optmization algorithm"""

    def __init__(self, x, y, params):
        self.x = x
        self.y = y

        self.grad_sqr_x, self.grad_sqr_y, self.s_x, self.s_y = 0, 0, 0, 0

        self.lr = params["lr"]
        self.eps = params["eps"]
        self.gamma = params["gamma"]

        self.x_hist = [x]
        self.y_hist = [y]
        self.z_hist = [function(x, y)]

    def minmize(self):

        grad_x, grad_y = grad_func(self.x, self.y)

        self.grad_sqr_x += np.square(grad_x)
        self.grad_sqr_y += np.square(grad_y)

        self.s_x += self.gamma * self.s_x + (1 - self.gamma) * self.grad_sqr_x
        self.s_y += self.gamma * self.s_y + (1 - self.gamma) * self.grad_sqr_y

        new_grad_x = self.lr * (1 / np.sqrt(self.eps + self.s_y)) * grad_x
        new_grad_y = self.lr * (1 / np.sqrt(self.eps + self.s_y)) * grad_y

        self.x = self.x - new_grad_x
        self.y = self.y - new_grad_y

        # for visualization purposes
        z = function(self.x, self.y)

        self.x_hist.append(self.x)
        self.y_hist.append(self.y)
        self.z_hist.append(z)


class Adam(object):
    def __init__(self, x, y, params):
        raise NotImplementedError

    def minmize(self):
        raise NotImplementedError
