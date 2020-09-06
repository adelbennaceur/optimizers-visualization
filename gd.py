"""
python implmentation of gradient descent algorithm.
"""

from math import pi
import numpy as np
import matplotlib.pyplot as plt


def function(x, y):
    return x ** 2 + y*np.sin(pi*y)


def grad_func(x, y):
    return (2 * x, 2 * y)


class GradientDescent(object):
    def __init__(self, x, y, lr):
        # initiale values
        self.x = x
        self.y = y
        self.lr = lr
        self.x_hist = [x]
        self.y_hist = [y]
        self.z_hist = [function(x,y)]

    def optimize(self):

        x_result = [self.x]
        y_result = [self.y]
        grad_x, grad_y = grad_func(self.x, self.y)

        self.x = self.x - self.lr * grad_x
        self.y = self.y - self.lr * grad_y

        #for visualization pusrpose
        z = function(self.x,self.y)
        self.x_hist.append(self.x)
        self.y_hist.append(self.y)
        self.z_hist.append(z)

        #print("x: ", self.x, "y: ", self.y)

    def visualize(self):
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        # plot in the range of x£[-6,6] and y£[-6,6]
        x = np.linspace(-6, 6, 30)
        y = np.linspace(-6, 6, 30)
        X, Y = np.meshgrid(x, y)
        Z = function(X, Y)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="jet", alpha=0.5)
        ax.set_title("Gradient descent")

        ax.plot(
            self.x_hist,
            self.y_hist,
            self.z_hist,
            marker="*",
            color="r",
            alpha=0.4,
            label="Gradient descent",
        )
        plt.show()


if __name__ == "__main__":
    optimizer = GradientDescent(x=-3, y=-5, lr=0.01)
    ## TODO:  PLot the Zero with a star
    n_epochs = 100
    for i in range(n_epochs):
        optimizer.optimize()
    optimizer.visualize()
