import os
import glob

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def function(x, y):
    return -5 * x * y * np.exp((-0.7 * x * x - y * y)) + np.exp(0.1 * y) + np.exp(x)


def grad_func(x, y):
    grad_x = (
        -5 * y * np.exp(-0.7 * x * x - y * y)
        + 7 * y * x * np.exp(-0.7 * x * x - y * y)
        + np.exp(x)
    )
    grad_y = (
        -5 * x * np.exp(-0.7 * x * x - y * y)
        + 10 * x * y * np.exp(-0.7 * x * x - y * y)
        + np.exp(y)
    )
    return (grad_x, grad_y)


def plot_function():

    ax = plt.axes(projection="3d")
    ax.view_init(elev=40, azim=200)
    x = np.linspace(-3, 1.5, 30)
    y = np.linspace(-3, 3, 30)
    X, Y = np.meshgrid(x, y)

    Z = function(X, Y)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="jet", alpha=0.7)

    plt.show()


def save_anim(epoch, x_hist, y_hist, z_hist,optim):

    for i in range(epoch):
        ax = plt.axes(projection="3d")
        ax.view_init(elev=40, azim=200)
        x = np.linspace(-3, 1.5, 30)
        y = np.linspace(-3, 3, 30)
        X, Y = np.meshgrid(x, y)

        Z = function(X, Y)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="jet", alpha=0.7)

        ax.plot(
            x_hist[0:i],
            y_hist[0:i],
            z_hist[0:i],
            marker="*",
            color="r",
            alpha=0.4,
            label=optim,
        )
        leg = plt.legend(loc='best', ncol=1)

        im_name = "./figures/" + str(i) + ".png"
        plt.savefig(im_name)
        plt.close()


def make_gif(epoch):
    images = []
    for i in range(epoch):
        exec("a" + str(i) + '= Image.open("' "./figures/" + str(i) + '.png")')
        images.append(eval("a" + str(i)))
        images[0].save(
            "./figures/anim.gif",
            save_all=True,
            append_images=images[1:],
            duration=epoch,
            loop=0,
        )

    f_imgs = glob.glob("./figures/*.png", recursive=True)
    for f in f_imgs:
        os.remove(f)
    print("[INFOS]gif saved...")


if __name__ == "__main__":
    plot_function()
