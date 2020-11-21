import torch
from torch.optim import Adam, SGD


from utils import save_anim, make_gif
from optimizers import GradientDescent

parameters = {
    "gd": {"lr": 0.02},
    "momentum": {"lr": 0.01, "gamma": 0.09},
    "adagrad": {"lr": 0.01, "eps": 1e-7},
    "adadelta": {"lr": 0.01, "mu": 0.95, "eps": 1e-05, "decay": 0},
    "rmsprop": {"lr": 0.01, "eps": 1e-6, "gamma": 0.09},
    "n_epochs": 75,
}


def main():
    # initial values
    x, y = 1, 2

    gd = GradientDescent(x, y, parameters["gd"])

    for _ in range(parameters["n_epochs"]):
        # test queues and multiprocessing
        gd.minimize()
    # save animation
    save_anim(parameters["n_epochs"], gd.x_hist, gd.y_hist, gd.z_hist)
    make_gif(parameters["n_epochs"])


if __name__ == "__main__":
    main()
