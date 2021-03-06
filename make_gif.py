from utils import save_anim, make_gif
from optimizers import GradientDescent , Adam

parameters = {
    "SGD": {"lr": 0.02},
    "Momentum": {"lr": 0.01, "gamma": 0.09},
    "Adagrad": {"lr": 0.01, "eps": 1e-7},
    "Adam": {"lr": 0.1, "beta1": 0.8, "beta2": 0.9, "eps":1e-3},
    "Rmsprop": {"lr": 0.01, "eps": 1e-6, "gamma": 0.09},
    "n_epochs": 100,
}


def main():
    # initial values
    x, y = 1, 2

    gd = GradientDescent(x, y, parameters["SGD"])
    
    
    for step in range(0,parameters["n_epochs"]):
        gd.minimize()
    # save animation
    save_anim(parameters["n_epochs"], gd.x_hist, gd.y_hist, gd.z_hist, "SGD")
    make_gif(parameters["n_epochs"])


if __name__ == "__main__":
    main()
