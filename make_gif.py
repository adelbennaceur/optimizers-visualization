from utils import save_anim, make_gif
from optimizers import GradientDescent , Adam

parameters = {
    "SGD": {"lr": 0.02},
    "Momentum": {"lr": 0.01, "gamma": 0.09},
    "Adagrad": {"lr": 0.01, "eps": 1e-7},
    "Adam": {"lr": 0.01, "beta1": 0.95, "beta2": 0.99, "eps":1e-6},
    "Rmsprop": {"lr": 0.01, "eps": 1e-6, "gamma": 0.09},
    "n_epochs": 200,
}


def main():
    # initial values
    x, y = 1, 2

    gd = GradientDescent(x, y, parameters["SGD"])
    adam = Adam(x,y,parameters["Adam"])
    
    for step in range(0,parameters["n_epochs"]):
        adam.minimize(step+1)
    # save animation
    save_anim(parameters["n_epochs"], adam.x_hist, adam.y_hist, adam.z_hist, "Adam")
    make_gif(parameters["n_epochs"])


if __name__ == "__main__":
    main()
