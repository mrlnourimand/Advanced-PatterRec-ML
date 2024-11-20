"""
DATA.ML.200
Week 3, title: MLP gradient descent.

(I wrote a gradient descent (GD) Python code for the MLP and train and
test how well it learns 3 different logical functions.)

Creator: Maral Nourimand
"""
import numpy as np
from scipy.special import expit
# from matplotlib import pyplot as plt


def init_weights():
    """
    to initialize the weights randomly for the first epoch

    :return: initialized weights
    """
    # Initial weights
    w10_t = np.random.normal(-1, 1)
    w11_t = np.random.normal(-1, 1)
    w12_t = np.random.normal(-1, 1)
    w13_t = np.random.normal(-1, 1)

    w20_t = np.random.normal(-1, 1)
    w21_t = np.random.normal(-1, 1)
    w22_t = np.random.normal(-1, 1)
    w23_t = np.random.normal(-1, 1)

    w1_t = np.random.normal(-1, 1)
    w2_t = np.random.normal(-1, 1)
    w0_t = np.random.normal(-1, 1)

    return w10_t, w11_t, w12_t, w13_t, w20_t, w21_t, w22_t, w23_t, w1_t, w2_t, w0_t


def training(x1, x2, x3, y):
    """
    train mlp with respect to x1, x2 and x3.

    :param x1: input
    :param x2: input
    :param x3: input
    :param y: the ground true output
    :return: the final predicted output, y_hat
    """
    w10_t, w11_t, w12_t, w13_t, w20_t, w21_t, w22_t, w23_t, w1_t, w2_t, w0_t = \
        init_weights()

    num_of_epochs = 5000
    lr = 0.1

    MSE = np.zeros([num_of_epochs, 1])
    y_h = 0

    # Main training loop
    for e in range(num_of_epochs):
        ## Forward pass

        y_1 = expit(w11_t * x1 + w12_t * x2 + w13_t * x3 + w10_t)
        y_2 = expit(w21_t * x1 + w22_t * x2 + w23_t * x3 + w20_t)
        y_h = expit(w1_t * y_1 + w2_t * y_2 + w0_t)

        ## Backward pass

        # Loss gradient
        nabla_L = -2 * (y - y_h)  # dMSE/dy_h

        # Output neuron gradient
        nabla_y_h_y1 = y_h * (1 - y_h) * w1_t
        nabla_y_h_y2 = y_h * (1 - y_h) * w2_t

        # # Hidden layer gradient, No Need for them in UPDATE
        # nabla_y1_x1 = y_1 * (1 - y_1) * w11_t
        # nabla_y1_x2 = y_1 * (1 - y_1) * w12_t
        # nabla_y1_x3 = y_1 * (1 - y_1) * w13_t
        #
        # nabla_y2_x1 = y_2 * (1 - y_2) * w21_t
        # nabla_y2_x2 = y_2 * (1 - y_2) * w22_t
        # nabla_y2_x3 = y_2 * (1 - y_2) * w23_t

        ## Update

        # Output weights
        w1_t = w1_t - lr * np.sum(nabla_L * y_h * (1 - y_h) * y_1)
        w2_t = w2_t - lr * np.sum(nabla_L * y_h * (1 - y_h) * y_2)
        w0_t = w0_t - lr * np.sum(nabla_L * y_h * (1 - y_h) * 1)

        # Hidden layer y_1 weights
        w11_t = w11_t - lr * np.sum(nabla_L * nabla_y_h_y1 * y_1 * (1 - y_1) * x1)
        w12_t = w12_t - lr * np.sum(nabla_L * nabla_y_h_y1 * y_1 * (1 - y_1) * x2)
        w13_t = w13_t - lr * np.sum(nabla_L * nabla_y_h_y1 * y_1 * (1 - y_1) * x3)
        w10_t = w10_t - lr * np.sum(nabla_L * nabla_y_h_y1 * y_1 * (1 - y_1) * 1)

        # Hidden layer y_2 weights
        w21_t = w21_t - lr * np.sum(nabla_L * nabla_y_h_y2 * y_2 * (1 - y_2) * x1)
        w22_t = w22_t - lr * np.sum(nabla_L * nabla_y_h_y2 * y_2 * (1 - y_2) * x2)
        w23_t = w23_t - lr * np.sum(nabla_L * nabla_y_h_y2 * y_2 * (1 - y_2) * x3)
        w20_t = w20_t - lr * np.sum(nabla_L * nabla_y_h_y2 * y_2 * (1 - y_2) * 1)
        # MSE[e] = np.sum((y - y_h) ** 2) / len(y)

        # if np.mod(e, 40) == 0 or e == 1:  # Print after every 40th epoch
        #     print(MSE[e])

    return y_h


def main():
    np.random.seed(42)

    # Inputs
    x1 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    x2 = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    x3 = np.array([0, 1, 0, 1, 0, 0, 1, 1])

    # Outputs
    y_OR = np.array([0, 1, 1, 1, 1, 1, 1, 1])  # OR x1 v x2 v x3
    y_AND = np.array([0, 0, 0, 0, 0, 0, 1, 0]) # AND  x1 ^ x2 ^ x3
    y_MIX = np.array([0, 0, 0, 1, 0, 0, 0, 1]) # ((x1 ^ ~x2) v (~x1 ^ x2)) ^ x3

    y_h_OR = training(x1, x2, x3, y_OR)
    y_h_AND = training(x1, x2, x3, y_AND)
    y_h_MIX = training(x1, x2, x3, y_MIX)

    np.set_printoptions(precision=3, suppress=True)
    print(f'OR: True values y={y_OR} and predicted values y_pred={y_h_OR}\n')
    print(f'AND: True values y={y_AND} and predicted values y_pred={y_h_AND}\n')
    print(f'3rd Function: True values y={y_MIX} and predicted values y_pred={y_h_MIX}')


if __name__ == "__main__":
    main()

