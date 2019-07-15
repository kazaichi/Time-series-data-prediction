import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
F0, gamma, omega, delta = 10, 0.1, np.pi / 3, 1.5 * np.pi
a, b = 1 / 4, 1 / 2
def duffing(var, t, gamma, a, b, F0, omega, delta):  # カオス時系の作成
    x_dot = var[1]
    p_dot = -gamma * var[1] + 2 * a * var[0] - 4 * b * var[0]**3 + F0 * np.cos(omega * t + delta)

    return np.array([x_dot, p_dot])

def create_duffing(length):
    var, var_lin = [[0, 1]] * 2

    # timescale
    t = np.arange(0, 20000, 2 * np.pi / omega)
    t_lin = np.linspace(0, length, length*10)

    # solve
    var = odeint(duffing, var, t, args=(gamma, a, b, F0, omega, delta))
    var_lin = odeint(duffing, var_lin, t_lin, args=(gamma, a, b, F0, omega, delta))

    x_lin = var_lin.T[0]
    return x_lin, t_lin

if __name__ == "__main__":
    x_lin, t_lin = create_duffing(1000)
    print(len(t_lin))
    plt.plot(x_lin[int(len(x_lin)*0.9):len(x_lin)])
    plt.show()