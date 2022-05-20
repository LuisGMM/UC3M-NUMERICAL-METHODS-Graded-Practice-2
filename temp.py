
from typing import Callable, Tuple, List

import numpy as np
from numba import njit

import matplotlib.pyplot as plt
from matplotlib.animation import  FuncAnimation



@njit
def hyperbolic(h: float, k: float, x0: float, xf: float, t0: float, tf: float) -> Tuple[np.ndarray]:

    rho = (k/h)**2

    x = np.arange(x0, xf+h, h)
    t = np.arange(t0, tf+k, k)

    LEN_X = len(x)
    LEN_T = len(t)

    sol = np.zeros((LEN_X, LEN_T))

    u0 = np.sin(4*np.pi*x)[1:-1]
    u0_plus_h = np.sin(4*np.pi*(x + h))[1:-1]
    u0_minus_h = np.sin(4*np.pi*(x - h))[1:-1]

    sol[1:-1, 0] = u0

    sol[1:-1, 1] = 1/2 * rho * (u0_plus_h - u0_minus_h) + (1 - rho) * u0

    for ti in range(2, LEN_T):
        sol[1:-1, ti] = rho*sol[2:, ti-1] + 2*(1 - rho)*sol[1:-1, ti-1] + rho*sol[:-2, ti-1]  - sol[1:-1, ti-2]

    return sol, x, t


dx = 0.01
sol, x, t = hyperbolic(dx, 0.01, x0 = 0, xf = 1, t0 = 0, tf = 100)
