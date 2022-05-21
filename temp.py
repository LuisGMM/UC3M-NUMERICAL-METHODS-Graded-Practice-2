
from typing import Callable, Tuple, List

import numpy as np
from numba import njit

import matplotlib.pyplot as plt
from matplotlib.animation import  FuncAnimation


def t_radiactive_rod(h: float, k: float, K: float, tau0: float, a: float,
                     r0: float, rf: float, t0: float, tf: float, u_t0: float,
                     u_rf:float) -> Tuple[np.ndarray]:

    s = k*K

    r = np.arange(r0+h, rf+h, h)
    t = np.arange(t0+k, tf+h, k)

    LEN_R = len(r)
    LEN_T = len(t)

    def c(t:float) -> float:    
        return np.array([ s/a**2 * np.exp(- t/tau0) if ri <= a else 0 for ri in r ])

    m = np.diag([1 + s/h**2] + [-s/h**2 + s/(ri*h) for ri in r[1:-1]], -1) \
      + np.diag([1 + s/h**2] + [1 + 2*s/h**2 - s/(ri*h) for ri in r[1:]], 0) \
      + np.diag([- s/h**2 for _ in r[1:]], 1)

    sol = np.zeros((LEN_R, LEN_T))
    sol[:, 0] = u_t0
    sol[-1, :] = u_rf

    m_inv = np.linalg.inv(m)
    
    print(m[0,0], m[1,0], m[1,1])
    # print(m)
    for ti in range(1, LEN_T):
        sol[:, ti] = m_inv @ (sol[:, ti-1] + c(t=ti))

    return sol, r, t



dt = 1
dr = 1

sol, r, t = t_radiactive_rod(dr, dt, K = 2*10**7, tau0 = 100, a = 25, r0 = 0, rf = 100, t0 = 0, tf = 100, u_t0 = 3, u_rf = 300)
