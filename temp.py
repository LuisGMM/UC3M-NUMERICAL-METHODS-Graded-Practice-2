
from typing import Callable, Tuple, List

import numpy as np
from numba import njit

import matplotlib.pyplot as plt
from matplotlib.animation import  FuncAnimation

@njit
def __create_m(v1: Callable[[float], float], v2: Callable[[float], float], v3: Callable[[float], float], N: int,
               set1: np.ndarray, set2: np.ndarray, set3: np.ndarray,
               k1: int = -1, k2: int = 0, k3: int = 1) -> np.ndarray:
    
    diag1 = [v1(x1) for x1 in set1[:N-abs(k1)]]
    diag2 = [v2(x2) for x2 in set1[:N-abs(k2)]]
    diag3 = [v3(x3) for x3 in set3[:N-abs(k3)]]

    return np.diag(diag1, k1) + np.diag(diag2, k2) + np.diag(diag3, k3)


def t_radiactive_rod(h: float, k: float, K: float, tau0: float, a: float,
                     r0: float, rf: float, t0: float, tf: float, u_t0: float,
                     u_rf:float, u_devr0:float) -> Tuple[np.ndarray]:

    s = k*K

    r = np.arange(r0+h, rf+h, h) # TODO: Ignored r0 (==0 and explodes)
    t = np.arange(t0+h, tf+h, k) # TODO: Ignored r0 (==0 and explodes)

    LEN_R = len(r)
    LEN_T = len(t)

    @njit
    def v1(r:float) -> float: # TODO: when taking the first r, r0, it gives -inf due to the s/r*h
        return s/h**2 - s/(r*h) # TODO: when taking the first r, r0, it gives -inf due to the s/r*h
    @njit
    def v2(r:float) -> float:
        return 1 - 2*s/h**2 + s/(r*h)
    @njit
    def v3(r:float) -> float:
        return s/h**2

    def c(t:float) -> float:    
        return np.array([- s/a**2 * np.exp(- t/tau0) if ri <= a else 0 for ri in r[:-1] ])

    m = __create_m(v1, v2, v3, LEN_R-1, set1=r, set2=r, set3=r) #-1 is to take into account the boundary condition

    m[0, 0] = 1 - s/h**2

    sol = np.zeros((LEN_R, LEN_T))
    sol[:, 0] = u_t0
    sol[-1, :] = u_rf

    for ti in range(1, LEN_T):
        sol[:-1, ti] = m @ sol[:-1, ti-1] + c(t=ti-1) # :-1 is to take into account the boundary condition

    return sol, r, t

def t_radiactive_rod_wraper(h: float, k: float, K: float = 2*10**7, tau0: float = 100, a: float = 25,
                     r0: float = 0, rf: float = 100, t0: float = 0, tf: float = 100, u_t0: float = 3,
                     u_rf:float = 300, u_devr0: float = 0) -> Tuple[np.ndarray]:
    
    return t_radiactive_rod(h=h, k=k, K=K, tau0=tau0, a=a, r0=r0, rf=rf, t0=t0, tf=tf, u_t0=u_t0,
                            u_rf=u_rf, u_devr0=u_devr0)



dt = 10
dr = 10

sol, r, t = t_radiactive_rod_wraper(dr, dt)
print(sol[:, 1])