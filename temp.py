
from typing import Callable, Tuple, List

import numpy as np
from numba import njit

import matplotlib.pyplot as plt
from matplotlib.animation import  FuncAnimation


@njit
def func(x):
    return np.sin(4*np.pi*x)


@njit
def hyperbolic(h: float, k: float, x0: float, xf: float, t0: float, tf: float, u0: Callable[[float], float]) -> Tuple[np.ndarray]:

    rho = (k/h)**2

    x = np.arange(x0, xf+h, h)
    t = np.arange(t0, tf+k, k)

    LEN_X = len(x)
    LEN_T = len(t)

    sol = np.zeros((LEN_X, LEN_T))

    sol[1:-1, 0] = u0(x)[1:-1]

    sol[1:-1, 1] = 1/2 * rho * (u0(x+h)[1:-1] - u0(x-h)[1:-1]) + (1 - rho) * u0(x)[1:-1]

    for ti in range(2, LEN_T):
        sol[1:-1, ti] = rho*sol[2:, ti-1] + 2*(1 - rho)*sol[1:-1, ti-1] + rho*sol[:-2, ti-1]  - sol[1:-1, ti-2]

    return sol, x, t

@njit
def hyperbolic_wraper(dx: float, dt: float, x0: float = 0, xf: float = 1, t0 : float = 0, tf: float = 100, u0: Callable[[float], float] = func) -> Tuple[np.ndarray]:
    return hyperbolic(h=dx, k=dt, x0=x0, xf=xf, t0=t0, tf=tf, u0=u0)



dx = 0.01
# sol, x, t = hyperbolic(dx, 0.01, x0 = 0, xf = 1, t0 = 0, tf = 100)
sol, x, t = hyperbolic_wraper(dx, 0.01)




@njit
def c(t:float, s, a, tau0, r) -> float:    
    return np.array([ s/a**2 * np.exp(- t/tau0) if ri <= a else 0 for ri in r ])


@njit
def t_radiactive_rod(h: float, k: float, K: float, tau0: float, a: float,
                     r0: float, rf: float, t0: float, tf: float, u_t0: float,
                     u_rf:float) -> Tuple[np.ndarray]:

    s = k*K

    r = np.arange(r0+h, rf+h, h)
    t = np.arange(t0+k, tf+h, k)

    LEN_R = len(r)
    LEN_T = len(t)

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
        sol[:, ti] = m_inv @ (sol[:, ti-1] + c(t=ti, s=s, a=a, tau0=tau0, r=r))

    return sol, r, t


def t_radiactive_rod_wraper(h: float, k: float, K: float = 2*10**7, tau0: float = 100, a: float = 25,
                     r0: float = 0, rf: float = 100, t0: float = 0, tf: float = 100, u_t0: float = 3,
                     u_rf:float = 300) -> Tuple[np.ndarray]:
    
    return t_radiactive_rod(h=h, k=k, K=K, tau0=tau0, a=a, r0=r0, rf=rf, t0=t0, tf=tf, u_t0=u_t0,
                            u_rf=u_rf)



dt = 1
dr = 1

sol, r, t = t_radiactive_rod(dr, dt, K = 2*10**7, tau0 = 100, a = 25, r0 = 0, rf = 100, t0 = 0, tf = 100, u_t0 = 3, u_rf = 300)
