import numpy as np
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle


def run_animation(dx, x, y):

    fig = plt.figure()
    ax = plt.axes(xlim=(-0.2, 1.2), ylim=(-0.025, 0.025))
    line, = ax.plot([], [])

    def init():
        line.set_data([], [])
        return line,

    def animate(i, *fargs):
        line.set_data(fargs[0], fargs[1][:, i])
        # print(i)
        return line,

    anim = animation.FuncAnimation(fig, animate, fargs=[x, y], init_func=init, frames=len(x), interval=400, blit=True)
    plt.show()


def hyperbolic(h: float, k: float, x0: float, xf: float, t0: float, tf: float, u0: Callable) -> Tuple[np.ndarray]:
    r'''Computes, a hyperbolic PDE of the kind:
    :math: `$$\begin{array}{l}
                U_{tt}=U_{xx} \\
                U(x0,t)=U(xf,t)=0 \; \textrm{ and } \; \frac{\partial U}{\partial t}(x, t0)=0 \\
                U(x,t0)=u0(x) \\
                \end{array}$$`

    over the interval :math: `$[t0,tf]$` for a stepsize `h` in x and `k` in t.

    Args:
        h (float): Step size in x.
        k (float): Step size in t.
        x0 (float): Initial position.
        xf (float): Final position.
        t0 (float): Initial time.
        tf (float): Final time.
        u0 (function): Function of x in t0. u(x, t0).

    Returns:
        Tuple[np.ndarray]: Solution of the PDE in those intervals, x mesh, t mesh.
    '''
    rho = (k/h)**2

    x = np.arange(x0, xf+h, h)
    t = np.arange(t0, tf+h, h)

    LEN_X = len(x)
    LEN_T = len(t)

    sol = np.zeros((LEN_X, LEN_T))

    sol[1:-1, 0] = u0(x)[1:-1]

    sol[1:-1, 1] = 1/2 * rho * (u0(x+h)[1:-1] - u0(x-h)[1:-1]) + (1 - rho) * u0(x)[1:-1]

    for ti in range(2, LEN_T):
        sol[1:-1, ti] = rho*sol[2:, ti] + 2*(1 - rho)*sol[1:-1, ti] + rho*sol[:-2, ti]  - sol[1:-1, ti-1]

    return sol, x, t


def hyperbolic_wraper(dx: float, dt: float) -> Tuple[np.ndarray]:
    return hyperbolic(h=dx, k=dt, x0=0, xf=1, t0=0, tf=100, u0=np.sin)




dt = 0.0001
sol, x, t = hyperbolic_wraper(0.001, dt)

run_animation(dt, x, sol)