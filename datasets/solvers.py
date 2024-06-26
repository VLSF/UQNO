import jax.numpy as jnp

from jax.experimental.sparse import BCOO
from scipy.special import roots_legendre
from jax import config
import numpy as np
import itertools
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from jax.lax import scan

config.update("jax_enable_x64", True)

def tent(x, center, h):
    left_mask = (x <= center) & (x >= (center - h))
    right_mask = (x > center) & (x <= (center + h))
    y = (x - center)/h
    return (1. - y) * right_mask + (1. + y) * left_mask

def d_tent(x, center, h):
    left_mask = (x < center) & (x >= (center - h))
    right_mask = (x > center) & (x <= (center + h))
    return (-1. * right_mask + left_mask) / h

def FEM_2D(N_points, F, M_int=8, backend="jax"):
    # dx (a1 dx) + dy (a2 dy) + dx (a12 dy) + dy (a21 dx) + dx b1 + dy b2 + c
    a1, a2, a12, a21, b1, b2, c, f = F
    coords, w = roots_legendre(M_int)
    w = jnp.array(w)
    w_ = jnp.hstack([w, w])
    ind = []
    values = []
    rhs = []
    h = 1/(N_points+1)
    lex = lambda i, j: i + N_points*j

    def discretization(carry, ind):
        w_, coords, h  = carry
        j, i = ind[0], ind[1]
        x, y = (i+1)*h, (j+1)*h
        x_ = jnp.hstack([x - h/2 + coords*h/2, x + h/2 + coords*h/2])
        y_ = jnp.hstack([y - h/2 + coords*h/2, y + h/2 + coords*h/2])
        X_, Y_ = jnp.meshgrid(x_, y_, indexing="ij")
        a_1 = - a1(X_, Y_) * d_tent(x_, x, h).reshape(-1, 1) * tent(y_, y, h).reshape(1, -1)
        a_12 = - a12(X_, Y_) * d_tent(x_, x, h).reshape(-1, 1) * tent(y_, y, h).reshape(1, -1)
        a_2 = - a2(X_, Y_) * d_tent(y_, y, h).reshape(1, -1) * tent(x_, x, h).reshape(-1, 1)
        a_21 = - a21(X_, Y_) * d_tent(y_, y, h).reshape(1, -1) * tent(x_, x, h).reshape(-1, 1)
        b_1 = - b1(X_, Y_) * d_tent(x_, x, h).reshape(-1, 1) * tent(y_, y, h).reshape(1, -1)
        b_2 = - b2(X_, Y_) * d_tent(y_, y, h).reshape(1, -1) * tent(x_, x, h).reshape(-1, 1)
        c_ = c(X_, Y_) * tent(x_, x, h).reshape(-1, 1) * tent(y_, y, h).reshape(1, -1)
        res = [] # row, col, value, i, j
        for p1, p2 in itertools.product([-1, 0, +1], repeat=2):
            v = a_1 * d_tent(x_, x + p1*h, h).reshape(-1, 1) * tent(y_, y + p2*h, h).reshape(1, -1)
            v += a_12 * d_tent(y_, y + p2*h, h).reshape(1, -1) * tent(x_, x + p1*h, h).reshape(-1, 1)
            v += a_2 * d_tent(y_, y + p2*h, h).reshape(1, -1) * tent(x_, x + p1*h, h).reshape(-1, 1)
            v += a_21 * d_tent(x_, x + p1*h, h).reshape(-1, 1) * tent(y_, y + p2*h, h).reshape(1, -1)
            v += (b_1 + b_2 + c_) * tent(y_, y + p2*h, h).reshape(1, -1) * tent(x_, x + p1*h, h).reshape(-1, 1)
            res += [[lex(i, j), lex(i+p1, j+p2), w_ @ v @ w_, i + p1, j + p2]]
        return carry, jnp.array(res)

    def rhs_discretization(carry, ind):
        w_, coords, h  = carry
        j, i = ind[0], ind[1]
        x, y = (i+1)*h, (j+1)*h
        x_ = jnp.hstack([x - h/2 + coords*h/2, x + h/2 + coords*h/2])
        y_ = jnp.hstack([y - h/2 + coords*h/2, y + h/2 + coords*h/2])
        X_, Y_ = jnp.meshgrid(x_, y_, indexing="ij")
        res = w_ @ (f(X_, Y_) * tent(x_, x, h).reshape(-1, 1) * tent(y_, y, h).reshape(1, -1)) @ w_
        return carry, res

    xs = jnp.array([*itertools.product(range(N_points), repeat=2)])
    carry = [w_, coords, h]
    carry, res = scan(discretization, carry, xs)
    carry, rhs = scan(rhs_discretization, carry, xs)

    for_mask = jnp.array(res[:, :, 3:], dtype=int)
    mask = ((for_mask[:, :, 0] < N_points) & (0 <= for_mask[:, :, 0]) & (for_mask[:, :, 1] < N_points) & (0 <= for_mask[:, :, 1])).reshape(-1, )
    res = res[:, :, :3].reshape(-1, 3)[mask]
    ind = jnp.array(res[:, :2], dtype=int)
    values = res[:, 2]
    if backend == "jax":
        return rhs, BCOO((values, ind), shape=(N_points**2, N_points**2))
    else:
        return np.array(rhs), coo_matrix((np.array(values), (np.array(ind[:, 0]), np.array(ind[:, 1]))), shape=(N_points**2, N_points**2)).tocsr()

def solve_BVP_2D_scipy(N_points, F):
    rhs, A = FEM_2D(N_points-2, F, M_int=8, backend="scipy")
    sol_ = spsolve(A, rhs)
    sol = np.zeros((N_points, N_points))
    sol[1:-1, 1:-1] = sol_.reshape((N_points-2, N_points-2))
    return jnp.array(sol.reshape(-1,))
