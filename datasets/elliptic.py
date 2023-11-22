from datasets import solvers

import jax.numpy as jnp
import itertools
from datasets import utilities
from jax import config, random

def random_polynomial_2D(x, y, coeff):
    res = 0
    for i, j in itertools.product(range(coeff.shape[0]), repeat=2):
        res += coeff[i, j]*jnp.exp(2*jnp.pi*x*i*1j)*jnp.exp(2*jnp.pi*y*j*1j)/(1+i+j)**2
    res = jnp.real(res)
    return res

def get_functions_1(key):
    c_ = random.normal(key, (5, 5, 5), dtype=jnp.complex128)
    alpha = lambda x, y, c=c_[0]: random_polynomial_2D(x, y, c)*0.1 + 1
    beta = lambda x, y, c=c_[1]: random_polynomial_2D(x, y, c)*0.1 + 1
    gamma = lambda x, y, c=c_[2]: random_polynomial_2D(x, y, c)
    c = lambda x, y, c=c_[3]: random_polynomial_2D(x, y, c)
    rhs = lambda x, y, c=c_[4]: random_polynomial_2D(x, y, c)
    a11 = lambda x, y: alpha(x, y)**2
    a12 = lambda x, y: alpha(x, y)*gamma(x, y)
    a22 = lambda x, y: beta(x, y)**2 + gamma(x, y)**2
    return a11, a22, a12, c, rhs

def get_functions_2(key):
    c_ = random.normal(key, (5, 5), dtype=jnp.complex128)
    def alpha(x, y, c=c_):
        g = random_polynomial_2D(x, y, c)
        return (g >= 0) * 10 + (g < 0)

    c = lambda x, y: jnp.zeros_like(x)
    rhs = lambda x, y: jnp.ones_like(x)
    a11 = lambda x, y: alpha(x, y)
    a12 = lambda x, y: jnp.zeros_like(x)
    a22 = lambda x, y: alpha(x, y)
    return a11, a22, a12, c, rhs

def get_functions_3(key):
    c_ = random.normal(key, (3, 5, 5), dtype=jnp.complex128)
    def alpha(x, y, c=c_[0]):
        g = random_polynomial_2D(x, y, c)
        return (g >= 0) * 10 + (g < 0)

    c = lambda x, y, c=c_[1]: random_polynomial_2D(x, y, c)
    rhs = lambda x, y, c=c_[2]: random_polynomial_2D(x, y, c)
    a11 = lambda x, y: alpha(x, y)
    a12 = lambda x, y: jnp.zeros_like(x)
    a22 = lambda x, y: alpha(x, y)
    return a11, a22, a12, c, rhs

def get_functions_4(key):
    c_ = random.normal(key, (2, 5, 5), dtype=jnp.complex128)
    def alpha(x, y, c=c_[0]):
        g = random_polynomial_2D(x, y, c)
        return (g >= 0) * 10 + (g < 0)

    def beta(x, y, c=c_[0]):
        g = random_polynomial_2D(x, y, c)
        return (g >= 0) + (g < 0) * 5

    def gamma(x, y, c=c_[0]):
        g = random_polynomial_2D(x, y, c)
        return (g >= 0) * 3 + (g < 0)

    c = lambda x, y, c=c_[1]: random_polynomial_2D(x, y, c)
    rhs = lambda x, y: jnp.ones_like(x)
    a11 = lambda x, y: alpha(x, y)
    a12 = lambda x, y: gamma(x, y)
    a22 = lambda x, y: beta(x, y)
    return a11, a22, a12, c, rhs

def get_functions_5(key):
    c_ = random.normal(key, (5, 5, 5), dtype=jnp.complex128)
    alpha = lambda x, y, c=c_[0]: random_polynomial_2D(x, y, c)*0.1 + 1
    beta = lambda x, y, c=c_[1]: random_polynomial_2D(x, y, c)*0.1 + 1
    gamma = lambda x, y, c=c_[2]: random_polynomial_2D(x, y, c)
    c = lambda x, y, c=c_[3]: jnp.zeros_like(x)
    rhs = lambda x, y, c=c_[4]: random_polynomial_2D(x, y, c)
    a11 = lambda x, y: alpha(x, y)**2
    a12 = lambda x, y: alpha(x, y)*gamma(x, y)
    a22 = lambda x, y: beta(x, y)**2 + gamma(x, y)**2
    return a11, a22, a12, c, rhs

def get_functions_6(key):
    c_ = random.normal(key, (3, 5, 5), dtype=jnp.complex128)
    def alpha(x, y, c=c_[0]):
        g = random_polynomial_2D(x, y, c)
        return (g >= 0) * 100 + (g < 0)

    c = lambda x, y, c=c_[1]: random_polynomial_2D(x, y, c)
    rhs = lambda x, y, c=c_[2]: random_polynomial_2D(x, y, c)
    a11 = lambda x, y: alpha(x, y)
    a12 = lambda x, y: jnp.zeros_like(x)
    a22 = lambda x, y: alpha(x, y)
    return a11, a22, a12, c, rhs

def get_functions_scale(key, scale):
    keys = random.split(key, 2)
    c = random.normal(keys[0], (5, 5), dtype=jnp.complex128)
    x = jnp.linspace(0, 1, 100)
    x, y = jnp.meshgrid(x, x)
    F = random_polynomial_2D(x, y, c)
    min_F, max_F = jnp.min(F), jnp.max(F)
    s = random.uniform(keys[1], minval=1.0, maxval=scale)
    def alpha(x, y, c=c, a=min_F, b=max_F, s=s):
        g = random_polynomial_2D(x, y, c)
        return s * (g - a) / (b - a) + 1
    c = lambda x, y: jnp.zeros_like(x)
    rhs = lambda x, y: jnp.ones_like(x)
    a11 = lambda x, y: alpha(x, y)
    a12 = lambda x, y: jnp.zeros_like(x)
    a22 = lambda x, y: alpha(x, y)
    return a11, a22, a12, c, rhs

def get_F(a11, a22, a12, c, rhs):
    return [a11, a22, a12, a12, lambda x, y: jnp.zeros_like(x), lambda x, y: jnp.zeros_like(x), lambda x, y: -c(x, y)**2, lambda x, y: -rhs(x, y)]

def get_data(J, F):
    N_points = 2**J + 1
    x = jnp.linspace(0, 1, N_points)
    X, Y = jnp.meshgrid(x, x, indexing="ij")
    solution = solvers.solve_BVP_2D_scipy(N_points, F).reshape(1, N_points, N_points, order="F")

    a = [F[0](X, Y), F[2](X, Y), F[1](X, Y)]
    b = jnp.sqrt(-F[6](X, Y))
    f = -F[-1](X, Y)

    return solution, a, b, f, x
