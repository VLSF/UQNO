import jax.numpy as jnp

from jax import config, jit, grad
from jax.lax import dot_general, scan
config.update("jax_enable_x64", True)

import optax
from functools import partial
import itertools

@jit
def d1(a, h):
    '''
    find derivative of a 1D functions given on uniform grid x

    a.shape = (N_features, N_x)
    h = grid spacing
    '''
    d_a = (jnp.roll(a, -1, axis=1) - jnp.roll(a, 1, axis=1)) / (2*h)
    d_a = d_a.at[:, 0].set((-3*a[:, 0]/2 + 2*a[:, 1] - a[:, 2]/2)/h) # 1/2	−2	3/2
    d_a = d_a.at[:, -1].set((a[:, -3]/2 - 2*a[:, -2] + 3*a[:, -1]/2)/h) # 1/2	−2	3/2
    return d_a

@partial(jit, static_argnums=(2,))
def d_fd(a, h, axis):
    '''
    find first derivative of nD function given on uniform grid

    a.shape = (N_features, N_x, N_y, N_z, ...) - input array for taking derivative
    h = grid spacing
    axis = dimension to take a derivative, 1 corresponds to dx, 2 orresponds to dy, ...
    '''
    d_a = d1(jnp.moveaxis(a, axis, 1), h)
    return jnp.moveaxis(d_a, axis, 1)

def energy_norm_a(u, a, x):
    # computes ||u||_a^2 with b = 0
    h = x[1] - x[0]
    a11, a12, a22 = a
    du_dx = d_fd(u, h, 1)[0]
    du_dy = d_fd(u, h, 2)[0]
    r = jnp.trapz(jnp.trapz(a11*du_dx**2 + 2*a12*du_dx*du_dy + a22*du_dy**2, x), x)
    return r

def energy_norm_b(u, b, x):
    # computes ||bu||_2^2
    r = jnp.trapz(jnp.trapz(b**2*u[0]**2, x), x)
    return r

def energy_norm(u, a, b, x):
    return energy_norm_a(u, a, x) + energy_norm_b(u, b, x)

def energy_norm_indicator(u, a, b, x):
    h = x[1] - x[0]
    a11, a12, a22 = a
    du_dx = d_fd(u, h, 1)[0]
    du_dy = d_fd(u, h, 2)[0]
    return a11*du_dx**2 + 2*a12*du_dx*du_dy + a22*du_dy**2 + b**2*u[0]**2

def compute_flux(u, a, x):
    # computes a\nabla u
    h = x[1] - x[0]
    a11, a12, a22 = a
    du_dx = d_fd(u, h, 1)[0]
    du_dy = d_fd(u, h, 2)[0]
    return jnp.stack([a11*du_dx + a12*du_dy, a12*du_dx + a22*du_dy], 0)

def inv_a_norm(u, a, x):
    # computes ||u||_{a^{-1}}^2
    h = x[1] - x[0]
    a11, a12, a22 = a
    d = a11*a22 - a12**2
    b11, b12, b22 = a22 / d, -a12 / d, a22 / d
    r = jnp.trapz(jnp.trapz(b11*u[0]**2 + 2*b12*u[0]*u[1] + b22*u[1]**2, x), x)
    return r

def inv_a_norm_indicator(u, a, x):
    h = x[1] - x[0]
    a11, a12, a22 = a
    d = a11*a22 - a12**2
    b11, b12, b22 = a22 / d, -a12 / d, a22 / d
    return b11*u[0]**2 + 2*b12*u[0]*u[1] + b22*u[1]**2

def min_eigenvalue(a):
    # computes \inf_x\min\lambda(a)
    a11, a12, a22 = a
    m = (a11 + a22)/2
    d = a11*a22 - a12**2
    lambda_min = m - jnp.sqrt(m**2 - d)
    return jnp.min(lambda_min)

def compute_C(a):
    # upper bound for C in 2D
    l_min = min_eigenvalue(a)
    return 1/(2*jnp.pi*jnp.sqrt(l_min))

def upper_bound(params, v, a, b, f, C, x):
    # complete upper bound for energy norm
    h = x[1] - x[0]
    y, beta = params
    a_v = compute_flux(v, a, x)
    R = f - b**2 * v[0] + d_fd(y[:1], h, 1)[0] + d_fd(y[1:], h, 2)[0]
    norm_a_ = inv_a_norm(a_v - y, a, x)*(1 + beta)/beta
    return jnp.trapz(jnp.trapz(R**2*C**2*(1+beta)/(C**2*(1+beta)*b**2 + 1), x), x) + norm_a_

def upper_bound_indicator(params, v, a, b, f, C, x):
    h = x[1] - x[0]
    y, beta = params
    a_v = compute_flux(v, a, x)
    R = f - b**2 * v[0] + d_fd(y[:1], h, 1)[0] + d_fd(y[1:], h, 2)[0]
    norm_a_ = inv_a_norm_indicator(a_v - y, a, x)*(1 + beta)/beta
    return norm_a_ + R**2*C**2*(1+beta)/(C**2*(1+beta)*b**2 + 1)

def compute_J(u, a, b, f, x):
    return energy_norm_a(u, a, x)/2 + energy_norm_b(u, b, x)/2 - jnp.trapz(jnp.trapz(f*u[0], x), x)

def compute_lower_bound(params, u, a, b, f, x):
    v, = params
    return 2*(compute_J(u, a, b, f, x) - compute_J(u+v, a, b, f, x))

def even_cubic(v):
    v_ = jnp.hstack([(5*v[0] + 15*v[1] - 5*v[2] + v[3])/16, (-v[:-3] + 9*jnp.roll(v, -1)[:-3] + 9*jnp.roll(v, -2)[:-3] - jnp.roll(v, -3)[:-3])/16, (v[-4] - 5*v[-3] + 15*v[-2] + 5*v[-1])/16])
    interpolated_coarse = jnp.zeros((v_.shape[0]+v.shape[0], ))
    interpolated_coarse = interpolated_coarse.at[::2].set(v)
    interpolated_coarse = interpolated_coarse.at[1::2].set(v_)
    return interpolated_coarse

def compute_residual(u, a, b, f, x):
    a11, a12, a22 = a
    h = x[1] - x[0]
    ux = d_fd(u, h, 1)
    uy = d_fd(u, h, 2)
    res = f - (-d_fd(ux*jnp.expand_dims(a11, 0), h, 1)[0] - d_fd(uy*jnp.expand_dims(a22, 0), h, 2)[0] - d_fd(ux*jnp.expand_dims(a12, 0), h, 2)[0] - d_fd(uy*jnp.expand_dims(a12, 0), h, 1)[0] + b**2 * u[0])
    return res

def compute_loss(params, problem_data):
    solution_coarse, a_coarse, b_coarse, f_coarse, C, x_coarse = problem_data
    return upper_bound([params[0], params[1]**2], solution_coarse, a_coarse, b_coarse, f_coarse, C, x_coarse)[0]

@partial(jit, static_argnums=(2,))
def make_step(params, problem_data, optim, opt_state):
    grads = grad(compute_loss)(params, problem_data)
    updates, opt_state = optim.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state

def compute_certificate(problem_data, optim, N_epoch=1000):
    flux = compute_flux(problem_data[0], problem_data[1], problem_data[-1])
    params = [flux, jnp.array([1.0,])]
    opt_state = optim.init(params)

    def opt_step(carry, ind):
        params, opt_state, problem_data = carry
        params, opt_state = make_step(params, problem_data, optim, opt_state)
        r = compute_loss(params, problem_data)
        carry = params, opt_state, problem_data
        return carry, r

    carry = params, opt_state, problem_data
    t_ = jnp.arange(N_epoch)
    res, errors = scan(opt_step, carry, t_)
    return [res[0][0], res[0][1]**2]

def compute_smoothed_loss(params, model, problem_data):
    solution_coarse, a_coarse, b_coarse, f_coarse, C, x_coarse = problem_data
    return upper_bound([model(params[0]), params[1]**2], solution_coarse, a_coarse, b_coarse, f_coarse, C, x_coarse)[0]

@partial(jit, static_argnums=(1, 3,))
def make_smoothed_step(params, model, problem_data, optim, opt_state):
    grads = grad(compute_smoothed_loss)(params, model, problem_data)
    updates, opt_state = optim.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state

def compute_smoothed_certificate(problem_data, optim, N_epoch=1000, N=7):
    flux = compute_flux(problem_data[0], problem_data[1], problem_data[-1])
    K = problem_data[0].shape[-1]
    x = jnp.linspace(0, 1, K)
    x, y = jnp.meshgrid(x, x)
    M = jnp.stack([(x**i*y**j).reshape(-1,) for i, j in itertools.product(range(N), repeat=2)], 0)
    model = lambda params: (params @ M).reshape(-1, K, K)
    Q, R = jnp.linalg.qr(M.T)
    p = flux.reshape(2, -1) @ Q
    params = [jnp.linalg.solve(R, p.T).T, jnp.array([1.0,])]
    opt_state = optim.init(params)

    def opt_step(carry, ind):
        params, opt_state, problem_data = carry
        params, opt_state = make_smoothed_step(params, model, problem_data, optim, opt_state)
        r = compute_smoothed_loss(params, model, problem_data)
        carry = params, opt_state, problem_data
        return carry, r

    carry = params, opt_state, problem_data
    t_ = jnp.arange(N_epoch)
    res, errors = scan(opt_step, carry, t_)
    return [model(res[0][0]), res[0][1]**2]
