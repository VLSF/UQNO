import jax.numpy as jnp
import equinox as eqx
import time
import optax
import itertools
import os.path
import argparse

from jax.nn import gelu
from jax.lax import scan
from jax import random, jit, vmap, grad
from jax.tree_util import tree_map, tree_flatten
from jax.lax import dot_general, scan, dynamic_slice_in_dim
from jax.nn import gelu

def normalize_conv(A):
    A = eqx.tree_at(lambda x: x.weight, A, A.weight*jnp.sqrt(2/A.weight.shape[1]))
    A = eqx.tree_at(lambda x: x.bias, A, jnp.zeros_like(A.bias))
    return A

class FFNO(eqx.Module):
    encoder: eqx.Module
    decoder: eqx.Module
    convs1: list
    convs2: list
    A: jnp.array

    def __init__(self, N_layers, N_features, N_modes, key, D=1):
        n_in, n_processor, n_out = N_features

        keys = random.split(key, 3 + 2*N_layers)
        self.encoder = normalize_conv(eqx.nn.Conv(D, n_in, n_processor, 1, key=keys[-1]))
        self.decoder = normalize_conv(eqx.nn.Conv(D, n_processor, n_out, 1, key=keys[-2]))
        self.convs1 = [normalize_conv(eqx.nn.Conv(D, n_processor, n_processor, 1, key=key)) for key in keys[:N_layers]]
        self.convs2 = [normalize_conv(eqx.nn.Conv(D, n_processor, n_processor, 1, key=key)) for key in keys[N_layers:2*N_layers]]
        self.A = random.normal(keys[-3], [N_layers, n_processor, n_processor, N_modes, D], dtype=jnp.complex64)*jnp.sqrt(2/n_processor)

    def __call__(self, u, x):
        u = jnp.concatenate([x, u], 0)
        u = self.encoder(u)
        for conv1, conv2, A in zip(self.convs1, self.convs2, self.A):
            u += gelu(conv2(gelu(conv1(self.spectral_conv(u, A)))))
        u = self.decoder(u)
        return u

    def spectral_conv(self, v, A):
        u = 0
        N = v.shape
        for i in range(A.shape[-1]):
            u_ = dynamic_slice_in_dim(jnp.fft.rfft(v, axis=i+1), 0, A.shape[-2], axis=i+1)
            u_ = dot_general(A[:, :, :, i], u_, (((1,), (0,)), ((2, ), (i+1, ))))
            u_ = jnp.moveaxis(u_, 0, i+1)
            u += jnp.fft.irfft(u_, axis=i+1, n=N[i+1])
        return u

class FFNO_RBF_d(eqx.Module):
    FFNO_branch: eqx.Module
        
    def __init__(self, N_layers, N_features, N_modes, key, D=1):
        self.FFNO_branch = FFNO(N_layers, N_features, N_modes, key, D=D)
        
    def __call__(self, x_point, u, x, sigma):
        values = self.FFNO_branch(u, x)
        weights = jnp.exp(-jnp.sum((x_point.reshape(-1, 1, 1) - x)**2 / sigma**2, axis=0) / 2)
        return jnp.sum(values[0]*weights) * jnp.sin(x_point[1]*jnp.pi) * jnp.sin(x_point[0]*jnp.pi)
    
class FFNO_RBF(eqx.Module):
    FFNO_branch: eqx.Module
        
    def __init__(self, N_layers, N_features, N_modes, key, D=1):
        self.FFNO_branch = FFNO(N_layers, N_features, N_modes, key, D=D)
            
    def __call__(self, x_point, u, x, sigma):
        values = self.FFNO_branch(u, x)
        weights = jnp.exp(-jnp.sum((x_point.reshape(-1, 1, 1) - x)**2 / sigma**2, axis=0) / 2)
        return jnp.sum(values[0]*weights)
    
class FFNO_RBF3(eqx.Module):
    models: list
    beta: jnp.array
    sigma: jnp.array
        
    def __init__(self, N_layers, N_features, N_modes, key, D=1, sigma=1.0):
        keys = random.split(key, 3)
        self.models = [FFNO_RBF(N_layers, N_features, N_modes, key, D=D) for key in keys[:2]]
        self.models += [FFNO_RBF_d(N_layers, N_features, N_modes, keys[-1], D=D)]
        self.beta = jnp.array([1.0,])
        self.sigma = jnp.array([sigma, sigma])
        for _ in range(D):
            self.sigma = jnp.expand_dims(self.sigma, -1)
        
    def __call__(self, x_point, u, x, i):
        return self.models[i](x_point, u, x, self.sigma)

def get_flux(model, x, features, coords_f):
    return grad(model, argnums=0)(x, features, coords_f)

def get_laplacian(model, x, features, coords_f):
    return grad(lambda x: grad(model, argnums=0)(x, features, coords_f)[0])(x)[0] + grad(lambda x: grad(model, argnums=0)(x, features, coords_f)[1])(x)[1]

def compute_loss_(model, coordinates, rhs, a, C_F, features, coords_f):
    flux = vmap(grad(model, argnums=0), in_axes=(0, None, None, None), out_axes=1)(coordinates, features, coords_f, 2)
    dx_y = [vmap(grad(model, argnums=0), in_axes=(0, None, None, None), out_axes=1)(coordinates, features, coords_f, i)[i] for i in [0, 1]]
    y1 = vmap(model, in_axes=(0, None, None, None))(coordinates, features, coords_f, 0)
    y2 = vmap(model, in_axes=(0, None, None, None))(coordinates, features, coords_f, 1)
    loss = (1 + model.beta[0]**2)*(C_F**2*(rhs + dx_y[0] + dx_y[1])**2 +  ((a*flux[0] - y1)**2 + (a*flux[1] - y2)**2) / (a*model.beta[0]**2))
    return jnp.linalg.norm(loss)

def compute_loss(model, coordinates, rhs, a, C_F, features, coords_f):
    return jnp.mean(vmap(compute_loss_, in_axes=(None, None, 0, 0, 0, 0, None))(model, coordinates, rhs, a, C_F, features, coords_f))

def compute_error_energy_norm(model, coordinates, a, dx_sol, dy_sol, weights, features, coords_f):
    flux = vmap(grad(model, argnums=0), in_axes=(0, None, None, None), out_axes=1)(coordinates, features, coords_f, 2)
    integrand = a * ((flux[0] - dx_sol)**2 + (flux[1] - dy_sol)**2)
    l = jnp.sum(jnp.sum(integrand.reshape(weights.size, weights.size)*weights, axis=1) * weights[0]) / 4
    return l

def compute_relative_error(model, coords_eval, target_eval, feature, coordinates):
    prediction = vmap(model, in_axes=(0, None, None, None))(coords_eval, feature, coordinates, 2)
    error = jnp.linalg.norm(prediction - target_eval) / jnp.linalg.norm(target_eval)
    return error

def compute_upper_bound(model, coordinates, rhs, a, C_F, features, coords_f, weights):
    flux = vmap(grad(model, argnums=0), in_axes=(0, None, None, None), out_axes=1)(coordinates, features, coords_f, 2)
    dx_y = [vmap(grad(model, argnums=0), in_axes=(0, None, None, None), out_axes=1)(coordinates, features, coords_f, i)[i] for i in [0, 1]]
    y1 = vmap(model, in_axes=(0, None, None, None))(coordinates, features, coords_f, 0)
    y2 = vmap(model, in_axes=(0, None, None, None))(coordinates, features, coords_f, 1)
    integrand = (1 + model.beta[0]**2)*(C_F**2*(rhs + dx_y[0] + dx_y[1])**2 +  ((a*flux[0] - y1)**2 + (a*flux[1] - y2)**2) / (a*model.beta[0]**2))
    l = jnp.sum(jnp.sum(integrand.reshape(weights.size, weights.size)*weights, axis=1) * weights[0]) / 4
    return l

compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)

def make_step_scan(carry, ind, optim, M):
    model, coordinates, a, rhs, features, coords_f, C_F, opt_state = carry
    loss, grads = compute_loss_and_grads(model, coordinates[ind[:M]], rhs[ind[M:], :][:, ind[:M]], a[ind[M:], :][:, ind[:M]], C_F[ind[M:]], features[ind[M:]], coords_f)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, coordinates, a, rhs, features, coords_f, C_F, opt_state], loss

def get_argparser():
    parser = argparse.ArgumentParser()
    args = {
        "-N_layers": {
            "default": [4,],
            "nargs": '+',
            "type": int,
            "help": "number of layers"
        },
        "-N_modes": {
            "default": [12,],
            "nargs": '+',
            "type": int,
            "help": "number of modes along each dimension"
        },
        "-N_h_features": {
            "default": [24,],
            "nargs": '+',
            "type": int,
            "help": "number of features in a hidden layer"
        },
        "-gamma": {
            "default": [0.5,],
            "nargs": '+',
            "type": float,
            "help": "decay parameter for the exponential decay of learning rate"
        },
        "-N_updates": {
            "default": [50000,],
            "nargs": '+',
            "type": int,
            "help": "number of updates of the model weights"
        },
        "-N_drop": {
            "default": [25000,],
            "nargs": '+',
            "type": int,
            "help": "number of updates after which learning rate is multiplied by chosen learning rate decay"
        },
        "-N_batch_x": {
            "default": [15,],
            "nargs": '+',
            "type": int,
            "help": "number of points used along each dimension for gradient estimation"
        },
        "-N_batch_s": {
            "default": [10,],
            "nargs": '+',
            "type": int,
            "help": "number of PDE instances in a single batch"
        },
       "-path_to_dataset": {
            "help": "path to dataset in the .npz format"
        },
        "-path_to_results": {
            "help": "path to folder where to save results"
        },
        "-learning_rate": {
            "default": [1e-4,],
            "nargs": '+',
            "type": float,
            "help": "learning rate"
        },
        "-sigma": {
            "default": 1.0,
            "type": float,
            "help": "initial value of sigma"
        }
    }

    for key in args:
        parser.add_argument(key, **args[key])

    return parser

if __name__ == "__main__":
    parser = get_argparser()
    args = vars(parser.parse_args())
    Ns_layers = args["N_layers"]
    Ns_modes = args["N_modes"]
    Ns_h_features = args["N_h_features"]
    gammas = args["gamma"]
    Ns_updates = args["N_updates"]
    Ns_drop = args["N_drop"]
    N_batch_x = args["N_batch_x"]
    N_batch_s = args["N_batch_s"]
    learning_rates = args["learning_rate"]
    dataset_path = args["path_to_dataset"]
    results_path = args["path_to_results"]
    sigma = args["sigma"]
    
    N_train = 800
    key = random.PRNGKey(11)
    keys = random.split(key, 3)
    
    data = jnp.load(dataset_path)
    a_train = data["a_train"]
    rhs_train = data["rhs_train"]
    dx_a_train = data["dx_a_train"]
    dy_a_train = data["dy_a_train"]
    a_eval_legendre = data["a_eval_legendre"]
    dx_sol_eval_legendre = data["dx_sol_eval_legendre"]
    dy_sol_eval_legendre = data["dy_sol_eval_legendre"]
    rhs_legendre = data["rhs_legendre"]
    a_legendre = data["a_legendre"]
    sol_eval = data["sol_eval"]
    C_F = data["C_F"]
    coords_train = data["coords_train"]
    weights = data["weights"]
    coords_legendre = data["coords_legendre"]
    coords_eval = data["coords_eval"]
    
    coords_train_ = coords_train.T.reshape(-1, 64, 64)
    features = jnp.stack([(a_train / jnp.max(jnp.abs(a_train), keepdims=True)).reshape(-1, 64, 64), (rhs_train / jnp.max(jnp.abs(rhs_train), keepdims=True)).reshape(-1, 64, 64)], 1)
    
    header = "sigma,N_updates,Nx,N_batch,N_drop,N_modes,N_layers,learning_rate,gamma,N_h_features,mean_train_relative_error,std_train_relative_error,mean_train_energy_norm,std_train_energy_norm,mean_test_relative_error,std_test_relative_error,mean_test_energy_norm,std_test_energy_norm,final_loss,mean_train_upper_bound,std_train_upper_bound,mean_test_upper_bound,std_test_upper_bound,training_time"
    if not os.path.isfile(results_path):
        with open(results_path, "w") as f:
            f.write(header)
    
    iterators = [Ns_updates, N_batch_x, N_batch_s, Ns_drop, Ns_modes, Ns_layers, learning_rates, gammas, Ns_h_features]
    for N_updates, Nx, N_batch, N_drop, N_modes, N_layers, learning_rate, gamma, N_h_features in itertools.product(*iterators):
        M = Nx*Nx
        inds = random.choice(keys[0], a_train.shape[-1], (N_updates, M))
        inds_ = random.choice(keys[1], N_train, (N_updates, N_batch))
        inds = jnp.concatenate([inds, inds_], 1)

        N_features = [features.shape[1]+coords_train_.shape[0], N_h_features, 1]
        model = FFNO_RBF3(N_layers, N_features, N_modes, keys[2], D=2, sigma=sigma)

        sc = optax.exponential_decay(learning_rate, N_drop, gamma)
        optim = optax.lion(learning_rate=sc)
        opt_state = optim.init(eqx.filter(model, eqx.is_array))

        carry = [model, coords_train, a_train, rhs_train, features, coords_train_, C_F, opt_state]

        make_step_scan_ = lambda a, b: make_step_scan(a, b, optim, M)

        start = time.time()
        carry, loss = scan(make_step_scan_, carry, inds)
        stop = time.time()
        training_time = stop - start
        model = carry[0]

        inds = jnp.arange(sol_eval.shape[0])
        compute_relative_error_scan = lambda carry, ind: (carry, compute_relative_error(carry[0], carry[1], carry[2][ind], carry[3][ind], carry[4]))
        _, relative_errors = scan(compute_relative_error_scan, [model, coords_eval, sol_eval, features, coords_train_], inds)
        compute_energy_norm_scan = lambda carry, ind: (carry, jnp.sqrt(compute_error_energy_norm(carry[0], carry[1], carry[2][ind], carry[3][ind], carry[4][ind], carry[5], carry[6][ind], carry[7])))
        _, energy_norms = scan(compute_energy_norm_scan, [model, coords_legendre, a_eval_legendre, dx_sol_eval_legendre, dy_sol_eval_legendre, weights, features, coords_train_], inds)
        compute_upper_bound_scan = lambda carry, ind: (carry, jnp.sqrt(compute_upper_bound(carry[0], carry[1], carry[2][ind], carry[3][ind], carry[4][ind], carry[5][ind], carry[6], weights)))
        _, upper_bounds = scan(compute_upper_bound_scan, [model, coords_legendre, rhs_legendre, a_eval_legendre, C_F, features, coords_train_], inds)

        mean_train_relative_error = jnp.mean(relative_errors[:N_train])
        std_train_relative_error = jnp.sqrt(jnp.var(relative_errors[:N_train]))
        mean_train_energy_norm = jnp.mean(energy_norms[:N_train])
        std_train_energy_norm = jnp.sqrt(jnp.var(energy_norms[:N_train]))
        mean_train_upper_bound = jnp.mean(upper_bounds[:N_train])
        std_train_upper_bound = jnp.sqrt(jnp.var(upper_bounds[:N_train]))
        mean_test_upper_bound = jnp.mean(upper_bounds[N_train:])
        std_test_upper_bound = jnp.sqrt(jnp.var(upper_bounds[N_train:]))
        mean_test_relative_error = jnp.mean(relative_errors[N_train:])
        std_test_relative_error = jnp.sqrt(jnp.var(relative_errors[N_train:]))
        mean_test_energy_norm = jnp.mean(energy_norms[N_train:])
        std_test_energy_norm = jnp.sqrt(jnp.var(energy_norms[N_train:]))
        final_loss = loss[-1]
        
        res = f"{sigma},{N_updates},{Nx},{N_batch},{N_drop},{N_modes},{N_layers},{learning_rate},{gamma},{N_h_features},{mean_train_relative_error},{std_train_relative_error},{mean_train_energy_norm},{std_train_energy_norm},{mean_test_relative_error},{std_test_relative_error},{mean_test_energy_norm},{std_test_energy_norm},{final_loss},{mean_train_upper_bound},{std_train_upper_bound},{mean_test_upper_bound},{std_test_upper_bound},{training_time}"
        with open(results_path, "a") as f:
            f.write(res)