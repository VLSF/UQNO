import warnings
warnings.filterwarnings('ignore')

import jax
import optax
import os, sys
import cloudpickle
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import matplotlib.pyplot as plt

from jax.nn import relu, leaky_relu, hard_tanh, gelu
from architectures import DilResNet, fSNO, UNet, ChebNO
from tqdm import tqdm
from IPython import display
from jax import config, random, grad, jit, hessian, vmap
from transforms import utilities, cheb
from jax.lax import scan
from functools import partial
from transforms import integrals_and_derivatives as int_diff

%matplotlib inline
%config InlineBackend.figure_format='retina'

def energy_norm_a(u, a):
    # computes ||u||_a^2 with b = 0
    h = 1 / u.shape[-1]
    a11, a12, a22 = a
    du_dx = int_diff.d_fd(u, h, 1)[0]
    du_dy = int_diff.d_fd(u, h, 2)[0]
    r = jnp.trapz(jnp.trapz(a11*du_dx**2 + 2*a12*du_dx*du_dy + a22*du_dy**2, dx=h), dx=h)
    return r

def energy_norm_b(u, b):
    # computes ||bu||_2^2
    h = 1 / u.shape[-1]
    r = jnp.trapz(jnp.trapz(b*(u[0]**2), dx=h), dx=h)
    return r

def energy_norm(u, a, b):
    return energy_norm_a(u, a) + energy_norm_b(u, b)

def compute_flux(u, a):
    # computes a\nabla u
    h = 1 / u.shape[-1]
    a11, a12, a22 = a
    du_dx = int_diff.d_fd(u, h, 1)[0]
    du_dy = int_diff.d_fd(u, h, 2)[0]
    return jnp.stack([a11*du_dx + a12*du_dy, a12*du_dx + a22*du_dy], 0)

def inv_a_norm(u, a):
    # computes ||u||_{a^{-1}}^2
    h = 1 / u.shape[-1]
    a11, a12, a22 = a
    d = a11*a22 - a12**2
    b11, b12, b22 = a22 / d, -a12 / d, a22 / d
    r = jnp.trapz(jnp.trapz(b11*u[0]**2 + 2*b12*u[0]*u[1] + b22*u[1]**2, dx=h), dx=h)
    return r

def upper_bound(params, v, a, b, f, C):
    # complete upper bound for energy norm
    h = 1 / v.shape[-1]
    # fixed beta
    beta = 1 
    y = params
    a_v = compute_flux(v, a)
    R = f - b**2 * v[0] + int_diff.d_fd(y[0], h, 0) + int_diff.d_fd(y[1], h, 1)
    norm_a_ = inv_a_norm(a_v - y, a)*(1 + beta)/beta
    return jnp.sqrt(jnp.trapz(jnp.trapz(R**2*C**2*(1+beta)/(C**2*(1+beta)*b**2 + 1), dx=h), dx=h) + norm_a_)

def compute_J(u, a, b, f):
    h = 1 / u.shape[-1]
    return energy_norm_a(u, a)/2 + energy_norm_b(u, b)/2 - jnp.trapz(jnp.trapz(f*u[0], dx=h), dx=h)

def loss_with_boudaries(params, v, a, b, f, C, weight):
    return upper_bound(params, v, a, b, f, C) + weight * jnp.sqrt(jnp.sum(v[0][:, 0]**2) + jnp.sum(v[0][:, -1]**2) 
                                                           + jnp.sum(v[0][0, 1:-1]**2) + jnp.sum(v[0][-1, 1:-1]**2))

def compute_loss(model, input, c, weight):
    output = vmap(model)(input)
    v, params = output[:, :1], output[:, 1:]
    a, b, f = input[:, :3], input[:, 3], input[:, 4]

    loss = jnp.mean(vmap(loss_with_boudaries, in_axes=(0, 0, 0, 0, 0, 0, None))(params, v, a, b, f, c, weight))
    return loss

compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)

@eqx.filter_jit
def make_step(model, input, c, optim, opt_state, weight):
    loss, grads = compute_loss_and_grads(model, input, c, weight)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

def compute_loss_fsno(model, input, c, analysis, synthesis, weight):
    output = vmap(lambda z: model(z, analysis, synthesis), in_axes=(0,))(input)
    v, params = output[:, :1], output[:, 1:]
    a, b, f = input[:, :3], input[:, 3], input[:, 4]

    loss = jnp.mean(vmap(loss_with_boudaries, in_axes=(0, 0, 0, 0, 0, 0, None))(params, v, a, b, f, c, weight))
    return loss

compute_loss_and_grads_fsno = eqx.filter_value_and_grad(compute_loss_fsno)

@eqx.filter_jit
def make_step_fsno(model, input, c, optim, opt_state, analysis, synthesis, weight):
    loss, grads = compute_loss_and_grads_fsno(model, input, c, analysis, synthesis, weight)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

def compute_exact_energy_integrand(u, f):
    h = 1 / u.shape[-1]
    return -jnp.trapz(jnp.trapz(f*u/2, dx=h), dx=h)

def compute_energy(a, b, u1, u2):
    return jnp.sqrt(energy_norm_a((u2 - u1), a) + energy_norm_b((u2 - u1), b))

def compute_energy_relative(a, b, u1, u2):
    return compute_energy(a, b, u1, u2) / jnp.sqrt(energy_norm_a((u1), a) + energy_norm_b((u1), b))


# -

# ## Models

def get_DilResNet(features_train, key):
    input = features_train[0]
    D = len(input.shape[1:])
    
    # Parameters of model
    kernel_size = 3
    channels = [input.shape[0], 32, 1+D]
    n_cells = 5
    
    model = DilResNet.DilatedResNet(key, channels, n_cells, D=D)
    
    # Parameters of training
    N_epoch = 500
    batch_size = 16
    learning_rate = 2e-3
    
    model_data = {
        "model_name": "DilResNet",
        "model": model,
        "compute_loss": compute_loss
    }
    
    optimization_specification = {
        "learning_rate": learning_rate,
        "make_step": make_step,
        "N_epochs": N_epoch,
        "batch_size": batch_size,
    }
    
    return model_data, optimization_specification


def get_ChebNO(features_train, key):
    input = features_train[0]
    D = len(input.shape[1:])

    # Parameters of model
    kernel_size = 3
    n_conv = 3
    channels = [input.shape[0], 32, 1+D]
    n_cells = 4
    N_modes = [16,]*D
    weights = [(-1)**jnp.arange(n) for n in input.shape[1:]]

    model = ChebNO.ChebNO(key, n_cells, channels, kernel_size, n_conv, D, N_modes, weights)
    
    # Parameters of training
    N_epoch = 500
    batch_size = 16
    learning_rate = 2e-3
    
    model_data = {
        "model_name": "ChebNO",
        "model": model,
        "compute_loss": compute_loss
    }
    
    optimization_specification = {
        "learning_rate": learning_rate,
        "make_step": make_step,
        "N_epochs": N_epoch,
        "batch_size": batch_size,
    }
    
    return model_data, optimization_specification


def get_SNO(features_train, key):
    input = features_train[0]
    D = len(input.shape[1:])
    
    # Parameters of model
    polynomials = ["Chebyshev_t", ] * D
    parameters = [[0.1, 0.1],] * D
    M_keep = [20, ] * D
    N_points = input.shape[1:]
    grids = N_points
    
    data = {
        "polynomials": polynomials,
        "parameters": parameters,
        "grids": grids,
        "M_keep": M_keep,
        "N_points": N_points
    }
    
    synthesis = utilities.get_operators("synthesis", **data)
    analysis = utilities.get_operators("analysis", **data)
    
    kernel_size = 3
    n_conv_layers = 3
    cell = lambda features, key: DilResNet.DilatedConvBlock([features,]*(n_conv_layers + 1), [[1,]*D, ]*n_conv_layers, [[kernel_size,]*D, ]*n_conv_layers, key, activation=lambda x: x)
    
    input_shape = input.shape
    N_features_out = 1+D
    N_features = 32
    N_layers = 4

    model = fSNO.fSNO(input_shape, N_features_out, N_layers, N_features, cell, key)
    
    # Parameters of training
    N_epoch = 500
    batch_size = 16
    learning_rate = 2e-3
    
    model_data = {
        "model_name": "fSNO",
        "model": model,
        "compute_loss": lambda model, input, C, weight: compute_loss_fsno(model, input, C, analysis, synthesis, weight),
        "analysis": analysis,
        "synthesis": synthesis
        }
    
    optimization_specification = {
        "learning_rate": learning_rate,
        "make_step": lambda model, input, C, optim, opt_state, weight: make_step_fsno(model, input, C, optim, opt_state, analysis, synthesis, weight),
        "N_epochs": N_epoch,
        "batch_size": batch_size
    }
    return model_data, optimization_specification


def get_UNet(features_train, key):
    input = features_train[0]
    D = len(input.shape[1:])
    
    N_convs = 2
    input_features = input.shape[0]
    internal_features = 10
    output_features = 1+D
    kernel_size = 3
    
    model = UNet.UNet(D, input.shape[1], [input_features, internal_features, output_features], kernel_size, N_convs, key, depth=4)
    
    # Parameters of training
    N_epoch = 500
    batch_size = 32
    learning_rate = 6e-3
    
    model_data = {
        "model_name": "UNet",
        "model": model,
        "compute_loss": compute_loss
    }
    
    optimization_specification = {
        "learning_rate": learning_rate,
        "make_step": make_step,
        "N_epochs": N_epoch,
        "batch_size": batch_size,
    }
    return model_data, optimization_specification


# ## Training

def batch_generator(x, c, batch_size, key, shuffle=True):
    N_samples = len(x)
    list_of_indeces = jnp.linspace(0, N_samples-1, N_samples, dtype=jnp.int64)

    if shuffle:
        random.shuffle(key, list_of_indeces)

    list_x = x[list_of_indeces]
    list_c = c[list_of_indeces]

    n_batches = N_samples // batch_size
    if N_samples % batch_size != 0:
        n_batches += 1

    for k in range(n_batches):
        this_batch_size = batch_size

        if k == n_batches - 1:
            if N_samples % batch_size > 0:
                this_batch_size = N_samples % batch_size

        This_X = list_x[k * batch_size : k * batch_size + this_batch_size]
        This_c = list_c[k * batch_size : k * batch_size + this_batch_size]
        x = jnp.array(This_X)
        c = jnp.array(This_c)

        yield x, c


def train_on_epoch(train_generator, model, optimizer, opt_state, make_step, weight):
    epoch_loss = []
    for it, (batch_of_x, batch_of_c) in enumerate(train_generator):
        batch_loss, model, opt_state = make_step(model, batch_of_x, batch_of_c, optimizer, opt_state, weight)
        epoch_loss.append(batch_loss.item())
        
    return epoch_loss, model, opt_state


def train_model(model_data, train_data, C, optimization_specification, weight, plot = True):
    model = model_data["model"]
    
    c = train_data[0].shape[0] // optimization_specification["batch_size"]
    keys = [value * c for value in np.arange(50, 1050, 50)]
    values = [0.5, ] * len(keys)
    dict_lr = dict(zip(keys, values))
    
    sc = optax.piecewise_constant_schedule(optimization_specification["learning_rate"], dict_lr)
    if model_data['model_name']=='FNO':
        optimizer = optax.experimental.split_real_and_imaginary(optax.adamw(sc, weight_decay=1e-2))
    else:
        optimizer = optax.adamw(sc, weight_decay=1e-2)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    train_losses = []    
    
    iterations = tqdm(range(optimization_specification["N_epochs"]), desc='N_epochs')
    iterations.set_postfix({'train epoch loss': jnp.nan})
    
    for it in iterations:
        key = random.PRNGKey(it)
        generator = batch_generator(x=train_data[0], c=C, batch_size=optimization_specification["batch_size"], key=key, shuffle=True)
        epoch_loss, model, opt_state = train_on_epoch(generator, model, optimizer, opt_state, optimization_specification['make_step'], weight)
        iterations.set_postfix({'train epoch loss': epoch_loss})
        train_losses.append(jnp.array(epoch_loss).mean())
        
        if plot:
            a, b, f = train_data[0][:, :3], train_data[0][:, 3], train_data[0][:, 4]
            if model_data["model_name"] == 'fSNO':
                output = vmap(lambda z: model(z, model_data['analysis'], model_data['synthesis']), in_axes=(0,))(train_data[0])[:, 0]
            else:    
                output = vmap(model)(train_data[0])[:, 0]
        
            energy_norm = vmap(compute_energy, in_axes=(0, 0, 0, 0))(a, b, train_data[1], output)
            display.clear_output(wait=True)
            fig, axes = plt.subplots(1, 4, figsize=(18, 4))

            axes[0].set_title(r'Loss')
            axes[0].set_yscale("log")
            axes[0].plot(train_losses, color='red')

            axes[1].set_title("Energy norm")
            axes[1].set_yscale("log")
            axes[1].plot(energy_norm, ".", color="red")

            axes[2].imshow(output[101,0])
            axes[2].set_title("Prediction")

            axes[3].imshow(targets[101,0])
            axes[3].set_title("Target")

            plt.tight_layout()
            plt.show()
    return model, train_losses


def train_run(model_name, dataset_path, train_size = None, weight = None, plot = True):
    dataset = jnp.load(dataset_path)
    features, targets = dataset["features"], dataset['exact_sol']
    C = dataset['C'][:, None]
    N = dataset["features"].shape[-1]
    
    if features.shape[1] == 2:
        features = jnp.concatenate([features[:, :1], jnp.zeros([features.shape[0], 1, features.shape[-1]]), features[:, 1:]], axis=1)
    
    if features.shape[0] == 3000:
        features_, targets_ = features[:2000], targets[:2000]
        C_ = C[:2000]
    else:
        features_, targets_ = features, targets
        C_ = C
    
    features_train, targets_train = features_[:train_size], targets_[:train_size]
    
    N_test = 1800
    features_test, targets_test = features_[N_test:], targets_[N_test:]
    
    if train_size != None:
        features_train, targets_train = features_train[:train_size], targets_train[:train_size]
    
    key = random.PRNGKey(123)
    if model_name == "DilResNet":
        model_data, optimization_specification = get_DilResNet(features_train, key)
    elif model_name == "ChebNO":
        model_data, optimization_specification = get_ChebNO(features_train, key)
    elif model_name == "fSNO":
        model_data, optimization_specification = get_SNO(features_train, key)
    elif model_name == 'UNet':
        model_data, optimization_specification = get_UNet(features_train, key)
    if train_size != None:
        C_train, C_test = C_[:train_size],  C_[N_test:]
    else:
        C_train, C_test = C_[:train_size], C_[N_test:]
        
    if weight == None:
        weight = 1    
        
    model, train_losses = train_model(model_data, [features_train, targets_train], C_train, optimization_specification, weight, plot=plot)
    
    return model, train_losses, [features_train, targets_train], [features_test, targets_test], C_train, C_test, model_data


def test_model(model, train_data, test_data, model_data):
    a, b, f = train_data[0][:, :3], train_data[0][:, 3], train_data[0][:, 4]
    ouput = jnp.zeros(train_data[1].shape)
    for i in range(train_data[1].shape[0]//10):
        if model_data["model_name"] == 'fSNO':
            ouput = ouput.at[i*10:(i+1)*10].set(vmap(lambda z: model(z, model_data['analysis'], model_data['synthesis']), in_axes=(0,))(train_data[0][i*10:(i+1)*10])[:, :1])
        else:    
            ouput = ouput.at[i*10:(i+1)*10].set(vmap(model)(train_data[0][i*10:(i+1)*10])[:, :1])
    train_errors = vmap(compute_energy, in_axes=(0, 0, 0, 0))(a, b, train_data[1], ouput)
    
    a, b, f = test_data[0][:, :3], test_data[0][:, 3], test_data[0][:, 4]
    ouput = jnp.zeros(test_data[1].shape)
    for i in range(test_data[1].shape[0]//10):
        if model_data["model_name"] == 'fSNO':
            ouput = ouput.at[i*10:(i+1)*10].set(vmap(lambda z: model(z, model_data['analysis'], model_data['synthesis']), in_axes=(0,))(test_data[0][i*10:(i+1)*10])[:, :1])
        else:    
            ouput = ouput.at[i*10:(i+1)*10].set(vmap(model)(test_data[0][i*10:(i+1)*10])[:, :1])
    test_errors = vmap(compute_energy, in_axes=(0, 0, 0, 0))(a, b, test_data[1], ouput)
    
    return train_errors, test_errors


def final_upper_bound(model, train_data, test_data, C_train, C_test, model_data):
    a, b, f = train_data[0][:, :3], train_data[0][:, 3], train_data[0][:, 4]
    ouput = jnp.zeros(a.shape)
    for i in range(train_data[1].shape[0]//10):
        if model_data["model_name"] == 'fSNO':
            ouput = ouput.at[i*10:(i+1)*10].set(vmap(lambda z: model(z, model_data['analysis'], model_data['synthesis']), in_axes=(0,))(train_data[0][i*10:(i+1)*10]))
        else:    
            ouput = ouput.at[i*10:(i+1)*10].set(vmap(model)(train_data[0][i*10:(i+1)*10]))
    upper_bound_train = vmap(upper_bound, in_axes=(0, 0, 0, 0, 0, 0))(ouput[:,1:], ouput[:,:1], a, b, f, C_train)

    a, b, f = test_data[0][:, :3], test_data[0][:, 3], test_data[0][:, 4]
    ouput = jnp.zeros(a.shape)
    for i in range(test_data[1].shape[0]//10):
        if model_data["model_name"] == 'fSNO':
            ouput = ouput.at[i*10:(i+1)*10].set(vmap(lambda z: model(z, model_data['analysis'], model_data['synthesis']), in_axes=(0,))(test_data[0][i*10:(i+1)*10]))
        else:    
            ouput = ouput.at[i*10:(i+1)*10].set(vmap(model)(test_data[0][i*10:(i+1)*10]))
    upper_bound_test = vmap(upper_bound, in_axes=(0, 0, 0, 0, 0, 0))(ouput[:,1:], ouput[:,:1], a, b, f, C_test)
    
    return upper_bound_train, upper_bound_test


def calculation_of_metrics(model, train_data, test_data, C_train, C_test, train_losses, model_data, path, dataset_name, weight=None):
    path = path if path[-1] == "/" else path + "/"
    os.makedirs(path, exist_ok=True)
    
    errors = test_model(model, train_data, test_data, model_data)
    
    upper_bounds = final_upper_bound(model, train_data, test_data, C_train, C_test, model_data)
    
    corr_train = jnp.corrcoef(jnp.log10(errors[0]), jnp.log10(upper_bounds[0]))[0, 1].item()
    corr_test = jnp.corrcoef(jnp.log10(errors[1]), jnp.log10(upper_bounds[1]))[0, 1].item()
    
    data = {
        "train_loss": train_losses,
        "train_error": errors[0],
        "test_error": errors[1],
        "train_upper_bounds": upper_bounds[0],
        "test_upper_bounds": upper_bounds[1],
        "train_R": corr_train,
        "test_R": corr_test
    }
    model_name = model_data["model_name"]
    if weight==None:
        with open(path + f"{dataset_name}_{model_name}_data.npz", "wb") as f:
            jnp.savez(f, **data)

        with open(path + f"{dataset_name}_{model_name}_model", "wb") as f:
            cloudpickle.dump(model, f)
    else:
        with open(path + f"{dataset_name}_{model_name}_{weight}_data.npz", "wb") as f:
            jnp.savez(f, **data)

        with open(path + f"{dataset_name}_{model_name}_{weight}_model", "wb") as f:
            cloudpickle.dump(model, f)
