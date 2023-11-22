import jax.numpy as jnp
import sys

import equinox as eqx
import optax
import cloudpickle

from jax import jit, vmap, grad, config, random
from jax.nn import relu
from jax.lax import dot_general
from functools import partial

import os, sys, shutil
from pathlib import Path
from datasets import functional_error_estimate

from transforms import utilities
from transforms import library_of_transforms as lft

from architectures import DilResNet, FNO, fSNO, MLP, UNet, ChebNO

config.update("jax_enable_x64", True)

def standard_data_loader(train_data, key, chunk_size):
    train_features, train_targets = train_data
    n = random.permutation(key, jnp.arange(train_features.shape[0], dtype=int))
    train_features, train_targets = train_features[n], train_targets[n]
    chunks = [*range(0, train_features.shape[0] + 1, chunk_size)]
    if chunks[-1] < train_features.shape[0]:
        chunks.append(train_features.shape[0])
    for i, j in zip(chunks[:-1], chunks[1:]):
        yield train_features[i:j], train_targets[i:j]

def get_DilResNet(key, features_train, targets_train):
    input, target = features_train[0], targets_train[0]
    D = len(input.shape[1:])

    kernel_size = 3
    channels = [input.shape[0], 24, target.shape[0]]
    n_cells = 4
    activation = relu

    model = DilResNet.DilatedResNet(key, channels, n_cells, activation=activation, kernel_size=kernel_size, D=D)

    N_batch = 30
    N_epoch = 1000
    learning_rate = 1e-3
    N_drop = 200 * features_train.shape[0] / N_batch # drop each 100 epochs
    sc = optax.exponential_decay(learning_rate, N_drop, 0.5)
    optim = optax.adamw(sc, weight_decay=1e-2)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    model_data = {
        "model": model,
        "compute_loss": DilResNet.compute_loss
    }

    optimization_specification = {
        "optimizer": optim,
        "make_step": DilResNet.make_step_m,
        "N_epochs": N_epoch,
        "opt_state": opt_state,
        "data_loader": lambda data, key: standard_data_loader(data, key, N_batch),
        "Save_each_N": 10000
    }

    return model_data, optimization_specification

def get_fSNO(key, features_train, targets_train):
    input, target = features_train[0], targets_train[0]
    D = len(input.shape[1:])

    polynomials = ["Chebyshev_t", ] * D
    parameters = [[0.1, 0.1],] * D
    M_keep = [features_train.shape[-1]//4, ] * D
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
    N_features_out = target.shape[0]
    N_features = 34
    N_layers = 4

    model = fSNO.fSNO(input_shape, N_features_out, N_layers, N_features, cell, key)

    N_batch = 30
    N_epoch = 1000
    learning_rate = 1e-3
    N_drop = 200 * features_train.shape[0] / N_batch # drop each 100 epochs
    sc = optax.exponential_decay(learning_rate, N_drop, 0.5)
    optim = optax.adamw(sc, weight_decay=1e-2)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    model_data = {
        "model": model,
        "compute_loss": lambda model, input, target: fSNO.compute_loss(model, input, target, analysis, synthesis)
        }

    optimization_specification = {
        "optimizer": optim,
        "make_step": lambda model, input, target, optim, opt_state: fSNO.make_step_m(model, input, target, analysis, synthesis, optim, opt_state),
        "N_epochs": N_epoch,
        "opt_state": opt_state,
        "data_loader": lambda data, key: standard_data_loader(data, key, N_batch),
        "Save_each_N": 10000
    }
    return model_data, optimization_specification, analysis, synthesis

def get_FNO(key, features_train, targets_train):
    input, target = features_train[0], targets_train[0]
    D = len(input.shape[1:])

    N_features = [input.shape[0], 24, target.shape[0]]
    N_layers = 4
    N_modes = features_train.shape[-1]//4

    if D == 1:
        x = jnp.expand_dims(jnp.linspace(0, 1, features_train.shape[-1]), 0)
        N_features[0] = N_features[0] + x.shape[0]
        model = FNO.FNO1D(N_layers, N_features, N_modes, key)
    elif D == 2:
        x = jnp.linspace(0, 1, features_train.shape[-1])
        x = jnp.stack(jnp.meshgrid(x, x), 0)
        N_features[0] = N_features[0] + x.shape[0]
        model = FNO.FNO2D(N_layers, N_features, N_modes, key)

    N_batch = 30
    N_epoch = 1000
    learning_rate = 1e-3
    N_drop = 200 * features_train.shape[0] / N_batch # drop each 100 epochs
    sc = optax.exponential_decay(learning_rate, N_drop, 0.5)
    optim = optax.experimental.split_real_and_imaginary(optax.adamw(sc, weight_decay=1e-2))
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    model_data = {
        "model": model,
        "compute_loss": lambda model, input, target: FNO.compute_loss(model, input, target, x)
    }

    optimization_specification = {
        "optimizer": optim,
        "make_step": lambda model, input, target, optim, opt_state: FNO.make_step(model, input, target, x, optim, opt_state),
        "N_epochs": N_epoch,
        "opt_state": opt_state,
        "data_loader": lambda data, key: standard_data_loader(data, key, N_batch),
        "Save_each_N": 10000
    }

    return model_data, optimization_specification, x

def reinterpolate_between_grids_(data, grid_in, grid_out):
    return vmap(jnp.interp, in_axes=(None, None, 0))(grid_out, grid_in, data)

def reinterpolate_between_grids(data, grid_in, grid_out):
    data_ = reinterpolate_between_grids_(data.reshape(-1, data.shape[-1]), grid_in, grid_out).reshape(data.shape)
    data_ = jnp.moveaxis(data_, 3, 2)
    data_ = reinterpolate_between_grids_(data_.reshape(-1, data.shape[-1]), grid_in, grid_out).reshape(data.shape)
    data_ = jnp.moveaxis(data_, 3, 2)
    return data_

def make_prediction(model, data, model_name):
    key = random.PRNGKey(11)
    keys = [*data.keys()]
    targets_name = 'targets' if 'targets' in keys else 'solution'

    if model_name == "fSNO":
        N = data["features"].shape[-1]
        grid_out = (lft.poly_data["Chebyshev_t"]["nodes"](N, [1, 1]) + 1)/2
        grid_in = jnp.linspace(0, 1, N)
        features_ = reinterpolate_between_grids(data["features"], grid_in, grid_out)
        _, _, analysis, synthesis = get_fSNO(key, features_, data[targets_name])
        apply_model = lambda x: model(x, analysis, synthesis)
    elif model_name == "FNO":
        features_ = data["features"]
        _, _, z = get_FNO(key, features_, data[targets_name])
        apply_model = lambda x: model(x, z)
    else:
        features_ = data["features"]
        apply_model = lambda x: model(x)

    predictions = vmap(eqx.filter_jit(apply_model))(features_)
    if model_name == "fSNO":
        polynomials = ["Chebyshev_t",]*2
        parameters = [[1, 1],]*2

        N_out = [N, N]
        grids = [2*grid_in - 1,]*2

        data_c = utilities.transform_data(predictions, polynomials, parameters, "analysis", grids) # transform in the space of coefficients
        predictions = utilities.transform_data(data_c, polynomials, parameters, "synthesis", grids) # interpolate
    return predictions

def compute_upper_certificates(data_for_upper_bound, C, x_coarse, optim):
    problem_data = [data_for_upper_bound[:1], data_for_upper_bound[1:4], data_for_upper_bound[4:5], data_for_upper_bound[5:], C, x_coarse]
    v, b = functional_error_estimate.compute_certificate(problem_data, optim, N_epoch=10000)
    return v, b

def compute_upper_bound(v, b, data_for_upper_bound, C, x_coarse):
    problem_data = [data_for_upper_bound[:1], data_for_upper_bound[1:4], data_for_upper_bound[4:5], data_for_upper_bound[5:], C, x_coarse]
    return jnp.sqrt(functional_error_estimate.upper_bound([v, b], *problem_data))[0]

def compute_energy_norms(exact_solution, data_for_upper_bound, x_coarse):
    return jnp.sqrt(functional_error_estimate.energy_norm(exact_solution - data_for_upper_bound[:1], data_for_upper_bound[1:4], data_for_upper_bound[4:5], x_coarse))[0]

def train_model(dataset, N_train, network, save_path, N_test=200):
    keys = [*dataset.keys()]
    targets_name = 'targets' if 'targets' in keys else 'solution'

    N = dataset["features"].shape[-1]
    grid_out = (lft.poly_data["Chebyshev_t"]["nodes"](N, [1, 1]) + 1)/2
    grid_in = jnp.linspace(0, 1, N)
    if network == "DilResNet" or network == "FNO":
        features_ = dataset["features"]
        targets_ = dataset[targets_name]
    elif network == "fSNO":
        features_ = reinterpolate_between_grids(dataset["features"], grid_in, grid_out)
        targets_ = reinterpolate_between_grids(dataset[targets_name], grid_in, grid_out)

    features_train, targets_train = features_[:N_train], targets_[:N_train]
    features_test, targets_test = features_[-N_test:], targets_[-N_test:]

    key = random.PRNGKey(11)
    if network == "DilResNet":
        model_data, optimization_specification = get_DilResNet(key, features_train, targets_train)
    elif network == "fSNO":
        model_data, optimization_specification, analysis, synthesis = get_fSNO(key, features_train, targets_train)
    elif network == "FNO":
        model_data, optimization_specification, x = get_FNO(key, features_train, targets_train)

    model, optimization_history, opt_state = ut.train_model(model_data, [features_train, targets_train], [features_test, targets_test], optimization_specification, key)
    if network == "DilResNet":
        apply_model = vmap(lambda z: model(z))
    elif network == "fSNO":
        apply_model = vmap(lambda z: model(z, analysis, synthesis))
    elif network == "FNO":
        apply_model = vmap(lambda z: model(z, x))

    errors = ut.test_model(apply_model, [features_train, targets_train], [features_test, targets_test], lambda x: x)
    Path(save_path).mkdir(parents=True, exist_ok=True)
    ut.save_results(model, errors, optimization_history, opt_state, save_path)

def process_solution(N_train, network, n, project_path, N_test=200):
    source = project_path + f"/{network}/{n}dataset/train_{N_train}/model"
    destination = project_path + "/model"
    shutil.copyfile(source, destination)
    with open(destination, "rb") as f:
        model = cloudpickle.load(f)

    # path to datasets
    data = jnp.load(f"DATASETS/{n}dataset.npz")

    predictions = make_prediction(model, data, model_name=network)
    predictions = jnp.concatenate([predictions[:N_train], predictions[-N_test:]], 0)

    exact_solutions = jnp.concatenate([data['exact_sol'][:N_train], data['exact_sol'][-N_test:]], 0)
    features = jnp.concatenate([data["features"][:N_train], data["features"][-N_test:]], 0)
    C = jnp.concatenate([data["C"][:N_train], data["C"][-N_test:]], 0)
    data_for_upper_bound = jnp.concatenate([predictions, features], 1)
    x_coarse = jnp.linspace(0, 1, data["features"][0].shape[-1])

    learning_rate = 1e-4
    weight_decay = 1e-10
    optim = optax.adamw(learning_rate, weight_decay=weight_decay)

    v, b = vmap(compute_upper_certificates, in_axes=(0, 0, None, None))(data_for_upper_bound, C, x_coarse, optim)
    b_batched = jnp.ones((v.shape[0], 1, v.shape[-2], v.shape[-1]))*b.reshape(-1, 1, 1, 1)
    targets = jnp.concatenate([v, b_batched], 1)

    u_b = vmap(compute_upper_bound, in_axes=(0, 0, 0, 0, None))(v, b, data_for_upper_bound, C, x_coarse)
    e = vmap(compute_energy_norms, in_axes=(0, 0, None))(exact_solutions, data_for_upper_bound, x_coarse)
    errors_and_bounds = jnp.stack([e, u_b], 0)

    new_dataset = {
        "features": data_for_upper_bound,
        "targets": targets
    }

    save_here = project_path + f"/{network}/{n}dataset/train_{N_train}"
    jnp.savez(save_here + "/dataset.npz", **new_dataset)
    jnp.save(save_here + "/errors_and_bounds.npy", errors_and_bounds)

def process_upper_bound(N_train, network, n, project_path, N_test=200):
    source = project_path + f"/{network}/{n}dataset/train_{N_train}/upper_bound/model"
    destination = project_path + '/model'
    shutil.copyfile(source, destination)
    with open(destination, "rb") as f:
        model = cloudpickle.load(f)

    data = jnp.load(project_path + f"/{network}/{n}dataset/train_{N_train}/dataset.npz")
    predictions = make_prediction(model, data, model_name=network)

    v = predictions[:, :2]
    b = jnp.mean(predictions[:, -1].reshape(predictions.shape[0], -1), axis=1)

    # path to datasets
    large_dataset = jnp.load(f"DATASETS/{n}dataset.npz")
    C = jnp.concatenate([large_dataset["C"][:N_train], large_dataset["C"][-N_test:]], 0)
    x_coarse = jnp.linspace(0, 1, data["features"].shape[-1])

    u_b = vmap(compute_upper_bound, in_axes=(0, 0, 0, 0, None))(v, b, dataset["features"], C, x_coarse)
    errors = jnp.load(project_path + f"/{network}/{n}dataset/train_{N_train}/errors_and_bounds.npy")

    data = "$N_{\\text{train}}$,$E_{\\text{train}}$,$E_{\\text{test}}$,$E^{\text{ub}}_{\\text{train}},$E^{\text{ub}}_{\\text{test}},$\\widetilde{E}^{\text{ub}}_{\\text{train}},$\\widetilde{E}^{\text{ub}}_{\\text{test}}$"
    data += f"\n{N_train},{jnp.mean(errors[0][:N_train])},{jnp.mean(errors[0][-N_test:])},{jnp.mean(errors[1][:N_train])},{jnp.mean(errors[1][-N_test:])},{jnp.mean(u_b[:N_train])},{jnp.mean(u_b[-N_test:])}"
    with open(project_path + f"/{network}/{n}dataset/train_{N_train}/upper_bound/results.csv", "w") as f:
        f.write(data)

    jnp.save(project_path + f"/{network}/{n}dataset/train_{N_train}/upper_bound/ub.npy", u_b)

if __name__ == "__main__":
    network = sys.argv[1]
    project_path = sys.argv[2]

    for N_train in [200, 400, 600, 800, 1000, 1200]:
        for n in [1, 2, 3, 4]:
            # path to datasets
            dataset = jnp.load(f"DATASETS/{n}dataset.npz")

            save_path = project_path + f"/{network}/{n}dataset/train_{N_train}"
            train_model(dataset, N_train, network, save_path)

            process_solution(N_train, network, n, project_path)

            dataset = jnp.load(project_path + f"/{network}/{n}dataset/train_{N_train}/dataset.npz")
            save_path = project_path + f"/{network}/{n}dataset/train_{N_train}/upper_bound"
            train_model(dataset, N_train, network, save_path)

            process_upper_bound(N_train, network, n, project_path)
