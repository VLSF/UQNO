import warnings
warnings.filterwarnings('ignore')

import jax
import optax
import os, sys
import argparse
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import matplotlib.pyplot as plt

sys.path.append('architectures')

from jax.nn import relu
from nD import DilResNet, fSNO, ChebNO, UNet
from jax import config, random, grad, jit, vmap
from transforms import utilities, cheb
from jax.lax import scan
from functools import partial
from utilities_2D import *

%matplotlib inline
%config InlineBackend.figure_format='retina'

def main(model_name, dataset_path, train_size = None, weight = None):
    dataset_name = dataset_path.split('/')[-1][:-4]
    model, train_losses, train_data, test_data, C_train, C_test, model_data = train_run(model_name, dataset_path, train_size = train_size,  weight = weight, plot = False)
    
    if weight == None:
        weight = 1
        print(f'Trained for {model_name}, {dataset_name}')
    else: 
        print(f'Trained for {model_name}, {dataset_name}, {weight}')
        
    if train_size != None:
        path = f'./experiments_2D/{train_size}'
    else:
        path = f'./experiments_2D/{N_train}'
    calculation_of_metrics(model, train_data, test_data, C_train, C_test, train_losses, model_data, path, dataset_name, weight)
    
    print(f'Done for {model_name}, {dataset_name}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pipeline')
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--trsize', type=int, help='train size')
    parser.add_argument('--cuda', type=int, help='device cuda')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    
    datasets = [
                f'datasets/{1}dataset.npz',
                f'datasets/{3}dataset.npz', 
                f'dataset/{4}dataset.npz',
                f'dataset/{6}dataset.npz'
               ]
    
    train_size = args.trsize
    
    for dataset in datasets:
        main(args.model, dataset, train_size = train_size, weight = 1)
