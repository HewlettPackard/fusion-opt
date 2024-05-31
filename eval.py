import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=""
import sys

sys.path.append('/lustre/ghorbanp/exbo/botorch')
sys.path.append(os.path.join(os.getcwd(), '..'))

import seaborn as sns
import pandas as pd
import time
import torch
import warnings
import matplotlib.pyplot as plt
from CoExBO._coexbo import CoExBOwithSimulation, StateManager
from CoExBO._utils import TensorManager
tm = TensorManager()
warnings.filterwarnings('ignore')


def michalewicz5d(x, m=10):
    """
    Michalewicz function (maximization version)
    :param x: torch tensor of shape (n,) representing the input vector
    :param m: a constant usually set to 10 (default)
    :return: The value of the Michalewicz function for the input vector x
    """
    x = torch.atleast_2d(x)
    batch_size, d = x.size()
    result = torch.zeros(batch_size, dtype=x.dtype, device=x.device)  # Initialize result tensor
    for k in range(batch_size):
        for i in range(d):
            result[k] += torch.sin(x[k, i]) * (torch.sin((i + 1) * x[k, i]**2 / np.pi) ** (2 * m))
    return result

def ackley4d(x):
    """
    Ackley function (maximization version)
    :param x: torch tensor of shape (batch_size, x) representing the input vector
    :return: The value of the Ackley function for the input vector x
    """
    x = torch.atleast_2d(x)
    batch_size, n = x.size()
    sum_sq_term = -0.2 * torch.sqrt((1/n) * torch.sum(x**2, dim=1))
    cos_term = (1/n) * torch.sum(torch.cos(2 * np.pi * x), dim=1)
    return 20 * torch.exp(sum_sq_term) + torch.exp(cos_term) - 20 - np.e


def styblinski3d(x):
    """
    Styblinski-Tang function
    :param x: torch tensor of shape (batch_size, d) representing the input vectors
    :return: The value of the Styblinski-Tang function for the input vectors x
    """
    x = torch.atleast_2d(x)
    return 0.5 * torch.sum(x**4 - 16*x**2 + 5*x, dim=1)

def holder_table_2d(x):
    """
    Hölder Table 2D function (maximization version)
    :param x: torch tensor of shape (batch_size, 2) representing the (x, y) coordinates for each batch
    :return: The value of the Hölder Table 2D function for each (x, y) coordinate pair in the batch
    """
    x = torch.atleast_2d(x)
    return torch.sin(x[:, 0]) * torch.cos(x[:, 1]) * torch.exp(torch.abs(1 - torch.sqrt(x[:, 0]**2 + x[:, 1]**2) / np.pi))


def rosenbrock_3d(x):
    """
    Rosenbrock function
    :param x: torch tensor of shape (batch_size, 3) representing the (x, y, z) coordinates for each batch
    :return: The value of the Rosenbrock function for each (x, y, z) coordinate triplet in the batch
    """
    a = 1
    b = 100
    c = 100
    x = torch.atleast_2d(x)
    x_dim, y_dim, z_dim = x[:, 0], x[:, 1], x[:, 2]
    # return (x_dim-1)**2 + b*(y_dim - x_dim**2)**2 + (y_dim-1)**2 + c*(z_dim - y_dim**2)**2
    return -((x_dim - 1)**2 + b*(y_dim - x_dim**2)**2 + (y_dim - 1)**2 + c*(z_dim - y_dim**2)**2)


def get_bounds(func_name):
    if func_name == "michalewicz5d":
        lower_limit = 0.0
        upper_limit = np.pi
        n_dims = 5  
    if func_name == "branin":
        lower_limit = -2
        upper_limit = 3
        n_dims = 2
    if func_name == "ackley4d":
        lower_limit = -1.0
        upper_limit = 1.0
        n_dims = 4  
    if func_name == "styblinski3d":
        lower_limit = -5.0
        upper_limit = 5.0
        n_dims = 3  
    if func_name == "holder_table_2d":
        lower_limit = 0.0
        upper_limit = 10.0
        n_dims = 2  
    if func_name == "rosenbrock_3d":
        lower_limit = -5.0
        upper_limit = 10.0
        n_dims= 3
        
    return lower_limit, upper_limit, n_dims



        
        
        
def run():    
    # set bounds
    lower_limit, upper_limit, n_dims= get_bounds("styblinski3d")
    mins = lower_limit * torch.ones(n_dims)
    maxs = upper_limit * torch.ones(n_dims)
    bounds = torch.vstack([mins, maxs]) # bounds

    # set domain
    from CoExBO._prior import Uniform    # Import prior from SOBER libraries
    domain = Uniform(bounds)
    true_function = styblinski3d

    # visualise domain
    samples = domain.sample(1000)
    # sns.pairplot(pd.DataFrame(tm.numpy(samples)))
    # plt.show()
    
    n_init_pref = 100      # number of initial random samples for preferential learning
    n_init_obj = 20        # number of initial random samples for objective function

   
    N_TRIALS = 10

    
    best_value_all = []
    results_all = []
    for trial in range(N_TRIALS):
        # coexbo = CoExBOwithSimulation(domain, true_function, sigma=0.1, n_suggestions=3, chosen_acf=["qMES", "qEI"])
        coexbo = CoExBOwithSimulation(domain, true_function, sigma=0.1)
        dataset_obj, dataset_duel = coexbo.initial_sampling(n_init_obj, n_init_pref)
    
    
        n_iterations = 10     # number of iterations
        seed = trial              # random seed for reproduce the results
        best_value=[]

        # initial setting
        torch.manual_seed(seed)
        np.random.seed(seed)
        state = StateManager(n_dims=n_dims, beta_init=0.8)
        results = []
        for t in range(n_iterations):
            beta, gamma = state(t)
            result, dataset_obj, dataset_duel = coexbo(
                dataset_obj, dataset_duel, beta, gamma,
            )
            print(f"{len(dataset_obj[0])}) Best value: {dataset_obj[1].max().item():.5e}")
            results.append(result)
            best_value.append(dataset_obj[1].max().item())
        results = torch.tensor(results)
        results_all.append(results)
        best_value_all.append(best_value)
    
    arr = np.array(best_value_all) 
    np.save('/lustre/ghorbanp/exbo/mExBo/rosenbrock_3d_1.npy', arr)  


run()   
       

   