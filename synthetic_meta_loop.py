import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))
import time
import torch
import warnings
import matplotlib.pyplot as plt
from CoExBO._utils import TensorManager
from botorch.models import SingleTaskGP
from botorch import fit_gpytorch_mll
from botorch.optim.fit import fit_gpytorch_mll_torch
from gpytorch import ExactMarginalLogLikelihood
tm = TensorManager()
warnings.filterwarnings('ignore')

dataset = "hydrogen"
exploration_rate = 0.5 ##EXPLORATION RATE
gamma_init = 0.01 #0.01 #1

if dataset == 'svm':
    n_dims = 8 
elif dataset == 'xgboost':
    n_dims = 16
elif dataset == 'ranger9':
    n_dims = 10
elif dataset == 'ranger':
    n_dims = 6
elif dataset == 'rpart':
    n_dims = 6
elif dataset == 'battery':
    n_dims = 3
elif dataset == 'hydrogen':
    n_dims = 10


seed = 1            # random seed for reproduce the results #SEED 8 rpart_preproc, SEED 3 for Ranger, SEED 1 for rpart (50000_ckpt), 5 for rpart val, 

class Benchmark():
    def __init__(self, X_data, Y_data, sample = False):
        # self.X_data = X_data
        # self.Y_data = Y_data
        if sample:
            self.X_data, self.Y_data = self.sample_for_training(X_data, Y_data)
        else:
            self.X_data, self.Y_data = X_data, Y_data
        print("Training Model")
        self.gp_model = SingleTaskGP(torch.tensor(self.X_data), torch.tensor(self.Y_data)).cuda()
        mll = ExactMarginalLogLikelihood(self.gp_model.likelihood, self.gp_model).cuda()
        fit_gpytorch_mll(mll=mll)
        print("Training Done")

    def find_closest_point(self, query, loop = False):
        y_set = []
        query = query.squeeze()
        self.gp_model.eval()
        if len(query.shape) > 1:
            for X in query:
                posterior = self.gp_model.posterior(X.unsqueeze(0).float().cuda())
                mean = posterior.mean.item()
                variance = posterior.variance.item()

                # distances = np.linalg.norm(self.X_data - X, axis=1)
                # closest_index = np.argmin(distances)
                # y = self.Y_data[closest_index]
                mean = np.clip(mean,0,1)
                y_set.append(mean)
            return torch.tensor(y_set, dtype=torch.float32).squeeze()
        else:
            X = query#.detach().numpy()
            posterior = self.gp_model.posterior(X.unsqueeze(0).float().cuda())
            mean = posterior.mean.item()
            variance = posterior.variance.item()
            mean = np.clip(mean,0,1)
            return torch.tensor(mean, dtype=torch.float32).reshape([1])

    def sample_for_training(self, X, Y):
        Y = Y.squeeze()
        yuniq, ycount = np.unique(Y, return_counts=True)
        # print(yuniq)
        # print(ycount)
        counts = {v: c for v, c in zip(yuniq, ycount)}

        # print(counts)
        # for i in range(len(Y)):
        #     print(Y[i])
        #     print(counts[Y[i]])
        logits = np.array([Y[i] / counts[Y[i]] for i in range(len(Y))])
        freq_idx = logits.argsort()[::-1]

        selected_rows = freq_idx[:(3 * len(yuniq))]
        np.random.shuffle(selected_rows)
        X = X[selected_rows]
        Y = Y[selected_rows]
        #stdY = (Y - Y.mean()) / Y.std()

        num_dims = list(np.arange(X.shape[-1]))
        cat_dims = []

        # Fit and save GP
        print(f'Fit GP on dataset {dataset} containing {X.shape[0]} points...')
        X_ = torch.from_numpy(X).to(dtype=torch.float64)
        Y_ = torch.from_numpy(Y).to(dtype=torch.float64)
        #print("Y SHAPE: ", Y.shape)
        return X_, Y_.unsqueeze(1)
    
    def clean(self):
        del self.gp_model
        torch.cuda.empty_cache()

lower_limit = 0
upper_limit = 1
colour_map = 'summer'
resolution = 100

import seaborn as sns
import pandas as pd

# set bounds

mins = lower_limit * torch.ones(n_dims)
maxs = upper_limit * torch.ones(n_dims)
bounds = torch.vstack([mins, maxs]) # bounds

# set domain
from CoExBO._prior import Uniform    # Import prior from SOBER libraries
domain = Uniform(bounds)

# visualise domain
samples = domain.sample(1000)
# sns.pairplot(pd.DataFrame(tm.numpy(samples)))
#plt.show()
from CoExBO._coexbo import CoExBOwithSimulation, StateManager
if dataset == 'svm':
    n_init_pref = 500
elif dataset == 'battery':
    n_init_pref = 50
else:
    n_init_pref = 100
n_init_obj = 1        # number of initial random samples for objective function

from importlib.machinery import SourceFileLoader
def load_module(filename):
    module_name = os.path.splitext(os.path.basename(filename))[0]
    return SourceFileLoader(module_name, filename).load_module(module_name)
import yaml
model_type = 'tnpa'
model_cls = getattr(load_module(f'./models/{model_type}.py'), model_type.upper())
with open(f'configs/{dataset}/{model_type}.yaml', 'r') as f:
    config = yaml.safe_load(f)
#if args.model in ["np", "anp", "cnp", "canp", "bnp", "banp", "tnpa", "tnpd", "tnpnd"]:
model = model_cls(**config)
model.cuda()
import os.path as osp
import torch
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.utils import t_batch_mode_transform
from CoExBO._utils import TensorManager

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
results_path = './results/'
model_name = 'tnpa'
if dataset == 'svm':
    ckp_point = '518000'
elif dataset == 'xgboost':
    ckp_point = '460000'
elif dataset == 'ranger9':
    ckp_point = '390000'
elif dataset == 'rpart':
    ckp_point = '130000'
elif dataset == 'ranger':
    ckp_point = '760000'
elif dataset == 'battery':
    ckp_point = '340000' 
elif dataset == 'hydrogen':
    ckp_point = '490000' 

ckpt_path = osp.join(results_path, dataset, model_name, f'{ckp_point}_ckpt.tar')

ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt.model)

def sample_from_data(X_train, y_train, max_points = 100):
    samples_X = []
    samples_y = []
    for i in range(len(X_train)):
        if len(X_train[i]) < max_points:
            points_to_sample = len(X_train[i])
        else:
            points_to_sample = max_points
        r_idx = np.random.randint(0,len(X_train[i]),points_to_sample)
        samples_X.append(X_train[i][r_idx])
        samples_y.append(y_train[i][r_idx])
    return samples_X, samples_y

n_iterations = 50     # number of iterations


# initial setting
torch.manual_seed(seed)

state = StateManager(n_dims=n_dims, gamma_init=gamma_init) #gamma_init = 1 svm for na9, default for battery, default for ranger 6D, ranger9 = 1, xgboost = default, rpart = default

results = []
noise_level = 0.2
import numpy as np
meta = True
X_test = np.load(f'./datasets/{dataset}/X_test.npz')
y_test = np.load(f'./datasets/{dataset}/y_test.npz')
X_validation = np.load(f'./datasets/{dataset}/X_validation.npz')
y_validation = np.load(f'./datasets/{dataset}/y_validation.npz')

X_test_ = np.array([X_test[key].astype('float32') for key in X_test.keys()])
y_test_ = np.array([y_test[key].astype('float32') for key in y_test.keys()])


X_validation_ = np.array([X_validation[key].astype('float32') for key in X_validation.keys()])
y_validation_ = np.array([y_validation[key].astype('float32') for key in y_validation.keys()])

#samples_val_X, samples_val_y = sample_from_data(X_validation_, y_validation_, max_points=100)

samples_val_X, samples_val_y = sample_from_data(X_validation_, y_validation_, max_points=100)



trajectories_over_tries = []
Y0_over_tries = []
Y1_over_tries = []
number_of_tries = 25

max_num_points = 1000 #



samplig = False
if dataset == 'svm':
    samplig = True

mean_seeds = []
for j in range(number_of_tries):
#for j in seeds:
    print("SEED: ", j)
    torch.random.manual_seed(j)
    Y_0_total = []
    Y_1_total = []
    best_values = []
    total_trajectories = []
    for i in range(len(X_test)):
        print("TEST TASK: ", i)
        benchmark = Benchmark(X_test_[i], y_test_[i], sample=samplig)

        true_function = benchmark.find_closest_point
        if len(X_test_[i]) > max_num_points:
            r_idx = np.random.randint(0,len(X_test_[i]),max_num_points)
            model.test_set_X = X_test_[i][r_idx]
        else:
            model.test_set_X = X_test_[i]
        model.w_exploration = exploration_rate
        min_test_idx = np.argmin(y_test_[i])
        #print("y_test_[i]: ", y_test_[i][min_test_idx])
        
        #print(" X_test_[i]: ", len(X_test_[i][r_idx]))
        x_val = torch.tensor(X_test_[i][min_test_idx]).unsqueeze(0)
        y_val = true_function(x_val)
        dataset_obj = (x_val,y_val)
        
        #true_function2 = benchmark2.find_closest_point
        coexbo = CoExBOwithSimulation(domain, true_function, sigma=0.1, hallucinate=False, meta = meta)
        #coexbo2 = CoExBOwithSimulation(domain, true_function2, sigma=0.1, hallucinate=False, meta = meta)
        #dataset_obj, _ = coexbo.initial_sampling(n_init_obj, n_init_pref)
        _, dataset_duel = coexbo.initial_sampling(n_init_obj, n_init_pref, torch.tensor(samples_val_X[i]))
        # print("dataset_duel: ", dataset_duel)
        #print(dataset_duel)
        #_, dataset_duel = coexbo.initial_sampling(n_init_obj, n_init_pref)
        max_bv = -1
        trajectory = [y_val]
        Y_0_values = []
        Y_1_values = []
        for t in range(n_iterations):
            beta, gamma = state(t)
            # Y_0_values.append(dataset_obj[1].detach().cpu().numpy())
            # Y_1_values.append(dataset_obj[1].detach().cpu().numpy())
            print("dataset_obj: ", dataset_obj[1])
            result, dataset_obj, dataset_duel, Y_returns = coexbo(
                dataset_obj, dataset_duel, beta, gamma, model_TPN = model, lower_limit= lower_limit, upper_limit=upper_limit
            )
            best_v = dataset_obj[1].max().item()
            if best_v > max_bv:
                max_bv = best_v
            trajectory.append(max_bv)
            print("Trajectory: ", trajectory)
            #print(f"{len(dataset_obj[0])}) Best value: {best_v:.5e}")
            print("results[2]: ", Y_returns)
            results.append(result)

            Y_0_values.append(Y_returns[0])
            Y_1_values.append(Y_returns[1])
        Y_0_total.append(Y_0_values)
        Y_1_total.append(Y_1_values)
        total_trajectories.append(dataset_obj[1].detach().cpu().numpy())
        #best_values.append(trajectory)
        #benchmark2.clean()
        benchmark.clean()
    print("MEAN SEED: ", np.array(total_trajectories).mean())
    mean_seeds.append(np.array(total_trajectories).mean())
    Y0_over_tries.append(Y_0_total)
    Y1_over_tries.append(Y_1_total)
    trajectories_over_tries.append(total_trajectories)
#results = torch.tensor(results)

import pickle
# Save the object as a pickle file 
print("MEAN SEEDS: ", mean_seeds)
print("GAMMA INIT = ", gamma_init)
print("EXPLORATION RATE = ", exploration_rate)
print("n_init_pref: ", n_init_pref)
with open(f'evaluations_loop/meta_{dataset}_na_loop_sv24.pkl', 'wb') as file: #rpart v1 evaluated on ckpt 50000, best one so far sample = False, na3 = we 0.5, na20 = 0.5
    pickle.dump(trajectories_over_tries, file)                       #ranger9 sv6 x_field sample=False 390k na20 we = 0.7
                                                                #ranger6D sv6 x_field sample=False 760k, sv9 test 100 samples test_X, sv9 validation 100, na3 we = 0.5 works, na20 we = 0.7 seems to be better.
                                                                #svm sv6 x_field sample = True 518k, na9 = 0.5, na10 = 0.7
                                                                #xgboost sv6 x_field sample=False 560k, na3 we = 0.5 works
                                                                #battery na11 we = 0.7, na12/6 we = 0.3
                                                                
with open(f'evaluations_loop/meta_{dataset}_na_loop_sv24_Y0.pkl', 'wb') as file: #ranger6D sv1 whole test_X, sv2 100 samples test_X, sv3 validation 100.
    pickle.dump(Y0_over_tries, file)

with open(f'evaluations_loop/meta_{dataset}_na_loop_sv24_Y1.pkl', 'wb') as file: 
    pickle.dump(Y1_over_tries, file)
