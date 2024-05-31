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
import seaborn as sns
import pandas as pd
from CoExBO._coexbo import CoExBOwithSimulation, StateManager
from importlib.machinery import SourceFileLoader
import yaml
tm = TensorManager()
warnings.filterwarnings('ignore')

n_dims = 10                      # number of dimensions
dataset = "ranger9"
seed = 1            # random seed for reproduce the results #SEED 8 rpart_preproc, SEED 3 for Ranger, SEED 1 for rpart (50000_ckpt), 5 for rpart val, 


lower_limit = 0
upper_limit = 1
colour_map = 'summer'
resolution = 200
ground_truth = torch.tensor([-1.02543108, -1.02543108])




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


n_init_pref = 100      # number of initial random samples for preferential learning
n_init_obj = 1        # number of initial random samples for objective function

class Benchmark():
    def __init__(self, X_data, Y_data, sample = False):
        # self.X_data = X_data
        # self.Y_data = Y_data
        print("Training Model")
        if sample:
            self.X_data, self.Y_data = self.sample_for_training(X_data, Y_data)
        else:
            self.X_data, self.Y_data = X_data, Y_data
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
            # distances = np.linalg.norm(self.X_data - X, axis=1)
            # closest_index = np.argmin(distances)
            # y = self.Y_data[closest_index]
            # print("X: ", X)
            # distances = np.linalg.norm(self.X_data - X, axis=1)
            # closest_index = np.argmin(distances)
            # print("closest_index: ", closest_index)
            # print("distances: ", distances[closest_index])
            # y = self.Y_data[closest_index]
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
    
    


def load_module(filename):
    module_name = os.path.splitext(os.path.basename(filename))[0]
    return SourceFileLoader(module_name, filename).load_module(module_name)

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
ckpt_path = osp.join(results_path, dataset, model_name, '390000_ckpt.tar') #rpart_50k was best #ranger9_v1: 290000,

ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt.model)

def sample_from_training(X_train, y_train, max_points = 100):
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
state = StateManager(n_dims=n_dims) # beta_init = 0.5 best working so far



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

#FLIP
# X_validation_ = np.array([X_test[key].astype('float32') for key in X_test.keys()])
# y_validation_ = np.array([y_test[key].astype('float32') for key in y_test.keys()])


# X_test_ = np.array([X_validation[key].astype('float32') for key in X_validation.keys()])
# y_test_ = np.array([y_validation[key].astype('float32') for key in y_validation.keys()])

samples_val_X, samples_val_y = sample_from_training(X_validation_, y_validation_, max_points=5)

samples_val_X = torch.tensor(np.vstack(samples_val_X))
samples_val_y = torch.tensor(np.vstack(samples_val_y))
samples_val_X = samples_val_X + np.random.normal(0, noise_level, size=samples_val_X.shape)
samples_val_y = samples_val_y + np.random.normal(0, noise_level, size=samples_val_y.shape)

#X_t_fit, y_t_fit = sample_from_training(X_test_,y_test_, max_points = 5000)
best_values = []
total_trajectories = []
for i in range(len(X_test)):
    
    torch.random.manual_seed(seed)
    #X_t_fit, y_test_ = sample_from_training(X_test_[i],y_test_[i], max_points = 1000)

    benchmark = Benchmark(X_test_[i], y_test_[i], sample=False)
    # print(y_test[i])
    benchmark2 = Benchmark(samples_val_X, samples_val_y)
    true_function = benchmark.find_closest_point
    model.test_set_X = X_test_[i]
    min_test_idx = np.argmin(y_test_[i])
    #print("y_test_[i]: ", y_test_[i][min_test_idx])
    x_val = torch.tensor(X_test_[i][min_test_idx]).unsqueeze(0)
    y_val = true_function(x_val)
    dataset_obj = (x_val,y_val)
    
    true_function2 = benchmark2.find_closest_point
    coexbo = CoExBOwithSimulation(domain, true_function, sigma=0.1, hallucinate=False, meta = meta)
    coexbo2 = CoExBOwithSimulation(domain, true_function2, sigma=0.1, hallucinate=False, meta = meta)
    #dataset_obj, _ = coexbo.initial_sampling(n_init_obj, n_init_pref)
    _, dataset_duel = coexbo2.initial_sampling(n_init_obj, n_init_pref)
    # print("dataset_duel: ", dataset_duel)
    #print(dataset_duel)
    #_, dataset_duel = coexbo.initial_sampling(n_init_obj, n_init_pref)
    max_bv = -1
    trajectory = [y_val]
    for t in range(n_iterations):
        beta, gamma = state(t)
        print("dataset_obj: ", dataset_obj[1])
        result, dataset_obj, dataset_duel = coexbo(
            dataset_obj, dataset_duel, beta, gamma, model_TPN = model, lower_limit= lower_limit, upper_limit=upper_limit
        )
        best_v = dataset_obj[1].max().item()
        if best_v > max_bv:
            max_bv = best_v
        trajectory.append(max_bv)
        print("Trajectory: ", trajectory)
        #print(f"{len(dataset_obj[0])}) Best value: {best_v:.5e}")
        results.append(result)
    total_trajectories.append(dataset_obj[1].detach().cpu().numpy())
    best_values.append(trajectory)
results = torch.tensor(results)


import pickle
# Save the object as a pickle file 
with open(f'evaluations/meta_{dataset}_ranger22.pkl', 'wb') as file: #rpart v1 evaluated on ckpt 50000, best one so far v2 best evaluation for rpart #ranger9 v2 100k, v3 100k flip
    pickle.dump(total_trajectories, file)