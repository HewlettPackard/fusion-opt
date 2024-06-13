import torch
import pickle
import numpy as np
import pandas as pd
from CoExBO._utils import TensorManager
from CoExBO._gp_regressor import set_and_fit_rbf_model, predict


import glob
from pathlib import Path
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
import importlib
files = glob.glob("../metadrl/data/warm_idealGAS/*.csv")



training_size= 10

class URoch(TensorManager):
    def __init__(
        self, 
        path_data="./experiments/AEM_training_data.csv",
        path_prior="./experiments/electrolyte_prior.pickle",
        feature_names=["picket_powers","foot_powers"],
        sigma=3.,
    ):
        """
        URoch Search problem
        
        Args:
        - path_data: string, the path to the training data
        - path_prior: string, the path to the experts' duel data
        - feature_names: list, list of feature names
        - sigma: float, Gaussian noise variance in experimental data feedback
        """
        TensorManager.__init__(self)
        df = pd.DataFrame()
        for f in files:
            csv = pd.read_csv(f)
            # df = df.append(csv)
            df = pd.concat([df, csv])
            
        data_ICF = df[['picket_powers','foot_powers','neutron_yield','RhoR']]
        spec = importlib.util.spec_from_file_location("optimization_tools", "../metadrl/tools/optimization_tools.py")
        OOT = importlib.util.module_from_spec(spec) # importing old optimization tools
        spec.loader.exec_module(OOT) # import old optimization tools
        data_idealGAS = OOT.simulation_data('../metadrl/data/warm_idealGAS')
        data_supercycle = OOT.simulation_data('../metadrl/data/warm_supercycle')
        self.rs_idealGAS = OOT.ResponseSurface(data_idealGAS,picket_domain=[0.5,12],foot_domain=[0.5,12],target_variable='neutron_yield')
        self.rs_supercycle = OOT.ResponseSurface(data_supercycle,picket_domain=[0.5,12],foot_domain=[0.5,12],target_variable='neutron_yield')
        self.reduced_data = data_idealGAS[['picket_powers','foot_powers','neutron_yield']]
        self.reduced_data["neutron_yield"] = self.reduced_data.neutron_yield.astype(float)
        self.loading_data(path_data)
        self.prior_duel = None#self.load_prior_duel(path_prior)
        self.feature_names = feature_names
        self.train_gp(self.X, self.Y)
        self.noise = torch.distributions.Normal(0, sigma)

        




    def target_f(self, x):
        return  torch.tensor(self.rs_supercycle.sample_batch(x[:,0], x[:,1]), dtype=self.dtype, device= self.device)
        
    def loading_data(self, path):
        """
        loading training data for interpolation
        
        Args:
        - path: string, the path to the training data
        """
        
        self.bounds =  torch.stack([torch.tensor([0.5,0.5], dtype=self.dtype, device=self.device), torch.tensor([12,12],dtype=self.dtype,device=self.device)])
        X_train =  np.array(self.reduced_data[['picket_powers','foot_powers']])
        self.X =   torch.tensor(X_train, dtype=self.dtype, device=self.device)
        self.Y = torch.tensor(np.array(self.reduced_data[['neutron_yield']]), dtype=self.dtype, device=self.device)
        


        
        
    def load_prior_duel(self, path):
        """
        loading the experts' duel data
        
        Args:
        - path: string, the path to experts' duel data
        
        Return:
        - dataset_duel: list, the list of experts' duel data
        """
        pass
        # with open(path, 'rb') as handle:
        #     dataset_duel = pickle.load(handle)
        # return dataset_duel
    
    def train_gp(self, X, Y):
        """
        Training the Gaussian process regression model to interpolate values.
        
        Args:
        - X: torch.tensor, the input variables
        - Y: torch.tensor, the output variables
        """
        self.model = set_and_fit_rbf_model(X, Y)
        
    def __call__(self, X):
        """
        Return the experimental values estimated by GP.
        
        Args:
        - X: torch.tensor, the input variables
        
        Return:
        - f + epsilon: torch.tensor, the noisy output variables
        - f: torch.tensor, the noiseless output variables
        """
        if len(X.shape) == 1:
            X = X.unsqueeze(0)
        # pred = predict(X, self.model)
        # f = pred.loc * self.Y.var() + self.Y.mean()
        f = self.target_f(X.detach().cpu().numpy())
        # epsilon = self.noise.sample(torch.Size([len(X)]))
        return  f