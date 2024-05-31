import torch
from botorch.acquisition import AnalyticAcquisitionFunction
from torch.distributions import Normal
from botorch.utils import t_batch_mode_transform
from ._utils import TensorManager, generate_meshgrid
from utils.utils_bo import twod_to_task, Dataset
from bayeso import acquisition, covariance
from scipy.stats import norm
import numpy as np
class CoExBO_UCB(AnalyticAcquisitionFunction, TensorManager):
    def __init__(
        self,
        model,
        prior_pref,
        beta,
        gamma,
        pi_augment=True,
    ):
        AnalyticAcquisitionFunction.__init__(
            self, 
            model=model,
            posterior_transform=None,
        )
        TensorManager.__init__(self)
        self.register_buffer("beta", self.tensor(beta))
        self.register_buffer("gamma", self.tensor(gamma))        
        self.pi_augment = pi_augment
        self.initialise(prior_pref, model)
        
    def initialise(self, prior_pref, model):
        if not hasattr(prior_pref, "y_test_mean"):
            prior_pref.normalising_constant()
        
        self.E_y_pref = prior_pref.y_test_mean.mean()
        self.std_y_pref = prior_pref.y_test_mean.std()
        self.E_y_obs = model.train_targets.mean()
        self.std_y_obs = model.train_targets.std()
        self.prior_pref = prior_pref
        
    def prior_gp(self, X):
        prior_mean, prior_std = self.prior_pref.probability(X, both=True)

        prior_mean_conv = (prior_mean - self.E_y_pref) / self.std_y_pref * self.std_y_obs + self.E_y_obs
        prior_std_conv = prior_std / self.std_y_pref * self.std_y_obs
        return prior_mean_conv, prior_std_conv
    
    def posterior_gp(self, X, likelihood_gp_mean, likelihood_gp_std):
        prior_gp_mean, prior_gp_std = self.prior_gp(X)
        prior_gp_std_max = (
            self.gamma * likelihood_gp_std.pow(2) + prior_gp_std.pow(2)
        ).sqrt()
        posterior_gp_std = (
            prior_gp_std_max.pow(2) * likelihood_gp_std.pow(2) / (
                prior_gp_std_max.pow(2) + likelihood_gp_std.pow(2)
            )
        ).sqrt()
        posterior_gp_mean = (
            posterior_gp_std.pow(2) / prior_gp_std_max.pow(2)
        ) * prior_gp_mean + (
            posterior_gp_std.pow(2) / likelihood_gp_std.pow(2)
        ) * likelihood_gp_mean
        return posterior_gp_mean, posterior_gp_std
        
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        likelihood_gp_mean, likelihood_gp_std = self._mean_and_sigma(X.float())
        if self.pi_augment:
            posterior_gp_mean, posterior_gp_std = self.posterior_gp(X, likelihood_gp_mean, likelihood_gp_std)
            return posterior_gp_mean + self.beta.sqrt() * posterior_gp_std
        else:
            return likelihood_gp_mean + self.beta.sqrt() * likelihood_gp_std

class CoExBO_UCB_Meta_EI(TensorManager):
    def __init__(
        self,
        model,
        prior_pref,
        beta,
        gamma,
        pi_augment=True,
    ):
        # AnalyticAcquisitionFunction.__init__(
        #     self, 
        #     model=model,
        #     posterior_transform=None,
        # )

        TensorManager.__init__(self)
        #self.register_buffer("beta", self.tensor(beta))
        #self.register_buffer("gamma", self.tensor(gamma))        
        self.pi_augment = pi_augment
        #self.initialise(prior_pref, model)
        self.model_preference = prior_pref
        self.model = model
        self.step = 0
        self.beta = self.tensor(beta)
        self.gamma = self.tensor(gamma)
        self.min_value = 1e-3
    
    def posterior_gp(self, likelihood_gp_mean, likelihood_gp_std, prior_gp_mean, prior_gp_std):
        
        prior_gp_std_max = (
            self.gamma * likelihood_gp_std.pow(2) + prior_gp_std.pow(2)
        ).sqrt()
        posterior_gp_std = (
            prior_gp_std_max.pow(2) * likelihood_gp_std.pow(2) / (
                prior_gp_std_max.pow(2) + likelihood_gp_std.pow(2)
            )
        ).sqrt()
        posterior_gp_mean = (
            posterior_gp_std.pow(2) / prior_gp_std_max.pow(2)
        ) * prior_gp_mean + (
            posterior_gp_std.pow(2) / likelihood_gp_std.pow(2)
        ) * likelihood_gp_mean

        return posterior_gp_mean, posterior_gp_std

    def get_expected_improvement(self, max_mean_y, mean_y,sigma_y_new, xi = 0.25): #1: 0.1, 2: 0.25, 3: 0.35
        diff_y = mean_y - max_mean_y - xi
        #print(diff_y)
        z = (diff_y) / sigma_y_new
        exp_imp = (diff_y) * norm.cdf(z) + sigma_y_new * norm.pdf(z)
        exp_imp = exp_imp.ravel()
        #print("exp_imp", exp_imp.shape)
        return torch.tensor(exp_imp)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):

        # w_exploration  = 0.5
        # epsilon =  1e-6
        # w_uncertainty = 0.5  # fixed weight which may not be optimal,  use the weight based uncertainty commented below
        # #X = X.squeeze(1)  # Remove unnecessary singleton dimension (if present)

        #########################################
        
        self.context_x = self.model.train_X
        self.context_y = self.model.train_targets
        x_field = torch.tensor(self.model.test_set_X)
        X_shape = self.context_x.shape
        context_x_ = self.context_x.reshape([1,X_shape[0],X_shape[1]]).float().cuda()
        context_y_ = self.context_y.reshape([1,len(self.context_y),1]).float().cuda()
        #mesh_grid = generate_meshgrid(self.model.lower_limit, self.model.upper_limit, dim=X_shape[1])
        #distribution = self.model.predict(context_x_,context_y_,mesh_grid.reshape([1,len(mesh_grid),X_shape[1]]).cuda())

        
        distribution = self.model.predict(context_x_,context_y_,x_field.reshape([1,len(x_field),X_shape[1]]).cuda())

        mean_single = torch.squeeze(distribution.mean.detach().cpu()[0])
        std_single = torch.squeeze(distribution.scale.detach().cpu()[0]) 
        # print("STD SINGLE shape: ", mean_single.shape)
        # print("STD SINGLE shape: ", std_single.shape)

        distribution = self.model.predict(context_x_,context_y_,context_x_)
        means_observed = distribution.mean.detach().cpu()[0]
        stds_observed = distribution.stddev.detach().cpu()[0]

        # print("means_observed shape: ", means_observed.shape)
        # print("stds_observed shape: ", stds_observed.shape)

        current_best = torch.max(means_observed)
        ##################################

        ##################################

        # PairwiseGP posterior for average predicted preference scores and uncertainties
        posterior_pairwise = self.model_preference.posterior(x_field)
        avg_predicted_pref = posterior_pairwise.mean.reshape(-1).detach().cpu()    # Average predicted preference scores
        std_pairwise = torch.sqrt(posterior_pairwise.covariance_matrix.diagonal(dim1=-2, dim2=-1)).detach().cpu()  # Extract diagonal elements


        

        #####
        # Expected Improvement based on predicted preference (modify for your model)
        # z = (avg_predicted_pref - avg_predicted_pref.max()) / std_pairwise  # Normalize by max score and std_pairwise
        # cdf = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0)).cdf(z)
        # pdf = torch.exp(torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0)).log_prob(z))
        # ei_preference = std_pairwise * (z * cdf + pdf)

        # use this instead of fixed weight 
        # Uncertainty weighting factor  - weight based on uncertainty
        # w_uncertainty = 1 / (std_pairwise + epsilon)
        
        
        
        # utility function for a better incorporation of  exploration incentive alongside maximizing the predicted performance, instead of using only mean_single

        # clipped version if getting nan values in utility function
        #clipped_mean_single = torch.clamp(mean_single, min=self.min_value)    # the default min_value is for Branin function came with this code. 
        #utility = w_exploration * torch.log(clipped_mean_single + self.min_value) 
        
        # use this one if you don't get nan values 
        # Logarithmic Utility function (adjust w and epsilon)
        #utility = w_exploration  * torch.log(mean_single + epsilon) #  
        
        if self.pi_augment:
            posterior_gp_mean, posterior_gp_std = self.posterior_gp(mean_single, std_single, avg_predicted_pref, std_pairwise)
            

            #aqcf = self.get_expected_improvement(current_best, posterior_gp_mean, posterior_gp_std)
            #point = torch.argmax(aqcf)


            acq_vals = -1.0 * acquisition.aei(np.ravel(posterior_gp_mean), np.ravel(posterior_gp_std), context_y_.reshape([len(self.context_y),1]).detach().cpu().numpy(), noise = 0.1)
            point = np.argmin(acq_vals)

            return x_field[point][:X_shape[1]]
        else:
            # aqcf = self.get_expected_improvement(current_best, mean_single, std_single)
            # point = torch.argmax(aqcf)
            #print("context_y_shape: ", context_y_.reshape([len(self.context_y),1]).detach().cpu().numpy())
            acq_vals = -1.0 * acquisition.aei(np.ravel(mean_single), np.ravel(std_single), context_y_.reshape([len(self.context_y),1]).detach().cpu().numpy(), noise = 0.1)
            point = np.argmin(acq_vals)

            return x_field[point][:X_shape[1]]
        
        # # Combined acquisition function (weighted sum)
        # acquisition_function = utility + w_uncertainty * ei_preference


        # # SingleTaskGP posterior (predictions and uncertainties)
        # point = torch.argmax(acquisition_function)
        # return mesh_grid[point][:X_shape[1]]


class CoExBO_H_Meta(TensorManager):
    
    def __init__(
        self,
        model,
        prior_pref,
        pref_pairwise,
        rf_clf,
        beta,
        gamma,
        pi_augment=True,
    ):

        TensorManager.__init__(self)
        # self.register_buffer("beta", self.tensor(beta))
        # self.register_buffer("gamma", self.tensor(gamma))        
        self.pi_augment = pi_augment
        self.initialise(prior_pref, pref_pairwise, rf_clf, model)
        self.model = model
        self.step = 0
        self.beta = self.tensor(beta)
        self.gamma = self.tensor(gamma)
       

    def initialise(self, prior_pref, pref_pairwise,  rf_clf, model):
        if not hasattr(prior_pref, "y_test_mean"):
            prior_pref.normalising_constant()
        
        self.E_y_pref = prior_pref.y_test_mean.mean()
        self.std_y_pref = prior_pref.y_test_mean.std()
        self.E_y_obs = model.train_targets.mean()
        self.std_y_obs = model.train_targets.std()
        self.prior_pref = prior_pref
        self.pref_pairwise = pref_pairwise
        self.model_preference = pref_pairwise
        self.rf_clf = rf_clf
        
    def prior_gp(self, X):
        prior_mean, prior_std = self.prior_pref.probability(X, both=True)

        prior_mean_conv = (prior_mean - self.E_y_pref) / self.std_y_pref * self.std_y_obs + self.E_y_obs
        prior_std_conv = prior_std / self.std_y_pref * self.std_y_obs
        return prior_mean_conv, prior_std_conv
    
    def posterior_gp(self, X, likelihood_gp_mean, likelihood_gp_std):
        
        
        prior_gp_mean, prior_gp_std = self.prior_gp(X)
        prior_gp_std_max = (
            self.gamma * likelihood_gp_std.pow(2) + prior_gp_std.pow(2)
        ).sqrt()
        posterior_gp_std = (
            prior_gp_std_max.pow(2) * likelihood_gp_std.pow(2) / (
                prior_gp_std_max.pow(2) + likelihood_gp_std.pow(2)
            )
        ).sqrt()
        posterior_gp_mean = (
            posterior_gp_std.pow(2) / prior_gp_std_max.pow(2)
        ) * prior_gp_mean + (
            posterior_gp_std.pow(2) / likelihood_gp_std.pow(2)
        ) * likelihood_gp_mean
        # print("likelihood_gp_std: ", likelihood_gp_std)
        # print("posterior_gp_std: ", posterior_gp_std)
        # print("prior_gp_std: ", prior_gp_std)

        # print("prior_gp_mean: ", prior_gp_mean)
        # print("likelihood_gp_mean: ", likelihood_gp_mean)
        # print("posterior_gp_mean: ", posterior_gp_mean)

        return posterior_gp_mean, posterior_gp_std
        
    
    def forward(self, X):
       
        #print("X IN FORWARD: ", X)
        x_field = torch.tensor(self.model.test_set_X)
        X = x_field
        self.context_x = self.model.train_X
        self.context_y = self.model.train_targets
        X_shape = self.context_x.shape
        context_x_ = self.context_x.reshape([1,X_shape[0],X_shape[1]]).cuda()
        context_y_ = self.context_y.reshape([1,len(self.context_y),1]).cuda()
        # = generate_meshgrid(self.model.lower_limit, self.model.upper_limit, dim=X_shape[1])
        #distribution = self.model.predict(context_x_.float().cuda(),context_y_.float().cuda(),mesh_grid.reshape([1,len(mesh_grid),X_shape[1]]).cuda())

        distribution = self.model.predict(context_x_,context_y_,x_field.reshape([1,len(x_field),X_shape[1]]).cuda())

        # likelihood_gp_mean = torch.squeeze(distribution.mean.detach().cpu()[0])
        
        # likelihood_gp_std = torch.squeeze(distribution.scale.detach().cpu()[0]) 

        likelihood_gp_mean = torch.squeeze(distribution.mean.detach()[0])
        
        likelihood_gp_std = torch.squeeze(distribution.scale.detach()[0]) 

        # print("likelihood_gp_std before: ", likelihood_gp_std)
        # print("self.beta.sqrt(): ", self.beta.sqrt())
        #likelihood_gp_mean, likelihood_gp_std = self._mean_and_sigma(X)
        w_exploration = self.model.w_exploration
        best_value = self.context_y.max() #self.model.train_targets.max()
        #w_exploration  = 0.5 #self.beta#0.8
        k = 1.0
        epsilon =  1e-6
        default_min_y = 0.0 
        default_max_y = 1.0

        if self.pi_augment:
            #w_uncertainty = 0.65  # fixed weight which may not be optimal,  use the weight based uncertainty commented below
            # X = X.squeeze(1)  # Remove unnecessary singleton dimension (if present)

            # SingleTaskGP posterior (predictions and uncertainties)
            # posterior_single = self.model_gp.posterior(X)
            mean_single = likelihood_gp_mean #posterior_single.mean.reshape(-1)  
            std_single =  likelihood_gp_std#torch.sqrt(posterior_single.covariance_matrix.diagonal(dim1=-2, dim2=-1))
            
            #posterior_gppair = self.model_preference.posterior(X)
            #avg_predicted_gppair = posterior_gppair.mean.reshape(-1)    # Average predicted preference scores
            #std_gppair = torch.sqrt(posterior_gppair.covariance_matrix.diagonal(dim1=-2, dim2=-1))  # Extract diagonal elements
            
            avg_predicted_pref, std_pairwise = self.prior_pref.probability(X.cuda(), both=True)
            
               
            min_pref, max_pref = torch.min(avg_predicted_pref), torch.max(avg_predicted_pref)
            normalized_mean_pref = (avg_predicted_pref - min_pref) / (max_pref - min_pref)
            
            # gpair_min, gpai_rmax = torch.min(avg_predicted_gppair), torch.max(avg_predicted_gppair)
            # normalized_mean_gppair = (avg_predicted_gppair - gpair_min) / (gpai_rmax - gpair_min)
            # normalized_mean_pref = avg_predicted_pref / std_pairwise
            
           
            
            # Calculate decaying weight factor (0 < weight_decay < 1)
       
            weight_decay = 1.0 / (1.0 + self.gamma)
            # Combined weight for preference models (decaying)
            weight_pref_models = weight_decay
            # Weight for model_gp (increasing with iteration)
            # w_model = 1.0 - w_preference  # Maintain sum of weights as 1
            
        
            
            # Combine posterior preferences with weights
            avg_predicted_pref =  normalized_mean_pref # + normalized_mean_gppair # 0.4, 0.6
            # Normalize performance (assuming access to true values)
            normalized_performance = (likelihood_gp_mean - best_value) / likelihood_gp_std
            

            # Standard deviation of the preference predictions
            # std_pref = torch.sqrt(0.5 * (torch.pow(normalized_mean_gppair - avg_predicted_pref, 2) + torch.pow(normalized_mean_pref - avg_predicted_pref, 2)))
            std_pref = std_pairwise
            # Option 1: Upper Confidence Bound (UCB) with uncertainty weighting
            # w_uncertainty = 1 / (std_pref + epsilon)
            # ucb_weight = w_exploration * std_single  # Assuming access to std_single for uncertainty weighting
            # ucb = normalized_performance + ucb_weight * torch.log(1  / (std_single**2 + epsilon))
            # acquisition_function_ucb = ucb + w_uncertainty * w_preference * (avg_predicted_pref - std_pref)  # Include std_pref for uncertainty

            # # Option 2: Expected Improvement (EI) using standard normal CDF
            # improvement = (w_model * normalized_performance + w_preference * avg_predicted_pref) - best_value
            # Improvement with combined weights
            improvement = weight_pref_models * avg_predicted_pref + (1.0 - weight_pref_models) * normalized_performance - best_value

            # uncertainty_pref = 1 / (std_pref + epsilon)
            # uncertainty_penalty = 1.0 / (0.6*(std_single**2 + epsilon) * 0.4* (std_pref**2 + epsilon))
            # Option 2: Separate uncertainty terms with weight decay (direct decay of preference uncertainty)
            model_uncertainty = 1.0 / (std_single**2 + epsilon)
            preference_uncertainty = 1.0 / (std_pref**2 + epsilon)
            uncertainty_penalty = (1.0 - weight_decay) * model_uncertainty + weight_decay * preference_uncertainty
          

            # uncertainty_penalty = 1.0 / ((std_single**2 + epsilon) * (std_pref**2 + epsilon))
            # uncertainty_penalty = 1.0 / ((std_single**2 ) * (std_pref**2) + epsilon)
            # improvement_penalized = improvement * uncertainty_penalty
            # improvement_penalized = improvement 
            standard_normal = Normal(torch.tensor(0.0), torch.tensor(1.0))
            #acquisition_function_ei = standard_normal.cdf(improvement  / std_single)  # Assuming access to std_single
            acquisition_function = standard_normal.cdf(improvement / std_single) + w_exploration * uncertainty_penalty
            
             # UCB exploration term (uses model uncertainty directly)
            exploration_term = w_exploration * torch.sqrt(torch.log(self.beta + 1.0) / uncertainty_penalty)
            acquisition_function = standard_normal.cdf(improvement / std_single) + exploration_term 
            
            point = torch.argmax(acquisition_function)
            return x_field[point][:X_shape[1]]

        else:
            mean_single = likelihood_gp_mean #posterior_single.mean.reshape(-1)  
            std_single =  likelihood_gp_std#torch.sqrt(posterior_single.covariance_matrix.diagonal(dim1=-2, dim2=-1))
            normalized_performance = (likelihood_gp_mean - best_value) / likelihood_gp_std
            improvement = normalized_performance - best_value
            model_uncertainty = 1.0 / (std_single**2 + epsilon)
            #preference_uncertainty = 1.0 / (std_pref**2 + epsilon)
            uncertainty_penalty = model_uncertainty
            standard_normal = Normal(torch.tensor(0.0), torch.tensor(1.0))
            #acquisition_function_ei = standard_normal.cdf(improvement  / std_single)  # Assuming access to std_single
            #acquisition_function = standard_normal.cdf(improvement / std_single) + w_exploration * uncertainty_penalty
            exploration_term = w_exploration * torch.sqrt(torch.log(self.beta.cpu() + 1.0) / uncertainty_penalty)
            acquisition_function = standard_normal.cdf(improvement / std_single) + exploration_term 
            
            point = torch.argmax(acquisition_function)
            return x_field[point][:X_shape[1]]

            # aqcf = likelihood_gp_mean + self.beta.sqrt() * likelihood_gp_std
            # point = torch.argmax(aqcf)
            # return x_field[point][:X_shape[1]]
            
    # def reshape(self, mesh):
    #     return mesh[:,:2].reshape(mesh[:,:2].shape[0],1,mesh[:,:2].shape[1])

class CoExBO_UCB_Meta(TensorManager):
    
    def __init__(
        self,
        model,
        prior_pref,
        pref_pairwise,
        beta,
        gamma,
        pi_augment=True,
    ):

        TensorManager.__init__(self)
        #self.register_buffer("beta", self.tensor(beta))
        #self.register_buffer("gamma", self.tensor(gamma))        
        self.pi_augment = pi_augment
        self.initialise(prior_pref, pref_pairwise, model)
        self.model = model
        self.step = 0
        self.beta = self.tensor(beta)
        self.gamma = self.tensor(gamma)
       

    def initialise(self, prior_pref, pref_pairwise,  model):
        if not hasattr(prior_pref, "y_test_mean"):
            prior_pref.normalising_constant()
        
        self.E_y_pref = prior_pref.y_test_mean.mean()
        self.std_y_pref = prior_pref.y_test_mean.std()
        self.E_y_obs = model.train_targets.mean()
        self.std_y_obs = model.train_targets.std()
        self.prior_pref = prior_pref
        self.pref_pairwise = pref_pairwise
        
    def prior_gp(self, X):
        prior_mean, prior_std = self.prior_pref.probability(X, both=True)

        prior_mean_conv = (prior_mean - self.E_y_pref) / self.std_y_pref * self.std_y_obs + self.E_y_obs
        prior_std_conv = prior_std / self.std_y_pref * self.std_y_obs
        return prior_mean_conv, prior_std_conv
    
    def posterior_gp(self, X, likelihood_gp_mean, likelihood_gp_std):
        
        
        prior_gp_mean, prior_gp_std = self.prior_gp(X)
        prior_gp_std_max = (
            self.gamma * likelihood_gp_std.pow(2) + prior_gp_std.pow(2)
        ).sqrt()
        posterior_gp_std = (
            prior_gp_std_max.pow(2) * likelihood_gp_std.pow(2) / (
                prior_gp_std_max.pow(2) + likelihood_gp_std.pow(2)
            )
        ).sqrt()
        posterior_gp_mean = (
            posterior_gp_std.pow(2) / prior_gp_std_max.pow(2)
        ) * prior_gp_mean + (
            posterior_gp_std.pow(2) / likelihood_gp_std.pow(2)
        ) * likelihood_gp_mean
        # print("likelihood_gp_std: ", likelihood_gp_std)
        # print("posterior_gp_std: ", posterior_gp_std)
        # print("prior_gp_std: ", prior_gp_std)

        # print("prior_gp_mean: ", prior_gp_mean)
        # print("likelihood_gp_mean: ", likelihood_gp_mean)
        # print("posterior_gp_mean: ", posterior_gp_mean)

        return posterior_gp_mean, posterior_gp_std
        
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        #print("X IN FORWARD: ", X)
        x_field = torch.tensor(self.model.test_set_X)
        self.context_x = self.model.train_X
        self.context_y = self.model.train_targets
        X_shape = self.context_x.shape
        context_x_ = self.context_x.reshape([1,X_shape[0],X_shape[1]]).cuda()
        context_y_ = self.context_y.reshape([1,len(self.context_y),1]).cuda()
        # = generate_meshgrid(self.model.lower_limit, self.model.upper_limit, dim=X_shape[1])
        #distribution = self.model.predict(context_x_.float().cuda(),context_y_.float().cuda(),mesh_grid.reshape([1,len(mesh_grid),X_shape[1]]).cuda())

        distribution = self.model.predict(context_x_,context_y_,x_field.reshape([1,len(x_field),X_shape[1]]).cuda())

        likelihood_gp_mean = torch.squeeze(distribution.mean.detach().cpu()[0])
        
        likelihood_gp_std = torch.squeeze(distribution.scale.detach().cpu()[0]) 
        # print("likelihood_gp_std before: ", likelihood_gp_std)
        # print("self.beta.sqrt(): ", self.beta.sqrt())
        #likelihood_gp_mean, likelihood_gp_std = self._mean_and_sigma(X)
        if self.pi_augment:
            posterior_gp_mean, posterior_gp_std = self.posterior_gp(x_field, likelihood_gp_mean, likelihood_gp_std)
            aqcf = posterior_gp_mean + self.beta.sqrt() * posterior_gp_std
            point = torch.argmax(aqcf)
            return x_field[point][:X_shape[1]]
        else:
            aqcf = likelihood_gp_mean + self.beta.sqrt() * likelihood_gp_std
            point = torch.argmax(aqcf)
            return x_field[point][:X_shape[1]]
    # def reshape(self, mesh):
    #     return mesh[:,:2].reshape(mesh[:,:2].shape[0],1,mesh[:,:2].shape[1])

class PiBO_UCB(AnalyticAcquisitionFunction, TensorManager):
    def __init__(
        self,
        model,
        prior_pref,
        beta,
        gamma,
        pi_augment=True,
    ):
        AnalyticAcquisitionFunction.__init__(
            self, 
            model=model,
            posterior_transform=None,
        )
        TensorManager.__init__(self)
        self.prior_pref = prior_pref
        self.pi_augment = pi_augment
        self.register_buffer("beta", self.tensor(beta))
        self.register_buffer("gamma", self.tensor(gamma))
        
    def prior_gp(self, X):
        prior_mean = self.prior_pref.pdf(X)
        return prior_mean
        
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        mean, sigma = self._mean_and_sigma(X)
        ucb = mean + self.beta.sqrt() * sigma
        if self.pi_augment:
            prior_mean = self.prior_gp(X)
            ucb = ucb * prior_mean.pow(self.gamma)
        return ucb
