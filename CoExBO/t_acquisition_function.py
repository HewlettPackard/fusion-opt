import torch
from torch.distributions import Normal
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.utils import t_batch_mode_transform
from ._utils import TensorManager
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition import qExpectedImprovement
from botorch.acquisition import ExpectedImprovement
from botorch.utils.sampling import draw_sobol_samples
from xgboost import DMatrix

class MEBH_expected_improvement(AnalyticAcquisitionFunction, TensorManager):
    
    # num_samples=100,
    def __init__(
        self,
        model,
        prior_pref,
        pref_pairwise,
        xgboost_model,
        beta ,
        gamma
      
    ):
        AnalyticAcquisitionFunction.__init__(
            self, 
            model=model,
            posterior_transform=None,
        )
        TensorManager.__init__(self)
        self.register_buffer("beta", self.tensor(beta))
        self.register_buffer("gamma", self.tensor(gamma))
        self.model_gp = model
        self.model_preference = pref_pairwise
        self.prior_pref = prior_pref
        self.pref_pairwise = pref_pairwise
        self.xgboost_model = xgboost_model
        # self.num_samples = num_samples
        self.min_value = 1e-3 #1e-6
        self.MC_SAMPLES = 512 
        self.initial_clipping_threshold = 100.0
        self.clipping_quantile = 0.99
        
        
        if not hasattr(prior_pref, "y_test_mean"):
            prior_pref.normalising_constant()
        
        self.E_y_pref = prior_pref.y_test_mean.mean()
        self.std_y_pref = prior_pref.y_test_mean.std()
     
    def clip_reciprocal_quantile(self, gamma, clipping_threshold=None):
        """Clips the reciprocal of gamma based on a quantile of observed values.

        Args:
            gamma: The current value of gamma.
            clipping_threshold: An optional overall clipping threshold (defaults to None).

        Returns:
            The clipped reciprocal of gamma.
        """
        if clipping_threshold is None:
            # Initialize or update the list of observed reciprocals
            if not hasattr(self, 'observed_reciprocals'):
                self.observed_reciprocals = []
            self.observed_reciprocals.append(1.0 / gamma)

            # Clip based on the quantile of observed reciprocals (if enough data is available)
            if len(self.observed_reciprocals) > 10:  # Adjust minimum data points as needed
                clipping_threshold = torch.quantile(torch.tensor(self.observed_reciprocals), self.clipping_quantile)
        
        # Clip the reciprocal with the threshold (or directly with gamma if no threshold)
        clipped_reciprocal = min(1.0 / gamma, clipping_threshold if clipping_threshold is not None else float('inf'))
        return clipped_reciprocal 

    def clip_reciprocal_decay(self, gamma):
        """Clips the reciprocal of gamma with adaptive decay.

        Args:
            gamma: The current value of gamma.

        Returns:
            The clipped reciprocal of gamma.
        """
        # Update the clipping threshold with decay
        
        decay_factor = 0.9
        self.clipping_threshold = max(self.clipping_threshold * decay_factor, self.initial_clipping_threshold)
        return min(1.0 / gamma, self.clipping_threshold)

    @t_batch_mode_transform(expected_q=1)
    def forward(self,X):
        best_value = self.model.train_targets.max()
        w_exploration  =  0.5#self.beta
        k = 1.0
        # w_preference =  0.5#self.beta
        epsilon =  1e-6
        # w_uncertainty =  0.65 #0.2  # fixed weight which may not be optimal,  use the weight based uncertainty commented below
        X = X.squeeze(1)  # Remove unnecessary singleton dimension (if present)

        # SingleTaskGP posterior (predictions and uncertainties)
        posterior_single = self.model_gp.posterior(X)
        mean_single = posterior_single.mean.reshape(-1)  
        std_single = torch.sqrt(posterior_single.covariance_matrix.diagonal(dim1=-2, dim2=-1))
        
        
        #fixing std with zero values
        # Check for zero values in std_single
        mask = torch.eq(std_single, 0.0)  # Find elements equal to zero
        # Add noise to elements with zero std_single
        noise_level=1e-3
        noise = torch.normal(mean=0.0, std=noise_level, size=std_single.size())
        std_single_noise = std_single + noise * mask
        std_single_noise = std_single 

        
     
        

        avg_predicted_pref, std_pairwise = self.prior_pref.probability(X, both=True)
        
     
        


        # Normalize performance using std_single with noise
        normalized_performance = (mean_single - best_value) / std_single_noise


        min_pref, max_pref = torch.min(avg_predicted_pref), torch.max(avg_predicted_pref)
        normalized_mean_pref = (avg_predicted_pref - min_pref) / (max_pref - min_pref)
     
        
     
        # self.gamma = self.gamma_init * (t**2) in which self.gamma_init=0.01 that's how self.gamma is implemented, t is the time step or iteration number
        # Calculate decaying weight based on gamma, potentially with a lower bound
        
        
        
       
        weight_decay = 1.0 / (1.0 + self.gamma)
        # Combined weight for preference models (decaying)
        weight_pref_models = weight_decay

     
        
        
        # Combine posterior preferences with weights
        avg_predicted_pref =  normalized_mean_pref 
        # avg_predicted_pref = (normalized_mean_pref  normalized_mean_gppair ) # 0.4, 0.6
        std_pref = std_pairwise 
        
        # print('std_single:', std_single)
        
        model_uncertainty = 1.0 / (std_single**2 + epsilon)
        preference_uncertainty = 1.0 / (std_pref**2 + epsilon)
        uncertainty_penalty = (1.0 - weight_decay) * model_uncertainty + weight_decay * preference_uncertainty 
        
        improvement = weight_pref_models * avg_predicted_pref + (1.0 - weight_pref_models) * normalized_performance - best_value
        standard_normal = Normal(torch.tensor(0.0), torch.tensor(1.0))
        exploration_term = w_exploration * torch.sqrt(torch.log(self.beta + 1.0) / uncertainty_penalty)
        # print('exploration_term:', exploration_term)
        acquisition_function =    standard_normal.cdf(improvement / std_single) + exploration_term   
        # print('improvement:', improvement)
              
              

        return acquisition_function
    

    @t_batch_mode_transform(expected_q=1)
    def forward__1(self,X):
        best_value = self.model.train_targets.max()
        w_exploration  = 0.8  # self.beta
        k = 1.0
        w_preference =  0.5#self.beta
        epsilon =  1e-6
        w_uncertainty =  0.65 #0.2  # fixed weight which may not be optimal,  use the weight based uncertainty commented below
        X = X.squeeze(1)  # Remove unnecessary singleton dimension (if present)

        # SingleTaskGP posterior (predictions and uncertainties)
        posterior_single = self.model_gp.posterior(X)
        mean_single = posterior_single.mean.reshape(-1)  
        std_single = torch.sqrt(posterior_single.covariance_matrix.diagonal(dim1=-2, dim2=-1))

        
        posterior_gppair = self.model_preference.posterior(X)
        avg_predicted_gppair = posterior_gppair.mean.reshape(-1)    # Average predicted preference scores
        std_gppair = torch.sqrt(posterior_gppair.covariance_matrix.diagonal(dim1=-2, dim2=-1))  # Extract diagonal elements
        

        avg_predicted_pref, std_pairwise = self.prior_pref.probability(X, both=True)
        
        # Predict preference probabilities (using trained XGBoost model)
        xgboost_x = DMatrix(X.numpy())
        predicted_probs = self.xgboost_model.predict(xgboost_x)
        # Convert results back to PyTorch tensors
        predicted_probs = torch.from_numpy(predicted_probs)
        std_preferences = torch.sqrt(predicted_probs * (1 - predicted_probs))
        
        
        # Normalize performance and preference scores
        normalized_mean_single = (mean_single - best_value) / std_single
        min_pref, max_pref = torch.min(avg_predicted_pref), torch.max(avg_predicted_pref)
        normalized_mean_pref = (avg_predicted_pref - min_pref) / (max_pref - min_pref)
        # normalized_mean_pref = avg_predicted_pref / std_pairwise
        normalized_mean_gppair = avg_predicted_gppair / std_gppair
        print('maxes:', normalized_mean_pref.max(), normalized_mean_gppair.max())
        
        # Calculate decaying weight factor (0 < weight_decay < 1)
        weight_decay = 1.0 / (1.0 + self.gamma)
        # Preference weight (decaying with iteration)
        w_preference = 0.6 * weight_decay
        # Weight for model_gp (increasing with iteration)
        w_model = 1.0 - w_preference  # Maintain sum of weights as 1
        
        
        # Combine posterior preferences with weights
        avg_predicted_pref = (0.33* normalized_mean_pref + 0.34 *normalized_mean_gppair + 0.33 * predicted_probs) # 0.4, 0.6

        # Standard deviation of the preference predictions
        # std_pref = torch.sqrt(0.5 * (torch.pow(normalized_mean_gppair - avg_predicted_pref, 2) + torch.pow(normalized_mean_pref - avg_predicted_pref, 2)))
         # Standard deviation of the preference predictions
        std_pref = torch.sqrt(0.5 * (torch.pow(normalized_mean_gppair - avg_predicted_pref, 2) + torch.pow(normalized_mean_pref - avg_predicted_pref, 2) + torch.pow(predicted_probs - avg_predicted_pref, 2)))
        

        
        
        improvement = (w_model * normalized_mean_single + w_preference * avg_predicted_pref) - best_value
        uncertainty_penalty = 1.0 / (0.6*(std_single**2 + epsilon) * 0.4* (std_pref**2 + epsilon))
        
        # Option 2: Separate uncertainty terms with weight decay (direct decay of preference uncertainty)
        model_uncertainty = 1.0 / (0.6 * std_single**2 + epsilon)
        preference_uncertainty = 1.0 / (0.4 * std_pref**2 + epsilon)
        uncertainty_penalty = (1.0 - weight_decay) * model_uncertainty + weight_decay * preference_uncertainty

        
        # Probability of improvement using standard normal CDF
        standard_normal = Normal(torch.tensor(0.0), torch.tensor(1.0))
        acquisition_function_ei = standard_normal.cdf(improvement / std_single)
        exploration_term = w_exploration * uncertainty_penalty
        acquisition_function = acquisition_function_ei  +  w_uncertainty  * exploration_term
        return acquisition_function


    @t_batch_mode_transform(expected_q=1)
    def forward___2(self,X):
        best_value = self.model.train_targets.max()
        w_exploration  = 0.8  # self.beta
        k = 1.0
        w_preference =  0.5#self.beta
        epsilon =  1e-6
        w_uncertainty =  0.65 #0.2  # fixed weight which may not be optimal,  use the weight based uncertainty commented below
        X = X.squeeze(1)  # Remove unnecessary singleton dimension (if present)

        # SingleTaskGP posterior (predictions and uncertainties)
        posterior_single = self.model_gp.posterior(X)
        mean_single = posterior_single.mean.reshape(-1)  
        std_single = torch.sqrt(posterior_single.covariance_matrix.diagonal(dim1=-2, dim2=-1))

        
        posterior_gppair = self.model_preference.posterior(X)
        avg_predicted_gppair = posterior_gppair.mean.reshape(-1)    # Average predicted preference scores
        std_gppair = torch.sqrt(posterior_gppair.covariance_matrix.diagonal(dim1=-2, dim2=-1))  # Extract diagonal elements
        

        avg_predicted_pref, std_pairwise = self.prior_pref.probability(X, both=True)
        
        # Predict preference probabilities (using trained XGBoost model)
        xgboost_x = DMatrix(X.numpy())
        predicted_probs = self.xgboost_model.predict(xgboost_x)
        # Convert results back to PyTorch tensors
        predicted_probs = torch.from_numpy(predicted_probs)
        std_preferences = torch.sqrt(predicted_probs * (1 - predicted_probs))
        
        
        # Normalize performance and preference scores
        normalized_mean_single = (mean_single - best_value) / std_single
        min_pref, max_pref = torch.min(avg_predicted_pref), torch.max(avg_predicted_pref)
        normalized_mean_pref = (avg_predicted_pref - min_pref) / (max_pref - min_pref)
        # normalized_mean_pref = avg_predicted_pref / std_pairwise
        normalized_mean_gppair = avg_predicted_gppair / std_gppair
        print('maxes:', normalized_mean_pref.max(), normalized_mean_gppair.max())
        
        # Calculate decaying weight factor (0 < weight_decay < 1)
        weight_decay = 1.0 / (1.0 + self.gamma)
        # Preference weight (decaying with iteration)
        w_preference = 0.6 * weight_decay
        # Weight for model_gp (increasing with iteration)
        w_model = 1.0 - w_preference  # Maintain sum of weights as 1
        
        
        # Combine posterior preferences with weights
        avg_predicted_pref = (0.4* normalized_mean_pref + 0.6 *normalized_mean_gppair ) # 0.4, 0.6

        # Standard deviation of the preference predictions
        # std_pref = torch.sqrt(0.5 * (torch.pow(normalized_mean_gppair - avg_predicted_pref, 2) + torch.pow(normalized_mean_pref - avg_predicted_pref, 2)))
         # Standard deviation of the preference predictions
        std_pref = torch.sqrt(0.5 * (torch.pow(normalized_mean_gppair - avg_predicted_pref, 2) + torch.pow(normalized_mean_pref - avg_predicted_pref, 2)))
        

        
        
        improvement = (w_model * normalized_mean_single + w_preference * avg_predicted_pref) - best_value
        uncertainty_penalty = 1.0 / (0.6*(std_single**2 + epsilon) * 0.4* (std_pref**2 + epsilon))
        
        # Option 2: Separate uncertainty terms with weight decay (direct decay of preference uncertainty)
        model_uncertainty = 1.0 / (0.6 * std_single**2 + epsilon)
        preference_uncertainty = 1.0 / (0.4 * std_pref**2 + epsilon)
        uncertainty_penalty = (1.0 - weight_decay) * model_uncertainty + weight_decay * preference_uncertainty

        
        # Probability of improvement using standard normal CDF
        standard_normal = Normal(torch.tensor(0.0), torch.tensor(1.0))
        acquisition_function_ei = standard_normal.cdf(improvement / std_single)
        exploration_term = w_exploration * uncertainty_penalty
        acquisition_function = acquisition_function_ei  +  w_uncertainty  * exploration_term
        return acquisition_function
    
    
    @t_batch_mode_transform(expected_q=1)
    def forward_4(self,X):
        best_value = self.model.train_targets.max()
        w_exploration  = 0.8  # self.beta
        k = 1.0
        w_preference =  0.5#self.beta
        epsilon =  1e-6
        w_uncertainty =  0.65 #0.2  # fixed weight which may not be optimal,  use the weight based uncertainty commented below
        X = X.squeeze(1)  # Remove unnecessary singleton dimension (if present)

        # SingleTaskGP posterior (predictions and uncertainties)
        posterior_single = self.model_gp.posterior(X)
        mean_single = posterior_single.mean.reshape(-1)  
        std_single = torch.sqrt(posterior_single.covariance_matrix.diagonal(dim1=-2, dim2=-1))

        
       
        

        avg_predicted_pref, std_pairwise = self.prior_pref.probability(X, both=True)
        
      
        
        
        # Normalize performance and preference scores
        normalized_mean_single = (mean_single - best_value) / std_single
        min_pref, max_pref = torch.min(avg_predicted_pref), torch.max(avg_predicted_pref)
        normalized_mean_pref = (avg_predicted_pref - min_pref) / (max_pref - min_pref)
        # normalized_mean_pref = avg_predicted_pref / std_pairwise

        
        # Calculate decaying weight factor (0 < weight_decay < 1)
        weight_decay = 1.0 / (1.0 + self.gamma)
        # Preference weight (decaying with iteration)
        w_preference = 0.6 * weight_decay
        # Weight for model_gp (increasing with iteration)
        w_model = 1.0 - w_preference  # Maintain sum of weights as 1
        
        
        # Combine posterior preferences with weights
        avg_predicted_pref = normalized_mean_pref #+ 0.34 *normalized_mean_gppair  # 0.4, 0.6

        # Standard deviation of the preference predictions
        # std_pref = torch.sqrt(0.5 * (torch.pow(normalized_mean_gppair - avg_predicted_pref, 2) + torch.pow(normalized_mean_pref - avg_predicted_pref, 2)))
         # Standard deviation of the preference predictions
        # std_pref = torch.sqrt(0.5 * (torch.pow(normalized_mean_gppair - avg_predicted_pref, 2) + torch.pow(normalized_mean_pref - avg_predicted_pref, 2) + torch.pow(predicted_probs - avg_predicted_pref, 2)))
        std_pref = std_pairwise
        

        
        
        improvement = (w_model * normalized_mean_single + w_preference * avg_predicted_pref) - best_value
        uncertainty_penalty = 1.0 / (0.6*(std_single**2 + epsilon) * 0.4* (std_pref**2 + epsilon))
        
        # Option 2: Separate uncertainty terms with weight decay (direct decay of preference uncertainty)
        model_uncertainty = 1.0 / (0.6 * std_single**2 + epsilon)
        preference_uncertainty = 1.0 / (0.4 * std_pref**2 + epsilon)
        uncertainty_penalty = (1.0 - weight_decay) * model_uncertainty + weight_decay * preference_uncertainty
        
        clipping_threshold = 100.0
        clipped_reciprocal = min(1.0 / self.gamma, clipping_threshold)
        # combined_uncertainty = 1.0 / (std_single**2 + 1.0 / self.gamma * std_pref**2 + epsilon)
        uncertainty_penalty = 1.0 / (std_single**2 + clipped_reciprocal * std_pref**2 + epsilon)

        
        # Probability of improvement using standard normal CDF
        standard_normal = Normal(torch.tensor(0.0), torch.tensor(1.0))
        acquisition_function_ei = standard_normal.cdf(improvement / std_single)
        exploration_term = w_exploration * uncertainty_penalty
        acquisition_function = acquisition_function_ei  +   exploration_term
        return acquisition_function
        

   

