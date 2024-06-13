import torch
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.utils import t_batch_mode_transform
from ._utils import TensorManager

class MEBH_expected_improvement__(AnalyticAcquisitionFunction, TensorManager):
    
    def __init__(
        self,
        model,
        prior_pref,
        num_samples=100,
        kappa=0.1,
      
    ):
        AnalyticAcquisitionFunction.__init__(
            self, 
            model=model,
            posterior_transform=None,
        )
        TensorManager.__init__(self)
        self.register_buffer("kappa", self.tensor(kappa))
        self.model_gp = model
        self.model_preference = prior_pref
        self.num_samples = num_samples
     
     
    # @t_batch_mode_transform(expected_q=1)
    # def forward(self, X):    
        

            

    #     # Get mean and variance predictions from the SingleTaskGP model
    #     mvn_gp = self.model_gp(X)
    #     # Extract mean and covariance matrix from MultivariateNormal distribution
    #     mean_gp = mvn_gp.mean
    #     cov_gp = mvn_gp.covariance_matrix
        
    #     # Calculate the variance from the diagonal of the covariance matrix
    #     var_gp = cov_gp.diagonal(dim1=-2, dim2=-1)
    #     # var_gp = var_gp.unsqueeze(-1)

        
    #     # Get mean and standard deviation predictions from the preference prior model
    #     mean_pref, std_pref = self.model_preference.probability(X, both=True)
        
    #     # Sample from the preference distribution
    #     pref_samples = torch.distributions.Normal(mean_pref, std_pref).sample((self.num_samples,))
    #     # Calculate expected improvement for each preference sample
    #     ei_samples = []
    #     for pref_sample in pref_samples:
    #         pref_mean_sample = pref_sample.mean(dim=-1)
    #         pref_std_sample = pref_sample.std(dim=-1)
    #         z = (mean_gp - pref_mean_sample) / pref_std_sample
    #         improvement = mean_gp - pref_mean_sample - self.kappa
    #         std_gp = torch.sqrt(var_gp)
    #         u = improvement / std_gp
    #         ei = improvement * torch.distributions.Normal(0, 1).cdf(u) + std_gp * torch.distributions.Normal(0, 1).log_prob(u)
    #         ei = ei * torch.distributions.Normal(0, 1).cdf(z)
    #         ei_samples.append(ei)
        
    #     # Compute the mean of expected improvements over samples
    #     ei_samples = torch.stack(ei_samples)
    #     ei_mean = ei_samples.mean(dim=0)
        
    #     # Squeeze extra dimension
    #     ei_mean = ei_mean.squeeze(dim=-1)
        
    #     return ei_mean

        
  
      
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):    
        

            

        # Get mean and variance predictions from the SingleTaskGP model
        mvn_gp = self.model_gp(X)
        # Extract mean and covariance matrix from MultivariateNormal distribution
        mean_gp = mvn_gp.mean
        cov_gp = mvn_gp.covariance_matrix
        
        # Calculate the variance from the diagonal of the covariance matrix
        var_gp = cov_gp.diagonal(dim1=-2, dim2=-1)
        # var_gp = var_gp.unsqueeze(-1)

        
        # Get mean and standard deviation predictions from the preference prior model
        mean_pref, std_pref = self.model_preference.probability(X, both=True)
        
        # Sample from the preference distribution
        pref_samples = torch.distributions.Normal(mean_pref, std_pref).sample((self.num_samples,))
        
        # Calculate the mean and standard deviation of the preference distribution
        # We assume a normal distribution for simplicity
        # pref_dist = torch.distributions.Normal(mean_pref, std_pref)
        pref_mean = mean_pref#pref_dist.mean
        pref_std =  std_pref#pref_dist.stddev
        
        # Calculate the standardization term
        z = (mean_gp - pref_mean) / pref_std
        
        # Calculate the improvement
        improvement = mean_gp - pref_mean - self.kappa
        
        # Calculate the expected improvement
        std_gp = torch.sqrt(var_gp)
        u = improvement / std_gp
        # ei = improvement * torch.distributions.Normal(0, 1).cdf(u) + std_gp * torch.distributions.Normal(0, 1).log_prob(u)
        # ei = ei * pref_dist.cdf(z)
        ei = improvement * torch.distributions.Normal(0, 1).cdf(u) + std_gp * torch.distributions.Normal(0, 1).log_prob(u)
        ei = ei * torch.distributions.Normal(0, 1).cdf(z)
        # Sum along the last dimension to get a single value for each data point
        ei = ei.sum(dim=-1)
        
        
        return ei


