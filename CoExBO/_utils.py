import torch
import warnings
from torch.quasirandom import SobolEngine
from torch.distributions.multivariate_normal import MultivariateNormal


class TensorManager:
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        #self.device = torch.device('cpu')
        self.dtype = torch.double
        
    def standardise_tensor(self, tensor):
        return tensor.to(self.device, self.dtype)
    
    def standardise_device(self, tensor):
        return tensor.to(self.device)
    
    def ones(self, n_samples, n_dims=None):
        if n_dims == None:
            return self.standardise_tensor(torch.ones(n_samples))
        else:
            return self.standardise_tensor(torch.ones(n_samples, n_dims))
    
    def zeros(self, n_samples, n_dims=None):
        if n_dims == None:
            return self.standardise_tensor(torch.zeros(n_samples))
        else:
            return self.standardise_tensor(torch.zeros(n_samples, n_dims))
    
    def rand(self, n_dims, n_samples, qmc=True):
        if qmc:
            random_samples = SobolEngine(n_dims, scramble=True).draw(n_samples)
        else:
            random_samples = torch.rand(n_samples, n_dims)
        return self.standardise_tensor(random_samples)
    
    def arange(self, length):
        return self.standardise_device(torch.arange(length))
    
    def null(self):
        return self.standardise_device(torch.tensor([]))
    
    def tensor(self, x):
        return self.standardise_tensor(torch.tensor(x))
    
    def randperm(self, length):
        return self.standardise_device(torch.randperm(length))
    
    def multinomial(self, weights, n):
        return self.standardise_device(torch.multinomial(weights, n))
    
    def sqrt(self, x):
        return self.tensor(x).sqrt()
    
    def exp(self, x):
        return self.tensor(x).exp()
    
    def numpy(self, x):
        return x.detach().cpu().numpy()
    
    def from_numpy(self, x):
        return self.standardise_tensor(torch.from_numpy(x))
    
    def is_cuda(self):
        if self.device == torch.device('cuda'):
            return True
        else:
            return False

class SafeTensorOperator(TensorManager):
    def __init__(self):
        TensorManager.__init__(self)
        self.eps = -torch.sqrt(torch.tensor(torch.finfo().max)).item()
        self.gpu_lim = int(5e5)
        self.max_iter = 10
        
    def remove_anomalies(self, y):
        """
        Args:
           - y: torch.tensor, observations

        Returns:
           - y: torch.tensor, observations whose anomalies have been removed.
        """
        y[y.isnan()] = self.eps
        y[y.isinf()] = self.eps
        y[y < self.eps] = self.eps
        return y

    def remove_anomalies_uniform(self, X, uni_min, uni_max):
        """
        Args:
           - X: torch.tensor, inputs
           - uni_min: torch.tensor, the minimum limit values of uniform distribution
           - uni_max: torch.tensor, the maximum limit values of uniform distribution

        Returns:
           - idx: bool, indices where the inputs X do not exceed the min-max limits
        """
        logic = torch.sum(torch.stack([torch.logical_or(
            X[:, i] < uni_min[i],
            X[:, i] > uni_max[i],
        ) for i in range(X.size(1))]), axis=0)
        return (logic == 0)

    def is_psd(self, mat):
        """
        Args:
           - mat: torch.tensor, symmetric matrix

        Returns:
           - flag: bool, flag to judge whether or not the given matrix is positive semi-definite
        """
        return bool((mat == mat.T).all() and (torch.linalg.eig(mat)[0].real >= 0).all())
        
    def make_cov_psd(self, cov):
        """
        Args:
           - cov: torch.tensor, covariance matrix of multivariate normal distribution

        Returns:
           - cov: torch.tensor, covariance matrix of multivariate normal distribution
        """
        if self.is_psd(cov):
            return cov
        else:
            warnings.warn("Estimated covariance matrix was not positive semi-definite. Conveting...")
            cov = torch.nan_to_num(cov)
            cov = torch.sqrt(cov * cov.T)
            if not self.is_psd(cov):
                n_dim = cov.size(0)
                r_increment = 2
                jitter = self.ones(n_dim) * 1e-5
                n_iter = 0
                while not self.is_psd(cov):
                    cov[range(n_dim), range(n_dim)] += jitter
                    jitter *= r_increment
                    n_iter += 1
                    if n_iter > self.max_iter:
                        cov = cov.diag().diag()
                        break
            return cov

    def safe_mvn_register(self, mu, cov):
        """
        Args:
           - mu: torch.tensor, mean vector of multivariate normal distribution
           - cov: torch.tensor, covariance matrix of multivariate normal distribution

        Returns:
           - mvn: torch.distributions, function of multivariate normal distribution
        """
        cov = self.make_cov_psd(cov)
        return MultivariateNormal(mu, cov)
        
    def safe_mvn_prob(self, mu, cov, X):
        """
        Args:
           - mu: torch.tensor, mean vector of multivariate normal distribution
           - cov: torch.tensor, covariance matrix of multivariate normal distribution
           - X: torch.tensor, the locations that we wish to calculate the probability density values

        Returns:
           - pdf: torch.tensor, the probability density values at given locations X.
        """
        mvn = self.safe_mvn_register(mu, cov)
        if X.size(0) > self.gpu_lim:
            warnings.warn("The matrix size exceeds the GPU limit. Splitting.")
            n_split = torch.tensor(X.size(0) / self.gpu_lim).ceil().long()
            _X = torch.tensor_split(X, n_split)
            Npdfs = torch.cat(
                [
                    mvn.log_prob(_X[i]).exp()
                    for i in range(n_split)
                ]
            )
        else:
            Npdfs = mvn.log_prob(X).exp()
        return Npdfs
        
class Utils(SafeTensorOperator):
    def __init__(self):
        super().__init__()

    
def generate_meshgrid(lb, ub, dim = 2, num_points = 40, step = 0.025):
    
    tensors = []
    for _ in range(dim):
        tensor = torch.linspace(lb, ub, num_points)
        tensors.append(tensor)
    
    # Stack the tensors along the second dimension to create a 2D tensor
    tensor_combined = torch.stack(tensors, dim=1)
    return tensor_combined