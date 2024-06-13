import torch
import torch.distributions as D
from ._prior import Uniform
from ._utils import TensorManager, convert_to_pair


class DuelFeedback(TensorManager):
    def __init__(
        self, 
        prior_init, 
        true_function,
        n_suggestions=2,
        noisy=False,
        
    ):
        """
        Class for duel feedback.
        
        Args:
        - prior_init: CoExBO._prior.BasePrior, the prior distribution over the domain.
        - true_function: class, the function that returns the true f values
        - noisy: bool, whether or not the feedback contains noisy observations
        """
        super().__init__()
        self.n_suggestions = n_suggestions
        self.prior_duel = Uniform(prior_init.bounds.repeat(1,n_suggestions))
        self.true_function = true_function
        self.noisy = noisy

    def initialise_variance(self, dataset_obj):
        """
        Compute the variance of objective values for transformation.
        
        Args:
        - dataset_obj: list, list of the observed samples for objective function.
        """
        if self.noisy:
            self.Y_var = dataset_obj[2].var()
        else:
            self.Y_var = dataset_obj[1].var()
            
            
    def n_feedback(self, X_n_wise, sigma=0, in_loop=True):
        num_initial_points = len(X_n_wise)
        dim = X_n_wise.shape[1] // self.n_suggestions
        
        
        chunks = torch.chunk(X_n_wise, dim=1, chunks=self.n_suggestions)
        noises = []
        y_vals = []
        for chunk_id in range(self.n_suggestions):
            chunk = chunks[chunk_id]    
            if not sigma == 0:
                noises.append(D.Normal(0, sigma*self.Y_var).sample(torch.Size([len(chunk)])))
            else:
                noises.append(self.zeros(len(chunk)))
                
            y = None
            if self.noisy:
                _, y = self.true_function(chunk.squeeze())
            else:
                y = self.true_function(chunk.squeeze())
            y_vals.append(y)
            
        
        X_pairwise = torch.tensor([], dtype=X_n_wise.dtype,  device=X_n_wise.device)    
        y_pairwise = torch.tensor([], dtype=torch.int8,  device=X_n_wise.device)
        y_pairwise_unsure = torch.tensor([], dtype=torch.int8,  device=X_n_wise.device)
        for i in range(self.n_suggestions):
            for j in range(i + 1, self.n_suggestions):
                
                curr_X_pairwise = torch.cat((chunks[i],chunks[j]),dim=1)
                X_pairwise = torch.cat((X_pairwise,curr_X_pairwise))
                
                y = y_vals[i] 
                noise = noises[i]
                
                y_prime = y_vals[j] 
                noise_prime =  noises[j]
                
                currnt_y_pairwise = (y + noise > y_prime + noise_prime).long()
                thresh = sigma * self.Y_var
                bool_unsure = ((y - y_prime).abs().pow(2) <= thresh)
                currnt_y_pairwise_unsure = bool_unsure.long()
                if in_loop:
                    currnt_y_pairwise[bool_unsure] = 1

                y_pairwise = torch.cat((y_pairwise, currnt_y_pairwise))
                y_pairwise_unsure = torch.cat((y_pairwise_unsure, currnt_y_pairwise_unsure))
                
                    

        return X_pairwise, y_pairwise, y_pairwise_unsure
        

    
    def feedback(self, X_n_wise, sigma=0, in_loop=True):
        """
        Synthetic human feedback function.
        
        Args:
        - X_pairwise: torch.tensor, a pairwise candidate for the next query.
        - sigma: float, Gaussian noise variance to the synthetic human selection process.
        - in_loop: bool, initial sampling if false, otherwise running in the human-in-the-loop.
        
        Return:
        - y_pairwise: torch.tensor, the observed preference result (sure)
        - y_pairwise_unsure: torch.tensor, the observed preference result (unsure)
        """
        X_pairwise = convert_to_pair(X_n_wise, self.n_suggestions) #Sahand
        X_pairwise = X_pairwise[:self.n_suggestions-1]
        u, u_prime = torch.chunk(X_pairwise, dim=1, chunks=2)
        if not sigma == 0:
            noise = D.Normal(0, sigma*self.Y_var).sample(torch.Size([len(u)]))
            noise_prime = D.Normal(0, sigma*self.Y_var).sample(torch.Size([len(u)]))
        else:
            noise = self.zeros(len(u))
            noise_prime = self.zeros(len(u_prime))
        
        if self.noisy:
            _, y = self.true_function(u.squeeze())
            _, y_prime = self.true_function(u_prime.squeeze())
        else:
            y = self.true_function(u.squeeze())
            y_prime = self.true_function(u_prime.squeeze())
        
        y_pairwise = (y + noise > y_prime + noise_prime).long()
        thresh = sigma * self.Y_var
        bool_unsure = ((y - y_prime).abs().pow(2) <= thresh)
        y_pairwise_unsure = bool_unsure.long()
        if in_loop:
            y_pairwise[bool_unsure] = 1  # Sahand check if this is correct for the case of in_loop with more n_suggestions > 2

        return y_pairwise, y_pairwise_unsure

    def augmented_feedback(self, X_pairwise, sigma=0, in_loop=True):
        """
        Skew-symmetric data augmentation for the feedback in one go.
        This heuristics is introduced in the following paper:
        Siu Lun Chau, Javier González, and Dino Sejdinovic.
        Learning inconsistent preferences with gaussian processes.
        In International Conference on Artificial Intelligence and Statistics, pp. 2266–2281. PMLR, 2022b.
        
        Args:
        - X_pairwise: torch.tensor, a pairwise candidate for the next query.
        - sigma: float, Gaussian noise variance to the synthetic human selection process.
        - in_loop: bool, initial sampling if false, otherwise running in the human-in-the-loop.
        
        Return:
        - X_pairwise: torch.tensor, the augmented pairwise candidate for the next query.
        - y_pairwise: torch.tensor, the augmented preference result (sure)
        - y_pairwise_unsure: torch.tensor, the augmented preference result (unsure)
        """
        X_pairwise, y_pairwise, y_pairwise_unsure = self.n_feedback(X_pairwise, sigma=sigma, in_loop=in_loop)
        X_pairwise, y_pairwise, y_pairwise_unsure = self.data_augment(X_pairwise, y_pairwise, y_pairwise_unsure)
        return X_pairwise, y_pairwise, y_pairwise_unsure

    def swap_columns(self, X_pairwise):
        """
        Swapping the columns for skew-symmetric data augmentation.
        
        Args:
        - X_pairwise: torch.tensor, a pairwise candidate for the next query.
        
        Return:
        - X_pairwise: torch.tensor, the swapped pairwise candidate for the next query.
        """
        dim = int(X_pairwise.shape[-1] / 2)
        index = torch.cat([torch.arange(dim,2*dim), torch.arange(dim)])
        return X_pairwise[:,index]

    def data_augment(self, X_n_wise, y_pairwise, y_pairwise_unsure):
        """
        Skew-symmetric data augmentation operation.
        
        Args:
        - X_pairwise: torch.tensor, a pairwise candidate for the next query.
        - y_pairwise_next: torch.tensor, a preference result (sure)
        - y_pairwise_unsure_next: torch.tensor, a preference result (unsure)
        
        Return:
        - X_pairwise: torch.tensor, the augmented pairwise candidate for the next query.
        - y_pairwise: torch.tensor, the augmented preference result (sure)
        - y_pairwise_unsure: torch.tensor, the augmented preference result (unsure)
        """
        X_pairwise = convert_to_pair(X_n_wise, self.n_suggestions)
        X_pairwise = X_pairwise[:self.n_suggestions-1]
        X_pairwise_swap = self.swap_columns(X_pairwise)
        X_cat = torch.vstack([X_pairwise, X_pairwise_swap])
        y_pairwise_swap = abs(y_pairwise_unsure - y_pairwise)
        y_cat = torch.cat([y_pairwise, y_pairwise_swap], dim=0)
        y_unsure = torch.cat([y_pairwise_unsure, y_pairwise_unsure], dim=0)
        return X_cat, y_cat, y_unsure
    
    def sample(self, n_sample):
        """
        Draw i.i.d. samples from the expanded pairwise domain.
        
        Args:
        - n_sample: int, number of random samples to draw.
        
        Return:
        - X_pairwise: torch.tensor, pairwise samples.
        """
        X_pairwise = self.prior_duel.sample(n_sample)
        return X_pairwise

    def sample_both(self, n_init, sigma=0, in_loop=True):
        """
        Do the following at once:
        1. random sampling
        2. skew-symmetric data augmentation
        3. synthetic human feedback
        
        Args:
        - n_init: int, number of initial random samples.
        - sigma: float, Gaussian noise variance to the synthetic human selection process.
        - in_loop: bool, initial sampling if false, otherwise running in the human-in-the-loop.
        
        Return:
        - X_pairwise: torch.tensor, the augmented pairwise candidate for the next query.
        - y_pairwise: torch.tensor, the augmented preference result (sure)
        - y_pairwise_unsure: torch.tensor, the augmented preference result (unsure)
        """
        X_pairwise = self.sample(n_init)
        X_pairwise, y_pairwise, y_pairwise_unsure = self.augmented_feedback(X_pairwise, sigma=sigma, in_loop=in_loop)
        return X_pairwise, y_pairwise, y_pairwise_unsure

    def update_and_augment_data(self, dataset_duel, dataset_duel_new):
        """
        Do the following at once:
        1. Do skew-symmetric data augmentation for the new query
        2. Merge old and new datasets
        
        Args:
        - dataset_duel: list, list of the observed samples for human preference.
        - dataset_duel_new: list, list of the newly observed samples for human preference.
        
        Return:
        - dataset_duel_new: list, list of the updated samples for human preference.
        """
        X_pairwise, y_pairwise, y_pairwise_unsure = dataset_duel
        X_pairwise_new, y_pairwise_new, y_pairwise_unsure_new = dataset_duel_new
        X_pairwise_new, y_pairwise_new, y_pairwise_unsure_new = self.data_augment(X_pairwise_new, y_pairwise_new, y_pairwise_unsure_new)
        X_cat = torch.vstack([X_pairwise, X_pairwise_new])
        y_cat = torch.cat([y_pairwise, y_pairwise_new], dim=0)
        y_unsure = torch.cat([y_pairwise_unsure, y_pairwise_unsure_new], dim=0)
        dataset_duel_updated = (X_cat, y_cat, y_unsure)
        return dataset_duel_updated
    
    def evaluate_correct_answer_rate(self, X_pairwise, y_pairwise):
        """
        Evaluate the correct answer rate.
        
        Args:
        - X_pairwise: torch.tensor, the pairwise candidate for the next query.
        - y_pairwise: torch.tensor, the preference result (sure)
        
        Return:
        - correct_answer_rate: float, the correct answer rate
        """
        y_true, _ = self.feedback(X_pairwise, sigma=0)
        n_total = len(y_pairwise)
        n_wrong = (y_pairwise - y_true).abs().sum().item()
        n_correct = (n_total - n_wrong)
        return n_correct / n_total
