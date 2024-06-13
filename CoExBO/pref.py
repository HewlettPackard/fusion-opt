import numpy as np
import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition import ExpectedImprovement, qExpectedImprovement

class PreferenceModelGP:
    def __init__(self, dim):
        self.dim = dim
        self.model = SingleTaskGP(torch.zeros(1, dim), torch.zeros(1))
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
    
    def update_preferences(self, x, preference):
        X = torch.tensor(x, dtype=torch.float).view(1, -1)
        y = torch.tensor([[preference]], dtype=torch.float)
        self.model.train_inputs = torch.cat([self.model.train_inputs, X])
        self.model.train_targets = torch.cat([self.model.train_targets, y])
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        self.fit_gpytorch_model()
    
    def fit_gpytorch_model(self):
        self.model.train()
        self.mll.train()
        self.model.zero_grad()
        self.mll.zero_grad()
        output = self.model(*self.mll.train_inputs)
        loss = -self.mll(output, self.mll.train_targets)
        loss.backward()
        self.model.likelihood.noise_covar.register_constraint("raw_noise", constraint=constraints.positive)
        self.model.likelihood.noise_covar.raw_noise.requires_grad = False
        self.model.likelihood.noise = 1e-4
        self.mll.optimizer.step()

def initialize_preference_model(dim, num_initial_points):
    preference_model = PreferenceModelGP(dim=dim)
    print("Please provide your feedback for the initial points:")
    for i in range(num_initial_points):
        x = np.random.rand(dim)
        print(f"Point {i+1}: {x}")
        feedback = get_feedback()
        feedback_normalized = 1.0 if feedback in [1, 2, 3] else 0.0
        preference_model.update_preferences(x, feedback_normalized)
    return preference_model

def get_feedback():
    while True:
        print("Please provide your feedback:")
        print("1. Like the first recommendation")
        print("2. Like the second recommendation")
        print("3. Like the third recommendation")
        print("4. Not sure / Need more information")
        choice = int(input("Enter your choice (1, 2, 3, 4): "))
        if choice in [1, 2, 3, 4]:
            return choice
        else:
            print("Invalid choice. Please try again.")

def recommend_three_choices(recommendations, model_qEI, preference_model_EI, best_observed_preference):
    # Use qEI to recommend two choices with the model for qEI
    qEI_acqf = qExpectedImprovement(model_qEI, best_f=best_observed_preference, q=2)
    candidates_qEI, _ = optimize_acqf(
        qEI_acqf, bounds=torch.tensor([[0.0] * model_qEI.dim, [1.0] * model_qEI.dim]),
        q=2, num_restarts=5, raw_samples=20,
    )
    next_recommendations_indices_qEI = candidates_qEI.squeeze().tolist()
    next_recommendations_qEI = [recommendations[i] for i in next_recommendations_indices_qEI]

    # Use EI for the third choice based on the separate preference model
    EI_acqf = ExpectedImprovement(preference_model_EI.model, best_f=best_observed_preference)
    candidates_EI, _ = optimize_acqf(
        EI_acqf, bounds=torch.tensor([[0.0] * preference_model_EI.dim, [1.0] * preference_model_EI.dim]),
        num_restarts=5, raw_samples=20,
    )
    next_recommendation_prefmodel_index = torch.argmax(candidates_EI)
    next_recommendation_prefmodel = recommendations[next_recommendation_prefmodel_index]

    return next_recommendations_qEI + [next_recommendation_prefmodel]

def main():
    dim = 2  # Dimensionality of the input space
    num_initial_points = 3  # Number of initial points to collect feedback for
    
    # Initialize preference model for EI
    preference_model_EI = initialize_preference_model(dim, num_initial_points)
    
    # Initial set of recommendations based on the preference model
    recommendations = recommend_three_choices([], SingleTaskGP(torch.zeros(1, dim), torch.zeros(1)), preference_model_EI, 0.0)
    
    # Initialize a model for qEI (e.g., SingleTaskGP)
    model_qEI = SingleTaskGP(torch.zeros(1, dim), torch.zeros(1))
    
    # Initialize the best observed preference score
    best_observed_preference = 0.0  # Adjust as needed based on your preference scale
    
    while True:
        # Get user's feedback
        feedback = get_feedback()
        
        # Ask for user's choice based on feedback
        if feedback == 1:
            user_choice = 0
        elif feedback == 2:
            user_choice = 1
        elif feedback == 3:
            user_choice = 2
        else:  # Not sure, need more information
            print("Please select your preferred option from the recommendations.")
            print("Here are some recommendations:")
            for i, rec in enumerate(recommendations):
                print(f"{i+1}. {rec}")
            user_choice = int(input("Please select your choice (1, 2, 3, ...): ")) - 1

        # Update preference model with user feedback for EI
        feedback_normalized = 1.0 if feedback in [1, 2, 3] else 0.0
        preference_model_EI.update_preferences(recommendations[user_choice], feedback_normalized)
        
        # Update recommendations using combination of qEI and preference model
        next_recommendations = recommend_three_choices(recommendations, model_qEI, preference_model_EI, best_observed_preference)
        
        # Update recommendations list
        recommendations.extend(next_recommendations)
        
        # Break the loop if needed
        if len(recommendations) >= 6:
            break
    
    print("Final recommendations based on your preferences:")
    for rec in recommendations:
        print(rec)

if __name__ == "__main__":
    main()
    
    
    
    import numpy as np
import torch
from botorch.models.pairwise import PairwiseGP
from botorch.models import SingleTaskGP
from botorch.acquisition import qExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf

# Global variables for preference model and objective model
preference_model = None
objective_model = None

def initialize_preference_model(dim, num_initial_points):
    global preference_model
    X = torch.tensor(np.random.rand(num_initial_points, dim), dtype=torch.float)
    pairs = torch.tensor([], dtype=torch.long)
    Y = torch.tensor([], dtype=torch.float)
    for i in range(num_initial_points):
        for j in range(i + 1, num_initial_points):
            pairs = torch.cat((pairs, torch.tensor([[i, j]], dtype=torch.long)))
            print(f"Comparison {i+1} vs {j+1}:")
            feedback = get_feedback()
            Y = torch.cat((Y, torch.tensor([feedback - 1], dtype=torch.float)))
    preference_model = PairwiseGP(X, Y, pairs=pairs)
    preference_model.to(X)

def initialize_objective_model(dim):
    global objective_model
    objective_model = SingleTaskGP(torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float))
    objective_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def get_feedback():
    while True:
        print("Please choose your preferred option:")
        print("1. Option 1")
        print("2. Option 2")
        print("3. Option 3")
        print("4. Uncertain")
        choice = int(input("Enter your choice (1, 2, 3, or 4): "))
        if choice in [1, 2, 3, 4]:
            return choice
        else:
            print("Invalid choice. Please try again.")

def recommend_three_choices(recommendations):
    global preference_model, objective_model
    qEI_acqf = qExpectedImprovement(objective_model)
    UCB_acqf = UpperConfidenceBound(preference_model, beta=0.1)  # Adjust beta as needed
    candidates_qEI, _ = optimize_acqf(
        acq_function=qEI_acqf,
        bounds=torch.tensor([[0.0] * objective_model.dim, [1.0] * objective_model.dim]),
        q=3,
        num_restarts=10,
        raw_samples=100,
    )
    candidates_UCB, _ = optimize_acqf(
        acq_function=UCB_acqf,
        bounds=torch.tensor([[0.0] * preference_model.dim, [1.0] * preference_model.dim]),
        q=3,
        num_restarts=10,
        raw_samples=100,
    )
    next_recommendations_indices = torch.cat((candidates_qEI, candidates_UCB), dim=0).squeeze().tolist()
    next_recommendations = [recommendations[i] for i in next_recommendations_indices]
    return next_recommendations

def main():
    dim = 2
    num_initial_points = 3
    
    initialize_preference_model(dim, num_initial_points)
    initialize_objective_model(dim)
    
    recommendations = np.random.rand(3, dim)
    
    while len(recommendations) < 6:
        feedback = get_feedback()
        if feedback == 4:  # Uncertain option selected
            print("You selected uncertain. Here are the recommendations:")
            acqf = UpperConfidenceBound(preference_model, beta=0.1)  # Adjust beta as needed
            candidates, _ = optimize_acqf(
                acq_function=acqf,
                bounds=torch.tensor([[0.0] * dim, [1.0] * dim]),
                q=3,
                num_restarts=10,
                raw_samples=100,
            )
            next_recommendations_indices = candidates.squeeze().tolist()
            next_recommendations = [recommendations[i] for i in next_recommendations_indices]
        else:
            user_choice = feedback - 1
            preference_model.update(X=torch.tensor(recommendations[user_choice], dtype=torch.float), Y=torch.tensor([user_choice], dtype=torch.float))
            next_recommendations = recommend_three_choices(recommendations)
        recommendations = np.vstack((recommendations, next_recommendations))
    
    print("Final recommendations based on your preferences:")
    for rec in recommendations:
        print(f"Recommendation: {rec}")

if __name__ == "__main__":
    main()
