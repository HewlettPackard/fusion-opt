import numpy as np
import torch
from botorch.models.pairwise import PairwiseGP
from botorch.models import SingleTaskGP
from botorch.acquisition import qExpectedImprovement, UpperConfidenceBound
from botorch.acquisition import MaxValueEntropySearch
from botorch.optim import optimize_acqf

# Global variables for preference model and objective model
preference_model = None
objective_model = None




def recommend_three_choices_mes(recommendations):
    global preference_model, objective_model
    qEI_acqf = qExpectedImprovement(objective_model)
    MES_acqf = MaxValueEntropySearch(objective_model, num_samples=128)  # Adjust num_samples as needed
    candidates_qEI, _ = optimize_acqf(
        acq_function=qEI_acqf,
        bounds=torch.tensor([[0.0] * objective_model.dim, [1.0] * objective_model.dim]),
        q=3,
        num_restarts=10,
        raw_samples=100,
    )
    candidates_MES, _ = optimize_acqf(
        acq_function=MES_acqf,
        bounds=torch.tensor([[0.0] * objective_model.dim, [1.0] * objective_model.dim]),
        q=3,
        num_restarts=10,
        raw_samples=100,
    )
    next_recommendations_indices = torch.cat((candidates_qEI, candidates_MES), dim=0).squeeze().tolist()
    next_recommendations = [recommendations[i] for i in next_recommendations_indices]
    return next_recommendations

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
            acqf = qExpectedImprovement(objective_model)
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
