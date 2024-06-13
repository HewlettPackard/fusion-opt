import torch
from torch import nn
from torch.distributions import Normal

class StdBNN(nn.Module):
  def __init__(self, input_dim, hidden_dim):
    super(SimpleBNN, self).__init__()
    self.dense1 = nn.Linear(input_dim, hidden_dim)
    self.dense2_mu = nn.Linear(hidden_dim, 1)  # Mean of output
    self.dense2_logvar = nn.Linear(hidden_dim, 1)  # Log variance for std dev

  def forward(self, x):
    x = self.dense1(x)
    x = F.relu(x)  # ReLU activation
    mu = self.dense2_mu(x)
    logvar = self.dense2_logvar(x)
    # Reparameterization trick for sampling from the distribution during training
    std = torch.sqrt(torch.exp(logvar))
    epsilon = torch.randn_like(std)
    return mu + epsilon * std

def build_and_train_bnn(x_train, sigma_m2_train, epochs=100):
  """
  Builds and trains a simple BNN for predicting standard deviation.

  Args:
      x_train: Training data (features) (tensor).
      sigma_m2_train: Training labels (true standard deviation of model M2) (tensor).
      epochs: Number of training epochs (default=100).

  Returns:
      Trained BNN model.
  """
  bnn = StdBNN(x_train.shape[1], 32)  # Adjust hidden layer size as needed
  optimizer = torch.optim.Adam(bnn.parameters(), lr=0.001)

  criterion = nn.GaussianNLLLoss()  # Gaussian Negative Log Likelihood

  for epoch in range(epochs):
    # Forward pass
    y_pred = bnn(x_train)
    loss = criterion(y_pred, sigma_m2_train.unsqueeze(1))  # Reshape for loss

    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  return bnn

# Example usage (assuming you have training data)
bnn = build_and_train_bnn(x_train, sigma_m2_train)
# Use the trained bnn(x) to get predicted mean and std dev for new data x
