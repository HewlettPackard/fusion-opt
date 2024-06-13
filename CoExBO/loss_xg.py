import torch    
# Define custom loss function for PyTorch (mimicking XGBoost objective)
def custom_loss(y_true, y_pred):
    """Custom objective function for XGBoost preference learning (PyTorch version).

    Args:
        y_true: True preference labels (NumPy array).
        y_pred: Predicted preference probabilities (NumPy array).

    Returns:
        tuple: (loss, grad)
            - loss: Average negative log-likelihood loss (Tensor).
            - grad: Gradient of the loss with respect to predictions (Tensor).
    """
    
    # Create DMatrices for XGBoost
    
    
    
    # Convert y_true and y_pred to tensors (assuming they are NumPy arrays)
    y_true_tensor = torch.from_numpy(y_true).float().requires_grad_(True)
    y_pred_tensor = torch.clamp(torch.from_numpy(y_pred).float().requires_grad_(True), 1e-8, 1 - 1e-8)  # Clamp for numerical stability

    # Calculate loss (using torch.log for PyTorch compatibility)
    loss = -(y_true_tensor * torch.log(y_pred_tensor) + (1 - y_true_tensor) * torch.log(1 - y_pred_tensor))

    # Calculate average loss for scalar output
    avg_loss = loss.mean()

    # Calculate gradient using autograd
    avg_loss.backward()
    grad = y_pred_tensor.grad.clone().detach()  # Detach gradient for further computations

    # Clear gradients for memory efficiency (optional)
    y_pred_tensor.grad.zero_()

    return avg_loss, grad  # Return average loss and gradient