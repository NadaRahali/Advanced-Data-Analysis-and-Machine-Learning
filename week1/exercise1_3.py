"""
Exercise 1.3: Gradient Descent

This exercise implements basic gradient descent optimization.
"""

import numpy as np


def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    """
    Perform gradient descent to find optimal parameters.
    
    Args:
        X: Input features (m x n matrix)
        y: Target values (m x 1 vector)
        learning_rate: Learning rate for gradient descent
        n_iterations: Number of iterations
        
    Returns:
        theta: Optimized parameters
        cost_history: History of cost values
    """
    m, n = X.shape
    theta = np.zeros((n, 1))
    cost_history = []
    
    for i in range(n_iterations):
        # Compute predictions
        predictions = X.dot(theta)
        
        # Compute error
        error = predictions - y
        
        # Compute cost (MSE)
        cost = (1 / (2 * m)) * np.sum(error**2)
        cost_history.append(cost)
        
        # Compute gradient
        gradient = (1 / m) * X.T.dot(error)
        
        # Update parameters
        theta = theta - learning_rate * gradient
        
        # Print progress every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost:.4f}")
    
    return theta, cost_history


def compute_cost(X, y, theta):
    """
    Compute the cost function (MSE).
    
    Args:
        X: Input features
        y: Target values
        theta: Parameters
        
    Returns:
        cost: Mean squared error
    """
    m = len(y)
    predictions = X.dot(theta)
    error = predictions - y
    cost = (1 / (2 * m)) * np.sum(error**2)
    return cost


if __name__ == "__main__":
    print("Exercise 1.3: Gradient Descent")
    print("-" * 50)
    
    # Generate sample data
    np.random.seed(42)
    m = 100  # number of samples
    X = 2 * np.random.rand(m, 1)
    y = 4 + 3 * X + np.random.randn(m, 1)
    
    # Add bias term
    X_b = np.c_[np.ones((m, 1)), X]
    
    print(f"Dataset: {m} samples")
    print(f"Features shape: {X_b.shape}")
    print(f"Target shape: {y.shape}")
    
    # Perform gradient descent
    print("\nTraining with Gradient Descent...")
    theta, cost_history = gradient_descent(X_b, y, learning_rate=0.1, n_iterations=1000)
    
    print(f"\nOptimized parameters:")
    print(f"  Intercept (theta_0): {theta[0][0]:.4f}")
    print(f"  Slope (theta_1): {theta[1][0]:.4f}")
    print(f"Final cost: {cost_history[-1]:.4f}")
