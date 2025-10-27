"""
Exercise 1.2: Activation Functions

This exercise explores different activation functions and their properties.
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """Derivative of sigmoid function."""
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    """Tanh activation function."""
    return np.tanh(x)


def tanh_derivative(x):
    """Derivative of tanh function."""
    return 1 - np.tanh(x)**2


def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)


def relu_derivative(x):
    """Derivative of ReLU function."""
    return (x > 0).astype(float)


def leaky_relu(x, alpha=0.01):
    """Leaky ReLU activation function."""
    return np.where(x > 0, x, alpha * x)


def leaky_relu_derivative(x, alpha=0.01):
    """Derivative of Leaky ReLU function."""
    return np.where(x > 0, 1, alpha)


if __name__ == "__main__":
    print("Exercise 1.2: Activation Functions")
    print("-" * 50)
    
    # Test the activation functions
    x = np.linspace(-5, 5, 100)
    
    print(f"Testing activation functions on range [{x.min()}, {x.max()}]")
    print(f"\nSample values at x=0:")
    print(f"  Sigmoid(0) = {sigmoid(0):.4f}")
    print(f"  Tanh(0) = {tanh(0):.4f}")
    print(f"  ReLU(0) = {relu(0):.4f}")
    
    print(f"\nSample values at x=2:")
    print(f"  Sigmoid(2) = {sigmoid(2):.4f}")
    print(f"  Tanh(2) = {tanh(2):.4f}")
    print(f"  ReLU(2) = {relu(2):.4f}")
    
    print(f"\nSample values at x=-2:")
    print(f"  Sigmoid(-2) = {sigmoid(-2):.4f}")
    print(f"  Tanh(-2) = {tanh(-2):.4f}")
    print(f"  ReLU(-2) = {relu(-2):.4f}")
