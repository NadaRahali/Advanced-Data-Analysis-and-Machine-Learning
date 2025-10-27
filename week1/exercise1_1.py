"""
Exercise 1.1: Basic Neural Network Implementation

This exercise implements a simple neural network from scratch using NumPy.
"""

import numpy as np


class SimpleNeuralNetwork:
    """
    A simple neural network with one hidden layer.
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the neural network with random weights.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of neurons in hidden layer
            output_size: Number of output neurons
        """
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def forward(self, X):
        """
        Forward propagation through the network.
        
        Args:
            X: Input data
            
        Returns:
            Output of the network
        """
        # Hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-z))


if __name__ == "__main__":
    # Example usage
    print("Exercise 1.1: Basic Neural Network Implementation")
    print("-" * 50)
    
    # Create a simple dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])  # XOR problem
    
    # Initialize network
    nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    
    # Forward pass
    output = nn.forward(X)
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Initial output (before training):\n{output}")
