import numpy as np


from model.utils import relu, relu_derivative, sigmoid


class Model:
    def __init__(self, input_size, hidden_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, 1) * 0.01
        self.b2 = np.zeros((1, 1))

    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        return (
            -np.sum(
                y_true * np.log(y_pred + 1e-8)
                + (1 - y_true) * np.log(1 - y_pred + 1e-8)
            )
            / m
        )

    def forward(self, X):
        self.Z1 = X.dot(self.W1) + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = self.A1.dot(self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)  # Output probability
        return self.A2

    def backward(self, X, y_true, learning_rate=0.01):
        m = X.shape[0]

        # Compute gradients
        dZ2 = self.A2 - y_true
        dW2 = self.A1.T.dot(dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = dZ2.dot(self.W2.T)
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = X.T.dot(dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update weights
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X, y, epochs=1000, learning_rate=0.01):
        for i in range(epochs + 1):
            y_pred = self.forward(X)
            loss = self.compute_loss(y_pred, y)
            self.backward(X, y, learning_rate)
            if i % 100 == 0:
                print(f"Epoch {i}, Loss: {loss:.4f}")

    def predict(self, X):
        """Make predictions (0 or 1)"""
        y_pred = self.forward(X)
        return (y_pred >= 0.5).astype(int)
