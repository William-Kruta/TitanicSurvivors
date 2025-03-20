import numpy as np


def one_hot_encode(y, num_classes: int):
    """
    Convert integer labels to one-hot encoded vectors.

    Parameters
    ----------
    y : np.ndarray
        Array of integer labels.
    num_classes : int
        Number of classes to classify.
    """
    return np.eye(num_classes)[y]


def create_label_mapping(y):
    unique_labels = np.unique(y)
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    y_int = np.array([label_to_int[label] for label in y])
    return y_int


def relu(z):
    """ReLU activation function.

    z : np.ndarray
        Input array to apply ReLU activation function to.

    Returns
    -------
    z : np.ndarray
        If z > 0, return z. If z < 0, return 0.
    """
    return np.maximum(0, z)


def relu_derivative(z):
    """Derivative of the ReLU activation function."""
    return (z > 0).astype(float)


def sigmoid(z):
    """
    Sigmoid activation function.

    Parameters
    ----------
    z : np.ndarray
        Input data to apply sigmoid activation function to.

    Returns
    -------
    z : np.ndarray
        Scaled data using the sigmoid function.
    """
    return 1 / (1 + np.exp(-z))


def min_max_scaler(X, feature_range: tuple = (0, 1)):
    """
    Custom Min-Max Scaler to scale values between a given range (default: 0 to 1).

    Parameters:
        X (numpy array): Input array to scale.
        feature_range (tuple): Desired range for scaled data (default is (0,1)).

    Returns:
        Scaled array (numpy array).
    """
    X_min = X.min(axis=0)  # Find the min value for each feature
    X_max = X.max(axis=0)  # Find the max value for each feature

    # Scale data
    X_scaled = (X - X_min) / (X_max - X_min)

    # If a custom range is specified (e.g., (a, b)), rescale:
    a, b = feature_range
    X_scaled = X_scaled * (b - a) + a  # Rescale to new range

    return X_scaled
