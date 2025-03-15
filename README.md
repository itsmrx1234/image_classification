Neural Network from Scratch: Multi-Layer Perceptron (MLP) Implementation

This project implements a fully connected neural network from scratch using NumPy. The model is designed to handle multi-class classification using the softmax activation in the output layer and cross-entropy loss as the cost function.
Features

✅ Customizable Network Architecture – Allows tuning of hidden layer size (n_h).
✅ Forward Propagation – Implements matrix operations for weight multiplication and activation functions.
✅ Activation Functions – Uses ReLU for hidden layers and Softmax for output layer.
✅ Cross-Entropy Loss – Measures how well predictions match actual labels.
✅ Gradient Descent Optimization – Updates weights and biases using backpropagation.
✅ Batch Processing – Handles multiple examples at once for efficient computation.
Model Architecture

    Input Layer: Accepts training data (X_train) of shape (n_features, m).
    Hidden Layer(s): Fully connected layer(s) using ReLU activation.
    Output Layer: Uses Softmax activation to output probabilities for multi-class classification.
    Loss Function: Uses cross-entropy loss to compute model performance.

Code Overview
1️⃣ Forward Propagation

def forward_propagation(X, parameters):
    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]

    z1 = np.dot(w1, X) + b1
    a1 = relu(z1)  # Activation function
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2)  # Output probabilities
    return a2

2️⃣ Cost Function (Cross-Entropy Loss)

def cost_function(a2, Y):
    m = Y.shape[1]  # Number of examples
    cost = -np.sum(np.multiply(Y, np.log(a2))) / m  # Averaged loss
    return cost

3️⃣ Training the Model

parameters = model(X_train, Y_train, n_h=1000, learning_rate=0.005, iterations=1000)

How to Run

    Clone the repository:

    git clone https://github.com/your-username/your-repo-name.git

    Install dependencies (NumPy required).
    Prepare your dataset (train_X.csv, train_label.csv, test_X.csv, test_label.csv).
    Run the training script in a Python environment.

Future Enhancements

    Implement Adam Optimizer for better convergence.
    Add batch normalization to stabilize training.
    Extend to deep neural networks with multiple hidden layers.
