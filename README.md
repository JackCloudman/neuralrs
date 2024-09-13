# Simple Neural Network in Rust

This project is a simple implementation of a neural network in Rust without using external libraries. The neural network is designed to solve the XOR problem using the backpropagation algorithm.

## Project Structure

- `main.rs`: Contains the entire implementation of the neural network, including forward propagation, backpropagation, and training loops.

## How the Neural Network Works

- **Architecture**: The network consists of:
  - Input layer: 2 nodes (for XOR inputs)
  - Hidden layer: 2 nodes (with sigmoid activation)
  - Output layer: 1 node (with sigmoid activation)

- **Training**: The network is trained using the backpropagation algorithm, minimizing the mean squared error (MSE) loss over time.

- **Activation Function**: The sigmoid function is used for the hidden and output layers. The derivative of the sigmoid function is used during the backward pass to calculate gradients.

## Getting Started

### Prerequisites

You need to have Rust installed on your system. You can install it by following the instructions on the official Rust website:

- [Install Rust](https://www.rust-lang.org/tools/install)

### Running the Code

1. Clone or download this repository.
2. Navigate to the project directory.
3. Run the following command to execute the neural network:

```bash
cargo run
```

The network will train on the XOR problem for 10,000 epochs. You will see the error printed every 1000 epochs, as well as the predictions after training.

### Expected Output

The network should correctly learn to classify the XOR problem. Below is an example of expected output:

```
Epoch 0: Error = 1.2342359039160524
Epoch 1000: Error = 0.7871489298170318
Epoch 2000: Error = 0.020227134155769183
Epoch 3000: Error = 0.00760094718304348
...
Epoch 9000: Error = 0.0014517111483755194

Results after training:
Input: [0.0, 0.0], Prediction: 0.0161, Target: 0
Input: [0.0, 1.0], Prediction: 0.9830, Target: 1
Input: [1.0, 0.0], Prediction: 0.9830, Target: 1
Input: [1.0, 1.0], Prediction: 0.0209, Target: 0
```

### Modifying the Code

You can experiment with the following parameters:
- **Learning rate**: You can change the learning rate to see how it affects the training speed and accuracy.
- **Epochs**: Adjust the number of training epochs to observe how the error decreases over time.
- **Network architecture**: Add more neurons or layers to see how it impacts the network's ability to solve the XOR problem or extend it to more complex problems.

## License

This project is licensed under the GNU General Public License v3.0. You can read the full license here: [GPL 3.0 License](https://www.gnu.org/licenses/gpl-3.0.en.html).
