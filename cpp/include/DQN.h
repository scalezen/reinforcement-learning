#include <torch/torch.h>

// A simple Multi-Layer Perceptron (MLP) for our GridWorld
struct DQN : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    DQN(int input_size, int hidden_size, int num_actions) {
        // Construct and register the layers
        fc1 = register_module("fc1", torch::nn::Linear(input_size, hidden_size));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_size, hidden_size));
        fc3 = register_module("fc3", torch::nn::Linear(hidden_size, num_actions));
    }

    // Forward pass: State tensor in, Q-values tensor out
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        return fc3->forward(x); // No activation on the final layer; Q-values are unbounded
    }
};
