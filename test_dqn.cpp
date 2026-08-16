#include <torch/torch.h>
#include <iostream>

// 1. The Network Definition
struct DQN : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    DQN(int input_size, int hidden_size, int num_actions) {
        fc1 = register_module("fc1", torch::nn::Linear(input_size, hidden_size));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_size, hidden_size));
        fc3 = register_module("fc3", torch::nn::Linear(hidden_size, num_actions));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        return fc3->forward(x);
    }
};

int main() {
    // 2. Hardware Test: Check for Apple Silicon GPU (MPS)
    torch::Device device(torch::kCPU);
    if (torch::hasMPS()) {
        device = torch::Device(torch::kMPS);
        std::cout << "[SUCCESS] Apple Silicon MPS detected. Using GPU." << std::endl;
    } else {
        std::cout << "[WARNING] MPS not detected. Falling back to CPU." << std::endl;
    }

    // 3. Initialize the network using a shared pointer (standard for LibTorch)
    // 16 inputs (4x4 grid flattened), 64 hidden nodes, 4 outputs (Up, Right, Down, Left)
    auto model = std::make_shared<DQN>(16, 64, 4);
    model->to(device); // Move network to the GPU

    // 4. Create a dummy state tensor (Batch Size of 1, 16 features)
    // We use torch::rand to simulate arbitrary state data
    torch::Tensor dummy_state = torch::rand({1, 16}).to(device);
    
    // 5. Run the Forward Pass
    torch::Tensor q_values = model->forward(dummy_state);

    // 6. Verify Shapes and Outputs
    std::cout << "\n--- Network Architecture ---" << std::endl;
    std::cout << "Input State Shape:  " << dummy_state.sizes() << std::endl;
    std::cout << "Output Q-Values Shape: " << q_values.sizes() << std::endl;
    
    std::cout << "\n--- Predicted Q-Values ---" << std::endl;
    std::cout << q_values << std::endl;

    // Optional: Find the action the network would pick
    torch::Tensor best_action = torch::argmax(q_values, /*dim=*/1);
    std::cout << "Greedy Action Chosen: " << best_action.item<int>() << std::endl;

    return 0;
}
