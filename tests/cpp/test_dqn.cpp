#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <algorithm>

#include "DQN.h"
#include "ReplayBuffer.h"


int main() {
    std::cout << "=== Testing Hardware ===" << std::endl;
    torch::Device device(torch::kCPU);
    if (torch::hasMPS()) {
        device = torch::Device(torch::kMPS);
        std::cout << "[PASS] Apple Silicon MPS detected. Using GPU." << std::endl;
    }

    std::cout << "\n=== Testing DQN Architecture ===" << std::endl;

    // 16 inputs (4x4 grid), 64 hidden nodes, 4 outputs (actions)
    auto model = std::make_shared<DQN>(16, 64, 4);
    model->to(device);
    
    // Simulate passing a single state through the network
    torch::Tensor dummy_state = torch::rand({1, 16}).to(device);
    torch::Tensor q_values = model->forward(dummy_state);
    
    std::cout << "Input State Shape:  " << dummy_state.sizes() << std::endl;
    std::cout << "Output Q-Values Shape: " << q_values.sizes() << std::endl;
    std::cout << "[PASS] Forward pass successful." << std::endl;

    std::cout << "\n=== Testing Replay Buffer ===" << std::endl;
    size_t max_capacity = 100;
    ReplayBuffer buffer(max_capacity);
    std::cout << "Initial buffer size: " << buffer.size() << std::endl;

    // Simulate pushing 150 transitions (50 over capacity) to test overwrite
    for (int i = 0; i < 150; ++i) {
        torch::Tensor s = torch::zeros({16}).to(device);
        torch::Tensor next_s = torch::ones({16}).to(device);
        buffer.push(s, i % 4, 1.0, next_s, false);
    }

    std::cout << "Buffer size after 150 pushes: " << buffer.size() << std::endl;
    if (buffer.size() == max_capacity) {
        std::cout << "[PASS] Circular overwrite successfully maintained capacity limit." << std::endl;
    }

    // Test sampling logic
    size_t batch_size = 32;
    auto batch = buffer.sample(batch_size);
    std::cout << "Sampled batch size: " << batch.size() << std::endl;
    if (batch.size() == batch_size) {
        std::cout << "[PASS] Correct batch size sampled." << std::endl;
    }

    return 0;
}
