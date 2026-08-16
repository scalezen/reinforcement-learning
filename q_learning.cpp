#pragma once
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <random>
#include <tuple>

#include "include/dqn.h"
#include "include/ReplayBuffer.h"

// --- Keep the DQN and ReplayBuffer structs from the previous code here ---

// --- Environment Setup ---
const int GOAL_STATE = 15;
const int TRAP_STATE = 5;

// Neural Networks need tensors, not integers. We one-hot encode the state.
torch::Tensor encode_state(int state_idx) {
    torch::Tensor t = torch::zeros({16});
    t[state_idx] = 1.0; // Set the current position to 1
    return t;
}

// The Environment logic
std::tuple<int, float, bool> env_step(int state, int action) {
    int row = state / 4;
    int col = state % 4;
    
    // 0=Up, 1=Right, 2=Down, 3=Left
    if (action == 0 && row > 0) row--;
    else if (action == 1 && col < 3) col++;
    else if (action == 2 && row < 3) row++;
    else if (action == 3 && col > 0) col--;
    
    int next_state = row * 4 + col;
    float reward = -1.0f; // Step penalty
    bool done = false;
    
    if (next_state == GOAL_STATE) { reward = 100.0f; done = true; }
    else if (next_state == TRAP_STATE) { reward = -100.0f; done = true; }
    
    return {next_state, reward, done};
}

int main() {
    torch::Device device(torch::hasMPS() ? torch::kMPS : torch::kCPU);
    ReplayBuffer buffer(1000);
    
    // 1. Initial State
    int current_state_idx = 0; // Start at (0,0)
    torch::Tensor current_state = encode_state(current_state_idx).to(device);
    
    // Agent chooses an action (hardcoded to '1' (Right) for this example)
    int action = 1;
    
    // ==========================================
    // STEP 2: Step (Interact with Environment)
    // ==========================================
    auto [next_state_idx, reward, done] = env_step(current_state_idx, action);
    torch::Tensor next_state = encode_state(next_state_idx).to(device);
    
    std::cout << "Action taken: " << action << " | Reward: " << reward << " | Done: " << done << std::endl;

    // ==========================================
    // STEP 3: Store (Save to Replay Buffer)
    // ==========================================
    buffer.push(current_state, action, reward, next_state, done);
    
    // Fast-forward: Let's fill the buffer with some random transitions so we have enough to sample
    for(int i = 0; i < 50; i++) {
        buffer.push(encode_state(i % 16).to(device), i % 4, 1.0f, encode_state((i + 1) % 16).to(device), false);
    }

    // ==========================================
    // STEP 4: Sample & Collate
    // ==========================================
    size_t batch_size = 32;
    std::vector<Transition> batch = buffer.sample(batch_size);
    
    // Create separate C++ vectors to hold the tensors
    std::vector<torch::Tensor> states_vec, next_states_vec, actions_vec, rewards_vec, dones_vec;
    
    for (const auto& t : batch) {
        states_vec.push_back(t.state);
        next_states_vec.push_back(t.next_state);
        
        // Convert primitive types to 1D tensors
        actions_vec.push_back(torch::tensor(t.action, torch::kInt64));
        rewards_vec.push_back(torch::tensor(t.reward, torch::kFloat32));
        
        // Convert boolean 'done' to a float (1.0 for true, 0.0 for false)
        // This is crucial because we will multiply by (1 - done) in the Bellman equation later
        dones_vec.push_back(torch::tensor(t.done ? 1.0f : 0.0f, torch::kFloat32)); 
    }
    
    // Stack the vectors into batched PyTorch tensors and move to GPU
    torch::Tensor batched_states = torch::stack(states_vec).to(device);
    torch::Tensor batched_next_states = torch::stack(next_states_vec).to(device);
    torch::Tensor batched_actions = torch::stack(actions_vec).to(device);
    torch::Tensor batched_rewards = torch::stack(rewards_vec).to(device);
    torch::Tensor batched_dones = torch::stack(dones_vec).to(device);
    
    // Output the tensor shapes to verify
    std::cout << "\n--- Batched Tensor Shapes ---" << std::endl;
    std::cout << "States shape:      " << batched_states.sizes() << std::endl;
    std::cout << "Actions shape:     " << batched_actions.sizes() << std::endl;
    std::cout << "Rewards shape:     " << batched_rewards.sizes() << std::endl;
    std::cout << "Next States shape: " << batched_next_states.sizes() << std::endl;
    std::cout << "Dones shape:       " << batched_dones.sizes() << std::endl;

    return 0;
}
