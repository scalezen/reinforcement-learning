#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <random>
#include <tuple>

#include "include/DQN.h"
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
    std::cout << "Training on: " << (torch::hasMPS() ? "MPS (Apple Silicon GPU)" : "CPU") << "\n\n";

    // Initialize Networks
    auto main_network = std::make_shared<DQN>(16, 64, 4);
    auto target_network = std::make_shared<DQN>(16, 64, 4);
    main_network->to(device);
    target_network->to(device);

    // Optimizer & Hyperparameters
    torch::optim::Adam optimizer(main_network->parameters(), /*lr=*/1e-3);
    ReplayBuffer buffer(10000);
    float gamma = 0.99f;
    size_t batch_size = 32;
    int sync_target_every = 500;
    int total_steps = 0;

    // ---------------------------------------------------------
    // PHASE 1: BURN-IN (Warm up the buffer with real data)
    // ---------------------------------------------------------
    std::cout << "--- Starting Burn-In Phase ---" << std::endl;
    int current_state_idx = 0;
    
    for (int i = 0; i < 1000; i++) {
        int action = rand() % 4; // Pure random exploration
        auto [next_state_idx, reward, done] = env_step(current_state_idx, action);
        
        buffer.push(encode_state(current_state_idx).to(device), action, reward, encode_state(next_state_idx).to(device), done);
        
        current_state_idx = done ? 0 : next_state_idx;
    }
    std::cout << "Burn-in complete. Buffer size: " << buffer.size() << "\n\n";

    // ---------------------------------------------------------
    // PHASE 2: TRAINING LOOP
    // ---------------------------------------------------------
    std::cout << "--- Starting Training Phase ---" << std::endl;
    current_state_idx = 0;

    // Train for 2000 environment steps as a demonstration
    for (int step = 0; step < 2000; step++) {
        torch::Tensor current_state = encode_state(current_state_idx).to(device);
        int action = 0;

        // Epsilon-Greedy Action Selection (10% exploration)
        if ((rand() % 100) < 10) {
            action = rand() % 4;
        } else {
            torch::NoGradGuard no_grad;
            torch::Tensor q_values = main_network->forward(current_state.unsqueeze(0));
            action = torch::argmax(q_values, 1).item<int>();
        }

        // Interact & Store
        auto [next_state_idx, reward, done] = env_step(current_state_idx, action);
        buffer.push(current_state, action, reward, encode_state(next_state_idx).to(device), done);
        
        current_state_idx = done ? 0 : next_state_idx;

        // --- SAMPLE & COLLATE ---
        auto batch = buffer.sample(batch_size);
        std::vector<torch::Tensor> states_vec, next_states_vec, actions_vec, rewards_vec, dones_vec;
        
        for (const auto& t : batch) {
            states_vec.push_back(t.state);
            next_states_vec.push_back(t.next_state);
            actions_vec.push_back(torch::tensor(t.action, torch::kInt64));
            rewards_vec.push_back(torch::tensor(t.reward, torch::kFloat32));
            dones_vec.push_back(torch::tensor(t.done ? 1.0f : 0.0f, torch::kFloat32)); 
        }
        
        torch::Tensor batched_states = torch::stack(states_vec).to(device);
        torch::Tensor batched_next_states = torch::stack(next_states_vec).to(device);
        torch::Tensor batched_actions = torch::stack(actions_vec).to(device);
        torch::Tensor batched_rewards = torch::stack(rewards_vec).to(device);
        torch::Tensor batched_dones = torch::stack(dones_vec).to(device);

        // --- THE BELLMAN OPTIMIZATION STEP ---
        // 1. Current Q-Values
        torch::Tensor q_values = main_network->forward(batched_states);
        torch::Tensor current_q = q_values.gather(1, batched_actions.unsqueeze(1)).squeeze(1);

        // 2. Target Q-Values (from Target Network)
        torch::Tensor max_next_q;
        {
            torch::NoGradGuard no_grad; 
            torch::Tensor next_q_values = target_network->forward(batched_next_states);
            max_next_q = std::get<0>(torch::max(next_q_values, 1));
        }
        torch::Tensor target_q = batched_rewards + (gamma * max_next_q * (1.0f - batched_dones));

        // 3. Backpropagate Loss
        torch::Tensor loss = torch::mse_loss(current_q, target_q);
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        // --- TARGET NETWORK SYNC ---
        total_steps++;
        if (total_steps % sync_target_every == 0) {
            torch::autograd::GradMode::set_enabled(false);
            auto new_params = main_network->named_parameters();
            auto target_params = target_network->named_parameters(true);
            
            for (auto& val : new_params) {
                auto name = val.key();
                auto* t = target_params.find(name);
                if (t != nullptr) t->copy_(val.value());
            }
            torch::autograd::GradMode::set_enabled(true);
            std::cout << "[Step " << total_steps << "] Target Network Synced. Recent Loss: " << loss.item<float>() << std::endl;
        }
    }
    
    std::cout << "\nTraining Complete!" << std::endl;

    return 0;
}
