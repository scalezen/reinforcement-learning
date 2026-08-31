#pragma once

#include <torch/torch.h>
//#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

struct Transition {
    torch::Tensor state, next_state;
    int action;
    float reward;
    bool done;
};

class ReplayBuffer {
private:
    std::vector<Transition> buffer;
    size_t capacity;
    size_t position;

public:
    ReplayBuffer(size_t cap) : capacity(cap), position(0) {}

    void push(torch::Tensor s, int a, float r, torch::Tensor next_s, bool d) {
        Transition t = {s, next_s, a, r, d};
        if (buffer.size() < capacity) {
            buffer.push_back(t);
        } else {
            // Overwrite oldest data if at capacity
            buffer[position] = t;
        }
        position = (position + 1) % capacity; 
    }

    std::vector<Transition> sample(size_t batch_size) {
        std::vector<Transition> batch;
        std::sample(buffer.begin(), buffer.end(), std::back_inserter(batch),
                    batch_size, std::mt19937{std::random_device{}()});
        return batch;
    }
    
    size_t size() { return buffer.size(); }
};
