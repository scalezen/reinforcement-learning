#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <iomanip>

#include "include/DQN.h"

using namespace std;

// Hyperparameters
const double ALPHA = 0.1;    // Learning Rate
const double GAMMA = 0.99;   // Discount Factor
const double EPSILON = 0.1;  // Exploration probability
const int EPISODES = 1000;

// Environment: 4x4 Grid
const int ROWS = 4;
const int COLS = 4;
const int NUM_STATES = ROWS * COLS;
const int NUM_ACTIONS = 4; // 0=Up, 1=Right, 2=Down, 3=Left

const int GOAL_STATE = 15; // Bottom-right corner
const int TRAP_STATE = 5;  // Somewhere in the middle

// Environment transition function
pair<int, double> step(int state, int action) {
    if (state == GOAL_STATE || state == TRAP_STATE) {
        return {state, 0.0}; // Terminal states
    }
    
    int row = state / COLS;
    int col = state % COLS;
    
    // Attempt move
    if (action == 0 && row > 0) row--;
    else if (action == 1 && col < COLS - 1) col++;
    else if (action == 2 && row < ROWS - 1) row++;
    else if (action == 3 && col > 0) col--;
    
    int next_state = row * COLS + col;
    double reward = -1.0; // Penalty for every step taken to encourage speed
    
    if (next_state == GOAL_STATE) reward = 100.0;
    else if (next_state == TRAP_STATE) reward = -100.0;
    
    return {next_state, reward};
}

int main() {
    // Initialize Q-Table to 0
    vector<vector<double>> q_table(NUM_STATES, vector<double>(NUM_ACTIONS, 0.0));
    
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);
    uniform_int_distribution<> act_dis(0, NUM_ACTIONS - 1);
    
    // Training Loop
    for (int ep = 0; ep < EPISODES; ++ep) {
        int state = 0; // Always start at (0,0)
        
        while (state != GOAL_STATE && state != TRAP_STATE) {
            int action = 0;
            
            // Epsilon-greedy action selection
            if (dis(gen) < EPSILON) {
                action = act_dis(gen); // Explore
            } else {
                // Exploit (choose best known action)
                action = distance(q_table[state].begin(), max_element(q_table[state].begin(), q_table[state].end()));
            }
            
            auto [next_state, reward] = step(state, action);
            
            // The Math: Bellman Q-Learning update rule
            double max_next_q = *max_element(q_table[next_state].begin(), q_table[next_state].end());
            q_table[state][action] = q_table[state][action] + ALPHA * (reward + GAMMA * max_next_q - q_table[state][action]);
            
            state = next_state;
        }
    }
    
    // Output the learned policy
    cout << "Learned Policy (U: Up, R: Right, D: Down, L: Left, G: Goal, T: Trap):\n\n";
    const char actions[] = {'U', 'R', 'D', 'L'};
    
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            int s = r * COLS + c;
            if (s == GOAL_STATE) cout << " G ";
            else if (s == TRAP_STATE) cout << " T ";
            else {
                int best_action = distance(q_table[s].begin(), max_element(q_table[s].begin(), q_table[s].end()));
                cout << " " << actions[best_action] << " ";
            }
        }
        cout << "\n";
    }
    return 0;
}
