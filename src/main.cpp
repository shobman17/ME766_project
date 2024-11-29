#include "brain.h" 
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>

int main(int argc, char* argv[]) {
    int n = 1000;
    if (argc > 1) {
        n = std::stoi(argv[1]);
    }
    int num_threads = 4;
    if (argc > 2) {
        num_threads = std::stoi(argv[2]);
    }
    omp_set_num_threads(num_threads);
    
    // Simulation parameters
    double T = 350;  // Total simulation time (ms)
    if (argc > 3) {
        T = std::stoi(argv[3]);
    }
    double dt = 0.01;  // Time step (ms)
    std::vector<double> time_arr;
    for (double t = 0; t <= T+dt; t += dt) {
        time_arr.push_back(t);
    }
    double pulse_end = 200;  // Duration of current pulse (ms)

    Network network;
    std::vector<int> indexes;

    // add neurons
    for (int i = 0; i < n; ++i) {
        int idx = network.add_neuron();
        indexes.push_back(idx);
    }
    
    // make connections with the following rule: 
    // any neuron n will be excitatory to n*2, n*3, etc if even
    // or inhibitory if odd
    for (int i = 1; i < network.num_neurons(); ++i) {
        FCM* neuron = network.neurons[i];
        int idx = neuron->get_index();
        for (int j = 2*i; j < network.num_neurons(); j += i) {
            FCM* post = network.neurons[j];
            if (idx % 2 == 0) {
                network.connect(idx, post->get_index(), idx, 1, 5, "excitatory");
            } else {
                network.connect(idx, post->get_index(), idx, 1, 5, "inhibitory");
            }
        }
    }

    std::cout << "total neurons: " << network.num_neurons() << std::endl;
    std::cout << "total synapses: " << network.synapses.size() << std::endl;
    std::cout << "simulation time (in ms): " << T << std::endl;
    std::cout << "number of threads: " << num_threads << std::endl;

    
    // Initialize arrays to store results
    std::vector<std::vector<double>> Vm_array(
        network.num_neurons(), 
        std::vector<double>(time_arr.size(), 0.0));
    for (int i = 0; i < network.num_neurons(); ++i) {
        Vm_array[i][0] = network.neurons[i]->get_V_rest();
    }
    
    // calculate input current
    std::vector<double> freq(network.num_neurons(), 10.0); // Input current frequency (Hz)
    std::vector<double> curr_amp(network.num_neurons(), 0.0); // Input current amplitude (uA)
    double curr_amp_max = 30.0;
    for (int i = 0; i < network.num_neurons(); ++i) {
        curr_amp[i] = curr_amp_max * (1.0 - (static_cast<double>(network.neurons[i]->get_index()) / network.num_neurons()));
    }
    
    std::vector<std::vector<double>> external_current(
        network.num_neurons(), 
        std::vector<double>(time_arr.size(), 0.0));
    std::vector<double> curr_input(network.num_neurons(), 0.0);

    for (size_t i = 0; i < time_arr.size(); ++i) {
        for (size_t j = 0; j < network.num_neurons(); ++j) {
            curr_input[j] = curr_amp[j] * std::sin(2 * M_PI * freq[j] * time_arr[i] * 1e-3);
        }
        
        for (size_t j = 0; j < network.num_neurons(); ++j) {
            external_current[j][i] = curr_input[j];
        }
    }

    std::cout << "External currents set" << std::endl;

    // Run simulation
    //auto start = std::chrono::high_resolution_clock::now();
    double start = omp_get_wtime();

    for (size_t i = 0; i < time_arr.size() - 1; ++i) {
        // Scale external current to microamps
        std::vector<double> scaled_current(network.num_neurons());
        for (size_t j = 0; j < network.num_neurons(); ++j) {
            scaled_current[j] = external_current[j][i] * 1e-6;
        }
        
        std::vector<double> Vm = network.step(scaled_current, time_arr[i], dt);
        
        for (size_t j = 0; j < network.num_neurons(); ++j) {
            Vm_array[j][i+1] = Vm[j];
        }
    }
    
    //auto end = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<double> diff = end - start;
    double end = omp_get_wtime();

    std::cout << "Time taken: " << (end - start) << " seconds" << std::endl;

    return 0;
}
