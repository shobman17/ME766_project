#include <vector>
#include <iostream>
#include <cmath>
#include "brain.h"
#include <omp.h>

//class Network {
//public:
//    std::vector<FCM*> neurons;
//    std::vector<Synapse*> synapses;
//    std::vector<double> input_currents;

    Network::Network() {
        // Constructor to initialize vectors
    }

    int Network::num_neurons() {
        return neurons.size();
    }

    int Network::add_neuron() {

        FCM* neuron = new FCM(0);
        neurons.push_back(neuron);
        neuron->set_index(neurons.size() - 1); // Index for easy access to input_current array when synaptic current has to be inserted
        input_currents.push_back(0.0); // Initial input is zero
        // std::cout << "Added neuron " << neuron->get_index() << std::endl;
        return neuron->get_index();
    }

    void Network::connect(int pre_index,
        int post_index,
        double weight,
        double delay,
        double tau,
        std::string type) {

        if (pre_index == post_index) {
            std::cout << "Select two unique neurons to connect" << std::endl;
            return;
        }
        FCM* pre_neuron = neurons[pre_index];
        FCM* post_neuron = neurons[post_index];
        Synapse* synapse = new Synapse(pre_neuron, post_neuron, weight, delay, tau, type);
        synapses.push_back(synapse);
        // std::cout << "Connected pre-neuron " << pre_index << " to post neuron " << post_index << " and type = " << synapse->get_type() << std::endl;
    }

    std::vector<double> Network::step(std::vector<double>& external_current, 
        double t, 
        double dt) {

        std::vector<double> Vm(num_neurons(), 0.0);

        // Update all neurons
        #pragma omp parallel for
        for (size_t n = 0; n < neurons.size(); n++) {
            FCM* neuron = neurons[n];
            neuron->step(input_currents[n] + external_current[n], dt);
            Vm[neuron->get_index()] = neuron->get_Vm();
        }

        // Set input currents for next step = 0
        std::fill(input_currents.begin(), input_currents.end(), 0.0);

        // Update all synapses and set synaptic currents for next time step
        #pragma omp parallel for
        for (auto& synapse : synapses) {
            synapse->step(t, dt);
            input_currents[synapse->get_post_neuron()->get_index()] += synapse->get_current() * 1e-6;
        }

        return Vm;
    }

