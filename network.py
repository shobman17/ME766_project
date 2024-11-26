import numpy as np
from FCM_intra import FCM
from synapse import Synapse

class Network:
    def __init__(self):
        self.neurons = []
        self.synapses = []
        # To keep track of synaptic and external currents
        self.input_currents = []

    def add_neuron(self):
        neuron = FCM()
        self.neurons.append(neuron)
        neuron.set_index(len(self.neurons) - 1) # index for easy access to input_current array when synaptic current has to be inserted
        input_currents.append(0) # initial input is just zero
        return neuron.index

    def connect(self, pre_index, post_index, weight=0.1, delay=1.0, tau=5.0, syn_type="excitatory"):
        if pre_index == post_index:
            print("Select two unique neurons to connect")
            return 
        pre_neuron = self.neurons[pre_index]
        post_neuron = self.neurons[post_index]
        synapse = Synapse(pre_neuron, post_neuron, weight, delay, tau, syn_type)
        self.synapses.append(synapse)

    def step(self, external_current, t, dt):
        """
        Simulate the network.
        
        Args:
        - external_current: array with external input current for neurons (given in terms of uA)
        - t: Time now (ms)
        - dt: Time step (ms).
        """

        # Update all neurons
        for n in range(len(self.neurons)):
            neuron = self.neurons[n]
            neuron.step(curr_input = self.input_currents[n][i], dt)
        
        # Update all synapses
        for synapse in self.synapses:
            synapse.step(t, dt)

        # Update input currents for next step


# Example usage
if __name__ == "__main__":
    # Create a network
    network = Network()

    # Add neurons to the network
    network.add_neuron(neuron1)
    network.add_neuron(neuron2)

    # Connect neurons with synapses
    network.connect(0, 1, weight=0.5, delay=1.0, tau=10.0, syn_type="excitatory")

    # Simulate the network
    network.run(sim_end=100, dt=0.1)

