import numpy as np
from FCM_intra import FCM
from synapse import Synapse
import matplotlib.pyplot as plt

class Network:
    def __init__(self):
        self.neurons = []
        self.synapses = []
        # To keep track of synaptic and external currents
        self.input_currents = []

    def num_neurons(self):
        return len(self.neurons)

    def add_neuron(self):
        neuron = FCM()
        self.neurons.append(neuron)
        neuron.set_index(len(self.neurons) - 1) # index for easy access to input_current array when synaptic current has to be inserted
        self.input_currents.append(0) # initial input is just zero
        print("Added neuron", neuron.index)
        return neuron.index

    def connect(self, pre_index, post_index, weight=0.1, delay=1.0, tau=5.0, syn_type="excitatory"):
        if pre_index == post_index:
            print("Select two unique neurons to connect")
            return 
        pre_neuron = self.neurons[pre_index]
        post_neuron = self.neurons[post_index]
        synapse = Synapse(pre_neuron, post_neuron, weight, delay, tau, syn_type)
        self.synapses.append(synapse)
        print("Connected pre-neuron", pre_index, "to post neuron", post_index, "and type =", synapse.syn_type)

    def step(self, external_current, t, dt):
        """
        Simulate the network.
        
        Args:
        - external_current: array with external input current for neurons (given in terms of uA)
        - t: Time now (ms)
        - dt: Time step (ms).
        """

        Vm = np.zeros([self.num_neurons(), 1])

        # Update all neurons
        for n in range(len(self.neurons)):
            neuron = self.neurons[n]
            neuron.step(curr_input = self.input_currents[n] + external_current[n], dt = dt)
            Vm[neuron.index] = neuron.Vm[0][0]
        
        # Set input currents for next step = 0
        for i in range(len(self.input_currents)):
            self.input_currents[i] = 0

        # Update all synapses and set synaptic currents for next time step
        for synapse in self.synapses:
            synapse.step(t = t, dt = dt)
            self.input_currents[synapse.post_neuron.index] += synapse.get_current()*1e-6
        
        return Vm

if __name__ == "__main__":
    # Create a network
    network = Network()

    # Add neurons to the network
    n1_index = network.add_neuron()
    n2_index = network.add_neuron()
    n3_index = network.add_neuron()

    # Connect neurons with synapses
    network.connect(n1_index, n2_index, weight=20, delay=1.0, tau=5, syn_type="excitatory")
    network.connect(n1_index, n3_index, weight = 20, delay = 1.0, tau = 5, syn_type = "inhibitory")

    # Simulation parameters
    T = 350  # Total simulation time (ms)
    dt = 0.01  # Time step (ms)
    time = np.arange(0, T+dt, dt)
    pulse_end = 200  # Duration of current pulse (ms)
    
    # Initialize arrays to store results
    Vm_array = np.zeros([network.num_neurons(), len(time)])
    for i in range(network.num_neurons()):
        Vm_array[i,0] = network.neurons[i].V_rest
    
    # calculate input current
    #freq = np.random.randint(low = -20, high = 20, size = network.num_neurons())
    freq = np.ones([network.num_neurons(), 1])*10 # Input current frequency (Hz)
    curr_amp = np.zeros([network.num_neurons(), 1]) # Input current amplitude (uA)
    curr_amp[0] = 20
    curr_amp[2] = 23
    external_current = np.zeros([network.num_neurons(), len(time)])
    curr_input = np.zeros([network.num_neurons()])

    for i in range(len(time)):
        if freq is not None:
            for j in range(network.num_neurons()):
                curr_input[j] = curr_amp[j]* np.sin(2*np.pi*freq[j]*time[i]*1e-3)
        else:
            curr_input = curr_amp if (2 <= time[i] <= pulse_end) else np.zeros([network.num_neurons(), 1])

    
        external_current[:,i] = curr_input

    print("External currents set")
    #print(external_current.shape)
    #plt.plot(time, external_current[0])
    #plt.show()

    # Run simulation
    for i in range(len(time)-1):

        Vm = network.step(external_current[:,i]*1e-6, t = time[i], dt = dt)
        Vm_array[:,i+1] = Vm.flatten()

        
    fig, axe = plt.subplots(network.num_neurons(), 1, figsize=(15,10), sharex=True)
    
    total_steps = int(T/dt)
    time = time[:total_steps]

    for i in range(network.num_neurons()):
        axe[i].plot(time, Vm_array[i][:total_steps])
        axe[i].grid()
        axe[i].set_xlabel("Time (ms)")
        axe[i].set_ylabel(f"Membrane potential of neuron {i} in mV")

    plt.tight_layout()
    plt.show()