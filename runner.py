from network import Network 
import sys
import numpy as np
import time

if __name__ == "__main__":
    n = 48
    if (len(sys.argv) > 1):
        n = int(sys.argv[1])
    network = Network()
    indexes = []

    # add neurons
    for i in range(n):
        idx = network.add_neuron()
        indexes.append(idx)
    
    # make connections with the following rule: 
    # any neuron n will be excitatory to n*2, n*3, etc if even
    # or inhibitory if odd
    # Technically, the bipolar cells are supposed to be what does "image processing" before the signals go to the brain
    # Thus their connection will represent an artificial neural net. We do not need to actually train it, just obtain speedup in calculations

    for i in range(1, network.num_neurons()):
        neuron = network.neurons[i]
        idx = neuron.index
        for j in range(2*i, network.num_neurons(), i):
            post = network.neurons[j]
            if idx % 2 == 0:
                network.connect(idx, post.index, weight = idx, delay = 1, tau = 5, syn_type="excitatory")
            else: 
                network.connect(idx, post.index, weight = idx, delay = 1, tau = 5, syn_type="inhibitory")

    print("total neurons:", network.num_neurons())
    print("total synapses:", len(network.synapses))

    # Simulation parameters
    T = 350  # Total simulation time (ms)
    dt = 0.01  # Time step (ms)
    time_arr = np.arange(0, T+dt, dt)
    pulse_end = 200  # Duration of current pulse (ms)
    
    # Initialize arrays to store results
    Vm_array = np.zeros([network.num_neurons(), len(time_arr)])
    for i in range(network.num_neurons()):
        Vm_array[i,0] = network.neurons[i].V_rest
    
    # calculate input current
    freq = np.ones([network.num_neurons(), 1])*10 # Input current frequency (Hz)
    curr_amp = np.zeros([network.num_neurons(), 1]) # Input current amplitude (uA)
    curr_amp_max = 30
    for i in range(network.num_neurons()):
        curr_amp[i] = curr_amp_max*(1 - (network.neurons[i].index/network.num_neurons()))
    external_current = np.zeros([network.num_neurons(), len(time_arr)])
    curr_input = np.zeros([network.num_neurons()])

    for i in range(len(time_arr)):
        if freq is not None:
            for j in range(network.num_neurons()):
                curr_input[j] = curr_amp[j]* np.sin(2*np.pi*freq[j]*time_arr[i]*1e-3)
        else:
            curr_input = curr_amp if (2 <= time_arr[i] <= pulse_end) else np.zeros([network.num_neurons(), 1])
        external_current[:,i] = curr_input

    print("External currents set")

    # Run simulation
    start = time.time()
    for i in range(len(time_arr)-1):

        Vm = network.step(external_current[:,i]*1e-6, t = time_arr[i], dt = dt)
        Vm_array[:,i+1] = Vm.flatten()
    end = time.time()
    print("Time taken:", (end - start))



