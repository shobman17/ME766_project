import numpy as np
from FCM_intra import FCM
import matplotlib.pyplot as plt

class Synapse:
    def __init__(self, pre_neuron, post_neuron, weight=0.1, delay=1.0, tau=5.0, syn_type="excitatory"):
        """
        Initialize the synapse.
        
        Args:
        - pre_neuron: The presynaptic neuron (an instance of the SCM class).
        - post_neuron: The postsynaptic neuron (an instance of the SCM class).
        - weight: Synaptic weight (strength of connection).
        - delay: Synapse relaxation delay in ms.
        - tau: Time constant for synaptic current decay in ms
        - syn_type: Type of synapse ('excitatory' or 'inhibitory').
        """
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.weight = weight
        self.delay = delay
        self.tau = tau
        self.syn_type = syn_type
        
        # Variables for synaptic dynamics
        self.syn_current = 0.0
        self.last_spike_time = -np.inf  # Keeps track of the last presynaptic spike

    def step(self, t, dt):
        """
        Update the synaptic current based on the presynaptic activity.
        
        Args:
        - t: Current simulation time (ms).
        - dt: Time step (ms).
        """
        # Check if the presynaptic neuron has spiked
        if self.pre_neuron.spiked() and (t - self.last_spike_time > self.delay):
            self.last_spike_time = t
            if self.syn_type == "excitatory":
                self.syn_current += self.weight
            elif self.syn_type == "inhibitory":
                self.syn_current -= self.weight

        # Decay the synaptic current
        self.syn_current *= np.exp(-dt / self.tau)

    def get_current(self):
        """
        Get the current generated by this synapse for the postsynaptic neuron.
        """
        return self.syn_current

if __name__ == "__main__":

    neuron1 = FCM()
    neuron2 = FCM()
    synapse = Synapse(neuron1, neuron2, weight = 20, delay = 1.0, tau = 5.0, syn_type="excitatory")
    
    # neuron1 will be given the input pulse

    # Simulation parameters
    T = 350  # Total simulation time (ms)
    dt = 0.01  # Time step (ms)
    time = np.arange(0, T+dt, dt)
    curr_amp = 20  # Input current amplitude (uA)
    pulse_end = 200  # Duration of current pulse (ms)
    
    # Initialize arrays to store results
    Vm1 = np.zeros([neuron1.num_rows-1, len(time)])
    Vm2 = np.zeros([neuron2.num_rows-1, len(time)])
    Vm1[:,0] = neuron1.Vm[:,0]  # Set initial voltage
    Vm2[:,0] = neuron2.Vm[:,0]  # Set initial voltage
    
    # initialize arrays for currents
    currents1 = {
        'i_na': [], 'i_ca_t': [], 'i_k_fast': [], 'i_k_slow': [],
        'i_leak': [], 'i_ca_l': [], 'i_hcn': []
    }
    currents2 = {
        'i_na': [], 'i_ca_t': [], 'i_k_fast': [], 'i_k_slow': [],
        'i_leak': [], 'i_ca_l': [], 'i_hcn': []
    }
    
    # calculate input current
    freq = 10.0
    input_current = []
    curr_input = 0

    for i in range(len(time)):
        if freq is not None:
            curr_input = curr_amp* np.sin(2*np.pi*freq*time[i]*1e-3)
        else:
            curr_input = curr_amp if (2 <= time[i] <= pulse_end) else 0
    
        input_current.append(curr_input) #value is stored in uA

    synaptic_current = []
    
    # Run simulation
    for i in range(len(time)-1):

        # neuron 1 gets input current
        step_currents1 = neuron1.step(input_current[i]*1e-6, dt = dt)
        # synapse gets updated
        synapse.step(t = time[i], dt = dt)
        # neuron2 gets updated
        step_currents2 = neuron2.step(synapse.get_current()*1e-6, dt = dt)
        synaptic_current.append(synapse.get_current())
        
        # Store currents and membrane voltages
        for key in currents1:
            currents1[key].append(step_currents1[key])
        for key in currents2:
            currents2[key].append(step_currents2[key])
        Vm1[:,i+1] = neuron1.Vm[:,0]
        Vm2[:,i+1] = neuron2.Vm[:,0]
        
    fig, axe = plt.subplots(3, 1, figsize=(10,10), sharex=True, constrained_layout = True)
    
    total_steps = int(T/dt)
    time = time[:total_steps]
    
    # axe[0].plot(time, currents["i_na"], label="I_Na")
    # axe[0].plot(time, currents["i_ca_t"], label="I_Ca_T")
    # axe[0].plot(time, currents["i_k_fast"], label="I_K_fast")
    # axe[0].plot(time, currents["i_k_slow"], label="I_K_slow")
    # axe[0].plot(time, currents["i_leak"], label="I_leakage")
    # axe[0].plot(time, currents["i_ca_l"], label="I_Ca_L")
    # axe[0].plot(time, currents["i_hcn"], label="I_HCN")
    axe[0].plot(time, input_current[:total_steps], label="Input current")
    axe[0].plot(time, synaptic_current[:total_steps], label="Synaptic current")
    # axe[0].plot(time, input_current[:total_steps], label="Input current")
    axe[0].legend()
    axe[0].grid()
    axe[0].set_ylabel("Current (uA)")
    
    axe[1].plot(time, Vm1[0][:total_steps])
    axe[1].grid()
    axe[1].set_xlabel("Time (ms)")
    axe[1].set_ylabel("Pre neuron potential (mV)")

    axe[2].plot(time, Vm2[0][:total_steps])
    axe[2].grid()
    axe[2].set_xlabel("Time (ms)")
    axe[2].set_ylabel(" Post neuron potential (mV)")
    
    # plt.tight_layout()
    plt.show()

