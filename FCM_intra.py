import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class FCM:
    def __init__(self):
        self.V_rest = -70.0
        self.E_l = -81.4
        self.Cm = 1.0

        self.E_po = -77.0
        self.E_Na = 50.0
        self.E_ca = 120.0
        self.E_HC = -40.0
        self.E_ct = 22.6
        self.RA = 0.1
        self.ro_ex = 1.0

        self.K_m_Na = 2.2**((31.0 - 20.0)/10.0)
        self.K_h_Na = 2.9**((31.0 - 20.0)/10.0)
        self.K_s_Na = 2.9**((31.0 - 20.0)/10.0)
        self.K_m_Ca_T = 5.0**((31.0 - 24.0)/10.0)
        self.K_h_Ca_T = 3.0**((31.0 - 24.0)/10.0)
        self.K_n_K_fast = 2.0**((31.0 - 6.3)/10.0)
        self.K_n_K_slow = 2.0**((31.0 - 6.3)/10.0)
        self.K_c_Ca_L = 1.0**((31.0 - 10.0)/10.0)
        self.K_y_HCN = 1.0**((31.0 - 37.0)/10.0)

        data=np.loadtxt('morphology_FCM.swc')
        ID = data[:,0]
        type_compartment=data[:,1]
        x_cord=data[:,2]*1e-4
        y_cord=data[:,3]*1e-4
        z_cord=data[:,4]*1e-4
        radius=data[:,5]*1e-4
        partner = data[:,6]

        num_rows=len(ID)
        Lenghts=np.zeros(num_rows)
        for i in range(num_rows):
            if partner[i]==-1:
                l=0
            else:
                l=np.sqrt((x_cord[int(ID[i]-1)]-x_cord[int(partner[i]-1)])**2 +\
                    (y_cord[int(ID[i]-1)]-y_cord[int(partner[i]-1)])**2 +\
                    (z_cord[int(ID[i]-1)]-z_cord[int(partner[i]-1)])**2 )
            Lenghts[i]=l
        x_mid=np.zeros(num_rows)
        y_mid=np.zeros(num_rows)
        z_mid=np.zeros(num_rows)
        for i in range(num_rows):
            if partner[i]==-1:
                x_mid[int(ID[i]-1)]=0
                y_mid[int(ID[i]-1)]
                z_mid[int(ID[i]-1)]
            else:
                x_mid[int(ID[i]-1)]=(x_cord[int(ID[i]-1)]+x_cord[int(partner[i]-1)])/2.
                y_mid[int(ID[i]-1)]=(y_cord[int(ID[i]-1)]+y_cord[int(partner[i]-1)])/2.
                z_mid[int(ID[i]-1)]=(z_cord[int(ID[i]-1)]+z_cord[int(partner[i]-1)])/2.

        ID=ID-1
        partner=partner-1
        ID=ID[1:]
        radius=radius[1:]
        x_mid=x_mid[1:]
        y_mid=y_mid[1:]
        z_mid=z_mid[1:]
        type_compartment=type_compartment[1:]
        Lenghts=Lenghts[1:]
        partner=partner[1:]
        for i in range(len(partner)):
            if partner[i]==0:
                partner[i]=1

        adj_list=[]
        for i in range(num_rows-1):
            adj_list.append([int(ID[i]),int(partner[i])])
        adj_list = adj_list[1:]
        G = nx.Graph(adj_list)
        adj_matrix= 1.*nx.adjacency_matrix(G)
        Con_Mat=adj_matrix.toarray()

        resistances=np.zeros(num_rows-1)
        for i in range(num_rows-1):
            resistances[i]=0.5*(self.RA*Lenghts[i])/(np.pi*radius[i]**2)
        surfaces=np.zeros(num_rows-1)
        for i in range(num_rows-1):
            surfaces[i]=2.*np.pi*radius[i]*Lenghts[i]

        self.gbar_l_vec=np.ones(num_rows-1)*0.033
        self.gbar_Na_vec=np.zeros(num_rows-1)
        self.gbar_Na_vec[1]=110.95
        self.gbar_ca_vec=np.zeros(num_rows-1)
        self.gbar_ca_vec[0]=1.01
        self.gbar_ca_vec[3]=1.01
        self.gbar_kd_vec=np.zeros(num_rows-1)
        self.gbar_kd_vec[1]=0.4757
        self.gbar_k7_vec=np.zeros(num_rows-1)
        self.gbar_k7_vec[0]=2.44
        self.gbar_k7_vec[3]=2.44
        self.gbar_HC_vec=np.zeros(num_rows-1)
        self.gbar_HC_vec[2]=3.69
        self.gbar_ct_vec=np.zeros(num_rows-1)
        self.gbar_ct_vec[2]=12.49

        for i in range(num_rows-1):
            for j in range(num_rows-1):
                if Con_Mat[i][j] == 1:
                    Con_Mat[i][j] = 1./(resistances[i]+resistances[j])
        for i in range(num_rows-1):
            element_ij = 0.
            for j in range(num_rows-1):
                if Con_Mat[i][j] != 0.:
                    element_ij=element_ij+Con_Mat[i][j]
            Con_Mat[i][i]=-1.*(element_ij)
        for i in range(num_rows-1):
            Con_Mat[i]=-1.*Con_Mat[i]/surfaces[i]

        self.Con_Mat = Con_Mat
        self.num_rows = num_rows
        self.surfaces = surfaces

    def alpha_n_K_fast(self, v):
        return -0.01*(v-5.0+55.)/(np.exp(-0.1*(v-5.0+55))-1)*1/5.0
    def beta_n_K_fast(self, v):
        return 0.125*np.exp((v-5.0+65)/-80)*1/5.0
    def alpha_n_K_slow(self, v):
        return -0.01*(v-0.0+55)/(np.exp(-0.1*(v-0.0+55))-1)*1/8.0
    def beta_n_K_slow(self, v):
        return 0.125*np.exp((v-0.0+65)/-80)*1/8.0
    def n_K_fast_inf(self, v):
        return self.alpha_n_K_fast(v)/(self.alpha_n_K_fast(v) + self.beta_n_K_fast(v))
    def tau_n_K_fast(self, v):
        return 1./(self.alpha_n_K_fast(v) + self.beta_n_K_fast(v))
    def n_K_slow_inf(self, v):
        return self.alpha_n_K_slow(v)/(self.alpha_n_K_slow(v) + self.beta_n_K_slow(v))
    def tau_n_K_slow(self, v):
        return 1./(self.alpha_n_K_slow(v) + self.beta_n_K_slow(v))
    def m_Na_inf(self, v):
        return 1./(1. + np.exp(-(v+27.2)/4.9))
    def h_Na_inf(self, v):
        return 1./(1. + np.exp(+(v+60.7)/7.7))
    def s_Na_inf(self, v):
        return (1. /(1. + np.exp(+(v+60.1)/5.4)))
    def tau_m_Na(self, v):
        return 0.15
    def tau_h_Na(self, v):
        return 0.25*(20.1*np.exp(-0.5*((v+61.4)/32.7)**2))
    def tau_s_Na(self, v):
        return 1.0*(1000*(106.7*np.exp(-0.5*((v+52.7)/18.3)**2)))
    def m_Ca_T_inf(self, v):
        return 1./(1. + np.exp(-(v+57.)/6.2))
    def h_Ca_T_inf(self, v):
        return 1./(1. + np.exp(+(v+81.)/4.))
    def tau_m_Ca_T(self, v):
        return (0.612+1./(np.exp(-(v+132.)/16.7)+np.exp((v+16.8)/18.2)))
    def tau_h_Ca_T(self, v):
        AA=np.zeros(len(v))
        for i in range(len(v)):
            if v[i]>-81.:
                aa=(28.+np.exp(-(v[i]+22.)/10.5))
            else:
                aa= np.exp((v[i]+467.)/66.6)
            AA[i]=aa
        return AA
    def alpha_y_HCN(self, v):
        return np.exp(-(v+23.)/20.)
    def beta_y_HCN(self, v):
        return np.exp((v+130.)/10.)
    def y_HCN_inf(self, v):
        return self.alpha_y_HCN(v)/(self.alpha_y_HCN(v) + self.beta_y_HCN(v))
    def tau_y_HCN(self, v):
        return 1./(self.alpha_y_HCN(v) + self.beta_y_HCN(v))
    def alpha_c_Ca_L(self, v):
        if type(v) != float:
            AA=np.zeros(len(v))
            for i in range(len(v)):
                aa= -0.4*(v[i]+10+70.0)/(-1. + np.exp(-0.1*(v[i]+18+70.0)))
                AA[i]=aa
            return AA
        else:
            aa= -0.4*(v+10+70.0)/(-1. + np.exp(-0.1*(v+18+70.0)))
            return aa
    def beta_c_Ca_L(self, v):
        return 10.*np.exp(-(v-38+38) / 12.6)
    def c_Ca_L_inf(self, v):
        return self.alpha_c_Ca_L(v)/(self.alpha_c_Ca_L(v) + self.beta_c_Ca_L(v))
    def tau_c_Ca_L(self, v):
        return 1./(self.alpha_c_Ca_L(v) + self.beta_c_Ca_L(v))

    def run(self, curr_amp, pulse_end, sim_end, freq=None):
        curr_amp = curr_amp*1e-6
        T     = sim_end
        dt    = 0.01
        time  = np.arange(0,T+dt,dt)
        Vm      = np.zeros([self.num_rows-1,len(time)])
        Vm[:,0] = self.V_rest
        dV      = np.zeros(self.num_rows-1)

        m_Na = np.ones(self.num_rows-1)*self.m_Na_inf(self.V_rest)
        h_Na = np.ones(self.num_rows-1)*self.h_Na_inf(self.V_rest)
        s_Na = np.ones(self.num_rows-1)*self.s_Na_inf(self.V_rest)
        m_Ca_T = np.ones(self.num_rows-1)*self.m_Ca_T_inf(self.V_rest)
        h_Ca_T = np.ones(self.num_rows-1)*self.h_Ca_T_inf(self.V_rest)
        n_K_fast = np.ones(self.num_rows-1)*self.n_K_fast_inf(self.V_rest)
        n_K_slow = np.ones(self.num_rows-1)*self.n_K_slow_inf(self.V_rest)
        c_Ca_L = np.ones(self.num_rows-1)*self.c_Ca_L_inf(self.V_rest)
        y_HCN = np.ones(self.num_rows-1)*self.y_HCN_inf(self.V_rest)

        II=np.identity(self.num_rows-1)
        II_Con_Mat=II+(self.Con_Mat*dt/self.Cm)
        Inv_Con_Mat=np.linalg.inv(II_Con_Mat)
        curr_Am=curr_amp/self.surfaces[0]
        III = np.zeros(len(time))
        if freq == None:
            for i, t in enumerate(time):
                if 2<=t<=pulse_end:
                    III[i] = curr_Am
        else:
            for i, t in enumerate(time):
                III[i] = curr_Am*np.sin(2*np.pi*freq*t*1e-3)


        I_Na=[np.sum(self.gbar_Na_vec*(m_Na**3)*h_Na*s_Na*(self.V_rest-self.E_Na))]
        I_Ca_T=[np.sum(self.gbar_ca_vec*(m_Ca_T**2)*h_Ca_T*(self.V_rest-self.E_ca))]
        I_K_fast=[np.sum(self.gbar_kd_vec*(n_K_fast**4)*(self.V_rest-self.E_po))]
        I_K_slow=[np.sum(self.gbar_k7_vec*(n_K_slow**4)*(self.V_rest-self.E_po))]
        I_leak=[np.sum(self.gbar_l_vec*(self.V_rest-self.E_l))]
        I_Ca_L=[0]
        I_HCN=[0]
        for i in range(0,len(time)-1):
            g_Na_asl = self.gbar_Na_vec*(m_Na**3)*h_Na*s_Na
            g_ca_asl = self.gbar_ca_vec*(m_Ca_T**2)*h_Ca_T
            g_kd_asl = self.gbar_kd_vec*(n_K_fast**4)
            g_k7_asl = self.gbar_k7_vec*(n_K_slow**4)
            g_ct_asl = self.gbar_ct_vec*(c_Ca_L**3)
            g_HC_asl = self.gbar_HC_vec*y_HCN
            g_l_asl  = self.gbar_l_vec

            m_Na = (m_Na + (dt*self.K_m_Na*self.m_Na_inf(Vm[:,i])/self.tau_m_Na(Vm[:,i])))/(1. + self.K_m_Na*dt/(self.tau_m_Na(Vm[:,i])))
            h_Na = (h_Na + (dt*self.K_h_Na*self.h_Na_inf(Vm[:,i])/self.tau_h_Na(Vm[:,i])))/(1. + self.K_h_Na*dt/(self.tau_h_Na(Vm[:,i])))
            s_Na = (s_Na + (dt*self.K_s_Na*self.s_Na_inf(Vm[:,i])/self.tau_s_Na(Vm[:,i])))/(1. + self.K_s_Na*dt/(self.tau_s_Na(Vm[:,i])))
            m_Ca_T = (m_Ca_T + (dt*self.K_m_Ca_T*self.m_Ca_T_inf(Vm[:,i])/self.tau_m_Ca_T(Vm[:,i])))/(1. + self.K_m_Ca_T*dt/(self.tau_m_Ca_T(Vm[:,i])))
            h_Ca_T = (h_Ca_T + (dt*self.K_h_Ca_T*self.h_Ca_T_inf(Vm[:,i])/self.tau_h_Ca_T(Vm[:,i])))/(1. + self.K_h_Ca_T*dt/(self.tau_h_Ca_T(Vm[:,i])))
            n_K_fast = (n_K_fast + (dt*self.K_n_K_fast*self.n_K_fast_inf(Vm[:,i])/self.tau_n_K_fast(Vm[:,i])))/(1. + self.K_n_K_fast*dt/(self.tau_n_K_fast(Vm[:,i])))
            n_K_slow = (n_K_slow + (dt*self.K_n_K_slow*self.n_K_slow_inf(Vm[:,i])/self.tau_n_K_slow(Vm[:,i])))/(1. + self.K_n_K_slow*dt/(self.tau_n_K_slow(Vm[:,i])))
            c_Ca_L = (c_Ca_L + (dt*self.K_c_Ca_L*self.c_Ca_L_inf(Vm[:,i])/self.tau_c_Ca_L(Vm[:,i])))/(1. + self.K_c_Ca_L*dt/(self.tau_c_Ca_L(Vm[:,i])))
            y_HCN = (y_HCN + (dt*self.K_y_HCN*self.y_HCN_inf(Vm[:,i])/self.tau_y_HCN(Vm[:,i])))/(1. + self.K_y_HCN*dt/(self.tau_y_HCN(Vm[:,i])))

            I_Na.append(np.sum(g_Na_asl*(Vm[:,i]-self.E_Na)*1e6*self.surfaces[:]))
            I_Ca_T.append(np.sum(g_ca_asl*(Vm[:,i]-self.E_ca)*1e6*self.surfaces[:]))
            I_K_fast.append(np.sum(g_kd_asl*(Vm[:,i]-self.E_po)*1e6*self.surfaces[:]))
            I_K_slow.append(np.sum(g_k7_asl*(Vm[:,i]-self.E_po)*1e6*self.surfaces[:]))
            I_leak.append(np.sum(g_l_asl*(Vm[:,i]-self.E_l)*1e6*self.surfaces[:]))
            I_Ca_L.append(np.sum(g_ct_asl*(Vm[:,i]-self.E_ct)*1e6*self.surfaces[:]))
            I_HCN.append(np.sum(g_HC_asl*((Vm[:,i] - self.E_Na)+(Vm[:,i] - self.E_po))*1e6*self.surfaces[:]))

            dV =-(g_Na_asl*(Vm[:,i]-self.E_Na)+ g_ca_asl*(Vm[:,i]-self.E_ca)+ g_HC_asl*((Vm[:,i] - self.E_Na)+(Vm[:,i] - self.E_po))+g_ct_asl*(Vm[:,i]-self.E_ct)
            + g_kd_asl*(Vm[:,i]-self.E_po)+ g_k7_asl*(Vm[:,i]-self.E_po)+ g_l_asl*(Vm[:,i]-self.E_l))
            dV[0] = dV[0]+III[i+1]

            Vm[:,i+1] = np.dot(Inv_Con_Mat,(Vm[:,i] + dt * dV / self.Cm))
        return time, Vm[0], I_Na, I_Ca_T, I_K_fast, I_K_slow, I_leak, I_Ca_L, I_HCN
    
    def step(self, curr_input, vm_current, m_Na, h_Na, s_Na, m_Ca_T, h_Ca_T, 
             n_K_fast, n_K_slow, c_Ca_L, y_HCN, dt=0.01):
        """
        Perform a single time step calculation for the FCM model.

        Parameters:
        -----------
        curr_input : float
            Input current in uA (will be converted to uA/cm^2 internally)
        vm_current : numpy.ndarray
            Current membrane voltage values for all compartments
        m_Na, h_Na, s_Na : numpy.ndarray
            Sodium channel gating variables
        m_Ca_T, h_Ca_T : numpy.ndarray
            T-type calcium channel gating variables
        n_K_fast, n_K_slow : numpy.ndarray
            Potassium channel gating variables
        c_Ca_L : numpy.ndarray
            L-type calcium channel gating variable
        y_HCN : numpy.ndarray
            HCN channel gating variable
        dt : float, optional
            Time step in milliseconds (default: 0.01)
        
        Returns:
        --------
        tuple:
            - vm_next : numpy.ndarray
                Next membrane voltage values
            - channel_states : dict
                Updated channel state variables
            - currents : dict
                Dictionary containing all ionic currents
        """
        # Convert input current from uA to uA/cm^2
        curr_density = curr_input/self.surfaces[0]
        
        # Calculate channel conductances
        g_Na_asl = self.gbar_Na_vec * (m_Na**3) * h_Na * s_Na
        g_ca_asl = self.gbar_ca_vec * (m_Ca_T**2) * h_Ca_T
        g_kd_asl = self.gbar_kd_vec * (n_K_fast**4)
        g_k7_asl = self.gbar_k7_vec * (n_K_slow**4)
        g_ct_asl = self.gbar_ct_vec * (c_Ca_L**3)
        g_HC_asl = self.gbar_HC_vec * y_HCN
        g_l_asl = self.gbar_l_vec
        
        # Update channel states
        m_Na_next = (m_Na + (dt*self.K_m_Na*self.m_Na_inf(vm_current)/self.tau_m_Na(vm_current))) / \
                    (1. + self.K_m_Na*dt/(self.tau_m_Na(vm_current)))
        
        h_Na_next = (h_Na + (dt*self.K_h_Na*self.h_Na_inf(vm_current)/self.tau_h_Na(vm_current))) / \
                    (1. + self.K_h_Na*dt/(self.tau_h_Na(vm_current)))
        
        s_Na_next = (s_Na + (dt*self.K_s_Na*self.s_Na_inf(vm_current)/self.tau_s_Na(vm_current))) / \
                    (1. + self.K_s_Na*dt/(self.tau_s_Na(vm_current)))
        
        m_Ca_T_next = (m_Ca_T + (dt*self.K_m_Ca_T*self.m_Ca_T_inf(vm_current)/self.tau_m_Ca_T(vm_current))) / \
                      (1. + self.K_m_Ca_T*dt/(self.tau_m_Ca_T(vm_current)))
        
        h_Ca_T_next = (h_Ca_T + (dt*self.K_h_Ca_T*self.h_Ca_T_inf(vm_current)/self.tau_h_Ca_T(vm_current))) / \
                      (1. + self.K_h_Ca_T*dt/(self.tau_h_Ca_T(vm_current)))
        
        n_K_fast_next = (n_K_fast + (dt*self.K_n_K_fast*self.n_K_fast_inf(vm_current)/self.tau_n_K_fast(vm_current))) / \
                        (1. + self.K_n_K_fast*dt/(self.tau_n_K_fast(vm_current)))
        
        n_K_slow_next = (n_K_slow + (dt*self.K_n_K_slow*self.n_K_slow_inf(vm_current)/self.tau_n_K_slow(vm_current))) / \
                        (1. + self.K_n_K_slow*dt/(self.tau_n_K_slow(vm_current)))
        
        c_Ca_L_next = (c_Ca_L + (dt*self.K_c_Ca_L*self.c_Ca_L_inf(vm_current)/self.tau_c_Ca_L(vm_current))) / \
                      (1. + self.K_c_Ca_L*dt/(self.tau_c_Ca_L(vm_current)))
        
        y_HCN_next = (y_HCN + (dt*self.K_y_HCN*self.y_HCN_inf(vm_current)/self.tau_y_HCN(vm_current))) / \
                     (1. + self.K_y_HCN*dt/(self.tau_y_HCN(vm_current)))
        
        # Calculate ionic currents
        I_Na = g_Na_asl * (vm_current - self.E_Na)
        I_Ca_T = g_ca_asl * (vm_current - self.E_ca)
        I_K_fast = g_kd_asl * (vm_current - self.E_po)
        I_K_slow = g_k7_asl * (vm_current - self.E_po)
        I_leak = g_l_asl * (vm_current - self.E_l)
        I_Ca_L = g_ct_asl * (vm_current - self.E_ct)
        I_HCN = g_HC_asl * ((vm_current - self.E_Na) + (vm_current - self.E_po))
        
        # Calculate voltage changes
        dV = -(I_Na + I_Ca_T + I_HCN + I_Ca_L + I_K_fast + I_K_slow + I_leak)
        dV[0] = dV[0] + curr_density
        
        # Calculate next voltage state
        II = np.identity(self.num_rows-1)
        II_Con_Mat = II + (self.Con_Mat*dt/self.Cm)
        Inv_Con_Mat = np.linalg.inv(II_Con_Mat)
        vm_next = np.dot(Inv_Con_Mat, (vm_current + dt * dV / self.Cm))
        
        # Package channel states for return
        channel_states = {
            'm_Na': m_Na_next,
            'h_Na': h_Na_next,
            's_Na': s_Na_next,
            'm_Ca_T': m_Ca_T_next,
            'h_Ca_T': h_Ca_T_next,
            'n_K_fast': n_K_fast_next,
            'n_K_slow': n_K_slow_next,
            'c_Ca_L': c_Ca_L_next,
            'y_HCN': y_HCN_next
        }
        
        # Package currents for return (convert to uA)
        currents = {
            'I_Na': np.sum(I_Na * 1e6 * self.surfaces[:]),
            'I_Ca_T': np.sum(I_Ca_T * 1e6 * self.surfaces[:]),
            'I_K_fast': np.sum(I_K_fast * 1e6 * self.surfaces[:]),
            'I_K_slow': np.sum(I_K_slow * 1e6 * self.surfaces[:]),
            'I_leak': np.sum(I_leak * 1e6 * self.surfaces[:]),
            'I_Ca_L': np.sum(I_Ca_L * 1e6 * self.surfaces[:]),
            'I_HCN': np.sum(I_HCN * 1e6 * self.surfaces[:])
        }
        
        return vm_next, channel_states, currents


# Initialize the model
fcm = FCM()

# Simulation parameters
T = 350  # Total simulation time (ms)
dt = 0.01  # Time step (ms)
time = np.arange(0, T+dt, dt)
curr_amp = 20  # Input current amplitude (uA)
pulse_end = 200  # Duration of current pulse (ms)

# Initialize arrays to store results
Vm = np.zeros([fcm.num_rows-1, len(time)])
Vm[:,0] = fcm.V_rest  # Set initial voltage

# Initialize channel states
channel_states = {
    'm_Na': np.ones(fcm.num_rows-1) * fcm.m_Na_inf(fcm.V_rest),
    'h_Na': np.ones(fcm.num_rows-1) * fcm.h_Na_inf(fcm.V_rest),
    's_Na': np.ones(fcm.num_rows-1) * fcm.s_Na_inf(fcm.V_rest),
    'm_Ca_T': np.ones(fcm.num_rows-1) * fcm.m_Ca_T_inf(fcm.V_rest),
    'h_Ca_T': np.ones(fcm.num_rows-1) * fcm.h_Ca_T_inf(fcm.V_rest),
    'n_K_fast': np.ones(fcm.num_rows-1) * fcm.n_K_fast_inf(fcm.V_rest),
    'n_K_slow': np.ones(fcm.num_rows-1) * fcm.n_K_slow_inf(fcm.V_rest),
    'c_Ca_L': np.ones(fcm.num_rows-1) * fcm.c_Ca_L_inf(fcm.V_rest),
    'y_HCN': np.ones(fcm.num_rows-1) * fcm.y_HCN_inf(fcm.V_rest)
}

# Initialize arrays for currents
currents = {
    'I_Na': [], 'I_Ca_T': [], 'I_K_fast': [], 'I_K_slow': [],
    'I_leak': [], 'I_Ca_L': [], 'I_HCN': []
}

freq = 10.0

input_current = [0]

# Run simulation
for i in range(len(time)-1):

    # Calculate input current
    if freq is not None:
        curr_input = curr_amp* 1e-6 * np.sin(2*np.pi*freq*time[i]*1e-3)
    else:
        curr_input = curr_amp * 1e-6 if (2 <= time[i] <= pulse_end) else 0

    input_current.append(curr_input)
    
    # Perform single step
    Vm[:,i+1], channel_states, step_currents = fcm.step(
        curr_input, Vm[:,i],
        channel_states['m_Na'], channel_states['h_Na'], channel_states['s_Na'],
        channel_states['m_Ca_T'], channel_states['h_Ca_T'],
        channel_states['n_K_fast'], channel_states['n_K_slow'],
        channel_states['c_Ca_L'], channel_states['y_HCN']
    )
    
    # Store currents
    for key in currents:
        currents[key].append(step_currents[key])

#time, Vm, I_Na, I_Ca_T, I_K_fast, I_K_slow, I_leak, I_Ca_L, I_HCN = fcm.run(20, 350, 350, 10)

fig, axe = plt.subplots(2, 1, figsize=(15,10), sharex=True)

total_steps = int(T/dt)
time = time[:total_steps]

axe[0].plot(time, currents["I_Na"], label="I_Na")
axe[0].plot(time, currents["I_Ca_T"], label="I_Ca_T")
axe[0].plot(time, currents["I_K_fast"], label="I_K_fast")
axe[0].plot(time, currents["I_K_slow"], label="I_K_slow")
axe[0].plot(time, currents["I_leak"], label="I_leakage")
axe[0].plot(time, currents["I_Ca_L"], label="I_Ca_L")
axe[0].plot(time, currents["I_HCN"], label="I_HCN")
axe[0].plot(time, input_current[:total_steps], label="Input current")
axe[0].legend()
axe[0].grid()
axe[0].set_ylabel("Current (uA)")

axe[1].plot(time, Vm[0][:total_steps])
axe[1].grid()
axe[1].set_xlabel("Time (ms)")
axe[1].set_ylabel("Membrane potential (mV)")

plt.tight_layout()
plt.show()




