#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include "brain.h"

//class FCM {
//public:

    // Constructor
    FCM::FCM(int idx){

        index = idx;

        // Initialize gating constants
        V_rest = -70.0;
        V_thresh = -30.0;
        E_l = -81.4;
        Cm = 1.0;
        E_po = -77.0;
        E_Na = 50.0;
        E_ca = 120.0;
        E_HC = -40.0;
        E_ct = 22.6;
        RA = 0.1;
        ro_ex = 1.0;

        // Initialize temperature-dependent constants
        K_m_Na = pow(2.2, (31.0 - 20.0) / 10.0);
        K_h_Na = pow(2.9, (31.0 - 20.0) / 10.0);
        K_s_Na = pow(2.9, (31.0 - 20.0) / 10.0);
        K_m_Ca_T = pow(5.0, (31.0 - 24.0) / 10.0);
        K_h_Ca_T = pow(3.0, (31.0 - 24.0) / 10.0);
        K_n_K_fast = pow(2.0, (31.0 - 6.3) / 10.0);
        K_n_K_slow = pow(2.0, (31.0 - 6.3) / 10.0);
        K_c_Ca_L = pow(1.0, (31.0 - 10.0) / 10.0);
        K_y_HCN = pow(1.0, (31.0 - 37.0) / 10.0);

        // Initialize connectivity matrix and its inverse
        Con_Mat = {
            {114.70698013, -2.38929269, -112.31768744, 0.0},
            {-6.130952, 9.10152754, 0.0, -2.97057554},
            {-1912.52907736, 0.0, 1912.52907736, 0.0},
            {0.0, -3.07782072, 0.0, 3.07782072}
        };
        Inv_Con_Mat = {
            {9.27335478e-01, 2.03249095e-02, 5.17538740e-02, 5.85738801e-04},
            {5.21539472e-02, 9.18466316e-01, 2.91067136e-03, 2.64690654e-02},
            {8.81257362e-01, 1.93149907e-02, 9.88710132e-02, 5.56634188e-04},
            {1.55727486e-03, 2.74246646e-02, 8.69103028e-05, 9.70931150e-01}
        };

        // Initialize surfaces
        surfaces = {3.55310988e-06, 1.38468209e-06, 2.08664584e-07, 1.33643351e-06};
        num_rows = 5;

        // Initialize conductance vectors
        gbar_l_vec = {0.033, 0.033, 0.033, 0.033};
        gbar_Na_vec = {0.0, 110.95, 0.0, 0.0};
        gbar_ca_vec = {1.01, 0.0, 0.0, 1.01};
        gbar_kd_vec = {0.0, 0.4757, 0.0, 0.0};
        gbar_k7_vec = {2.44, 0.0, 0.0, 2.44};
        gbar_HC_vec = {0.0, 0.0, 3.69, 0.0};
        gbar_ct_vec = {0.0, 0.0, 12.49, 0.0};

        // Initialize membrane potentials
        Vm = std::vector<double>(num_rows - 1, V_rest);

        // Initialize channel states
        channel_states.m_Na = std::vector<double>(num_rows - 1, m_Na_inf(V_rest));
        channel_states.h_Na = std::vector<double>(num_rows - 1, h_Na_inf(V_rest));
        channel_states.s_Na = std::vector<double>(num_rows - 1, s_Na_inf(V_rest));
        channel_states.m_Ca_T = std::vector<double>(num_rows - 1, m_Ca_T_inf(V_rest));
        channel_states.h_Ca_T = std::vector<double>(num_rows - 1, h_Ca_T_inf(V_rest));
        channel_states.n_K_fast = std::vector<double>(num_rows - 1, n_K_fast_inf(V_rest));
        channel_states.n_K_slow = std::vector<double>(num_rows - 1, n_K_slow_inf(V_rest));
        channel_states.c_Ca_L = std::vector<double>(num_rows - 1, c_Ca_L_inf(V_rest));
        channel_states.y_HCN = std::vector<double>(num_rows - 1, y_HCN_inf(V_rest));
    }

    //getter and setter functions 
    void FCM::set_index(int new_index) {
        index = new_index;
    }

    int FCM::get_index() {
        return index;
    }

    double FCM::get_Vm(){
        return Vm[0];
    }

    double FCM::get_V_rest(){
        return V_rest;
    }

    // Gating functions
    double FCM::alpha_n_K_fast(double v) {
        double denom = exp(-0.1 * (v - 5.0 + 55.0)) - 1.0;
        return (denom == 0.0) ? 0.0 : -0.01 * (v - 5.0 + 55.0) / denom * 1.0 / 5.0;
    }

    double FCM::beta_n_K_fast(double v) {
        return 0.125 * exp((v - 5.0 + 65.0) / -80.0) * 1.0 / 5.0;
    }

    double FCM::alpha_n_K_slow(double v) {
        double denom = exp(-0.1 * (v - 0.0 + 55.0)) - 1.0;
        return (denom == 0.0) ? 0.0 : -0.01 * (v - 0.0 + 55.0) / denom * 1.0 / 8.0;
    }

    double FCM::beta_n_K_slow(double v) {
        return 0.125 * exp((v - 0.0 + 65.0) / -80.0) * 1.0 / 8.0;
    }

    double FCM::n_K_fast_inf(double v) {
        double alpha = alpha_n_K_fast(v);
        double beta = beta_n_K_fast(v);
        return alpha / (alpha + beta);
    }

    double FCM::tau_n_K_fast(double v) {
        double alpha = alpha_n_K_fast(v);
        double beta = beta_n_K_fast(v);
        return 1.0 / (alpha + beta);
    }

    double FCM::n_K_slow_inf(double v) {
        double alpha = alpha_n_K_slow(v);
        double beta = beta_n_K_slow(v);
        return alpha / (alpha + beta);
    }

    double FCM::tau_n_K_slow(double v) {
        double alpha = alpha_n_K_slow(v);
        double beta = beta_n_K_slow(v);
        return 1.0 / (alpha + beta);
    }

    double FCM::FCM::m_Na_inf(double v) {
        return 1.0 / (1.0 + exp(-(v + 27.2) / 4.9));
    }

    double FCM::h_Na_inf(double v) {
        return 1.0 / (1.0 + exp((v + 60.7) / 7.7));
    }

    double FCM::s_Na_inf(double v) {
        return 1.0 / (1.0 + exp((v + 60.1) / 5.4));
    }

    double FCM::tau_m_Na(double v) {
        return 0.15;
    }

    double FCM::tau_h_Na(double v) {
        return 0.25 * (20.1 * exp(-0.5 * pow((v + 61.4) / 32.7, 2)));
    }

    double FCM::tau_s_Na(double v) {
        return 1000.0 * (106.7 * exp(-0.5 * pow((v + 52.7) / 18.3, 2)));
    }

    double FCM::m_Ca_T_inf(double v) {
        return 1.0 / (1.0 + exp(-(v + 57.0) / 6.2));
    }

    double FCM::h_Ca_T_inf(double v) {
        return 1.0 / (1.0 + exp((v + 81.0) / 4.0));
    }

    double FCM::tau_m_Ca_T(double v) {
        return 0.612 + 1.0 / (exp(-(v + 132.0) / 16.7) + exp((v + 16.8) / 18.2));
    }

    double FCM::tau_h_Ca_T(double v) {
        if (v > -81.0) {
            return 28.0 + exp(-(v + 22.0) / 10.5);
        } else {
            return exp((v + 467.0) / 66.6);
        }
    }

    double FCM::alpha_y_HCN(double v) {
        return exp(-(v + 23.0) / 20.0);
    }

    double FCM::beta_y_HCN(double v) {
        return exp((v + 130.0) / 10.0);
    }

    double FCM::y_HCN_inf(double v) {
        double alpha = alpha_y_HCN(v);
        double beta = beta_y_HCN(v);
        return alpha / (alpha + beta);
    }

    double FCM::tau_y_HCN(double v) {
        double alpha = alpha_y_HCN(v);
        double beta = beta_y_HCN(v);
        return 1.0 / (alpha + beta);
    }

    double FCM::alpha_c_Ca_L(double v) {
        double denom = exp(-0.1 * (v + 18.0 + 70.0)) - 1.0;
        return (fabs(denom) < 1e-6) ? 0.0 : -0.4 * (v + 10.0 + 70.0) / denom;
    }

    double FCM::beta_c_Ca_L(double v) {
        return 10.0 * exp(-(v - 38.0 + 38.0) / 12.6);
    }

    double FCM::c_Ca_L_inf(double v) {
        double alpha = alpha_c_Ca_L(v);
        double beta = beta_c_Ca_L(v);
        return alpha / (alpha + beta);
    }

    double FCM::tau_c_Ca_L(double v) {
        double alpha = alpha_c_Ca_L(v);
        double beta = beta_c_Ca_L(v);
        return 1.0 / (alpha + beta);
    }

    // Check for spike condition
    bool FCM::spiked() const {
        return Vm[0] > V_thresh;
    }

    // Step method for membrane potential update
    void FCM::step(double curr_input, double dt) {
        // Convert input current from uA to uA/cm^2
        double curr_density = curr_input / surfaces[0];

        // Local references for readability
        std::vector<double>& m_Na = channel_states.m_Na;
        std::vector<double>& h_Na = channel_states.h_Na;
        std::vector<double>& s_Na = channel_states.s_Na;
        std::vector<double>& m_Ca_T = channel_states.m_Ca_T;
        std::vector<double>& h_Ca_T = channel_states.h_Ca_T;
        std::vector<double>& n_K_fast = channel_states.n_K_fast;
        std::vector<double>& n_K_slow = channel_states.n_K_slow;
        std::vector<double>& c_Ca_L = channel_states.c_Ca_L;
        std::vector<double>& y_HCN = channel_states.y_HCN;

        std::vector<double> g_Na_asl(num_rows - 1);
        std::vector<double> g_ca_asl(num_rows - 1);
        std::vector<double> g_kd_asl(num_rows - 1);
        std::vector<double> g_k7_asl(num_rows - 1);
        std::vector<double> g_ct_asl(num_rows - 1);
        std::vector<double> g_HC_asl(num_rows - 1);
        std::vector<double> g_l_asl(num_rows - 1);

        // Calculate channel conductances
        #pragma omp parallel for 
        for (int i = 0; i < num_rows - 1; i++) {
            g_Na_asl[i] = gbar_Na_vec[i] * pow(m_Na[i], 3) * h_Na[i] * s_Na[i];
            g_ca_asl[i] = gbar_ca_vec[i] * pow(m_Ca_T[i], 2) * h_Ca_T[i];
            g_kd_asl[i] = gbar_kd_vec[i] * pow(n_K_fast[i], 4);
            g_k7_asl[i] = gbar_k7_vec[i] * pow(n_K_slow[i], 4);
            g_ct_asl[i] = gbar_ct_vec[i] * pow(c_Ca_L[i], 3);
            g_HC_asl[i] = gbar_HC_vec[i] * y_HCN[i];
            g_l_asl[i] = gbar_l_vec[i];
        }

        // Update channel states
        #pragma omp parallel
        {
            #pragma omp sections
            {
            #pragma omp section
            {
                update_channel_state(m_Na, K_m_Na, dt, &m_Na_inf, &tau_m_Na);
                update_channel_state(h_Na, K_h_Na, dt, &h_Na_inf, &tau_h_Na);
            }
            #pragma omp section
            {
                update_channel_state(s_Na, K_s_Na, dt, &s_Na_inf, &tau_s_Na);
                update_channel_state(m_Ca_T, K_m_Ca_T, dt, &m_Ca_T_inf, &tau_m_Ca_T);
            }
            #pragma omp section
            {
                update_channel_state(h_Ca_T, K_h_Ca_T, dt, &h_Ca_T_inf, &tau_h_Ca_T);
                update_channel_state(n_K_fast, K_n_K_fast, dt, &n_K_fast_inf, &tau_n_K_fast);
            }
            #pragma omp section
            {
                update_channel_state(n_K_slow, K_n_K_slow, dt, &n_K_slow_inf, &tau_n_K_slow);
                update_channel_state(c_Ca_L, K_c_Ca_L, dt, &c_Ca_L_inf, &tau_c_Ca_L);
            }
            #pragma omp section
            {
                update_channel_state(y_HCN, K_y_HCN, dt, &y_HCN_inf, &tau_y_HCN);
            }
            }
        }

        // Calculate currents and voltage changes
        std::vector<double> dV(num_rows - 1, 0.0);
        //#pragma omp parallel for 
        for (int i = 0; i < num_rows - 1; i++) {
            double I_Na = g_Na_asl[i] * (Vm[i] - E_Na);
            double I_Ca_T = g_ca_asl[i] * (Vm[i] - E_ca);
            double I_K_fast = g_kd_asl[i] * (Vm[i] - E_po);
            double I_K_slow = g_k7_asl[i] * (Vm[i] - E_po);
            double I_leak = g_l_asl[i] * (Vm[i] - E_l);
            double I_Ca_L = g_ct_asl[i] * (Vm[i] - E_ct);
            double I_HCN = g_HC_asl[i] * ((Vm[i] - E_Na) + (Vm[i] - E_po));

            dV[i] = -(I_Na + I_Ca_T + I_HCN + I_Ca_L + I_K_fast + I_K_slow + I_leak);
        }

        dV[0] += curr_density;

        // Update membrane potentials manually without matrix multiplication
        std::vector<double> vm_next(num_rows - 1, 0.0);
        //#pragma omp parallel for
        for (int i = 0; i < num_rows - 1; i++) {
            for (int j = 0; j < num_rows - 1; j++) {
                vm_next[i] += Inv_Con_Mat[i][j] * (Vm[j] + dt * dV[j] / Cm);
            }
        }
        Vm = vm_next;
    }

    // Helper function for channel state update
    void FCM::update_channel_state(std::vector<double>& state, double K, double dt,
                              double (*inf_func)(double), double (*tau_func)(double)) {
        for (size_t i = 0; i < state.size(); i++) {
            state[i] = (state[i] + (dt * K * inf_func(Vm[i]) / tau_func(Vm[i]))) /
                       (1.0 + K * dt / tau_func(Vm[i]));
        }
    }
