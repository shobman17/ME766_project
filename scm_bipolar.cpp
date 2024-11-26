#include <vector>
#include <cmath>
#include <algorithm>

class SCM {
private:
    // Constants
    double V_rest;
    double E_l;
    double Cm;
    
    double g_Na;
    double g_l;
    double g_Ca_T;
    double g_K_fast;
    double g_K_slow;
    double g_HCN;
    double g_Ca_L;
    double E_K;
    double E_Na;
    double E_Ca_T;
    double E_HCN;
    double E_Ca_L;

    // Temperature scaling factors
    double K_m_Na;
    double K_h_Na;
    double K_s_Na;
    double K_m_Ca_T;
    double K_h_Ca_T;
    double K_n_K_fast;
    double K_n_K_slow;
    double K_c_Ca_L;
    double K_y_HCN;

public:
    SCM() {
        V_rest = -70.0;
        E_l = -81.3;
        Cm = 1.0;

        g_Na = 23.55;
        g_l = 0.033;
        g_Ca_T = 0.76;
        g_K_fast = 0.1;
        g_K_slow = 1.83;
        g_HCN = 0.11;
        g_Ca_L = 0.38;
        E_K = -77.0;
        E_Na = 50.0;
        E_Ca_T = 120.0;
        E_HCN = -40.0;
        E_Ca_L = 22.6;

        K_m_Na = pow(2.2, (31.0 - 20.0)/10.0);
        K_h_Na = pow(2.9, (31.0 - 20.0)/10.0);
        K_s_Na = pow(2.9, (31.0 - 20.0)/10.0);
        K_m_Ca_T = pow(5.0, (31.0 - 24.0)/10.0);
        K_h_Ca_T = pow(3.0, (31.0 - 24.0)/10.0);
        K_n_K_fast = pow(2.0, (31.0 - 6.3)/10.0);
        K_n_K_slow = pow(2.0, (31.0 - 6.3)/10.0);
        K_c_Ca_L = pow(1.0, (31.0 - 10.0)/10.0);
        K_y_HCN = pow(1.0, (31.0 - 37.0)/10.0);
    }

    double alpha_n_K_fast(double v) {
        return 0.1 * -0.1 * (v - 5.0 + 55.0) / (exp(-0.1 * (v - 5.0 + 55.0)) - 1.0) / 5.0;
    }

    double beta_n_K_fast(double v) {
        return 0.125 * exp(-(v - 5.0 + 65.0)/80.0) / 5.0;
    }

    double alpha_n_K_slow(double v) {
        return -0.01 * (v + 55.0) / (exp(-0.1 * (v + 55.0)) - 1.0) / 8.0;
    }

    double beta_n_K_slow(double v) {
        return 0.125 * exp(-(v + 65.0)/80.0) / 8.0;
    }

    double n_K_fast_inf(double v) {
        return alpha_n_K_fast(v) / (alpha_n_K_fast(v) + beta_n_K_fast(v));
    }

    double tau_n_K_fast(double v) {
        return 1.0 / (alpha_n_K_fast(v) + beta_n_K_fast(v));
    }

    double n_K_slow_inf(double v) {
        return alpha_n_K_slow(v) / (alpha_n_K_slow(v) + beta_n_K_slow(v));
    }

    double tau_n_K_slow(double v) {
        return 1.0 / (alpha_n_K_slow(v) + beta_n_K_slow(v));
    }

    double m_Na_inf(double v) {
        return 1.0 / (1.0 + exp(-(v + 27.2)/4.9));
    }

    double tau_m_Na(double) {
        return 0.15;
    }

    double h_Na_inf(double v) {
        return 1.0 / (1.0 + exp(+(v + 60.7)/7.7));
    }

    double tau_h_Na(double v) {
        return 0.25 * (20.1 * exp(-0.5 * pow((v + 61.4)/32.7, 2)));
    }

    double s_Na_inf(double v) {
        return 1.0 / (1.0 + exp(+(v + 60.1)/5.4));
    }

    double tau_s_Na(double v) {
        return (1000 * (106.7 * exp(-0.5 * pow((v + 52.7)/18.3, 2))));
    }

    double m_Ca_T_inf(double v) {
        return 1.0 / (1.0 + exp(-(v + 57.0)/6.2));
    }

    double tau_m_Ca_T(double v) {
        return 0.612 + 1.0 / (exp(-(v + 132.0)/16.7) + exp((v + 16.8)/18.2));
    }

    double h_Ca_T_inf(double v) {
        return 1.0 / (1.0 + exp(+(v + 81.0)/4.0));
    }

    double tau_h_Ca_T(double v) {
        return v > -81.0 ? 28.0 + exp(-(v + 22.0)/10.5) : exp((v + 467.0)/66.6);
    }

    double alpha_y_HCN(double v) {
        return exp(-(v + 23.0)/20.0);
    }

    double beta_y_HCN(double v) {
        return exp((v + 130.0)/10.0);
    }

    double y_HCN_inf(double v) {
        return alpha_y_HCN(v) / (alpha_y_HCN(v) + beta_y_HCN(v));
    }

    double tau_y_HCN(double v) {
        return 1.0 / (alpha_y_HCN(v) + beta_y_HCN(v));
    }

    double alpha_c_Ca_L(double v) {
        return -0.4 * (v + 10.0 + 70.0) / (-1.0 + exp(-0.1 * (v + 18.0 + 70.0)));
    }

    double beta_c_Ca_L(double v) {
        return 10.0 * exp(-(v - 38.0 + 38.0)/12.6);
    }

    double c_Ca_L_inf(double v) {
        return alpha_c_Ca_L(v) / (alpha_c_Ca_L(v) + beta_c_Ca_L(v));
    }

    double tau_c_Ca_L(double v) {
        return 1.0 / (alpha_c_Ca_L(v) + beta_c_Ca_L(v));
    }

    struct SimulationResults {
        std::vector<double> time;
        std::vector<double> Vm;
        std::vector<double> i_Na;
        std::vector<double> i_Ca_T;
        std::vector<double> i_K_fast;
        std::vector<double> i_K_slow;
        std::vector<double> i_leak;
        std::vector<double> i_Ca_L;
        std::vector<double> i_HCN;
        std::vector<double> current_trace;
    };

    SimulationResults run(double curr_amp, double pulse_end, double sim_end, double freq = 0.0) {
        double dt = 0.01;
        int n_steps = static_cast<int>(sim_end/dt) + 1;
        
        SimulationResults results;
        // Initialize all vectors with n_steps size
        results.time.resize(n_steps);
        results.Vm.resize(n_steps);
        results.i_Na.resize(n_steps);
        results.i_Ca_T.resize(n_steps);
        results.i_K_fast.resize(n_steps);
        results.i_K_slow.resize(n_steps);
        results.i_leak.resize(n_steps);
        results.i_Ca_L.resize(n_steps);
        results.i_HCN.resize(n_steps);
        results.current_trace.resize(n_steps);

        // Generate time points
        for(int i = 0; i < n_steps; i++) {
            results.time[i] = i * dt;
        }

        // Initialize membrane potential
        results.Vm[0] = V_rest;

        // External current setup
        double surface_area = 6.50e-6;
        double curr_Am = curr_amp * 1e-6 / surface_area;
        
        // Set up current trace
        if (freq == 0.0) {
            for(int i = 0; i < n_steps; i++) {
                double t = results.time[i];
                if (t >= 2.0 && t <= pulse_end) {
                    results.current_trace[i] = curr_Am;
                }
            }
        } else {
            for(int i = 0; i < n_steps; i++) {
                results.current_trace[i] = curr_Am * sin(2.0 * M_PI * freq * results.time[i] * 1e-3);
            }
        }

        // Initialize state variables
        double m_Na = m_Na_inf(V_rest);
        double h_Na = h_Na_inf(V_rest);
        double s_Na = s_Na_inf(V_rest);
        double m_Ca_T = m_Ca_T_inf(V_rest);
        double h_Ca_T = h_Ca_T_inf(V_rest);
        double n_K_fast = n_K_fast_inf(V_rest);
        double n_K_slow = n_K_slow_inf(V_rest);
        double c_Ca_L = c_Ca_L_inf(V_rest);
        double y_HCN = y_HCN_inf(V_rest);

        // Initialize current vectors
        results.i_Na[0] = 0.0;
        results.i_Ca_T[0] = 0.0;
        results.i_K_fast[0] = 0.0;
        results.i_K_slow[0] = 0.0;
        results.i_leak[0] = 0.0;
        results.i_Ca_L[0] = 0.0;
        results.i_HCN[0] = 0.0;

        // Main simulation loop
        for(int i = 0; i < n_steps - 1; i++) {
            // Calculate conductances
            double g_Na = this->g_Na * pow(m_Na, 3) * h_Na * s_Na;
            double g_Ca_T = this->g_Ca_T * pow(m_Ca_T, 2) * h_Ca_T;
            double g_K_fast = this->g_K_fast * pow(n_K_fast, 4);
            double g_K_slow = this->g_K_slow * pow(n_K_slow, 4);
            double g_Ca_L = this->g_Ca_L * pow(c_Ca_L, 3);
            double g_HC = this->g_HCN * y_HCN;
            double g_l = this->g_l;

            // Update state variables
            m_Na = (m_Na + dt * K_m_Na * m_Na_inf(results.Vm[i]) / tau_m_Na(results.Vm[i])) /
                   (1.0 + K_m_Na * dt / tau_m_Na(results.Vm[i]));
            
            // TODO: Add similar updates for h_Na, s_Na, m_Ca_T, h_Ca_T, n_K_fast, n_K_slow, c_Ca_L, y_HCN

            // Calculate currents
            results.i_Na[i + 1] = g_Na * (results.Vm[i] - E_Na) * 1e6 * surface_area;
            // TODO: Add similar calculations for other currents (i_Ca_T, i_K_fast, etc.)

            // Calculate voltage update
            double dV = -(
                g_Na * (results.Vm[i] - E_Na) +
                g_Ca_T * (results.Vm[i] - E_Ca_T) +
                g_HC * (results.Vm[i] - E_Na + results.Vm[i] - E_K) +
                g_Ca_L * (results.Vm[i] - E_Ca_L) +
                g_K_fast * (results.Vm[i] - E_K) +
                g_K_slow * (results.Vm[i] - E_K) +
                g_l * (results.Vm[i] - E_l)
            ) + results.current_trace[i + 1];

            results.Vm[i + 1] = results.Vm[i] + dt * dV / Cm;
        }

        return results;
    }
};
