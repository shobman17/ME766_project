#include <iostream>     // For input/output (e.g., std::cout)
#include <vector>       // For std::vector
#include <map>          // For std::map to store channel states and currents
#include <eigen3/Eigen/Dense>  // For Eigen matrix operations
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <cmath>        // For mathematical functions (e.g., pow, sin, etc.)
#include <numeric>

class FCM {
private:
    int index;

    // Threshold voltages
    double V_rest = -70.0, V_thresh = -30.0, E_l = -81.4, Cm = 1.0;
    double E_po = -77.0, E_Na = 50.0, E_ca = 120.0, E_HC = -40.0, E_ct = 22.6
    double RA = 0.1, ro_ex = 1.0;
    double K_m_Na, K_h_Na, K_s_Na, K_m_Ca_T, K_h_Ca_T, K_n_K_fast, K_n_K_slow, K_c_Ca_L, K_y_HCN;

    Eigen::MatrixXd Con_Mat;
    int num_rows = 5;
    std::vector<double> surfaces;

    Eigen::VectorXd gbar_l_vec, gbar_Na_vec, gbar_ca_vec;
    Eigen::VectorXd gbar_kd_vec, gbar_k7_vec, gbar_HC_vec, gbar_ct_vec;

    Eigen::VectorXd Vm;
    std::unordered_map<std::string, std::vector<double>> channel_states;

public:
    FCM(int idx = 0) : index(idx) {
        K_m_Na = std::pow(2.2, (31.0 - 20.0) / 10.0);
        K_h_Na = std::pow(2.9, (31.0 - 20.0) / 10.0);
        K_s_Na = std::pow(2.9, (31.0 - 20.0) / 10.0);
        K_m_Ca_T = std::pow(5.0, (31.0 - 24.0) / 10.0);
        K_h_Ca_T = std::pow(3.0, (31.0 - 24.0) / 10.0);
        K_n_K_fast = std::pow(2.0, (31.0 - 6.3) / 10.0);
        K_n_K_slow = std::pow(2.0, (31.0 - 6.3) / 10.0);
        K_c_Ca_L = std::pow(1.0, (31.0 - 10.0) / 10.0);
        K_y_HCN = std::pow(1.0, (31.0 - 37.0) / 10.0);

        Con_Mat.resize(4, 4);
        Con_Mat << 114.70698013, -2.38929269, -112.31768744, 0.0,
                   -6.130952, 9.10152754, 0.0, -2.97057554,
                   -1912.52907736, 0.0, 1912.52907736, 0.0,
                   0.0, -3.07782072, 0.0, 3.07782072;

        surfaces = {3.55310988e-06, 1.38468209e-06, 2.08664584e-07, 1.33643351e-06};

        // Initialize conductance vectors
        gbar_l_vec = Eigen::VectorXd::Constant(num_rows - 1, 0.033);
        gbar_Na_vec = Eigen::VectorXd::Zero(num_rows - 1);
        gbar_Na_vec(1) = 110.95;
        gbar_ca_vec = Eigen::VectorXd::Zero(num_rows - 1);
        gbar_ca_vec(0) = 1.01;
        gbar_ca_vec(3) = 1.01;
        gbar_kd_vec = Eigen::VectorXd::Zero(num_rows - 1);
        gbar_kd_vec(1) = 0.4757;
        gbar_k7_vec = Eigen::VectorXd::Zero(num_rows - 1);
        gbar_k7_vec(0) = 2.44;
        gbar_k7_vec(3) = 2.44;
        gbar_HC_vec = Eigen::VectorXd::Zero(num_rows - 1);
        gbar_HC_vec(2) = 3.69;
        gbar_ct_vec = Eigen::VectorXd::Zero(num_rows - 1);
        gbar_ct_vec(2) = 12.49;

        Vm = Eigen::VectorXd::Constant(num_rows - 1, V_rest);

        channel_states["m_Na"] = std::vector<double>(num_rows - 1, m_Na_inf(V_rest));
        channel_states["h_Na"] = std::vector<double>(num_rows - 1, h_Na_inf(V_rest));
        channel_states["s_Na"] = std::vector<double>(num_rows - 1, s_Na_inf(V_rest));
        channel_states["m_Ca_T"] = std::vector<double>(num_rows - 1, m_Ca_T_inf(V_rest));
        channel_states["h_Ca_T"] = std::vector<double>(num_rows - 1, h_Ca_T_inf(V_rest));
        channel_states["n_K_fast"] = std::vector<double>(num_rows - 1, n_K_fast_inf(V_rest));
        channel_states["n_K_slow"] = std::vector<double>(num_rows - 1, n_K_slow_inf(V_rest));
        channel_states["c_Ca_L"] = std::vector<double>(num_rows - 1, c_Ca_L_inf(V_rest));
        channel_states["y_HCN"] = std::vector<double>(num_rows - 1, y_HCN_inf(V_rest));
    }

    void set_index(int idx) {
        index = idx;
    }

    double alpha_n_K_fast(double v) {
        return -0.01 * (v - 5.0 + 55.0) / (std::exp(-0.1 * (v - 5.0 + 55.0)) - 1) * 1 / 5.0;
    }

    double beta_n_K_fast(double v) {
        return 0.125 * std::exp((v - 5.0 + 65.0) / -80.0) * 1 / 5.0;
    }

    double alpha_n_K_slow(double v) {
        return -0.01 * (v - 0.0 + 55.0) / (std::exp(-0.1 * (v - 0.0 + 55.0)) - 1) * 1 / 8.0;
    }

    double beta_n_K_slow(double v) {
        return 0.125 * std::exp((v - 0.0 + 65.0) / -80.0) * 1 / 8.0;
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
        return 1.0 / (1.0 + std::exp(-(v + 27.2) / 4.9));
    }

    double h_Na_inf(double v) {
        return 1.0 / (1.0 + std::exp((v + 60.7) / 7.7));
    }

    double s_Na_inf(double v) {
        return 1.0 / (1.0 + std::exp((v + 60.1) / 5.4));
    }

    double tau_m_Na(double /*v*/) {
        return 0.15;
    }

    double tau_h_Na(double v) {
        return 0.25 * (20.1 * std::exp(-0.5 * std::pow((v + 61.4) / 32.7, 2)));
    }

    double tau_s_Na(double v) {
        return 1.0 * (1000 * (106.7 * std::exp(-0.5 * std::pow((v + 52.7) / 18.3, 2))));
    }

    double m_Ca_T_inf(double v) {
        return 1.0 / (1.0 + std::exp(-(v + 57.0) / 6.2));
    }

    double h_Ca_T_inf(double v) {
        return 1.0 / (1.0 + std::exp((v + 81.0) / 4.0));
    }

    double tau_m_Ca_T(double v) {
        return (0.612 + 1.0 / (std::exp(-(v + 132.0) / 16.7) + std::exp((v + 16.8) / 18.2)));
    }

    std::vector<double> tau_h_Ca_T(const std::vector<double>& v) {
        std::vector<double> result(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            if (v[i] > -81.0) {
                result[i] = 28.0 + std::exp(-(v[i] + 22.0) / 10.5);
            } else {
                result[i] = std::exp((v[i] + 467.0) / 66.6);
            }
        }
        return result;
    }

    double alpha_y_HCN(double v) {
        return std::exp(-(v + 23.0) / 20.0);
    }

    double beta_y_HCN(double v) {
        return std::exp((v + 130.0) / 10.0);
    }

    double y_HCN_inf(double v) {
        return alpha_y_HCN(v) / (alpha_y_HCN(v) + beta_y_HCN(v));
    }

    double tau_y_HCN(double v) {
        return 1.0 / (alpha_y_HCN(v) + beta_y_HCN(v));
    }

    double alpha_c_Ca_L(double v) {
        if (v == v) { // For single double inputs
            return -0.4 * (v + 10.0 + 70.0) / (-1.0 + std::exp(-0.1 * (v + 18.0 + 70.0)));
        }
        // For vector handling, implement overload or handle separately
        return 0.0; 
    }

    double beta_c_Ca_L(double v) {
        return 10.0 * std::exp(-(v - 38.0 + 38.0) / 12.6);
    }

    double c_Ca_L_inf(double v) {
        return alpha_c_Ca_L(v) / (alpha_c_Ca_L(v) + beta_c_Ca_L(v));
    }

    double tau_c_Ca_L(double v) {
        return 1.0 / (alpha_c_Ca_L(v) + beta_c_Ca_L(v));
    }

    bool spiked() {
        return Vm(0) > V_thresh;
    }

    std::map<std::string, Eigen::VectorXd> step(double curr_input, double dt = 0.01) {
    // Convert input current from uA to uA/cm^2
    double curr_density = curr_input / surfaces[0];

    Eigen::VectorXd vm_current = Vm; // Assuming Vm is a vector

    // Channel states
    auto& m_Na = channel_states["m_Na"];
    auto& h_Na = channel_states["h_Na"];
    auto& s_Na = channel_states["s_Na"];
    auto& m_Ca_T = channel_states["m_Ca_T"];
    auto& h_Ca_T = channel_states["h_Ca_T"];
    auto& n_K_fast = channel_states["n_K_fast"];
    auto& n_K_slow = channel_states["n_K_slow"];
    auto& c_Ca_L = channel_states["c_Ca_L"];
    auto& y_HCN = channel_states["y_HCN"];
    
    // Calculate channel conductances
    Eigen::VectorXd g_Na_asl = gbar_Na_vec.cwiseProduct(m_Na.array().pow(3) * h_Na.array() * s_Na.array());
    Eigen::VectorXd g_ca_asl = gbar_ca_vec.cwiseProduct(m_Ca_T.array().pow(2) * h_Ca_T.array());
    Eigen::VectorXd g_kd_asl = gbar_kd_vec.cwiseProduct(n_K_fast.array().pow(4));
    Eigen::VectorXd g_k7_asl = gbar_k7_vec.cwiseProduct(n_K_slow.array().pow(4));
    Eigen::VectorXd g_ct_asl = gbar_ct_vec.cwiseProduct(c_Ca_L.array().pow(3));
    Eigen::VectorXd g_HC_asl = gbar_HC_vec.cwiseProduct(y_HCN.array());
    Eigen::VectorXd g_l_asl = gbar_l_vec;

    // Update channel states using Euler method
    Eigen::VectorXd m_Na_next = (m_Na + (dt * K_m_Na * m_Na_inf(vm_current) / tau_m_Na(vm_current))) /
                                 (1.0 + K_m_Na * dt / tau_m_Na(vm_current));

    Eigen::VectorXd h_Na_next = (h_Na + (dt * K_h_Na * h_Na_inf(vm_current) / tau_h_Na(vm_current))) /
                                 (1.0 + K_h_Na * dt / tau_h_Na(vm_current));

    Eigen::VectorXd s_Na_next = (s_Na + (dt * K_s_Na * s_Na_inf(vm_current) / tau_s_Na(vm_current))) /
                                 (1.0 + K_s_Na * dt / tau_s_Na(vm_current));

    Eigen::VectorXd m_Ca_T_next = (m_Ca_T + (dt * K_m_Ca_T * m_Ca_T_inf(vm_current) / tau_m_Ca_T(vm_current))) /
                                   (1.0 + K_m_Ca_T * dt / tau_m_Ca_T(vm_current));

    Eigen::VectorXd h_Ca_T_next = (h_Ca_T + (dt * K_h_Ca_T * h_Ca_T_inf(vm_current) / tau_h_Ca_T(vm_current))) /
                                   (1.0 + K_h_Ca_T * dt / tau_h_Ca_T(vm_current));

    Eigen::VectorXd n_K_fast_next = (n_K_fast + (dt * K_n_K_fast * n_K_fast_inf(vm_current) / tau_n_K_fast(vm_current))) /
                                     (1.0 + K_n_K_fast * dt / tau_n_K_fast(vm_current));

    Eigen::VectorXd n_K_slow_next = (n_K_slow + (dt * K_n_K_slow * n_K_slow_inf(vm_current) / tau_n_K_slow(vm_current))) /
                                     (1.0 + K_n_K_slow * dt / tau_n_K_slow(vm_current));

    Eigen::VectorXd c_Ca_L_next = (c_Ca_L + (dt * K_c_Ca_L * c_Ca_L_inf(vm_current) / tau_c_Ca_L(vm_current))) /
                                   (1.0 + K_c_Ca_L * dt / tau_c_Ca_L(vm_current));

    Eigen::VectorXd y_HCN_next = (y_HCN + (dt * K_y_HCN * y_HCN_inf(vm_current) / tau_y_HCN(vm_current))) /
                                  (1.0 + K_y_HCN * dt / tau_y_HCN(vm_current));

    // Calculate ionic currents
    Eigen::VectorXd I_Na = g_Na_asl.array() * (vm_current.array() - E_Na);
    Eigen::VectorXd I_Ca_T = g_ca_asl.array() * (vm_current.array() - E_ca);
    Eigen::VectorXd I_K_fast = g_kd_asl.array() * (vm_current.array() - E_po);
    Eigen::VectorXd I_K_slow = g_k7_asl.array() * (vm_current.array() - E_po);
    Eigen::VectorXd I_leak = g_l_asl.array() * (vm_current.array() - E_l);
    Eigen::VectorXd I_Ca_L = g_ct_asl.array() * (vm_current.array() - E_ct);
    Eigen::VectorXd I_HCN = g_HC_asl.array() * ((vm_current.array() - E_Na) + (vm_current.array() - E_po));
    
    // Calculate voltage changes
    Eigen::VectorXd dV = -(I_Na + I_Ca_T + I_HCN + I_Ca_L + I_K_fast + I_K_slow + I_leak);
    dV(0) += curr_density;  // Adjust for current density

    // Calculate next voltage state
    Eigen::MatrixXd Inv_Con_Mat(4, 4);
    Inv_Con_Mat << 9.27335478e-01, 2.03249095e-02, 5.17538740e-02, 5.85738801e-04,
                   5.21539472e-02, 9.18466316e-01, 2.91067136e-03, 2.64690654e-02,
                   8.81257362e-01, 1.93149907e-02, 9.88710132e-02, 5.56634188e-04,
                   1.55727486e-03, 2.74246646e-02, 8.69103028e-05, 9.70931150e-01;

    Eigen::VectorXd vm_next = Inv_Con_Mat * (vm_current + dt * dV / Cm);

    // Package channel states for return
    std::map<std::string, Eigen::VectorXd> channel_states_next;
    channel_states_next["m_Na"] = m_Na_next;
    channel_states_next["h_Na"] = h_Na_next;
    channel_states_next["s_Na"] = s_Na_next;
    channel_states_next["m_Ca_T"] = m_Ca_T_next;
    channel_states_next["h_Ca_T"] = h_Ca_T_next;
    channel_states_next["n_K_fast"] = n_K_fast_next;
    channel_states_next["n_K_slow"] = n_K_slow_next;
    channel_states_next["c_Ca_L"] = c_Ca_L_next;
    channel_states_next["y_HCN"] = y_HCN_next;

    // Package currents for return (convert to uA)
    std::map<std::string, double> currents;
    double surfaces_sum = std::accumulate(surfaces.begin(), surfaces.end(), 0.0);
    currents["i_na"] = (I_Na.sum() * 1e6 * surfaces_sum);
    currents["i_ca_t"] = (I_Ca_T.sum() * 1e6 * surfaces_sum);
    currents["i_k_fast"] = (I_K_fast.sum() * 1e6 * surfaces_sum);
    currents["i_k_slow"] = (I_K_slow.sum() * 1e6 * surfaces_sum);
    currents["i_leak"] = (I_leak.sum() * 1e6 * surfaces_sum);
    currents["i_ca_l"] = (I_Ca_L.sum() * 1e6 * surfaces_sum);
    currents["i_hcn"] = (I_HCN.sum() * 1e6 * surfaces_sum);

    // Update state
    Vm = vm_next;
    channel_states = channel_states_next;

    return currents;
}
};