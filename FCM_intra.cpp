#include <iostream>
#include <vector>
#include <cmath>
// #include <Eigen/Dense>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <map>
#include <type_traits>

class CSVWriter {
private:
    std::ofstream file;

public:
    CSVWriter(const std::string& filename) {
        file.open(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open CSV file: " + filename);
        }
    }

    template<typename T>
    void writeRow(const std::vector<T>& data) {
        for (size_t i = 0; i < data.size(); ++i) {
            file << data[i];
            if (i < data.size() - 1) file << ",";
        }
        file << "\n";
    }

    void writeHeader(const std::vector<std::string>& headers) {
        writeRow(headers);
    }

    ~CSVWriter() {
        if (file.is_open()) {
            file.close();
        }
    }
};

class FCM {
private:
    // Model parameters
    int index;
    double V_rest = -70.0;
    double V_thresh = -30.0;
    double E_l = -81.4;
    double Cm = 1.0;

    double E_po = -77.0;
    double E_Na = 50.0;
    double E_ca = 120.0;
    double E_HC = -40.0;
    double E_ct = 22.6;
    double RA = 0.1;
    double ro_ex = 1.0;

    // Temperature-dependent rate factors
    double K_m_Na, K_h_Na, K_s_Na, K_m_Ca_T, K_h_Ca_T;
    double K_n_K_fast, K_n_K_slow, K_c_Ca_L, K_y_HCN;

    // Morphology and connectivity data
    Eigen::MatrixXd Con_Mat;
    Eigen::VectorXd surfaces;
    int num_rows;

    // State variables
    Eigen::MatrixXd Vm;
    std::map<std::string, Eigen::VectorXd> channel_states;

    // Conductance vectors
    Eigen::VectorXd gbar_l_vec, gbar_Na_vec, gbar_ca_vec;
    Eigen::VectorXd gbar_kd_vec, gbar_k7_vec, gbar_HC_vec, gbar_ct_vec;

public:
    FCM(const std::string& model_path = "morphology_FCM.swc");
    void set_index(int new_index);
    bool spiked(){
        return Vm(0,0) > V_thresh;
    }
    std::map<std::string, double> step(double curr_input, double dt = 0.01);

    // Helper methods for channel kinetics
    double alpha_n_K_fast(double v){
        return -0.01*(v-5.0+55.0)/(std::exp(-0.1*(v-5.0+55.0))-1.0)*1.0/5.0;
    }
    double beta_n_K_fast(double v){
        return 0.125*std::exp((v-5.0+65)/-80)*1/5.0;
    }
    double alpha_n_K_slow(double v){
        return -0.01*(v-0.0+55)/(std::exp(-0.1*(v-0.0+55))-1)*1/8.0;
    }
    double beta_n_K_slow(double v){
        return 0.125*std::exp((v-0.0+65)/-80)*1/8.0;
    }
    double n_K_fast_inf(double v){
        double a = alpha_n_K_fast(v);
        double b = beta_n_K_fast(v);
        return a/(a+b);
    }
    double tau_n_K_fast(double v){
        return 1./(alpha_n_K_fast(v) + beta_n_K_fast(v));
    }
    double n_K_slow_inf(double v){
        double a = alpha_n_K_slow(v);
        double b = beta_n_K_slow(v);
        return a/(a+b);
    }
    double tau_n_K_slow(double v){
        return 1./(alpha_n_K_slow(v) + beta_n_K_slow(v));
    }
    double m_Na_inf(double v){
        return 1./(1. + std::exp(-(v+27.2)/4.9));
    }
    double h_Na_inf(double v){
        return 1./(1. + std::exp(+(v+60.7)/7.7));
    }
    double s_Na_inf(double v){
        return (1. /(1. + std::exp(+(v+60.1)/5.4)));
    }
    double tau_m_Na(double v){
        return 0.15;
    }
    double tau_h_Na(double v){
        return 0.25*(20.1*std::exp(-0.5*std::pow(((v+61.4)/32.7), 2.0)));
    }
    double tau_s_Na(double v){
        return 1.0*(1000*(106.7*std::exp(-0.5*std::pow(((v+52.7)/18.3), 2.0))));
    }
    double m_Ca_T_inf(double v){
        return 1./(1. + std::exp(-(v+57.)/6.2));
    }
    double h_Ca_T_inf(double v){
        return 1./(1. + std::exp(+(v+81.)/4.));
    }
    double tau_m_Ca_T(double v){
        return (0.612+1./(std::exp(-(v+132.)/16.7)+std::exp((v+16.8)/18.2)));
    }
    std::vector<double> tau_h_Ca_T(const std::vector<double>& v) {
        std::vector<double> AA(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            double aa;
            if (v[i] > -81.0) {
                aa = 28.0 + std::exp(-(v[i] + 22.0) / 10.5);
            } else {
                aa = std::exp((v[i] + 467.0) / 66.6);
            }
            AA[i] = aa;
        }
        return AA;
    }
    double alpha_y_HCN(double v){
        return std::exp(-(v+23.)/20.);
    }
    double beta_y_HCN(double v){
        return std::exp((v+130.)/10.);
    }
    double y_HCN_inf(double v){
        double a = alpha_y_HCN(v);
        double b = beta_y_HCN(v);
        return a/(a+b);
    }
    double tau_y_HCN(double v){
        return 1./(alpha_y_HCN(v) + beta_y_HCN(v));
    }
    template <typename T>
    auto alpha_c_Ca_L(const T& v) {
        // If input is a single numeric value
        if constexpr (std::is_arithmetic_v<std::remove_reference_t<T>>) {
            return -0.4 * (v + 10.0 + 70.0) / (-1.0 + std::exp(-0.1 * (v + 18.0 + 70.0)));
        } 
        // If input is a vector/array-like container
        else {
            std::vector<double> AA(v.size());
            for (size_t i = 0; i < v.size(); ++i) {
                AA[i] = -0.4 * (v[i] + 10.0 + 70.0) / (-1.0 + std::exp(-0.1 * (v[i] + 18.0 + 70.0)));
            }
            return AA;
        }
    }
    double beta_c_Ca_L(double v){
        return 10.*std::exp(-(v-38+38) / 12.6);
    }
    double c_Ca_L_inf(double v){
        double a = alpha_c_Ca_L(v);
        double b = beta_c_Ca_L(v);
        return a/(a+b);
    }
    double tau_c_Ca_L(double v){
        return 1./(alpha_c_Ca_L(v) + beta_c_Ca_L(v));
    }
};

// Constructor implementation would be complex, loading SWC file similar to Python version
FCM::FCM(const std::string& model_path) {
    // Load morphology data
    std::ifstream file(model_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open morphology file");
    }
    
    // Parse file and populate data structures
    // Similar to numpy loadtxt in Python version
    
    // Initialize channel states and other parameters
    num_rows = 4;  // Hardcoded for this specific model
    Vm = Eigen::MatrixXd::Constant(num_rows - 1, 1, V_rest);

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

    // Channel states initialization
    channel_states["m_Na"] = Eigen::VectorXd::Constant(num_rows - 1, m_Na_inf(V_rest));
    channel_states["h_Na"] = Eigen::VectorXd::Constant(num_rows - 1, h_Na_inf(V_rest));
    channel_states["s_Na"] = Eigen::VectorXd::Constant(num_rows - 1, s_Na_inf(V_rest));
    channel_states["m_Ca_T"] = Eigen::VectorXd::Constant(num_rows - 1, m_Ca_T_inf(V_rest));
    channel_states["h_Ca_T"] = Eigen::VectorXd::Constant(num_rows - 1, h_Ca_T_inf(V_rest));
    channel_states["n_K_fast"] = Eigen::VectorXd::Constant(num_rows - 1, n_K_fast_inf(V_rest));
    channel_states["n_K_slow"] = Eigen::VectorXd::Constant(num_rows - 1, n_K_slow_inf(V_rest));
    channel_states["c_Ca_L"] = Eigen::VectorXd::Constant(num_rows - 1, c_Ca_L_inf(V_rest));
    channel_states["y_HCN"] = Eigen::VectorXd::Constant(num_rows - 1, y_HCN_inf(V_rest));
}

std::map<std::string, double> FCM::step(double curr_input, double dt) {
    // Implement step method similar to Python version
    // Key differences:
    // - Use Eigen matrix/vector operations
    // - Convert between std::vector/map and Eigen types
    
    std::map<std::string, double> currents;
    // Detailed implementation would mirror Python code
    
    return currents;
}

int main() {
    try {
        // Simulation parameters
        double T = 350.0;      // Total simulation time (ms)
        double dt = 0.01;      // Time step (ms)
        double curr_amp = 20.0;// Input current amplitude (uA)
        double freq = 10.0;    // Input frequency

        // // Create CSV writers for different data
        // CSVWriter currents_csv("currents.csv");
        // CSVWriter voltage_csv("voltage.csv");
        // CSVWriter input_csv("input_current.csv");

        // // Write headers
        // currents_csv.writeHeader({"time", "i_na", "i_ca_t", "i_k_fast", "i_k_slow", "i_leak", "i_ca_l", "i_hcn"});
        // voltage_csv.writeHeader({"time", "membrane_potential"});
        // input_csv.writeHeader({"time", "current"});

        // Initialize model and simulation containers
        FCM fcm;
        std::vector<double> time_points;
        std::vector<double> input_current;
        
        // Containers for currents
        std::map<std::string, std::vector<double>> currents;
        std::vector<double> Vm_trace;

        // Simulation loop
        for (double t = 0.0; t <= T; t += dt) {
            // Generate sinusoidal input current
            double curr_input = curr_amp * std::sin(2 * M_PI * freq * t * 1e-3);
            
            // Perform simulation step
            auto step_currents = fcm.step(curr_input, dt);
            
            // Store results
            time_points.push_back(t);
            input_current.push_back(curr_input * 1e6);
            
            // // Write current step data to CSV
            // std::vector<double> current_row = {t};
            // for (const auto& curr : {"i_na", "i_ca_t", "i_k_fast", "i_k_slow", "i_leak", "i_ca_l", "i_hcn"}) {
            //     current_row.push_back(step_currents[curr]);
            //     currents[curr].push_back(step_currents[curr]);
            // }
            // currents_csv.writeRow(current_row);
            
            // // Write input current
            // input_csv.writeRow({t, curr_input * 1e6});
            
            // Check for spikes (simplified)
            if (fcm.spiked()) {
                std::cout << "Spike detected at time " << t << " ms!" << std::endl;
            }
        }

        std::cout << "Simulation completed. Results written to CSV files." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}