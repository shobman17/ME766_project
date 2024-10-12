#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

#define NUM_NEURONS 10
#define V_REST -65.0
#define C_M 1.0
#define G_NA 120.0
#define G_K 36.0
#define G_L 0.3
#define E_NA 50.0
#define E_K -77.0
#define E_L -54.387

// Gating variable functions (same as before)
double alpha_n(double v) { return 0.01 * (v + 55.0) / (1.0 - exp(-(v + 55.0) / 10.0)); }
double beta_n(double v) { return 0.125 * exp(-(v + 65.0) / 80.0); }
double alpha_m(double v) { return 0.1 * (v + 40.0) / (1.0 - exp(-(v + 40.0) / 10.0)); }
double beta_m(double v) { return 4.0 * exp(-(v + 65.0) / 18.0); }
double alpha_h(double v) { return 0.07 * exp(-(v + 65.0) / 20.0); }
double beta_h(double v) { return 1.0 / (1.0 + exp(-(v + 35.0) / 10.0)); }

typedef struct {
    double v, n, m, h;
} Neuron;

typedef struct {
    Neuron neurons[NUM_NEURONS];
    double connections[NUM_NEURONS][NUM_NEURONS];
} Network;

void initialize_network(Network* net) {
    #pragma omp parallel for
    for (int i = 0; i < NUM_NEURONS; i++) {
        net->neurons[i] = (Neuron){V_REST, 0.0, 0.0, 0.0};
        for (int j = 0; j < NUM_NEURONS; j++) {
            net->connections[i][j] = (i == j) ? 0.0 : ((double)rand() / RAND_MAX) * 0.1;
        }
    }
}

void update_network(Network* net, double dt, double* i_ext) {
    #pragma omp parallel for
    for (int i = 0; i < NUM_NEURONS; i++) {
        Neuron* neuron = &net->neurons[i];
        double v = neuron->v;
        double n = neuron->n;
        double m = neuron->m;
        double h = neuron->h;

        double i_na = G_NA * m * m * m * h * (v - E_NA);
        double i_k = G_K * n * n * n * n * (v - E_K);
        double i_l = G_L * (v - E_L);

        double i_syn = 0.0;
        for (int j = 0; j < NUM_NEURONS; j++) {
            i_syn += net->connections[j][i] * (net->neurons[j].v - v);
        }

        neuron->v += dt * (i_ext[i] - i_na - i_k - i_l + i_syn) / C_M;
        neuron->n += dt * (alpha_n(v) * (1 - n) - beta_n(v) * n);
        neuron->m += dt * (alpha_m(v) * (1 - m) - beta_m(v) * m);
        neuron->h += dt * (alpha_h(v) * (1 - h) - beta_h(v) * h);
    }
}

int main() {
    Network net;
    initialize_network(&net);

    double t, dt = 0.01, t_max = 50.0;
    double i_ext[NUM_NEURONS] = {0};
    i_ext[0] = 10.0;  // Stimulate only the first neuron

    #pragma omp parallel
    {
        #pragma omp single
        for (t = 0; t < t_max; t += dt) {
            update_network(&net, dt, i_ext);
            
            #pragma omp critical
            {
                printf("%f", t);
                for (int i = 0; i < NUM_NEURONS; i++) {
                    printf(" %f", net.neurons[i].v);
                }
                printf("\n");
            }
        }
    }

    return 0;
}
