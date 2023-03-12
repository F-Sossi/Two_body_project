#pragma once

#include <cuda_runtime.h>
#include "cuda_runtime_api.h"
#include <cmath>
#include <random>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

const double G = 6.6743e-11;

__global__ void calculate_halfstep_velocity(int num_bodies, double dt, const double *d_forces, double *d_velocities);

__global__ void update_positions(int num_bodies, double dt, const double *d_velocities, double *d_positions);

__global__ void update_velocities(int num_bodies, double dt, const double *d_forces, double *d_velocities);

__global__ void calculate_forces(int num_bodies, const double *d_positions, const double *d_masses, double *d_forces);

class LeapFrogIntegrator
{
public:
    LeapFrogIntegrator(int num_bodies);
    void step(int num_steps, double dt);
    std::vector<double> get_positions() const;

private:
    std::vector<double> positions, velocities, forces, masses;
    double *d_positions, *d_velocities, *d_forces, *d_masses;
    int num_bodies;
    int write_freq = 1;

    void write_positions_to_file(const std::string& filename, int step);
};
