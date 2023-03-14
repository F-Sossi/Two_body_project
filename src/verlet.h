//---------------------------------------------------------------------------
// verlet.h - Header file for verlet.cu
// Author: Frank Sossi
// Author: Amalaye Oyake
//
// File contains:
// 1. VerletIntegrator class
//
//---------------------------------------------------------------------------
#pragma once
#include <cuda_runtime.h>
#include "cuda_runtime_api.h"
#include <cmath>
#include <random>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

// gravitational constant
const float VER_G = 6.6743e-11;

__global__ void update_positions_ver(int num_bodies, float dt, const float *d_velocities, const float *d_forces,
                                     float *d_positions, const float *d_masses);
__global__ void calculate_forces_ver(int num_bodies, const float *d_positions, const float *d_masses, float *d_forces);

__global__ void calculate_velocities_ver(int num_bodies, float dt, const float *d_forces, const float *d_old_forces,
                                         const float *d_masses, float *d_velocities);

class VerletIntegrator
{
public:
    VerletIntegrator(int num_bodies);
    void step(int num_steps, float dt);
    std::vector<float> get_positions() const;

private:
    std::vector<float> positions, velocities, forces, masses;
    float *d_positions, *d_velocities, *d_forces, *d_masses;
    int num_bodies;
    int write_freq = 1;

    void write_positions_to_file(const std::string& filename, int step);
};
