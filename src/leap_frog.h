//---------------------------------------------------------------------------
// leap_frog.h - Header file for leap_frog.cu
// Author: Frank Sossi
// Author: Amalaye Oyake
//
// File contains:
// 1. LeapFrogIntegrator class
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

const float G = 6.6743e-11;

__global__ void calculate_halfstep_velocity(int num_bodies, float dt, const float *d_forces, float *d_velocities);

__global__ void update_positions(int num_bodies, float dt, const float *d_velocities, float *d_positions);

__global__ void update_velocities(int num_bodies, float dt, const float *d_forces, float *d_velocities);

__global__ void calculate_forces(int num_bodies, const float *d_positions, const float *d_masses, float *d_forces);

class LeapFrogIntegrator
{
public:
    LeapFrogIntegrator(int num_bodies);
    void step(int num_steps, float dt);
    std::vector<float> get_positions() const;

private:
    std::vector<float> positions, velocities, forces, masses;
    float *d_positions, *d_velocities, *d_forces, *d_masses;
    int num_bodies;
    int write_freq = 1;

    void write_positions_to_file(const std::string& filename, int step);
};
