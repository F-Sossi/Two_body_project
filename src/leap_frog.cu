//---------------------------------------------------------------------------
// leap_frog.cu - code file for leap_frog.h
// Author: Frank Sossi
// Author: Amalaye Oyake
//
// File contains:
// 1. LeapFrogIntegrator class
//  1.1 LeapFrogIntegrator constructor
//  1.2 LeapFrogIntegrator step function
//  1.3 LeapFrogIntegrator get_positions function
//  1.4 LeapFrogIntegrator write_positions_to_file function
// 2. calculate_halfstep_velocity kernel
// 3. update_positions_ver kernel
// 4. update_velocities kernel
// 5. calculate_forces kernel
//---------------------------------------------------------------------------
#include "leap_frog.h"
#include <cuda_runtime.h>
#include <cmath>
#include <random>
#include <iostream>
#include <fstream>
#include <vector>

constexpr int BLOCK_SIZE_LEAP = 1024; // optimal block size for leap frog integrator on 3050Ti

//---------------------------------------------------------------------------
// Kernel function to calculate the half step velocity
// Input: num_bodies - number of bodies in the simulation
//        dt - time step
//        d_forces - array of forces on each body
//        d_velocities - array of velocities of each body
// Output: none
//---------------------------------------------------------------------------
__global__ void calculate_halfstep_velocity(int num_bodies, float dt, const float *d_forces, float *d_velocities)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_bodies * 3)
    {
        d_velocities[i] += 0.5 * d_forces[i] * dt;
    }
}

//---------------------------------------------------------------------------
// Kernel function to update the positions of the bodies
// Input:
//        num_bodies - number of bodies in the simulation
//        dt - time step
//        d_velocities - array of velocities of each body
//        d_positions - array of positions of each body
// Output: none
//---------------------------------------------------------------------------
__global__ void update_positions(int num_bodies, float dt, const float *d_velocities, float *d_positions)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_bodies * 3)
    {
        d_positions[i] += d_velocities[i] * dt;
    }
}
//---------------------------------------------------------------------------
// Kernel function to update the velocities of the bodies
// Input:
//        num_bodies - number of bodies in the simulation
//        dt - time step
//        d_forces - array of forces on each body
//        d_velocities - array of velocities of each body
// Output: none
//---------------------------------------------------------------------------
__global__ void update_velocities(int num_bodies, float dt, const float *d_forces, float *d_velocities)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_bodies * 3)
    {
        d_velocities[i] += d_forces[i] * dt;
    }
}

//---------------------------------------------------------------------------
// Kernel function to calculate the forces on each body
// Input:
//        num_bodies - number of bodies in the simulation
//        d_positions - array of positions of each body
//        d_masses - array of masses of each body
//        d_forces - array of forces on each body
// Output: none
//---------------------------------------------------------------------------
__global__ void calculate_forces(int num_bodies, const float *d_positions, const float *d_masses, float *d_forces)
{
    __shared__ float s_positions[BLOCK_SIZE_LEAP * 3];
    __shared__ float s_masses[BLOCK_SIZE_LEAP];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float xi = d_positions[3 * i], yi = d_positions[3 * i + 1], zi = d_positions[3 * i + 2];
    float mi = d_masses[i / 3];

    float fx = 0.0, fy = 0.0, fz = 0.0;
    for (int j = 0; j < num_bodies; j += blockDim.x * gridDim.x) {
        // Load positions and masses into shared memory.
        int k = j + threadIdx.x;
        if (k < num_bodies) {
            s_positions[3 * threadIdx.x] = d_positions[3 * k];
            s_positions[3 * threadIdx.x + 1] = d_positions[3 * k + 1];
            s_positions[3 * threadIdx.x + 2] = d_positions[3 * k + 2];
            s_masses[threadIdx.x] = d_masses[k / 3];
        }
        __syncthreads();

        // Compute forces using shared memory for positions and masses.
        for (int l = 0; l < blockDim.x && j + l < num_bodies; l++) {
            if (i != j + l) {
                float xj = s_positions[3 * l], yj = s_positions[3 * l + 1], zj = s_positions[3 * l + 2];
                float mj = s_masses[l];
                float dx = xj - xi, dy = yj - yi, dz = zj - zi;
                float dist = sqrt(dx * dx + dy * dy + dz * dz);
                float f = G * mi * mj / (dist * dist * dist);
                fx += f * dx;
                fy += f * dy;
                fz += f * dz;
            }
        }
        __syncthreads();
    }
    d_forces[3 * i] = fx;
    d_forces[3 * i + 1] = fy;
    d_forces[3 * i + 2] = fz;
}

//---------------------------------------------------------------------------
// Constructor to initialize the LeapFrogIntegrator class
// Input:
//        num_bodies - number of bodies in the simulation
// Output:  none
//---------------------------------------------------------------------------
LeapFrogIntegrator::LeapFrogIntegrator(int num_bodies)
    : num_bodies(num_bodies)
{
    // Initialize particle positions.
    std::default_random_engine rng;
    std::uniform_real_distribution<float> distribution(1000.0, 10000.0);
    positions.resize(num_bodies * 3);
    for (int i = 0; i < num_bodies; i++)
    {
        positions[i * 3] = distribution(rng);     // x position of ith particle
        positions[i * 3 + 1] = distribution(rng); // y position of ith particle
        positions[i * 3 + 2] = distribution(rng); // z position of ith particle
    }

    // Initialize particle velocities.
    std::normal_distribution<float> velocity_distribution(0.0, 5000.0);
    velocities.resize(num_bodies * 3);
    for (int i = 0; i < num_bodies; i++)
    {
        velocities[i * 3] = velocity_distribution(rng);     // x velocity of ith particle
        velocities[i * 3 + 1] = velocity_distribution(rng); // y velocity of ith particle
        velocities[i * 3 + 2] = velocity_distribution(rng); // z velocity of ith particle
    }

    // Initialize particle masses.
    std::uniform_real_distribution<float> mass_distribution(1000.0, 10000.0);
    masses.resize(num_bodies);
    for (int i = 0; i < num_bodies; i++)
    {
        masses[i] = mass_distribution(rng);
    }

    // Initialize particle forces.
    forces.resize(num_bodies * 3);
    std::fill(forces.begin(), forces.end(), 0.0);
}

//---------------------------------------------------------------------------
// Method to perform a single step of the LeapFrog integrator
// Input:
//        num_steps - number of steps to perform
//        dt - time step
// Output: none
//---------------------------------------------------------------------------
void LeapFrogIntegrator::step(int num_steps, float dt)
{
    // Allocate device memory.
    float *d_positions, *d_velocities, *d_forces, *d_masses;
    cudaMalloc(&d_positions, positions.size() * sizeof(float));
    cudaMalloc(&d_velocities, velocities.size() * sizeof(float));
    cudaMalloc(&d_forces, forces.size() * sizeof(float));
    cudaMalloc(&d_masses, masses.size() * sizeof(float));

    // Copy initial data to device.
    cudaMemcpy(d_positions, positions.data(), positions.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocities, velocities.data(), velocities.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_forces, forces.data(), forces.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_masses, masses.data(), masses.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Set up kernel launch configuration.
    int block_size = BLOCK_SIZE_LEAP;
    int num_blocks = (num_bodies + block_size - 1) / block_size;

    // set up modified kernel launch configuration for calculate forces optimal for 3050ti
//    int block_size2 = 640;
//    int num_blocks2 = (positions.size() + block_size2 - 1) / block_size2;


    // Main simulation loop.
    for (int step = 0; step < num_steps; step++)
    {
        // Calculate half-step velocity.
        calculate_halfstep_velocity<<<num_blocks, block_size>>>(num_bodies, dt, d_forces, d_velocities);

        // Update positions.
        update_positions<<<num_blocks, block_size>>>(num_bodies, dt, d_velocities, d_positions);

        // Calculate forces at new positions.
        calculate_forces<<<num_blocks, block_size>>>(num_bodies, d_positions, d_masses, d_forces);

        // Update velocities with full-step forces.
        update_velocities<<<num_blocks, block_size>>>(num_bodies, dt, d_forces, d_velocities);

        // Write particle positions to file.
        if ((step+1) % write_freq == 0) {
            std::string filename = "../data/positions_" + std::to_string(step) + ".txt";
            write_positions_to_file(filename, step+1);
        }
    }

    // Copy final data back to host.
    cudaMemcpy(positions.data(), d_positions, positions.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(velocities.data(), d_velocities, velocities.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(forces.data(), d_forces, forces.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory.
    cudaFree(d_positions);
    cudaFree(d_velocities);
    cudaFree(d_forces);
    cudaFree(d_masses);
}

//---------------------------------------------------------------------------
// Method to write particle positions to file
// Input:
//        filename - name of file to write to
//        step - current step of the simulation
// Output: positionsX.txt file where X is the step number
//---------------------------------------------------------------------------
void LeapFrogIntegrator::write_positions_to_file(const std::string& filename, int step) {
    // Open file for writing particle positions.
    std::ofstream output_file(filename);
    
    // Copy particle positions to host.
    cudaMemcpy(positions.data(), d_positions, positions.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Write particle positions to file.
    for (size_t i = 0; i < positions.size(); i += 3) {
        output_file << positions[i] << " " << positions[i+1] << " " << positions[i+2] << std::endl;
    }
    
    // Close output file.
    output_file.close();
}

//---------------------------------------------------------------------------
// Method to get particle positions
// Input: none
// Output: vector of particle positions
//---------------------------------------------------------------------------
std::vector<float> LeapFrogIntegrator::get_positions() const
{
    return positions;
}

