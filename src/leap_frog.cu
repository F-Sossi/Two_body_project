
#include "leap_frog.h"
#include <cuda_runtime.h>
#include <cmath>
#include <random>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

constexpr int BLOCK_SIZE_LEAP = 1024;


__global__ void calculate_halfstep_velocity(int num_bodies, double dt, const double *d_forces, double *d_velocities)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_bodies * 3)
    {
        d_velocities[i] += 0.5 * d_forces[i] * dt;
    }
}

__global__ void update_positions(int num_bodies, double dt, const double *d_velocities, double *d_positions)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_bodies * 3)
    {
        d_positions[i] += d_velocities[i] * dt;
    }
}

__global__ void update_velocities(int num_bodies, double dt, const double *d_forces, double *d_velocities)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_bodies * 3)
    {
        d_velocities[i] += d_forces[i] * dt;
    }
}

__global__ void calculate_forces(int num_bodies, const double *d_positions, const double *d_masses, double *d_forces)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_bodies * 3)
    {
        double fx = 0.0, fy = 0.0, fz = 0.0;
        double xi = d_positions[3 * i], yi = d_positions[3 * i + 1], zi = d_positions[3 * i + 2];
        double mi = d_masses[i / 3];
        for (int j = 0; j < num_bodies; j++)
        {
            if (j != i / 3)
            {
                double xj = d_positions[3 * j], yj = d_positions[3 * j + 1], zj = d_positions[3 * j + 2];
                double mj = d_masses[j];
                double dx = xj - xi, dy = yj - yi, dz = zj - zi;
                double dist = sqrt(dx * dx + dy * dy + dz * dz);
                double f = G * mi * mj / (dist * dist * dist);
                fx += f * dx;
                fy += f * dy;
                fz += f * dz;
            }
        }
        d_forces[3 * i] = fx;
        d_forces[3 * i + 1] = fy;
        d_forces[3 * i + 2] = fz;
    }
}

LeapFrogIntegrator::LeapFrogIntegrator(int num_bodies)
{
    // Initialize particle positions.
    std::default_random_engine rng;
    std::uniform_real_distribution<double> distribution(1000.0, 10000.0);
    positions.resize(num_bodies * 3);
    for (int i = 0; i < num_bodies; i++)
    {
        positions[i * 3] = distribution(rng);     // x position of ith particle
        positions[i * 3 + 1] = distribution(rng); // y position of ith particle
        positions[i * 3 + 2] = distribution(rng); // z position of ith particle
    }

    // Initialize particle velocities.
    std::normal_distribution<double> velocity_distribution(0.0, 5000.0);
    velocities.resize(num_bodies * 3);
    for (int i = 0; i < num_bodies; i++)
    {
        velocities[i * 3] = velocity_distribution(rng);     // x velocity of ith particle
        velocities[i * 3 + 1] = velocity_distribution(rng); // y velocity of ith particle
        velocities[i * 3 + 2] = velocity_distribution(rng); // z velocity of ith particle
    }

    // Initialize particle masses.
    std::uniform_real_distribution<double> mass_distribution(1000.0, 10000.0);
    masses.resize(num_bodies);
    for (int i = 0; i < num_bodies; i++)
    {
        masses[i] = mass_distribution(rng);
    }

    // Initialize particle forces.
    forces.resize(num_bodies * 3);
    std::fill(forces.begin(), forces.end(), 0.0);
}

void LeapFrogIntegrator::step(int num_steps, double dt)
{
    // Allocate device memory.
    double *d_positions, *d_velocities, *d_forces, *d_masses;
    cudaMalloc(&d_positions, positions.size() * sizeof(double));
    cudaMalloc(&d_velocities, velocities.size() * sizeof(double));
    cudaMalloc(&d_forces, forces.size() * sizeof(double));
    cudaMalloc(&d_masses, masses.size() * sizeof(double));

    // Copy initial data to device.
    cudaMemcpy(d_positions, positions.data(), positions.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocities, velocities.data(), velocities.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_forces, forces.data(), forces.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_masses, masses.data(), masses.size() * sizeof(double), cudaMemcpyHostToDevice);

    // Set up kernel launch configuration.
    int block_size = BLOCK_SIZE_LEAP;
    int num_blocks = (positions.size() + block_size - 1) / block_size;


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
    cudaMemcpy(positions.data(), d_positions, positions.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(velocities.data(), d_velocities, velocities.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(forces.data(), d_forces, forces.size() * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory.
    cudaFree(d_positions);
    cudaFree(d_velocities);
    cudaFree(d_forces);
    cudaFree(d_masses);
}

void LeapFrogIntegrator::write_positions_to_file(const std::string& filename, int step) {
    // Open file for writing particle positions.
    std::ofstream output_file(filename);
    
    // Copy particle positions to host.
    cudaMemcpy(positions.data(), d_positions, positions.size() * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Write particle positions to file.
    for (size_t i = 0; i < positions.size(); i += 3) {
        output_file << positions[i] << " " << positions[i+1] << " " << positions[i+2] << std::endl;
    }
    
    // Close output file.
    output_file.close();
}

std::vector<double> LeapFrogIntegrator::get_positions() const
{
    return positions;
}