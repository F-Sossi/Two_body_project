//---------------------------------------------------------------------------
// verlet.cu - code file for verlet.h
// Author: Frank Sossi
// Author: Amalaye Oyake
//
// File contains:
// 1. VerletIntegrator class
// 2. VerletIntegrator constructor
// 3. VerletIntegrator step function
// 4. VerletIntegrator get_positions function
// 5. VerletIntegrator write_positions_to_file function
// 6. update_positions_ver Kernel
// 7. calculate_forces_ver Kernel
// 8. calculate_velocities_ver Kernel
//---------------------------------------------------------------------------
#include "verlet.h"
#include <cuda_runtime.h>
#include <vector_types.h>

constexpr int BLOCK_SIZE_VER = 512;

//---------------------------------------------------------------------------
// Kernel function to update the positions of the bodies using the verlet method
// Input:
//      num_bodies - number of bodies in the simulation
//      dt - time step
//      d_velocities - array of velocities of the bodies
//      d_forces - array of forces on the bodies
//      d_positions - array of positions of the bodies
//      d_masses - array of masses of the bodies
// Output: none
//---------------------------------------------------------------------------
__global__ void update_positions_ver(int num_bodies, float dt, const float *d_velocities, const float *d_forces,
                                     float *d_positions, const float *d_masses)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // For each body being updated by this thread
    for (int i = tid; i < num_bodies; i += stride) {

        // get the position and of the body
        float px = d_positions[i * 3];
        float py = d_positions[i * 3 + 1];
        float pz = d_positions[i * 3 + 2];

        // get the velocity of the body
        float vx = d_velocities[i * 3];
        float vy = d_velocities[i * 3 + 1];
        float vz = d_velocities[i * 3 + 2];

        // get the force of the body
        float fx = d_forces[i * 3];
        float fy = d_forces[i * 3 + 1];
        float fz = d_forces[i * 3 + 2];

        // get the mass of the body
        float m = d_masses[i];

        // Update the position of the body using the verlet method
        px += vx * dt + 0.5f * fx / m * dt * dt;
        py += vy * dt + 0.5f * fy / m * dt * dt;
        pz += vz * dt + 0.5f * fz / m * dt * dt;

        // Update the position of the body
        d_positions[i * 3] = px;
        d_positions[i * 3 + 1] = py;
        d_positions[i * 3 + 2] = pz;
    }
}

//---------------------------------------------------------------------------
// Kernel function to calculate the forces on the bodies using the verlet method
// Input:
//      num_bodies - number of bodies in the simulation
//      d_positions - array of positions of the bodies
//      d_masses - array of masses of the bodies
//      d_forces - array of forces on the bodies
// Output: none
//---------------------------------------------------------------------------
__global__ void calculate_forces_ver(int num_bodies, const float *d_positions, const float *d_masses, float *d_forces)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // For each body being updated by this thread
    for (int i = tid; i < num_bodies; i += stride) {
        float fx = 0.0f;
        float fy = 0.0f;
        float fz = 0.0f;

        // get the position and mass of the body
        float xi = d_positions[i * 3];
        float yi = d_positions[i * 3 + 1];
        float zi = d_positions[i * 3 + 2];
        float mi = d_masses[i];

        // Loop over all other bodies
        for (int j = 0; j < num_bodies; j++) {
            // Skip self-interaction
            if (i == j) {
                continue;
            }

            // Get the position and mass of the other body
            float xj = d_positions[j * 3];
            float yj = d_positions[j * 3 + 1];
            float zj = d_positions[j * 3 + 2];
            float mj = d_masses[j];

            // Calculate the distance between the bodies
            float dx = xj - xi;
            float dy = yj - yi;
            float dz = zj - zi;

            // Calculate the force between the bodies
            float dist = sqrtf(dx*dx + dy*dy + dz*dz);
            float dist_cubed = dist * dist * dist;

            // Calculate the force accumulated on this body
            fx += VER_G * mj * dx / dist_cubed;
            fy += VER_G * mj * dy / dist_cubed;
            fz += VER_G * mj * dz / dist_cubed;
        }

        // Store the force for this body
        d_forces[i * 3] = fx;
        d_forces[i * 3 + 1] = fy;
        d_forces[i * 3 + 2] = fz;
    }
}

//---------------------------------------------------------------------------
// Kernel function to calculate the velocities of the bodies using the verlet method
// Input:
//      num_bodies - number of bodies in the simulation
//      dt - time step
//      d_forces - array of forces on the bodies
//      d_old_forces - array of forces on the bodies from the previous time step
//      d_masses - array of masses of the bodies
//      d_velocities - array of velocities of the bodies
// Output: none
//---------------------------------------------------------------------------
__global__ void calculate_velocities_ver(int num_bodies, float dt, const float *d_forces, const float *d_old_forces,
                                         const float *d_masses, float *d_velocities)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // Loop over the bodies for this thread
    for (int i = tid; i < num_bodies; i += stride) {

        // Get the forces and velocities for this body
        float vx = d_velocities[i * 3];
        float vy = d_velocities[i * 3 + 1];
        float vz = d_velocities[i * 3 + 2];

        // Get the forces for this body
        float fx = d_forces[i * 3];
        float fy = d_forces[i * 3 + 1];
        float fz = d_forces[i * 3 + 2];

        // Get the old forces for this body
        float fx_old = d_old_forces[i * 3];
        float fy_old = d_old_forces[i * 3 + 1];
        float fz_old = d_old_forces[i * 3 + 2];

        // Get the mass for this body
        float m = d_masses[i];

        // Update the velocities using the Verlet method
        vx += 0.5f * (fx_old + fx) / m * dt;
        vy += 0.5f * (fy_old + fy) / m * dt;
        vz += 0.5f * (fz_old + fz) / m * dt;


        // Store the new velocities
        d_velocities[i * 3] = vx;
        d_velocities[i * 3 + 1] = vy;
        d_velocities[i * 3 + 2] = vz;
    }
}

//---------------------------------------------------------------------------
// VerletIntegrator::step method - performs a passed number fo steps of the
//                   simulation
// Input:
//      num_steps - number of steps to perform
//      dt - time step
// Output: none
//---------------------------------------------------------------------------
void VerletIntegrator::step(int num_steps, float dt)
{
    // Allocate device memory.
    float *d_positions, *d_velocities, *d_forces, *d_old_forces, *d_masses;
    cudaMalloc(&d_positions, positions.size() * sizeof(float));
    cudaMalloc(&d_velocities, velocities.size() * sizeof(float));
    cudaMalloc(&d_forces, forces.size() * sizeof(float));
    cudaMalloc(&d_old_forces, forces.size() * sizeof(float));
    cudaMalloc(&d_masses, masses.size() * sizeof(float));

    // Copy initial data to device.
    cudaMemcpy(d_positions, positions.data(), positions.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocities, velocities.data(), velocities.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_forces, forces.data(), forces.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_old_forces, forces.data(), forces.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_masses, masses.data(), masses.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Set up kernel launch configuration.
    int block_size = BLOCK_SIZE_VER;
    int num_blocks = (num_bodies + block_size - 1) / block_size;

    // Main simulation loop.
    for (int step = 0; step < num_steps; step++)
    {
        // Calculate forces at old positions.
        calculate_forces_ver<<<num_blocks, block_size>>>(num_bodies, d_positions, d_masses, d_forces);

        // Update positions using Verlet method.
        update_positions_ver<<<num_blocks, block_size>>>(num_bodies, dt, d_velocities, d_forces, d_positions, d_masses);

        // Calculate forces at new positions.
        calculate_forces_ver<<<num_blocks, block_size>>>(num_bodies, d_positions, d_masses, d_forces);

        // Update velocities using Verlet method.
        calculate_velocities_ver<<<num_blocks, block_size>>>(num_bodies, dt, d_forces, d_old_forces, d_masses, d_velocities);


        // Swap forces.
        float *temp = d_forces;
        d_forces = d_old_forces;
        d_old_forces = temp;

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
    cudaFree(d_old_forces);
    cudaFree(d_masses);
}

//---------------------------------------------------------------------------
// VerletIntegrator::Constructor method - initializes the bodies in the
//                   simulation
// Input:
//      num_bodies - number of bodies to simulate
// Output: none
//---------------------------------------------------------------------------
VerletIntegrator::VerletIntegrator(int num_bodies)
        : num_bodies(num_bodies)
{
    // Initialize particle positions.
    std::default_random_engine rng;
    std::uniform_real_distribution<float> distribution(1000.0, 10000.0);
    positions.resize(num_bodies * 2);
    for (int i = 0; i < num_bodies; i++)
    {
        positions[i * 2] = distribution(rng);     // x position of ith particle
        positions[i * 2 + 1] = distribution(rng); // y position of ith particle
    }

    // Initialize particle velocities.
    std::normal_distribution<float> velocity_distribution(0.0, 5000.0);
    velocities.resize(num_bodies * 2);
    for (int i = 0; i < num_bodies; i++)
    {
        velocities[i * 2] = velocity_distribution(rng);     // x velocity of ith particle
        velocities[i * 2 + 1] = velocity_distribution(rng); // y velocity of ith particle
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
// VerletIntegrator::write_positions_to_file method - writes the positions
//                   of the bodies to a file for visualization
// Input:
//      filename - name of the file to write to
//      step - current step of the simulation
// Output: none
//---------------------------------------------------------------------------
void VerletIntegrator::write_positions_to_file(const std::string& filename, int step)
{
    // Open file for writing particle positions.
    std::ofstream output_file(filename);

    // Copy particle positions to host.
    cudaMemcpy(positions.data(), d_positions, positions.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Write particle positions to file.
    for (size_t i = 0; i < positions.size(); i += 2) {
        output_file << positions[i] << " " << positions[i+1] << " " << positions[i+2] << std::endl;
    }

    // Close output file.
    output_file.close();
}

//---------------------------------------------------------------------------
// VerletIntegrator::get_positions method - returns the positions of the
//                   bodies
// Input: none
// Output: positions of the bodies in a float vector
//---------------------------------------------------------------------------
std::vector<float> VerletIntegrator::get_positions() const
{
    return positions;
}



