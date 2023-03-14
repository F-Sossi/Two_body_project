#include "verlet.h"
#include <cuda_runtime.h>
#include <vector_types.h>

constexpr int BLOCK_SIZE_VER = 256;


__global__ void update_positions_ver(int num_bodies, float dt, const float *d_velocities, const float *d_forces, float *d_positions, const float *d_masses)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < num_bodies; i += stride) {
        float px = d_positions[i * 3];
        float py = d_positions[i * 3 + 1];
        float pz = d_positions[i * 3 + 2];

        float vx = d_velocities[i * 3];
        float vy = d_velocities[i * 3 + 1];
        float vz = d_velocities[i * 3 + 2];

        float fx = d_forces[i * 3];
        float fy = d_forces[i * 3 + 1];
        float fz = d_forces[i * 3 + 2];

        float m = d_masses[i];

        px += vx * dt + 0.5f * fx / m * dt * dt;
        py += vy * dt + 0.5f * fy / m * dt * dt;
        pz += vz * dt + 0.5f * fz / m * dt * dt;

        d_positions[i * 3] = px;
        d_positions[i * 3 + 1] = py;
        d_positions[i * 3 + 2] = pz;
    }
}

__global__ void calculate_forces_ver(int num_bodies, const float *d_positions, const float *d_masses, float *d_forces)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < num_bodies; i += stride) {
        float fx = 0.0f;
        float fy = 0.0f;
        float fz = 0.0f;

        float xi = d_positions[i * 3];
        float yi = d_positions[i * 3 + 1];
        float zi = d_positions[i * 3 + 2];
        float mi = d_masses[i];

        for (int j = 0; j < num_bodies; j++) {
            if (i == j) {
                continue;
            }

            float xj = d_positions[j * 3];
            float yj = d_positions[j * 3 + 1];
            float zj = d_positions[j * 3 + 2];
            float mj = d_masses[j];

            float dx = xj - xi;
            float dy = yj - yi;
            float dz = zj - zi;

            float dist = sqrtf(dx*dx + dy*dy + dz*dz);
            float dist_cubed = dist * dist * dist;

            fx += VER_G * mj * dx / dist_cubed;
            fy += VER_G * mj * dy / dist_cubed;
            fz += VER_G * mj * dz / dist_cubed;
        }

        d_forces[i * 3] = fx;
        d_forces[i * 3 + 1] = fy;
        d_forces[i * 3 + 2] = fz;
    }
}


__global__ void calculate_velocities_ver(int num_bodies, float dt, const float *d_forces, const float *d_old_forces, const float *d_masses, float *d_velocities)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < num_bodies; i += stride) {
        float vx = d_velocities[i * 3];
        float vy = d_velocities[i * 3 + 1];
        float vz = d_velocities[i * 3 + 2];

        float fx = d_forces[i * 3];
        float fy = d_forces[i * 3 + 1];
        float fz = d_forces[i * 3 + 2];

        float fx_old = d_old_forces[i * 3];
        float fy_old = d_old_forces[i * 3 + 1];
        float fz_old = d_old_forces[i * 3 + 2];

        float m = d_masses[i];

        vx += 0.5f * (fx_old + fx) / m * dt;
        vy += 0.5f * (fy_old + fy) / m * dt;
        vz += 0.5f * (fz_old + fz) / m * dt;

        d_velocities[i * 3] = vx;
        d_velocities[i * 3 + 1] = vy;
        d_velocities[i * 3 + 2] = vz;
    }
}


//__global__ void calculate_forces_ver(int num_bodies, const float *d_positions, const float *d_masses, float *d_forces)
//{
//    __shared__ float s_positions[BLOCK_SIZE_VER * 3];
//    __shared__ float s_masses[BLOCK_SIZE_VER];
//
//    int i = blockIdx.x * blockDim.x + threadIdx.x;
//    float xi = d_positions[3 * i], yi = d_positions[3 * i + 1], zi = d_positions[3 * i + 2];
//    float mi = d_masses[i / 3];
//
//    float fx = 0.0, fy = 0.0, fz = 0.0;
//    for (int j = 0; j < num_bodies; j += blockDim.x * gridDim.x) {
//        // Load positions and masses into shared memory.
//        int k = j + threadIdx.x;
//        if (k < num_bodies) {
//            s_positions[3 * threadIdx.x] = d_positions[3 * k];
//            s_positions[3 * threadIdx.x + 1] = d_positions[3 * k + 1];
//            s_positions[3 * threadIdx.x + 2] = d_positions[3 * k + 2];
//            s_masses[threadIdx.x] = d_masses[k / 3];
//        }
//        __syncthreads();
//
//        // Compute forces using shared memory for positions and masses.
//        for (int l = 0; l < blockDim.x && j + l < num_bodies; l++) {
//            if (i != j + l) {
//                float xj = s_positions[3 * l], yj = s_positions[3 * l + 1], zj = s_positions[3 * l + 2];
//                float mj = s_masses[l];
//                float dx = xj - xi, dy = yj - yi, dz = zj - zi;
//                float dist = sqrt(dx * dx + dy * dy + dz * dz);
//                float f = VER_G * mi * mj / (dist * dist * dist);
//                fx += f * dx;
//                fy += f * dy;
//                fz += f * dz;
//            }
//        }
//        __syncthreads();
//    }
//    d_forces[3 * i] = fx;
//    d_forces[3 * i + 1] = fy;
//    d_forces[3 * i + 2] = fz;
//}

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
    int block_size = 256;
    int num_blocks = (num_bodies + block_size - 1) / block_size;

    // Main simulation loop.
    for (int step = 0; step < num_steps; step++)
    {
        // Calculate forces at old positions.
        calculate_forces<<<num_blocks, block_size>>>(num_bodies, d_positions, d_masses, d_forces);

        // Update positions using Verlet method.
        update_positions_ver<<<num_blocks, block_size>>>(num_bodies, dt, d_velocities, d_forces, d_positions, d_masses);

        // Calculate forces at new positions.
        calculate_forces_ver<<<num_blocks, block_size>>>(num_bodies, d_positions, d_masses, d_forces);

        // Update velocities using Verlet method.
        calculate_velocities_ver<<<num_blocks, block_size>>>(num_bodies, dt, d_forces, d_old_forces, d_masses,
                                                             d_velocities);

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
    forces.resize(num_bodies * 2);
    std::fill(forces.begin(), forces.end(), 0.0);
}

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
std::vector<float> VerletIntegrator::get_positions() const
{
    return positions;
}



