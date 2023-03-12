#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <random>
#include <fstream>
#include <cuda_runtime.h>

constexpr float SOFTENING    = 0.00125f;
const float GRAVITY_CONSTANT = 6.67430e-11f;

struct Body
{
    float mass;
    float4 position;
    float4 velocity;
    float4 acceleration;
};

unsigned int getNumThreads(unsigned int n)
{
    unsigned int num_threads = 2;

    while((n > 2) && (num_threads < 1024))
    {
      n = n >> 1;
      num_threads   = num_threads << 1; 
    }

    return num_threads;
}

unsigned int getNumBlocks(unsigned int n)
{
    if (n % 1024 == 0)
    {
        return  n / 1024;
    }
    else
    {
        return n / 1024 + 1;
    }
}

void initBodies(Body *bodies, int numBodies) 
{
   for (int i = 0; i < numBodies; i++) 
    {
        bodies[i].position.x = 400.0f * (rand() / (float)RAND_MAX) - 400.0f;
        bodies[i].position.y = 400.0f * (rand() / (float)RAND_MAX) - 400.0f;
        bodies[i].position.z = 400.0f * (rand() / (float)RAND_MAX) - 400.0f;
        bodies[i].position.w = 1.0f;

        bodies[i].velocity.x = 1112.1f;
        bodies[i].velocity.y = 1112.1f;
        bodies[i].velocity.z = 1112.1f;
        bodies[i].velocity.w = 1112.1f;
    }
}

void initBodiesTest(Body *bodies, int numBodies) 
{
   for (int i = 0; i < numBodies; i++) 
    {
        bodies[i].position.x = 400.0f * i;
        bodies[i].position.y = 400.0f * i;
        bodies[i].position.z = 400.0f * i;
        bodies[i].position.w = 1.0f;

        bodies[i].velocity.x = 1100.1f;
        bodies[i].velocity.y = 200.1f;
        bodies[i].velocity.z = 30.1f;
        bodies[i].velocity.w = 0.1f;
        bodies[i].mass = 10.0f * i;
    }
}

void initBodies2(Body *bodies, int numBodies) 
{
    // Create random number generator for each item
    std::mt19937_64 gen(std::random_device{}());
    std::uniform_real_distribution<float> posDist(0.0f, 400.0f);
    std::uniform_real_distribution<float> velDist(0.0f, 500.1f);
    std::uniform_real_distribution<float> mass(0.0f, 10.0f);

    for (int i = 0; i < numBodies; i++) 
    {
        // Generate random values for position
        bodies[i].position.x = posDist(gen);
        bodies[i].position.y = posDist(gen);
        bodies[i].position.z = posDist(gen);
        bodies[i].position.w = 1.0f;

        // Generate random values for velocity
        bodies[i].velocity.x = velDist(gen);
        bodies[i].velocity.y = velDist(gen);
        bodies[i].velocity.z = velDist(gen);
        bodies[i].velocity.w = 0.1f;

        // Generate random value for mass
        bodies[i].mass = mass(gen) * i;
    }
}

__host__ __device__ float4 operator-(const float4& a, const float4& b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__global__
void integrate(Body *bodies, int numBodies, float deltaTime, float damping)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    unsigned int row = (index / numBodies); // 0 to N down
    unsigned int col = (index % numBodies); // 0 to N across

    // We will have (NxN) - ((N-1)x(N-1)) wasted threads
    // Also the worker thread has to be less than NxN
    if ((row != col) || (index < (numBodies * numBodies)))
    {
        float4 position      = bodies[row].position;
        float4 next_position = bodies[col].position;

        float4 delta_p = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 delta_v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 delta_acceleration = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        // Compute distance
        float4 distance  = next_position - position;

        // Compute distance squared
        float distance_squared = distance.x * distance.x + distance.y * distance.y + distance.z * distance.z;

        // Only calculate gravity if particles are close enough
        if (distance_squared < 10 * SOFTENING) 
        {
            // Calculate the force of gravity between the two particles reciprocal square root
            float invDist = rsqrtf(distance_squared + SOFTENING);

            // Calculate the force of gravity between the two particles
            float s = bodies[blockIdx.x].mass * powf(invDist, 3.0f);

            delta_acceleration.x = distance.x * s;
            delta_acceleration.y = distance.y * s;
            delta_acceleration.z = distance.z * s;
        }

        // Update the acceleration
        atomicAdd(&(bodies[blockIdx.x].acceleration.x), delta_acceleration.x);
        atomicAdd(&(bodies[blockIdx.x].acceleration.y), delta_acceleration.y);
        atomicAdd(&(bodies[blockIdx.x].acceleration.z), delta_acceleration.z);
        atomicAdd(&(bodies[blockIdx.x].acceleration.w), delta_acceleration.w);

        // Wait for completion
        __syncthreads();
    }
}

__global__
void update(Body *bodies_n0, Body *bodies_n1, float deltaTime)
{
    // The acceleration had been previously computed
    // among the N bodies. Now the velocity and position
    // must be updated.
    float4 acceleration = bodies_n0[blockIdx.x].acceleration;

    // Update velocity using acceleration and damping
    atomicAdd(&(bodies_n0[blockIdx.x].velocity.x), acceleration.x * deltaTime);
    atomicAdd(&(bodies_n0[blockIdx.x].velocity.y), acceleration.y * deltaTime);
    atomicAdd(&(bodies_n0[blockIdx.x].velocity.z), acceleration.z * deltaTime);
    atomicAdd(&(bodies_n0[blockIdx.x].velocity.w), 0);

    __syncthreads();

    float4 velocity = bodies_n0[blockIdx.x].velocity;

    atomicAdd(&(bodies_n0[blockIdx.x].velocity.x), velocity.x * deltaTime);
    atomicAdd(&(bodies_n0[blockIdx.x].velocity.y), velocity.y * deltaTime);
    atomicAdd(&(bodies_n0[blockIdx.x].velocity.z), velocity.z * deltaTime);
    atomicAdd(&(bodies_n0[blockIdx.x].velocity.w), 0);

    // Data will stay persistent in-between kernel calls
    bodies_n1[blockIdx.x] = bodies_n0[blockIdx.x];
}

void integrateNbodySystem(Body *bodies_n0, Body *bodies_n1, 
                          float deltaTime, float damping, int numBodies)
{
    unsigned int numThreads = numBodies * numBodies;
  
    unsigned int numBlocks       = getNumBlocks(numThreads);
    unsigned int threadsPerBlock = getNumThreads(numThreads);

    integrate<<<numBlocks, threadsPerBlock>>>(bodies_n0, numBodies, deltaTime, damping);

    cudaDeviceSynchronize();

    update<<<numBodies, 1>>>(bodies_n0, bodies_n1, deltaTime);

    cudaDeviceSynchronize();
}

void writePositionDataToFile(Body *bodies, int numBodies, const char* fileName) 
{
    std::ofstream outFile("../data/" + std::string(fileName));

    for (int i = 0; i < numBodies; i++)
    {
        outFile << bodies[i].position.x << " " << bodies[i].position.y << " " << bodies[i].position.z << std::endl;
    }
    
    outFile.close();
}

void runNBodySimulationParallel(int numBodies, int numIterations, float deltaTime, float damping)
{
    // Initial conditions
    Body *bodies_h; // Host body data

    // N-body system variables used for integration
    Body *bodies_n0;
    Body *bodies_n1;
    
    // Allocate the memory for the host
    bodies_h = new Body[numBodies];

    cudaMalloc(&bodies_n0, numBodies * sizeof(Body));
    cudaMalloc(&bodies_n1, numBodies * sizeof(Body));

    // Write particle positions to file for visualization
    char fileName[100];

    // Initialize the data
    initBodies2(bodies_h, numBodies);

    // Set up the initial conditions    
    cudaMemcpy(bodies_n0, bodies_h, numBodies * sizeof(Body), cudaMemcpyHostToDevice);

    // Run simulation for the specified number of iterations
    for (int i = 0; i < numIterations; i++) 
    {
        // Integrate the N-body system
        integrateNbodySystem(bodies_n0, bodies_n1, deltaTime, damping, numBodies);

        // nCopy particle position (n1) back to host
        cudaMemcpy(bodies_h, bodies_n1, numBodies * sizeof(Body), cudaMemcpyDeviceToHost);

        // Create filename with current iteration number
        sprintf(fileName, "positions_%d.txt", i);

        writePositionDataToFile(bodies_h, numBodies, fileName);

        // Print the positions of the first few particles for debugging purposes
        for (int j = 0; j < 10; j++) 
        {
            printf("Particle %d position: (%f, %f, %f)\n", j, bodies_h[i].position.x, 
                    bodies_h[i].position.y, bodies_h[i].position.z);
        }
    }

    // Cleanup
    delete[] bodies_h;
    cudaFree(bodies_n0);
    cudaFree(bodies_n1);
}
