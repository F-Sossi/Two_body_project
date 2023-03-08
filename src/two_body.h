#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <cuda_runtime.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_gl_interop.h>

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
    float4 velocity     = bodies_n0[blockIdx.x].velocity;
    float4 position     = bodies_n0[blockIdx.x].position;

    // Update velocity using acceleration and damping
    velocity.x += deltaTime * acceleration.x;
    velocity.y += deltaTime * acceleration.y;
    velocity.z += deltaTime * acceleration.z;

    bodies_n1[blockIdx.x].velocity = velocity;

    // Update position using velocity
    position.x = velocity.x * deltaTime;
    position.y = velocity.y * deltaTime;
    position.y = velocity.z * deltaTime;

    bodies_n1[blockIdx.x].position = position;

    // Wait for completion across all blocks
    __syncthreads();
}

void integrateNbodySystem(Body *bodies_n0, Body *bodies_n1, 
                          float deltaTime, float damping, int numBodies,
                          Body *bodies_d)
{
    unsigned int numThreads = numBodies * numBodies;
  
    //cudaEvent_t is_complete;
    //cudaEventCreate(&is_complete);

    unsigned int numBlocks       = getNumBlocks(numThreads);
    unsigned int threadsPerBlock = getNumThreads(numThreads);

    integrate<<<numBlocks, threadsPerBlock>>>(bodies_n0, numBodies, deltaTime, damping);

    // Signal that the work is done
    //cudaEventRecord(is_complete);

    //cudaEventSynchronize(is_complete);

    cudaDeviceSynchronize();

    update<<<numBodies, 1>>>(bodies_n0, bodies_n1, deltaTime);

    //cudaDeviceSynchronize();

    // Swap old and new position/velocity arrays
    Body *temp   = bodies_n0;
    bodies_n0    = bodies_n1;
    bodies_n1    = temp;

    // Copy updated position and velocity arrays back to device for output
    cudaMemcpy(bodies_d, bodies_n0, numBodies * sizeof(Body), cudaMemcpyHostToDevice);

    //cudaEventDestroy(is_complete);
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

void simulateNbodySystem(int numBodies, int numIterations, float deltaTime, float damping)
{
    // Initial conditions
    Body *bodies_h; // Host body data
    Body *bodies_d; // device body data

    // N-body system variables used for integration
    Body *bodies_n0;
    Body *bodies_n1;
    
    // Allocate the memory for the host
    bodies_h = new Body[numBodies];

    cudaMalloc(&bodies_d,  numBodies * sizeof(Body));
    cudaMalloc(&bodies_n0, numBodies * sizeof(Body));
    cudaMalloc(&bodies_n1, numBodies * sizeof(Body));

    // Initialize the data
    initBodies(bodies_h, numBodies);

    // Next, copy particle data to device to start the run
    cudaMemcpy(bodies_d, bodies_h, numBodies * sizeof(float4), cudaMemcpyHostToDevice);

    // Set up the initial conditions    
    cudaMemcpy(bodies_n0, bodies_h, numBodies * sizeof(Body), cudaMemcpyHostToDevice);

    // Run simulation for the specified number of iterations
    for (int i = 0; i < numIterations; i++) 
    {
        // Integrate the N-body system
        integrateNbodySystem(bodies_n0, bodies_n1, deltaTime, damping, numBodies, bodies_d);

        // Copy particle data back to host
        cudaMemcpy(bodies_h, bodies_d, numBodies * sizeof(Body), cudaMemcpyDeviceToHost);

        // Write particle positions to file for visualization
        char fileName[100];

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
    cudaFree(bodies_d);
    cudaFree(bodies_n0);
    cudaFree(bodies_n1);
}
