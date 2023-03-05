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
}


void initParticles(float* hPos, float* hVel, int numBodies) 
{
   for (int i = 0; i < numBodies; i++) 
    {
        hPos[i * 4]     = 400.0f * (rand() / (float)RAND_MAX) - 400.0f;
        hPos[i * 4 + 1] = 400.0f * (rand() / (float)RAND_MAX) - 400.0f;
        hPos[i * 4 + 2] = 400.0f * (rand() / (float)RAND_MAX) - 400.0f;
        hPos[i * 4 + 3] = 1.0f;

        hVel[i * 4]     = 1112.1f;
        hVel[i * 4 + 1] = 1112.1f;
        hVel[i * 4 + 2] = 1112.1f;
        hVel[i * 4 + 3] = 1112.1f;
    }
}

__host__ __device__ float4 operator-(const float4& a, const float4& b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__global__
void integrate(NBodyList bodies, int numBodies, float damping, float deltaTime, cudaEvent_t is_complete)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float4 pos     = NBodyList(blockIdx).pos;
    float4 nextpos = NBodyList((threadIdx % numBodies) + 1).pos;
    float4 vel     = NBodyList(threadIdx).vel;

    float4 delta_p = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 delta_v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 delta_a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float4 delta  = nextpos - pos;

    float distSqr = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;

    // Only calculate gravity if particles are close enough
    if (distSqr < 10 * SOFTENING) 
    {
        // Calculate the force of gravity between the two particles reciprocal square root
        float invDist = rsqrtf(distSqr + SOFTENING);

        // Calculate the force of gravity between the two particles
        float s = mass * powf(invDist, 3.0f);

        delta_a = delta.x * s;
        delta_a = delta.y * s;
        delta_a = delta.z * s;
    }

    // Update the acceleration
    AtomicAdd(NBodyList(blockIdx).accel, delta_a);
    cudaEventRecord(is_complete);
}

__global__
void update(NBodyList bodies, float deltaTime)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float4 accel   = bodies(blockIdx).accel;
    float4 vel     = bodies(blockIdx).vel;
    float4 pos     = bodies(blockIdx).pos;

    // Update velocity using acceleration and damping
    vel.x += deltaTime * accel.x;
    vel.y += deltaTime * accel.y;
    vel.z += deltaTime * accel.z;

    NBodyList(blockIdx).vel = vel;

    // Update position using velocity
    pos.x = vel.x * deltaTime;
    pos.y = vel.y * deltaTime;
    pos.y = vel.z * deltaTime;

    NBodyList(blockIdx).pos = pos;
}

void integrateNbodySystem(Body *bodies_n0, Body *bodies_n1, 
                          float deltaTime, float damping, int numBodies,
                          Body *bodies_d)
{
    cudaEvent_t is_complete;
    cudaEventCreate(&is_complete);

    integrate<<<numBlocks, numBlocks>>>(bodies_n0, bodies_n1, deltaTime, damping, 1, numBodies, is_complete);

    cudaEventSynchronize(is_complete);

    update<<numBlocks, 1>>(NBodyList bodies, float deltaTime);

    // Swap old and new position/velocity arrays
    Bodies* temp = bodies_n0;
    bodies_n0    = bodies_n1;
    bodies_n1    = temp;

    // Copy updated position and velocity arrays back to device
    cudaMemcpy(bodies_d, bodies_n0, numBodies * sizeof(Body), cudaMemcpyHostToDevice);
}

void writePositionDataToFile(float* hPos, int numBodies, const char* fileName) 
{
    std::ofstream outFile("../data/" + std::string(fileName));
    for (int i = 0; i < numBodies; i++)
    {
        outFile << hPos[i * 4] << " " << hPos[i * 4 + 1] << " " << hPos[i * 4 + 2] << std::endl;
    }
    outFile.close();
}


void simulateNbodySystem(int numBodies, int numIterations, float deltaTime, float damping)
{

    Body *bodies_h = new Body[numBodies];
    Body *bodies_d;
    
    cudaMalloc(&bodies_d, numBodies * sizeof(Body));

    // Initialize particle data
    // initParticles(hPos, hVel, numBodies);

    initBodies(bodies_h, numBodies);

    // Copy particle data to device
    cudaMemcpy(dPos, hPos, numBodies * sizeof(float4), cudaMemcpyHostToDevice);

    // Initialize variables for N-body system integration
    Body *bodies_n0;
    Body *bodies_n1;

    cudaMalloc(bodies_n0, numBodies * sizeof(Body));
    cudaMalloc(bodies_n1, numBodies * sizeof(Body));
    
    cudaMemcpy(bodies_n0, bodies_h, numBodies * sizeof(Body), cudaMemcpyHostToDevice);

    // Run simulation for the specified number of iterations
    for (int i = 0; i < numIterations; i++) 
    {
        // Integrate the N-body system
        integrateNbodySystem(bodies_n0, bodies_n1, deltaTime, damping, numBodies, dPos, dVel);

        // Copy particle data back to host
        cudaMemcpy(bodies_h, bodies_d, numBodies * sizeof(Body), cudaMemcpyDeviceToHost);

        // Write particle positions to file for visualization
        char fileName[100];

        // Create filename with current iteration number
        sprintf(fileName, "positions_%d.txt", i);
        writePositionDataToFile(hPos, numBodies, fileName);

        // Print the positions of the first few particles for debugging purposes
        for (int j = 0; j < 10; j++) 
        {
            printf("Particle %d position: (%f, %f, %f)\n", j, hPos[j * 4], hPos[j * 4 + 1], hPos[j * 4 + 2]);
        }
    }

    // Cleanup
    delete[] hPos;
    delete[] hVel;
    cudaFree(dPos);
    cudaFree(dVel);
    cudaFree(newPos);
    cudaFree(newVel);
    cudaFree(oldPos);
    cudaFree(oldVel);

}








