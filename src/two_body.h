#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <cuda_runtime.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_gl_interop.h>

constexpr float SOFTENING = 0.00125f;
const float GRAVITY_CONSTANT = 6.67430e-11f;




// void initParticles(float* hPos, float* hVel, int numBodies) {
//     for (int i = 0; i < numBodies; i++) {
//         hPos[i * 4] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
//         hPos[i * 4 + 1] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
//         hPos[i * 4 + 2] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
//         hPos[i * 4 + 3] = 1.0f;
//         hVel[i * 4] = 0.0f;
//         hVel[i * 4 + 1] = 0.0f;
//         hVel[i * 4 + 2] = 0.0f;
//         hVel[i * 4 + 3] = 0.0f;
//     }
// }

void initParticles(float* hPos, float* hVel, int numBodies) {
    for (int i = 0; i < numBodies; i++) {
        hPos[i * 4] = 400.0f * (rand() / (float)RAND_MAX) - 400.0f;
        hPos[i * 4 + 1] = 400.0f * (rand() / (float)RAND_MAX) - 400.0f;
        hPos[i * 4 + 2] = 400.0f * (rand() / (float)RAND_MAX) - 400.0f;
        hPos[i * 4 + 3] = 1.0f;
        hVel[i * 4] = 1112.1f;
        hVel[i * 4 + 1] = 1112.1f;
        hVel[i * 4 + 2] = 1112.1f;
        hVel[i * 4 + 3] = 1112.1f;
    }
}

__host__ __device__ float4 operator-(const float4& a, const float4& b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

// __global__
// void integrateBodies(float4* newPos, float4* newVel, float4* oldPos, float4* oldVel, 
//                      float deltaTime, float damping, int numBodies) 
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     if (idx < numBodies) {
//         float4 pos = oldPos[idx];
//         float4 vel = oldVel[idx];
//         float4 accel = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

//         for (int i = 0; i < numBodies; i++) {
//             if (i != idx) {
//                 float4 pos2 = oldPos[i];
//                 float4 delta = pos2 - pos;

//                 float distSqr = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
//                 float invDist = rsqrtf(distSqr + SOFTENING);

//                 float s = powf(invDist, 3.0f);

//                 accel.x += delta.x * s;
//                 accel.y += delta.y * s;
//                 accel.z += delta.z * s;
//             }
//         }

//         // Update velocity using acceleration and damping
//         vel.x += deltaTime * accel.x;
//         vel.y += deltaTime * accel.y;
//         vel.z += deltaTime * accel.z;

//         vel.x *= damping;
//         vel.y *= damping;
//         vel.z *= damping;

//         // Update position using velocity
//         pos.x += vel.x * deltaTime;
//         pos.y += vel.y * deltaTime;
//         pos.z += vel.z * deltaTime;

//         newPos[idx] = pos;
//         newVel[idx] = vel;
//     }
// }

// __global__
// void integrateBodies(float4* newPos, float4* newVel, float4* oldPos, float4* oldVel, 
//                      float deltaTime, float damping, float mass, int numBodies) 
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     if (idx < numBodies) {
//         float4 pos = oldPos[idx];
//         float4 vel = oldVel[idx];
//         float4 accel = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

//         for (int i = 0; i < numBodies; i++) {
//             if (i != idx) {
//                 float4 pos2 = oldPos[i];
//                 float4 delta = pos2 - pos;

//                 float distSqr = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
//                 if (distSqr < 10 * SOFTENING) { // only calculate gravity if particles are close enough
//                     float invDist = rsqrtf(distSqr + SOFTENING);

//                     float s = mass * powf(invDist, 3.0f);

//                     accel.x += delta.x * s;
//                     accel.y += delta.y * s;
//                     accel.z += delta.z * s;
//                 }
//             }
//         }

//         // Update velocity using acceleration and damping
//         vel.x += deltaTime * accel.x;
//         vel.y += deltaTime * accel.y;
//         vel.z += deltaTime * accel.z;

//         vel.x *= damping;
//         vel.y *= damping;
//         vel.z *= damping;

//         // Update position using velocity
//         pos.x += vel.x * deltaTime;
//         pos.y += vel.y * deltaTime;
//         pos.z += vel.z * deltaTime;

//         newPos[idx] = pos;
//         newVel[idx] = vel;
//     }
// }

__global__
void integrateBodies(float4* newPos, float4* newVel, float4* oldPos, float4* oldVel,
                     float deltaTime, float damping, float mass, int numBodies)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numBodies) {
        float4 pos = oldPos[idx];
        float4 vel = oldVel[idx];
        float4 accel = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        // Determine the starting index and stride for this thread
        int start = idx;
        int stride = gridDim.x * blockDim.x;

        // Loop over the chunk of memory that this thread is responsible for
        for (int i = start; i < numBodies; i += stride) {
            if (i != idx) {
                float4 pos2 = oldPos[i];
                float4 delta = pos2 - pos;

                float distSqr = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
                if (distSqr < 10 * SOFTENING) { // only calculate gravity if particles are close enough
                    
                    // Calculate the force of gravity between the two particles reciprocal square root
                    float invDist = rsqrtf(distSqr + SOFTENING);
                    // Calculate the force of gravity between the two particles
                    float s = mass * powf(invDist, 3.0f);

                    accel.x += delta.x * s;
                    accel.y += delta.y * s;
                    accel.z += delta.z * s;
                }
            }
        }

        // Update velocity using acceleration and damping
        vel.x += deltaTime * accel.x;
        vel.y += deltaTime * accel.y;
        vel.z += deltaTime * accel.z;

        vel.x *= damping;
        vel.y *= damping;
        vel.z *= damping;

        // Update position using velocity
        pos.x += vel.x * deltaTime;
        pos.y += vel.y * deltaTime;
        pos.z += vel.z * deltaTime;

        newPos[idx] = pos;
        newVel[idx] = vel;
    }
}



void integrateNbodySystem(float4* newPos, float4* newVel, float4* oldPos, float4* oldVel, 
                          float deltaTime, float damping, int numBodies, int p,
                          float* dPos, float* dVel)
{
    int threadsPerBlock = p;
    int numBlocks = (numBodies + threadsPerBlock - 1) / threadsPerBlock;

    integrateBodies<<<numBlocks, threadsPerBlock>>>(newPos, newVel, oldPos, oldVel, 
                                                     deltaTime, damping, 1, numBodies);

    // Swap old and new position/velocity arrays
    float4* temp = oldPos;
    oldPos = newPos;
    newPos = temp;

    temp = oldVel;
    oldVel = newVel;
    newVel = temp;

    // Copy updated position and velocity arrays back to device
    cudaMemcpy(dPos, oldPos, numBodies * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(dVel, oldVel, numBodies * sizeof(float4), cudaMemcpyHostToDevice);
}

void writePositionDataToFile(float* hPos, int numBodies, const char* fileName) {
    std::ofstream outFile("../data/" + std::string(fileName));
    for (int i = 0; i < numBodies; i++) {
        outFile << hPos[i * 4] << " " << hPos[i * 4 + 1] << " " << hPos[i * 4 + 2] << std::endl;
    }
    outFile.close();
}



void simulateNbodySystem(int numBodies, int numIterations, int p, float deltaTime, float damping)
{
    // Allocate memory for particle data
    float* hPos = new float[numBodies * 4];
    float* hVel = new float[numBodies * 4];
    float* dPos;
    float* dVel;
    cudaMalloc(&dPos, numBodies * sizeof(float4));
    cudaMalloc(&dVel, numBodies * sizeof(float4));

    // Initialize particle data
    initParticles(hPos, hVel, numBodies);

    // Copy particle data to device
    cudaMemcpy(dPos, hPos, numBodies * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(dVel, hVel, numBodies * sizeof(float4), cudaMemcpyHostToDevice);

    // Initialize variables for N-body system integration
    float4* newPos;
    float4* newVel;
    float4* oldPos;
    float4* oldVel;
    cudaMalloc(&newPos, numBodies * sizeof(float4));
    cudaMalloc(&newVel, numBodies * sizeof(float4));
    cudaMalloc(&oldPos, numBodies * sizeof(float4));
    cudaMalloc(&oldVel, numBodies * sizeof(float4));
    cudaMemcpy(oldPos, hPos, numBodies * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(oldVel, hVel, numBodies * sizeof(float4), cudaMemcpyHostToDevice);

    // Run simulation for the specified number of iterations
    for (int i = 0; i < numIterations; i++) {
        // Integrate the N-body system
        integrateNbodySystem(newPos, newVel, oldPos, oldVel, deltaTime, damping, numBodies, p, dPos, dVel);

        // Copy particle data back to host
        cudaMemcpy(hPos, dPos, numBodies * sizeof(float4), cudaMemcpyDeviceToHost);
        cudaMemcpy(hVel, dVel, numBodies * sizeof(float4), cudaMemcpyDeviceToHost);

        // Write particle positions to file for visualization
        char fileName[100];
        sprintf(fileName, "positions_%d.txt", i);  // Create filename with current iteration number
        writePositionDataToFile(hPos, numBodies, fileName);

        // Print the positions of the first few particles for debugging purposes
        // for (int j = 0; j < 10; j++) {
        //     printf("Particle %d position: (%f, %f, %f)\n", j, hPos[j * 4], hPos[j * 4 + 1], hPos[j * 4 + 2]);
        // }
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








