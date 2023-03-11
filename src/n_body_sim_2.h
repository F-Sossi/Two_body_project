#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <random>
#include <string.h>
#include <fstream>
#include <cuda_runtime.h>
#include "n_body_sim_1.h"

// constexpr float SOFTENING = 0.00125f;
// const float GRAVITY_CONSTANT = 6.67430e-11f;
constexpr int NUM_THREADS = 32;


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

void initBodiesTest2(Body *bodies, int numBodies) 
{
    // Create random number generator for each item
    std::mt19937_64 gen(std::random_device{}());
    std::uniform_real_distribution<float> posDist(0.0f, 40.0f);
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


__global__
void integrate2(Body *bodies, Body* upd_bodies, int numBodies, float deltaTime, float damping)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numBodies) {
        Body currentBody = bodies[idx];
        float4 accel = currentBody.acceleration;

        // Determine the starting index and stride for this thread
        int start = idx;
        //int stride = gridDim.x * blockDim.x;

        // Loop over the chunk of memory that this thread is responsible for
        for (int i = start; i < numBodies; i++) {
            if (i != idx) {
                Body otherBody = bodies[i];

                // Compute distance
                float4 delta_p = otherBody.position - currentBody.position;

                // Compute distance squared
                float distSqr = delta_p.x * delta_p.x + delta_p.y * delta_p.y + delta_p.z * delta_p.z;

                // Only calculate gravity if particles are close enough
                if (distSqr < 10 * SOFTENING) {
                    // Calculate the force of gravity between the two particles reciprocal square root
                    float invDist = rsqrtf(distSqr + SOFTENING);

                    // Calculate the force of gravity between the two particles
                    float s = otherBody.mass * powf(invDist, 3.0f);

                    accel.x += delta_p.x * s;
                    accel.y += delta_p.y * s;
                    accel.z += delta_p.z * s;
                }
            }
        }

        // Update velocity using acceleration and damping
        currentBody.velocity.x += deltaTime * accel.x;
        currentBody.velocity.y += deltaTime * accel.y;
        currentBody.velocity.z += deltaTime * accel.z;

        currentBody.velocity.x *= damping;
        currentBody.velocity.y *= damping;
        currentBody.velocity.z *= damping;

        // Update position using velocity
        currentBody.position.x += currentBody.velocity.x * deltaTime;
        currentBody.position.y += currentBody.velocity.y * deltaTime;
        currentBody.position.z += currentBody.velocity.z * deltaTime;

        //__syncthreads();

        // Update the body in the array
        upd_bodies[idx] = currentBody;
    }
}


// void integrateNbodySystem2(Body *bodies_n0, Body *bodies_n1, 
//                           float deltaTime, float damping, int numBodies,
//                           Body *bodies_d)
// {
//     unsigned int numThreads = numBodies * numBodies;


//     int threadsPerBlock = NUM_THREADS;
//     int numBlocks = (numBodies + threadsPerBlock - 1) / threadsPerBlock;
//     // unsigned int numBlocks       = getNumBlocks(numThreads);
//     // unsigned int threadsPerBlock = getNumThreads(numThreads);

//     integrate2<<<numBlocks, threadsPerBlock>>>(bodies_n0, numBodies, deltaTime, damping);

//     cudaDeviceSynchronize();

//     // Swap old and new position/velocity arrays
//     Body *temp   = bodies_n0;
//     bodies_n0    = bodies_n1;
//     bodies_n1    = temp;

//     // Copy updated position and velocity arrays back to device for output
//     cudaMemcpy(bodies_d, bodies_n0, numBodies * sizeof(Body), cudaMemcpyHostToDevice);

//     //cudaEventDestroy(is_complete);
// }

void integrateNbodySystem2(Body *&bodies_n0, Body *&bodies_n1, 
                          float deltaTime, float damping, int numBodies,
                          Body *bodies_d)
{
    unsigned int numThreads = numBodies * numBodies;

    int threadsPerBlock = NUM_THREADS;
    int numBlocks = (numBodies + threadsPerBlock - 1) / threadsPerBlock;
    // unsigned int numBlocks       = getNumBlocks(numThreads);
    // unsigned int threadsPerBlock = getNumThreads(numThreads);

    integrate2<<<numBlocks, threadsPerBlock>>>(bodies_n0, bodies_n1, numBodies, deltaTime, damping);

    cudaDeviceSynchronize();


    // Swap old and new position/velocity arrays
    Body *temp   = bodies_n1;
    bodies_n1    = bodies_n0;
    bodies_n1    = temp;

    // Copy updated position and velocity arrays back to device for output
    cudaMemcpy(bodies_d, bodies_n1, numBodies * sizeof(Body), cudaMemcpyHostToDevice);

    //cudaEventDestroy(is_complete);
}


void writePositionDataToFile2(float* hPos, int numBodies, const char* fileName) {
    std::ofstream outFile("../data/" + std::string(fileName));
    for (int i = 0; i < numBodies; i++) {
        outFile << hPos[i * 4] << " " << hPos[i * 4 + 1] << " " << hPos[i * 4 + 2] << std::endl;
    }
    outFile.close();
}

void simulateNbodySystem2(int numBodies, int numIterations, float deltaTime, float damping)
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
    initBodiesTest2(bodies_h, numBodies);

    // Next, copy particle data to device to start the run
    cudaMemcpy(bodies_d, bodies_h, numBodies * sizeof(float4), cudaMemcpyHostToDevice);

    // Set up the initial conditions    
    cudaMemcpy(bodies_n0, bodies_h, numBodies * sizeof(Body), cudaMemcpyHostToDevice);

    // Run simulation for the specified number of iterations
    for (int i = 0; i < numIterations; i++) 
    {
        // Integrate the N-body system
        //integrateNbodySystem2(bodies_n0, bodies_n1, deltaTime, damping, numBodies, bodies_d);
        integrateNbodySystem2(bodies_n0, bodies_n1, deltaTime, damping, numBodies, bodies_d);


        // Copy particle data back to host
        cudaMemcpy(bodies_h, bodies_d, numBodies * sizeof(Body), cudaMemcpyDeviceToHost);

        // Write particle positions to file for visualization
        char fileName[100];

        // Create filename with current iteration number
        sprintf(fileName, "positions_%d.txt", i);

        writePositionDataToFile(bodies_h, numBodies, fileName);

        // for (int j = 0; j < 10; j++) 
        // {
        //     printf("Particle %d position: (%f, %f, %f)\n", j, bodies_h[j].position.x, 
        //             bodies_h[j].position.y, bodies_h[j].position.z);
        // }
    }

    // Cleanup
    delete[] bodies_h;
    free (bodies_h);
    cudaFree(bodies_d);
    cudaFree(bodies_n0);
    cudaFree(bodies_n1);

}


// Old code

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