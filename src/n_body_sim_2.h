//---------------------------------------------------------------------------
// n_body_sim_2.h
// Author: Frank Sossi
// Author: Amalaye Oyake
//
// File contains:
// 1. simulateNbodySystem2 function
// 2. runNbodySimulation function
// 3. device functions for working with float4
// 4. kernel functions for integration and force calculation
// 5 NbodySystem class
//  5.1 NbodySystem::integrate function
//  5.2 NbodySystem::calculateForces function
// 6. writePositionsToFile function
//---------------------------------------------------------------------------

#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <random>
#include <string.h>
#include <cfloat>
#include <cmath>
#include <fstream>
#include <cuda_runtime.h>
#include "n_body_sim_1.h"

constexpr int NUM_THREADS = 1024;
constexpr float Epsilon = 1e-4f;

// Addition
__host__ __device__ float4 operator+(const float4& a, const float4& b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

// Scalar multiplication
__host__ __device__ float4 operator*(const float4& a, const float& b)
{
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

// Scalar multiplication
__host__ __device__ float4 operator*(const float& b, const float4& a)
{
    return a * b;
}

// Dot product
__host__ __device__ float dot(const float4& a, const float4& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// Cross product
__host__ __device__ float4 cross(const float4& a, const float4& b)
{
    return make_float4(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x, 0.0f);
}

// Magnitude
__host__ __device__ float length(const float4& a)
{
    return sqrtf(dot(a, a));
}


// Normalize
__host__ __device__ float4 normalize(const float4& a)
{
    float invLen = 1.0f / length(a);
    return make_float4(a.x * invLen, a.y * invLen, a.z * invLen, a.w * invLen);
}

// Scalar division
__host__ __device__ float4 operator/(const float4& a, const float& b)
{
    return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}


// Addition assignment
__host__ __device__ float4& operator+=(float4& a, const float4& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
}

// Subtraction assignment
__host__ __device__ float4& operator-=(float4& a, const float4& b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
    return a;
}

// Multiplication assignment
__host__ __device__ float4& operator*=(float4& a, const float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
    return a;
}
// forward declarations---------------------------------------------------------------------------
void integrateNbodySystem2(Body *&bodies_n0, Body *&bodies_n1, float deltaTime, float damping, int numBodies,
                           Body *bodies_d);
__global__ void integrate2(Body *bodies, Body* upd_bodies, int numBodies, float deltaTime, float damping, float restitution, float radius);
__global__ void integrate3(Body *bodies, Body* upd_bodies, int numBodies, float deltaTime, float damping, float restitution, float radius);
__global__ void integrate4(Body *bodies, Body* upd_bodies, int numBodies, float deltaTime, float damping, float restitution, float radius);
void initBodiesTest2(Body *bodies, int numBodies);

//---------------------------------------------------------------------------
// Function: initBodiesTest2 initializes the bodies to random values
// Input:
// 1. bodies: pointer to the array of bodies
// 2. numBodies: number of bodies
// Output: none
//---------------------------------------------------------------------------
void initBodiesTest2(Body *bodies, int numBodies) {
    std::mt19937_64 gen(std::random_device{}());
    std::uniform_real_distribution<float> posDist(1000.0f, 50000.0f);
    std::uniform_real_distribution<float> velDist(20.0f, 50.1f);
    std::uniform_real_distribution<float> mass(100000.0f, 1000000.0f);

    for (int i = 0; i < numBodies; i++) {
        // Generate random values for position
        bodies[i].position.x = posDist(gen);
        bodies[i].position.y = posDist(gen);
        bodies[i].position.z = posDist(gen);
        bodies[i].position.w = 1;
        
        // Generate random vector for velocity direction
        float vx = std::uniform_real_distribution<float>(-1.0f, 1.0f)(gen);
        float vy = std::uniform_real_distribution<float>(-1.0f, 1.0f)(gen);
        float vz = std::uniform_real_distribution<float>(-1.0f, 1.0f)(gen);

        // Scale the velocity direction vector to the desired magnitude
        float velocityMag = velDist(gen);
        float invLength = 1.0f / sqrt(vx*vx + vy*vy + vz*vz);
        vx *= velocityMag * invLength;
        vy *= velocityMag * invLength;
        vz *= velocityMag * invLength;

        // Set the velocity values for the body
        bodies[i].velocity.x = vx;
        bodies[i].velocity.y = vy;
        bodies[i].velocity.z = vz;
        bodies[i].velocity.w = 0.1f;

        // Generate random value for mass
        bodies[i].mass = mass(gen) * i;
    }
}


//---------------------------------------------------------------------------
// Function: integrateNbodySystem2 integrates the system of bodies
// Input:
// 1. bodies_n0: pointer to the array of bodies at time n
// 2. bodies_n1: pointer to the array of bodies at time n+1
// 3. deltaTime: time step
// 4. damping: damping factor
// 5. numBodies: number of bodies
// 6. bodies_d: pointer to the array of bodies on the device
// Output: none
//---------------------------------------------------------------------------
void integrateNbodySystem2(Body *&bodies_n0, Body *&bodies_n1, 
                           float deltaTime, float damping, int numBodies,
                           Body *bodies_d) {


    int threadsPerBlock = NUM_THREADS;
    int numBlocks = (numBodies + threadsPerBlock - 1) / threadsPerBlock;

    integrate3<<<numBlocks, threadsPerBlock>>>(bodies_n0, bodies_n1, numBodies,
                                                                 deltaTime, damping, 1.0f, 100.0f);

    cudaDeviceSynchronize();

    // Swap old and new position/velocity arrays
    Body *temp   = bodies_n1;
    bodies_n1    = bodies_n0;
    bodies_n0    = temp;

    // Copy updated position and velocity arrays back to device for output
    cudaMemcpy(bodies_d, bodies_n1, numBodies * sizeof(Body), cudaMemcpyHostToDevice);
}

//---------------------------------------------------------------------------
// Function: writePositionDataToFile writes the position data to a file
// Input:
// 1. hPos: pointer to the array of positions
// 2. numBodies: number of bodies
// 3. fileName: name of the file
// Output: positionsX.txt file containing the position data where X is step
//         number
//---------------------------------------------------------------------------
void writePositionDataToFile(float* hPos, int numBodies, const char* fileName) {
    std::ofstream outFile("../data/" + std::string(fileName));
    for (int i = 0; i < numBodies; i++) {
        outFile << hPos[i * 4] << " " << hPos[i * 4 + 1] << " " << hPos[i * 4 + 2] << std::endl;
    }
    outFile.close();
}

//---------------------------------------------------------------------------
// Function: integrate4 integrates the system of bodies
// Input:
// 1. bodies: pointer to the array of bodies
// 2. upd_bodies: pointer to the array of updated bodies
// 3. numBodies: number of bodies
// 4. deltaTime: time step
// 5. damping: damping factor
// 6. restitution: restitution factor
// 7. radius: radius of the bodies
// Output: none
//---------------------------------------------------------------------------
__global__ void integrate4(Body *bodies, Body* upd_bodies, int numBodies, float deltaTime, float damping,
                           float restitution, float radius) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numBodies) {

        Body currentBody = bodies[idx];
        float4 accel = currentBody.acceleration;

        // Determine the starting index and stride for this thread
        int start = idx;

        // Loop over the chunk of memory that this thread is responsible for
        for (int i = start; i < numBodies; i++) {
            if (i != idx) {
                Body otherBody = bodies[i];

                // Compute distance
                float4 delta_p = otherBody.position - currentBody.position;

                // Scale delta_p to prevent overflow
                float max_delta = fmaxf(fmaxf(fabsf(delta_p.x), fabsf(delta_p.y)), fabsf(delta_p.z));
                if (max_delta > FLT_MAX / sqrtf(3.0f)) {
                    delta_p *= FLT_MAX / max_delta / sqrtf(3.0f);
                }

                float distSqr = delta_p.x * delta_p.x + delta_p.y * delta_p.y + delta_p.z * delta_p.z;

                // Only calculate gravity if particles are close enough
                if (distSqr < 1000) {
                    // Calculate the force of gravity between the two particles reciprocal square root
                    float invDist = rsqrtf(max(distSqr, 0.0f) + Epsilon);
                    float s = otherBody.mass * powf(invDist, 3.0f);

                    accel.x += delta_p.x * s;
                    accel.y += delta_p.y * s;
                    accel.z += delta_p.z * s;
                }

                // Check for collision
                float dist = sqrtf(distSqr);
                if (dist < 2 * radius) {
                    // Calculate the relative velocity of the two bodies
                    float4 delta_v = otherBody.velocity - currentBody.velocity;

                    // Calculate the normal vector between the two bodies
                    float4 normal = delta_p / dist;

                    // Calculate the impulse magnitude
                    float impulse_mag = dot(delta_v, normal) * (1 + restitution) / (1 / currentBody.mass + 1 / otherBody.mass);

                    // Calculate the impulse
                    float4 impulse = impulse_mag * normal;

                    // Update the velocities of the two bodies
                    currentBody.velocity += impulse / currentBody.mass;
                    otherBody.velocity -= impulse / otherBody.mass;

                    // Move the bodies apart to avoid overlapping
                    float4 separation = (2 * radius - dist) * normal;
                    currentBody.position -= separation / 2;
                    otherBody.position += separation / 2;
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

        // Update the body in the array
        upd_bodies[idx] = currentBody;
    }
}

//---------------------------------------------------------------------------
// Function: integrate2 integrates the system of bodies
// Input:
// 1. bodies: pointer to the array of bodies
// 2. upd_bodies: pointer to the array of updated bodies
// 3. numBodies: number of bodies
// 4. deltaTime: time step
// 5. damping: damping factor
// 6. restitution: restitution factor
// 7. radius: radius of the bodies
// Output: none
//---------------------------------------------------------------------------
__global__ void integrate2(Body *bodies, Body* upd_bodies, int numBodies, float deltaTime, float damping,
                           float restitution, float radius){
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numBodies) {

        Body currentBody = bodies[idx];
        float4 accel = currentBody.acceleration;

        // Determine the starting index and stride for this thread
        int start = 0;

        // Loop over the chunk of memory that this thread is responsible for
        for (int i = start; i < numBodies; i++) {
            if (i != idx) {
                Body otherBody = bodies[i];

                // Compute distance
                float4 delta_p = otherBody.position - currentBody.position;

                // Compute distance squared
                float distSqr = delta_p.x * delta_p.x + delta_p.y * delta_p.y + delta_p.z * delta_p.z;

                // Only calculate gravity if particles are close enough
                if (distSqr < 1000) {
                    // Calculate the force of gravity between the two particles reciprocal square root
                    float invDist = rsqrtf(distSqr + SOFTENING);

                    // Calculate the force of gravity between the two particles
                    float s = otherBody.mass * powf(invDist, 3.0f);

                    accel.x += delta_p.x * s;
                    accel.y += delta_p.y * s;
                    accel.z += delta_p.z * s;
                }

                // Check for collision
                float dist = sqrtf(distSqr);
                if (dist < 2 * radius) {
                    // Calculate the relative velocity of the two bodies
                    float4 delta_v = otherBody.velocity - currentBody.velocity;

                    // Calculate the normal vector between the two bodies
                    float4 normal = delta_p / dist;

                    // Calculate the impulse magnitude
                    float impulse_mag = dot(delta_v, normal) * (1 + restitution) / (1 / currentBody.mass + 1 / otherBody.mass);

                    // Calculate the impulse
                    float4 impulse = impulse_mag * normal;

                    // Update the velocities of the two bodies
                    currentBody.velocity += impulse / currentBody.mass;
                    otherBody.velocity -= impulse / otherBody.mass;

                    // Move the bodies apart to avoid overlapping
                    float4 separation = (2 * radius - dist) * normal;
                    currentBody.position -= separation / 2;
                    otherBody.position += separation / 2;
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

//---------------------------------------------------------------------------
// Function: integrate3 integrates the system of bodies
// Input:
// 1. bodies: pointer to the array of bodies
// 2. upd_bodies: pointer to the array of updated bodies
// 3. numBodies: number of bodies
// 4. deltaTime: time step
// 5. damping: damping factor
// 6. restitution: restitution factor
// 7. radius: radius of the bodies
// Output: none
//---------------------------------------------------------------------------
__global__ void integrate3(Body *bodies, Body* upd_bodies, int numBodies, float deltaTime, float damping,
                           float restitution, float radius) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numBodies) {

        Body currentBody = bodies[idx];
        float4 accel = currentBody.acceleration;

        // Determine the starting index and stride for this thread
        int start = idx;

        // Loop over the chunk of memory that this thread is responsible for
        for (int i = start; i < numBodies; i++) {
            if (i != idx) {
                Body otherBody = bodies[i];

                // Compute distance
                float4 delta_p = otherBody.position - currentBody.position;

                // Scale delta_p to prevent overflow
                float max_delta = fmaxf(fmaxf(fabsf(delta_p.x), fabsf(delta_p.y)), fabsf(delta_p.z));
                if (max_delta > FLT_MAX / sqrtf(3.0f)) {
                    delta_p *= FLT_MAX / max_delta / sqrtf(3.0f);
                }

                float distSqr = delta_p.x * delta_p.x + delta_p.y * delta_p.y + delta_p.z * delta_p.z;

                // Only calculate gravity if particles are close enough
                if (distSqr < 1000) {
                    // Calculate the force of gravity between the two particles reciprocal square root
                    float invDist = rsqrtf(max(distSqr, 0.0f) + Epsilon);
                    float s = otherBody.mass * powf(invDist, 3.0f);

                    accel.x += delta_p.x * s;
                    accel.y += delta_p.y * s;
                    accel.z += delta_p.z * s;
                }
            }
        }

        // Check for NaN values in accel and set to zero if found
        if (isnan(accel.x) || isnan(accel.y) || isnan(accel.z)) {
            accel.x = 0.0f;
            accel.y = 0.0f;
            accel.z = 0.0f;
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

        // Update the body in the array
        upd_bodies[idx] = currentBody;
    }
}

//---------------------------------------------------------------------------
// Class: NbodySystem
// Input:
// 1. numBodies: number of bodies
// Output: none
//---------------------------------------------------------------------------
class NbodySystem {
private:
    Body *bodies_h;
    Body *bodies_d;

    Body *bodies_n0;
    Body *bodies_n1;

    int numBodies;

public:
    NbodySystem(int numBodies) : numBodies(numBodies) {
        cudaMalloc(&bodies_d,  numBodies * sizeof(Body));
        cudaMalloc(&bodies_n0, numBodies * sizeof(Body));
        cudaMalloc(&bodies_n1, numBodies * sizeof(Body));

        // Allocate the memory for the host
        bodies_h = new Body[numBodies];

        // Initialize the data
        initBodiesTest2(bodies_h, numBodies);

        // Next, copy particle data to device to start the run
        cudaMemcpy(bodies_d, bodies_h, numBodies * sizeof(float4), cudaMemcpyHostToDevice);

        // Set up the initial conditions    
        cudaMemcpy(bodies_n0, bodies_h, numBodies * sizeof(Body), cudaMemcpyHostToDevice);
    }
    //---------------------------------------------------------------------------
    // Method: simulate runs the simulation
    // Input:
    // 1. numIterations: number of iterations
    // 2. deltaTime: time step
    // 3. damping: damping factor
    // Output: none
    //---------------------------------------------------------------------------
    void simulate(int numIterations, float deltaTime, float damping) {
        // Run simulation for the specified number of iterations
        for (int i = 0; i < numIterations; i++) {
            // Integrate the N-body system
            integrateNbodySystem2(bodies_n0, bodies_n1, deltaTime, damping, numBodies, bodies_d);

            // Copy particle data back to host
            cudaMemcpy(bodies_h, bodies_d, numBodies * sizeof(Body), cudaMemcpyDeviceToHost);

            // Write particle positions to file for visualization
            char fileName[100];

            // Create filename with current iteration number
            sprintf(fileName, "positions_%d.txt", i);

            writePositionDataToFile((float *)bodies_h, numBodies, fileName);
        }
    }

    ~NbodySystem() {
        delete[] bodies_h;
        //firee(bodies_h);
        cudaFree(bodies_d);
        cudaFree(bodies_n0);
        cudaFree(bodies_n1);
    }
};

//---------------------------------------------------------------------------
// Function: runNbodySimulation runs the simulation
// Input:
// 1. numBodies: number of bodies
// 2. numIterations: number of iterations
// 3. deltaTime: time step
// 4. damping: damping factor
// Output: none
//---------------------------------------------------------------------------
void runNbodySimulation(int numBodies, int numIterations, float deltaTime, float damping) {
    NbodySystem nbodySystem(numBodies);

    nbodySystem.simulate(numIterations, deltaTime, damping);
}

