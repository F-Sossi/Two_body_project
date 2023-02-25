//---------------------------------------------------------------------------
// kernel.cu
// Author: Frank Sossi
// 
// This program simulates the motion of two bodies in space. The bodies are
// represented by spheres and the motion is simulated by using the Euler
// method.
//
// Complile with cmake: 
//
// mkdir build
// cd build
// cmake ..
// make
// ./my_project
// 
// To compile: 
//    cd build
//    make
// To run:
//    ./two_body.exe
//
//---------------------------------------------------------------------------
#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "device_launch_parameters.h"
#include "two_body.h"
#include <Python.h>

#define SIMULATION_COUNT 100
#define DEBUG

void display(Body* bodies);

int main(int argc, char** argv) {


    // Initialize bodies
    Body bodies[N];

    bodies[0].x = 0.0f;
    bodies[0].y = 0.0f;
    bodies[0].z = 0.0f;
    bodies[0].vx = 0.0f;
    bodies[0].vy = 0.0f;
    bodies[0].vz = 0.0f;
    bodies[0].m = 1.0f;

    bodies[1].x = 1.0f;
    bodies[1].y = 0.0f;
    bodies[1].z = 0.0f;
    bodies[1].vx = 0.0f;
    bodies[1].vy = 1.0f;
    bodies[1].vz = 0.0f;
    bodies[1].m = 1.0f;

    std::vector<std::vector<float>> positions;

    Body *dev_bodies;
    cudaMalloc((void **)&dev_bodies, N * sizeof(Body));
    cudaMemcpy(dev_bodies, bodies, N * sizeof(Body), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    for (int i = 0; i < SIMULATION_COUNT; i++) {

    simulate<<<numBlocks, blockSize>>>(dev_bodies);

    cudaMemcpy(bodies, dev_bodies, N * sizeof(Body), cudaMemcpyDeviceToHost);

    #ifdef DEBUG

    // Display the result
    for (int i = 0; i < N; i++) {
        std::cout << "Body " << i << ": " << std::endl;
        std::cout << "x = " << bodies[i].x << std::endl;
        std::cout << "y = " << bodies[i].y << std::endl;
        std::cout << "z = " << bodies[i].z << std::endl;
        std::cout << "vx = " << bodies[i].vx << std::endl;
        std::cout << "vy = " << bodies[i].vy << std::endl;
        std::cout << "vz = " << bodies[i].vz << std::endl;
        std::cout << "m = " << bodies[i].m << std::endl;
    }

    #endif

    // Save the positions to the vector
    std::vector<float> pos;
        pos.push_back(bodies[0].x);
        pos.push_back(bodies[0].y);
        pos.push_back(bodies[0].z);
        pos.push_back(bodies[1].x);
        pos.push_back(bodies[1].y);
        pos.push_back(bodies[1].z);
        positions.push_back(pos);
    }


    cudaFree(dev_bodies);

    return 0;
}






