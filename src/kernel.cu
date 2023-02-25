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
// To compile: nvcc kernel.cu -o tbp.exe -lGL -lGLU -lglut
// To run: ./lab3
// Note: install glut and opengl libraries
// sudo apt-get install mesa-common-dev libglu1-mesa-dev freeglut3-dev
//---------------------------------------------------------------------------
#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "device_launch_parameters.h"
#include "two_body.h"

int main(int argc, char** argv) {

  Body bodies[N];
  
  // Initialize bodies
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
  
  Body *dev_bodies;
  cudaMalloc((void **)&dev_bodies, N * sizeof(Body));
  cudaMemcpy(dev_bodies, bodies, N * sizeof(Body), cudaMemcpyHostToDevice);
  
  // Launch kernel on GPU
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  simulate<<<numBlocks, blockSize>>>(dev_bodies);
  
  // Copy result from GPU to host
  cudaMemcpy(bodies, dev_bodies, N * sizeof(Body), cudaMemcpyDeviceToHost);
  
  // Free GPU memory
  cudaFree(dev_bodies);
  
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
  
  return 0;
}

