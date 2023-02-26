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
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuda_runtime_api.h"
#include "two_body.h"


struct Vec3 {
    float x;
    float y;
    float z;

    __device__ Vec3 operator+(const Vec3& other) const {
        return { x + other.x, y + other.y, z + other.z };
    }

    __device__ Vec3 operator-(const Vec3& other) const {
        return { x - other.x, y - other.y, z - other.z };
    }

    __device__ Vec3 operator*(float scalar) const {
        return { x * scalar, y * scalar, z * scalar };
    }

    __device__ Vec3 operator/(float scalar) const {
        return { x / scalar, y / scalar, z / scalar };
    }

    __device__ float dot(const Vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    __device__ float length() const {
        return sqrtf(x * x + y * y + z * z);
    }

    __device__ Vec3 normalized() const {
        float len = length();
        return { x / len, y / len, z / len };
    }
};

struct Body {
    Vec3 position;
    Vec3 velocity;
};

__device__ void updateBodyVelocity(Body& body, const Vec3& force, float dt, float sign) {
    body.velocity = body.velocity + force * dt * sign;
}


__device__ Vec3 computeForce(const Body& a, const Body& b) {
    Vec3 r = b.position - a.position;
    float r2 = r.dot(r);
    float r3 = r2 * sqrtf(r2);
    return r / r3;
}

__global__ void allPairsKernel(Body* bodies, int N, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N && i < j) {
        const Body& a = bodies[i];
        const Body& b = bodies[j];
        Vec3 force = computeForce(a, b);
        updateBodyVelocity(bodies[i], force, dt, 1.0f);
        updateBodyVelocity(bodies[j], force, dt, -1.0f);
    }
}

int main() {
    int N = 1000; // number of bodies
    int threadsPerBlock = 32;

    // Allocate memory on the host for the array of bodies
    Body* bodies = new Body[N];

    // Initialize the position and velocity of each body
    for (int i = 0; i < N; i++) {
        bodies[i].position = { static_cast<float>(rand()), static_cast<float>(rand()), static_cast<float>(rand()) };
        bodies[i].velocity = { static_cast<float>(rand()), static_cast<float>(rand()), static_cast<float>(rand()) };
    }

    // Allocate memory on the device for the array of bodies
    Body* d_bodies;
    cudaMalloc((void**)&d_bodies, N * sizeof(Body));

    // Copy the array of bodies from the host to the device
    cudaMemcpy(d_bodies, bodies, N * sizeof(Body), cudaMemcpyHostToDevice);

    // Launch the kernel on the device
    dim3 blocksPerGrid((N + threadsPerBlock - 1) / threadsPerBlock, (N + threadsPerBlock - 1) / threadsPerBlock);
    float dt = 0.01f;
    allPairsKernel<<<blocksPerGrid, dim3(threadsPerBlock, threadsPerBlock)>>>(d_bodies, N, dt);

    // Copy the updated array of bodies from the device to the host
    cudaMemcpy(bodies, d_bodies, N * sizeof(Body), cudaMemcpyDeviceToHost);

    // Print the position and velocity of the first body
    std::cout << "Position: (" << bodies[0].position.x << ", " << bodies[0].position.y << ", " << bodies[0].position.z << ")" << std::endl;
    std::cout << "Velocity: (" << bodies[0].velocity.x << ", " << bodies[0].velocity.y << ", " << bodies[0].velocity.z << ")" << std::endl;

    // Free the memory on the host and device
    delete[] bodies;
    cudaFree(d_bodies);

    return 0;
}









