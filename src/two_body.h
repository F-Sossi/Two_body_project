//---------------------------------------------------------------------------
// two_body.h - Header file for two body problem
// Author: Frank Sossi
// 
// File contains: 
//     1. Class for Body - contains position, velocity, and mass of a body
//     2. Class for TwoBodyProblem - contains two bodies and a time step
// 
//---------------------------------------------------------------------------
#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <vector>
#include <fstream>
#include <chrono>

#define DEBUG

// Size of the vector
constexpr int n = 10;
constexpr int THREAD_PER_BLOCK = 32;
constexpr int BLOCK_SIZE = 32;

//---------------------------------------------------------------------------
// Class for Body - contains position, velocity, and mass of a body
// input - position, velocity, and mass
//---------------------------------------------------------------------------
class Body {
public:
  std::vector<float> position;
  std::vector<float> velocity;
  float mass;

  __device__ void initialize(std::vector<float> pos, std::vector<float> vel, float m) {
    position = pos;
    velocity = vel;
    mass = m;
  }

  __device__ void computeForce(Body otherBody) {
    // Compute the force between this body and the other body.
  }

  __device__ void updatePosition(float dt) {
    // Update the position of this body based on its velocity and the time step.
  }

  __device__ void updateVelocity(float dt) {
    // Update the velocity of this body based on the force and the time step.
  }
};

class TwoBodyProblem {
public:
  Body body1;
  Body body2;
  float dt;

  __device__ void initialize(std::vector<float> pos1, std::vector<float> vel1, float m1, std::vector<float> pos2, std::vector<float> vel2, float m2, float timeStep) {
    body1.initialize(pos1, vel1, m1);
    body2.initialize(pos2, vel2, m2);
    dt = timeStep;
  }

  void simulate() {
    // Simulate the two-body problem on the GPU.
  }
};


