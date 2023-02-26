#pragma once

// #include <cuda_runtime.h>
// #include <iostream>
// #include <math.h>

// #define N 2 // number of bodies
// #define dt 0.01 // time step
// #define steps 10000 // number of steps

// constexpr int THREAD_PER_BLOCK = 128;

// // struct Body {
// //   float x, y, z; // position
// //   float vx, vy, vz; // velocity
// //   float m; // mass
// // };

// __global__ void simulate(Body *bodies) {
//   int i = blockIdx.x * blockDim.x + threadIdx.x;
//   if (i >= N) return;
  
//   for (int step = 0; step < steps; step++) {
//     // Calculate the gravitational force between the two bodies
//     float dx = bodies[1].x - bodies[0].x;
//     float dy = bodies[1].y - bodies[0].y;
//     float dz = bodies[1].z - bodies[0].z;
//     float distance = sqrt(dx * dx + dy * dy + dz * dz);
//     float force = bodies[0].m * bodies[1].m / (distance * distance);
    
//     // Update the velocities of the two bodies
//     bodies[0].vx -= force * dx / distance * dt / bodies[0].m;
//     bodies[0].vy -= force * dy / distance * dt / bodies[0].m;
//     bodies[0].vz -= force * dz / distance * dt / bodies[0].m;
//     bodies[1].vx += force * dx / distance * dt / bodies[1].m;
//     bodies[1].vy += force * dy / distance * dt / bodies[1].m;
//     bodies[1].vz += force * dz / distance * dt / bodies[1].m;
    
//     // Update the positions of the two bodies
//     bodies[0].x += bodies[0].vx * dt;
//     bodies[0].y += bodies[0].vy * dt;
//     bodies[0].z += bodies[0].vz * dt;
//     bodies[1].x += bodies[1].vx * dt;
//     bodies[1].y += bodies[1].vy * dt;
//     bodies[1].z += bodies[1].vz * dt;
//   }
// }






