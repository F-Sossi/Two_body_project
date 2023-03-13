//---------------------------------------------------------------------------
// kernel.cu
// Author: Frank Sossi
// Author: Amalaye Oyake
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "n_body_sim_1.h"
#include "n_body_sim_2.h"
#include "leap_frog.h"


int main()
{
    const int numIterations = 10;

    //determines the time step used for each iteration in the simulation. 
    //Lower value = more accurate simulation higher value = faster simulation
    const float deltaTime = 10.1;

    // Lowers the vleocity of particles over time to simulate friction
    const float damping = 0.999;
       int numBodies = 20480;

    LeapFrogIntegrator integrator(numBodies);

    integrator.step(numIterations, deltaTime);



    // Call the simulation function with the user's inputs
    //simulateNbodySystem2(numBodies, numIterations, deltaTime, damping);

    //runNbodySimulation(numBodies, numIterations, deltaTime, damping);

    return 0;

}









