//---------------------------------------------------------------------------
// kernel.cu
// Author: Frank Sossi
// Author: Amalaye Oyake
// 
// This program simulates the motion of two bodies in space. The bodies are
// represented by spheres and the motion is simulated by using the Euler
// method. Also implements the leapfrog method which is more accurate.
//
// Compile with cmake:
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
#include <chrono>
#include <cuda_runtime.h>
#include "n_body_sim_1.h"
#include "n_body_sim_2.h"
#include "leap_frog.h"
#include "verlet.h"

//// Runs timing tests for each simulation loop
//#define TEST
// Run individual version of simulation
#define CUSTOM


int main()
{

#ifdef TEST
     std::vector<int>testSizes = { 8192, 100435};
     std::vector<long long>testTimesSim1;
     std::vector<long long>testTimesSim2;
     std::vector<long long>testTimesSim3;
     std::vector<long long>testTimesSim4;

    for (int i = 0; i < testSizes.size(); i++)
    {
        const int numIterations = 1;
        const float deltaTime = 10.1;
        const float damping = 0.999;

        //print Running simulation1 for each test size:
        std::cout << "Running simulation1 for test size: " << testSizes[i] << std::endl;

        //calculate the running time for each simulation
        auto startsim1 = std::chrono::high_resolution_clock::now();
        runNBodySimulationParallel(testSizes[i], numIterations, deltaTime, damping);
        auto endsim1 = std::chrono::high_resolution_clock::now();

        // calculate the time taken for the simulation in nano seconds
        auto durationSim1 = std::chrono::duration_cast<std::chrono::nanoseconds>(endsim1 - startsim1);
        testTimesSim1.push_back(durationSim1.count());

//        // print Running simulation2 for each test size:
//        std::cout << "Running simulation2 for test size: " << testSizes[i] << std::endl;
//
//        auto startsim2 = std::chrono::high_resolution_clock::now();
//        runNbodySimulation(testSizes[i], numIterations, deltaTime, damping);
//        auto endsim2 = std::chrono::high_resolution_clock::now();
//
//        auto durationSim2 = std::chrono::duration_cast<std::chrono::nanoseconds>(endsim2 - startsim2);
//        testTimesSim2.push_back(durationSim2.count());

        // print Running simulation3 for each test size:
        std::cout << "Running simulation3 for test size: " << testSizes[i] << std::endl;

        auto startsim3 = std::chrono::high_resolution_clock::now();
        LeapFrogIntegrator integrator1(testSizes[i]);
        integrator1.step(numIterations, deltaTime);
        auto endsim3 = std::chrono::high_resolution_clock::now();

        auto durationSim3 = std::chrono::duration_cast<std::chrono::nanoseconds>(endsim3 - startsim3);
        testTimesSim3.push_back(durationSim3.count());

        // print Running simulation4 for each test size:
        std::cout << "Running simulation4 for test size: " << testSizes[i] << std::endl;

        auto startsim4 = std::chrono::high_resolution_clock::now();
        VerletIntegrator integrator2(testSizes[i]);
        integrator2.step(numIterations, deltaTime);
        auto endsim4 = std::chrono::high_resolution_clock::now();

        auto durationSim4 = std::chrono::duration_cast<std::chrono::nanoseconds>(endsim4 - startsim4);
        testTimesSim4.push_back(durationSim4.count());

        // print test size and time taken for each simulation
        std::cout << "Test Size: " << testSizes[i] << std::endl;
        std::cout << "Time taken for simulation 1: " << durationSim1.count() << " nanoseconds" << std::endl;
        //std::cout << "Time taken for simulation 2: " << durationSim2.count() << " nanoseconds" << std::endl;
        std::cout << "Time taken for simulation 3: " << durationSim3.count() << " nanoseconds" << std::endl;
        std::cout << "Time taken for simulation 4: " << durationSim4.count() << " nanoseconds" << std::endl;
        std::cout << std::endl;
    }

#endif

#ifdef CUSTOM

    const int num_bodies    = 2000;
    const int numIterations = 100;
    const float deltaTime   = 10.0;
    const float damping     = 0.999;

//    // Run simulation 1 from n_body_sim_1.h
    std::cout << "Running simulation1 for test size: " << num_bodies << std::endl;
    runNBodySimulationParallel(num_bodies, numIterations, deltaTime, damping);

//    // Run simulation 2 from n_body_sim_2.h
//    std::cout << "Running simulation2 for test size: " << num_bodies << std::endl;
//    runNbodySimulation(num_bodies, numIterations, deltaTime, damping);

//    //Run simulation 3 from leap_frog.h
//    std::cout << "Running LeapFrog for test size: " << num_bodies << std::endl;
//    LeapFrogIntegrator integrator1(num_bodies);
//    integrator1.step(numIterations, deltaTime);

//    // Run simulation 4 from verlet.h
//    std::cout << "Running Verlet for test size: " << num_bodies << std::endl;
//    VerletIntegrator integrator2(num_bodies);
//    integrator2.step(numIterations, deltaTime);

#endif

    return 0;

}









