﻿//---------------------------------------------------------------------------
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_gl_interop.h>
#include "two_body.h"




int main()
{
    const int numIterations = 1000;
    const int numBodiesOptions[] = {1024, 2048, 4096, 8192};
    const int numBodiesOptionCount = sizeof(numBodiesOptions) / sizeof(numBodiesOptions[0]);
    const int p = 32;
    const int q = 1;
    const float deltaTime = 0.1;
    const float damping = 0.95;

    int numBodies = 1024;

    // // Prompt the user to select an option and loop until a valid option is entered
    // int option = 0;
    // while (option < 1 || option > numBodiesOptionCount) {
    //     printf("Select an option:\n");
    //     printf("1. %d bodies\n", numBodiesOptions[0]);
    //     printf("2. %d bodies\n", numBodiesOptions[1]);
    //     printf("3. %d bodies\n", numBodiesOptions[2]);
    //     printf("4. %d bodies\n", numBodiesOptions[3]);
    //     if (scanf("%d", &option) != 1 || option < 1 || option > numBodiesOptionCount) {
    //         printf("Invalid input. Please select a valid option.\n");
    //     }
    // }

    // // Get the number of bodies from the selected option
    // int numBodies = numBodiesOptions[option - 1];

    // print the user's inputs
    //printf("Number of bodies: %d \t Number of iterations: %d \t p: %d \t q: %d  \n", numBodies, numIterations, p, q);

    // Call the simulation function with the user's inputs
    simulateNbodySystem(numBodies, numIterations, p, q, deltaTime, damping);

    return 0;
}









