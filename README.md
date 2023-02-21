# Two_body_project

Project implementing CUDA operations for Orbital mechanics

![PlantUML model](https://www.plantuml.com/plantuml/png/FOsx2i90401xlq8BIxDYQPNAMj0-NEuMitAtU-o3uD-h25R3mC2CvaazMjtiY7EDWg5rkxXtEmDeMeS7-H9p66eMTcGViZIu76vYHd1VPPgblAQkyiJiq18nN7-oFnqbl1BZlnlwzpKYbm_V)

![PlantUML model](https://www.plantuml.com/plantuml/png/ZP5HIWCn48RVSugXJnNP2nIH2gAlOht0a1stmMIpp4mMYlRkreIcn0hhKy8t_-IV-RCLHTOKl9dGVk10iDgva3ogY-CAFWs0zIXomgZalLCg5E1sYk9-P1kOoMhaGcVoqJezjFGziYNP03BZeI0RmvIKdd9bVLEW6vK6HgKCZkRYdtN5kpPVgvuPPwTvUlJ_QtdG46NQ4plxo3WiwdVH8xzYxxBB2vD8ucMRjpqdqKLPctwMSXhFyM3VrtzOo_csWGhhSvL0rzr3Ji90EU7kg-lWEgcs_YkMjlXt1gscB7nYbvNoB7KoSpHR6liK_3S0)

Note: install glut and opengl libraries for visualisation
sudo apt-get install mesa-common-dev libglu1-mesa-dev freeglut3-dev

Added Cmake support for the project: sudo apt install cmake

Complile with cmake: 

mkdir build
cd build
cmake ..
make
./my_project

To view the .md files install markdown preview enhanced


To compile: nvcc kernel.cu -o tbp.exe -lGL -lGLU -lglut
To run: ./lab3

