cmake_minimum_required(VERSION 3.17)
project(my_project)

find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(PythonLibs REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --use_fast_math -arch=sm_86 -g -G")

file(GLOB KERNEL_SOURCES "src/kernel.cu")
file(GLOB N_BODY_SIM_1_SOURCES "src/n_body_sim_1.h")
file(GLOB N_BODY_SIM_2_SOURCES "src/n_body_sim_2.h")
file(GLOB LEAP_FROG_SOURCES "src/leap_frog.h" "src/leap_frog.cu")

cuda_add_executable(n_body.exe ${KERNEL_SOURCES} ${N_BODY_SIM_1_SOURCES} ${N_BODY_SIM_2_SOURCES} ${LEAP_FROG_SOURCES})

include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})

target_link_libraries(n_body.exe ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${PYTHON_LIBRARIES})




