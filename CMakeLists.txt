cmake_minimum_required(VERSION 3.17)
project(my_project)

find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(PythonLibs REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_86 -g -G")


file(GLOB KERNEL_SOURCES "src/kernel.cu")
file(GLOB TWO_BODY_SOURCES "src/two_body.h")


cuda_add_executable(two_body.exe ${KERNEL_SOURCES} ${TWO_BODY_SOURCES} ${SIMULATION_SOURCES})

include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})

target_link_libraries(two_body.exe ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${PYTHON_LIBRARIES})


