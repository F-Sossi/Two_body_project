cmake_minimum_required(VERSION 3.17)
project(my_project)

find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_86")

file(GLOB KERNEL_SOURCES "src/*.cu")
file(GLOB VIEWER_SOURCES "src/viewer.h" "src/viewer.cpp")
file(GLOB TWO_BODY_SOURCES "src/two_body.h" "src/two_body.cpp")


cuda_add_executable(two_body.exe ${KERNEL_SOURCES} ${VIEWER_SOURCES} ${TWO_BODY_SOURCES})

include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})

target_link_libraries(two_body.exe ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})
