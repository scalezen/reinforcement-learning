cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(grid_world_dqn)

# Tell CMake where to find LibTorch
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch")
find_package(Torch REQUIRED)

# Set standard compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Define the executable and link it to LibTorch
add_executable(q_learning q_learning.cpp)


# Link libtorch
target_link_libraries(q_learning "${TORCH_LIBRARIES}")
set_property(TARGET q_learning PROPERTY CXX_STANDARD 17)
