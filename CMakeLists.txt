cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(grid_world_dqn)

# Tell CMake where to find LibTorch
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch")
find_package(Torch REQUIRED)

# Set standard compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# fetch googletest automatically
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)

# enable ctest, cmake's testing tool
enable_testing()

# Define the executable and link it to LibTorch
add_executable(q_learning q_learning.cpp)

# Link libtorch
target_link_libraries(q_learning "${TORCH_LIBRARIES}")
set_property(TARGET q_learning PROPERTY CXX_STANDARD 17)

# 4. Define the test executable
add_executable(dqn_test test_dqn.cpp)

# 5. Link GTest (gtest_main) and LibTorch to the test executable
target_link_libraries(dqn_test gtest_main "${TORCH_LIBRARIES}")
set_property(TARGET dqn_test PROPERTY CXX_STANDARD 17)

# register the test executable with ctest
include(GoogleTest)
gtest_discover_tests(dqn_test)
