cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(grid_world_dqn)

# LibTorch is only needed for the DQN targets (q_learning, dqn_test). Look for it
# but don't require it, so CI can configure/build/test the splitmix64 targets
# without downloading the ~300MB libtorch distribution.
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch")
find_package(Torch QUIET)

if(Torch_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
else()
    message(STATUS "LibTorch not found under ${CMAKE_CURRENT_SOURCE_DIR}/libtorch — skipping q_learning and dqn_test targets")
endif()

set(CPP_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cpp/include")

# fetch googletest automatically
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(googletest)

# enable ctest, cmake's testing tool
enable_testing()
include(GoogleTest)

if(Torch_FOUND)
    # Define the executable and link it to LibTorch
    add_executable(q_learning cpp/q_learning.cpp)
    target_include_directories(q_learning PRIVATE "${CPP_INCLUDE_DIR}")

    # Link libtorch
    target_link_libraries(q_learning "${TORCH_LIBRARIES}")
    set_property(TARGET q_learning PROPERTY CXX_STANDARD 17)

    # Define the test executable
    add_executable(dqn_test tests/cpp/test_dqn.cpp)
    target_include_directories(dqn_test PRIVATE "${CPP_INCLUDE_DIR}")

    # Link GTest (gtest_main) and LibTorch to the test executable
    target_link_libraries(dqn_test gtest_main "${TORCH_LIBRARIES}")
    set_property(TARGET dqn_test PROPERTY CXX_STANDARD 17)

    # register the test executable with ctest
    gtest_discover_tests(dqn_test)
endif()

# splitmix64 standalone executable (no LibTorch dependency)
add_executable(splitmix64 cpp/splitmix64.cpp)
target_include_directories(splitmix64 PRIVATE "${CPP_INCLUDE_DIR}")
set_property(TARGET splitmix64 PROPERTY CXX_STANDARD 17)

# splitmix64 gtest suite
add_executable(splitmix64_test tests/cpp/test_splitmix64.cpp)
target_include_directories(splitmix64_test PRIVATE "${CPP_INCLUDE_DIR}")
target_link_libraries(splitmix64_test gtest_main)
set_property(TARGET splitmix64_test PROPERTY CXX_STANDARD 17)
gtest_discover_tests(splitmix64_test)
