cmake_minimum_required(VERSION 3.14)
project(pytorch_gpu_test)

find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 14)

include_directories(${TORCH_INCLUDE_DIRS})
include_directories(/home/len/pytorch1.2/vision/install/include)
link_directories(/home/len/pytorch1.2/vision/install/lib)

add_executable(pytorch_gpu_test main.cpp)

target_link_libraries(pytorch_gpu_test
        ${TORCH_CUDA_LIBRARIES}
        ${TORCH_LIBRARIES}
        torchvision)