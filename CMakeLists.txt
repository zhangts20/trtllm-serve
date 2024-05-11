cmake_minimum_required(VERSION 3.12)

project(my_test)

## CUDA
find_package(CUDAToolkit REQUIRED)
if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "11")
    add_definitions("-DENABLE_BF16")
    message(
      STATUS
        "CUDA version is greater or equal than 11.0, enable -DENABLE_BF16 flag"
    )
endif()

if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "11.8")
  add_definitions("-DENABLE_FP8")
  message(
    STATUS
      "CUDA version is greater or equal than 11.8, enable -DENABLE_FP8 flag"
  )
endif()

include_directories(${CUDAToolkit_INCLUDE_DIRS})
link_directories(${CUDAToolkit_LIBRARY_DIR})

## TensorRT
if(NOT DEFINED TRT_DIR)
    message(FATAL_ERROR "You must specify the directory of TensorRT.")
endif()
include_directories(${TRT_DIR}/include/)
link_directories(${TRT_DIR}/lib/)

## TensorRT-LLM
if(NOT DEFINED TRTLLM_DIR)
    message(FATAL_ERROR "You must specify the directory of TensorRT-LLM.")
endif()
include_directories(${TRTLLM_DIR}/cpp/)
include_directories(${TRTLLM_DIR}/cpp/include/)
link_directories(${TRTLLM_DIR}/tensorrt_llm/libs/)
link_directories(${TRTLLM_DIR}/cpp/build/tensorrt_llm/plugins/)

## C++ ABI
if(NOT DEFINED USE_CXX11_ABI)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} "-c"
                "import torch; print(torch.compiled_with_cxx11_abi(),end='');"
        RESULT_VARIABLE _PYTHON_SUCCESS
        OUTPUT_VARIABLE USE_CXX11_ABI)
    if(USE_CXX11_ABI)
        set(USE_CXX11_ABI 1)
    else()
        set(USE_CXX11_ABI 0)
    endif()
    message(STATUS "USE_CXX11_ABI is set by python Torch to ${USE_CXX11_ABI}")
endif()
if(NOT USE_CXX11_ABI)
    add_compile_options("-D_GLIBCXX_USE_CXX11_ABI=0")
endif()

add_executable(demo main.cpp)
target_link_libraries(demo libnvinfer_plugin_tensorrt_llm.so tensorrt_llm th_common cudart)
