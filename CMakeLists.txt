cmake_minimum_required(VERSION 3.10)

project(tensorrt_llm_cpp)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

set(TRTLLM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../TensorRT-LLM" CACHE PATH "Path to TensorRT-LLM directory")
set(TRTLLM_LIB_PATH "${TRTLLM_DIR}/tensorrt_llm/libs/libtensorrt_llm.so")
set(TRTLLM_PLUGIN_PATH "${TRTLLM_DIR}/tensorrt_llm/libs/libnvinfer_plugin_tensorrt_llm.so")
set(TRTLLM_INCLUDE_DIR "${TRTLLM_DIR}/cpp/include")

## Adapted from examples/cpp/executor/CMakeLists.txt
execute_process(
    COMMAND bach -c "nm -f posix -$ ${TRTLLM_LIB_PATH} | grep __cxx11"
    RESULT_VARIABLE GLIB_CXX11_FOUND
    OUTPUT_QUIET)
if(GLIB_CXX11_FOUND EQUAL 0)
    set(USE_CXX11_ABI 1)
else()
    set(USE_CXX11_ABI 0)
endif()
add_compile_definitions("-D_GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI}")
##

find_package(CUDAToolkit REQUIRED)
if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "11")
  add_definitions("-DENABLE_BF16")
  message(
    STATUS
      "CUDAToolkit_VERSION ${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR} is greater or equal than 11.0, enable -DENABLE_BF16 flag"
  )
endif()

if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "11.8")
  add_definitions("-DENABLE_FP8")
  message(
    STATUS
      "CUDAToolkit_VERSION ${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR} is greater or equal than 11.8, enable -DENABLE_FP8 flag"
  )
endif()

# Set the root directory of TensorRT
if(NOT DEFINED TRT_DIR)
    message(FATAL_ERROR "The root of tensorrt -DTRT_DIR is not defined.")
endif()
if(NOT EXISTS ${TRT_DIR})
    message(FATAL_ERROR "The required TensorRT ${TRT_DIR} does not exist.")
endif()
set(TRT_INCLUDE_DIR "${TRT_DIR}/include")
set(TRT_LIB_PATH "${TRT_DIR}/lib/libnvinfer.so")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(CUDAToolkit REQUIRED)

# The tensorrt_llm shared lib
add_library(tensorrt_llm SHARED IMPORTED)
set_property(TARGET tensorrt_llm PROPERTY IMPORTED_LOCATION ${TRTLLM_LIB_PATH})
set_property(
    TARGET tensorrt_llm PROPERTY IMPORTED_LINK_INTERFACE_LIBRARIES
    CUDA::cuda_driver CUDA::cudart_static CUDA::nvml
)

# The nvinfer_plugin_tensorrt_llm shared lib
add_library(nvinfer_plugin_tensorrt_llm SHARED IMPORTED)
set_property(
    TARGET nvinfer_plugin_tensorrt_llm PROPERTY IMPORTED_LOCATION ${TRTLLM_PLUGIN_PATH}
)
set_property(
    TARGET nvinfer_plugin_tensorrt_llm PROPERTY IMPORTED_LINK_INTERFACE_LIBRARIES 
    tensorrt_llm
)

# Add submodule
set(JSON_INCLUDE_DIR "3rdparty/json/include")
set(CXXOPTS_INCLUDE_DIR "3rdparty/cxxopts/include")
set(SENTENCEPIECE_INCLUDE_DIR "3rdparty/tokenizers-cpp/include")
set(CPP_HTTPLIB_INCLUDE_DIR "3rdparty/cpp-httplib")
include_directories(
    ${TRT_INCLUDE_DIR} 
    ${TRTLLM_INCLUDE_DIR} 
    ${CUDA_INCLUDE_DIRS} 
    ${JSON_INCLUDE_DIR}
    ${CXXOPTS_INCLUDE_DIR} 
    ${SENTENCEPIECE_INCLUDE_DIR}
    ${CPP_HTTPLIB_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/trtllm_src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
)
add_subdirectory("3rdparty/tokenizers-cpp")
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/build/3rdparty/tokenizers-cpp/)

find_package(MPI REQUIRED)
find_package(OpenSSL REQUIRED)

file(GLOB TRTLLM_SRC "src/trtllm_src/*.cpp")
file(GLOB UTILS_SRC "src/utils/*_utils.cpp")
list(APPEND CPP_FILES ${TRTLLM_SRC} ${UTILS_SRC})

set(SOURCES
    infer.cpp
    ${CPP_FILES})
add_executable(tllm-infer ${SOURCES})
target_link_libraries(
    tllm-infer 
    nvinfer_plugin_tensorrt_llm 
    ${MPI_LIBRARIES} 
    ${TRT_LIB_PATH} 
    tokenizers_cpp
    ${OPENSSL_SSL_LIBRARY}
)

set(SOURCES
    serve.cpp
    ${CPP_FILES})
add_executable(tllm-serve ${SOURCES})
target_link_libraries(
    tllm-serve 
    nvinfer_plugin_tensorrt_llm 
    ${MPI_LIBRARIES} 
    ${TRT_LIB_PATH} 
    tokenizers_cpp
    OpenSSL::SSL OpenSSL::Crypto 
)