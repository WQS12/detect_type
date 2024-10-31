# 指定 CMake 的最低版本要求
cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(MyOpenCVONNXProject)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 查找 OpenCV 包
find_package(OpenCV REQUIRED)

# 包含 OpenCV 头文件
include_directories(${OpenCV_INCLUDE_DIRS})

# 设置 ONNX Runtime 路径 
set(ONNXRUNTIME_DIR "/home/hja/onnxruntime/")

# 包含 ONNX Runtime 头文件
include_directories(${ONNXRUNTIME_DIR}/include)

# 链接 ONNX Runtime 库
link_directories(${ONNXRUNTIME_DIR}/lib)
link_libraries(onnxruntime)

# 将源文件添加到可执行文件中
add_executable(MyOpenCVONNXProject demo.cpp)

# 链接 OpenCV 和 ONNX Runtime 库
target_link_libraries(MyOpenCVONNXProject ${OpenCV_LIBS} onnxruntime)

# 打印 OpenCV 和 ONNX Runtime 版本信息
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "ONNX Runtime include dirs: ${ONNXRUNTIME_DIR}/include")
message(STATUS "ONNX Runtime libraries: ${ONNXRUNTIME_DIR}/lib")
