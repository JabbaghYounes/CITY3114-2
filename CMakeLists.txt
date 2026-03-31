cmake_minimum_required(VERSION 3.14)
project(svm_classifier LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(svm_classifier
    src/main.cpp
    src/data_loader.cpp
    src/kernel.cpp
    src/svm.cpp
    src/evaluation.cpp
)

target_include_directories(svm_classifier PRIVATE src)
