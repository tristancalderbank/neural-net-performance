cmake_minimum_required(VERSION 3.20)

project(HelloWorld LANGUAGES CXX)

add_executable(hello_world
    src/main.cpp
    src/image_loader.cpp
)

set_target_properties(hello_world PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

target_compile_features(hello_world PRIVATE cxx_std_17)
