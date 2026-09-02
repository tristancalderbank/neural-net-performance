cmake_minimum_required(VERSION 3.20)

project(HelloWorld LANGUAGES CXX)

add_executable(training
    src/training.cpp
    src/network.cpp
    src/network.h
    src/model_io.cpp
    src/model_io.h
    src/image_loader.cpp
    src/image_loader.h
)

set_target_properties(training PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

target_compile_features(training PRIVATE cxx_std_17)

add_executable(viewer
    src/viewer.cpp
    src/network.cpp
    src/network.h
    src/model_io.cpp
    src/model_io.h
    src/image_loader.cpp
    src/image_loader.h
    external/imgui/imgui.cpp
    external/imgui/imgui_draw.cpp
    external/imgui/imgui_tables.cpp
    external/imgui/imgui_widgets.cpp
    external/imgui/backends/imgui_impl_glfw.cpp
    external/imgui/backends/imgui_impl_opengl3.cpp
)

target_include_directories(viewer PRIVATE
    external/imgui
    external/imgui/backends
    external/glfw/include
)

target_link_libraries(viewer PRIVATE
    "${CMAKE_SOURCE_DIR}/external/glfw/lib/glfw3.lib"
    opengl32
    user32
    gdi32
    shell32
    legacy_stdio_definitions
)

set_target_properties(viewer PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

target_compile_features(viewer PRIVATE cxx_std_17)
