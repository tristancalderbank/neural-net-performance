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
    XCODE_GENERATE_SCHEME TRUE
    XCODE_SCHEME_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

target_compile_features(training PRIVATE cxx_std_17)

if(APPLE)
    target_compile_definitions(training PRIVATE NNP_PLATFORM_MACOS=1)
elseif(WIN32)
    target_compile_definitions(training PRIVATE NNP_PLATFORM_WINDOWS=1)
endif()

if(MSVC)
    target_compile_options(training PRIVATE /arch:AVX2 /Qvec-report:2 /EHsc
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:RelWithDebInfo>:/O2 /Zi>
    )
    target_link_options(training PRIVATE $<$<CONFIG:RelWithDebInfo>:/DEBUG>)
endif()

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
)

if(WIN32)
    target_include_directories(viewer PRIVATE external/glfw/include)
    target_link_libraries(viewer PRIVATE
        "${CMAKE_SOURCE_DIR}/external/glfw/lib/glfw3.lib"
        opengl32
        user32
        gdi32
        shell32
        legacy_stdio_definitions
    )
elseif(APPLE)
    find_package(glfw3 CONFIG REQUIRED)
    find_package(OpenGL REQUIRED)
    target_link_libraries(viewer PRIVATE glfw OpenGL::GL)
else()
    message(FATAL_ERROR "The viewer is currently supported only on Windows and macOS.")
endif()

set_target_properties(viewer PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    XCODE_GENERATE_SCHEME TRUE
    XCODE_SCHEME_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

target_compile_features(viewer PRIVATE cxx_std_17)

if(APPLE)
    target_compile_definitions(viewer PRIVATE NNP_PLATFORM_MACOS=1)
elseif(WIN32)
    target_compile_definitions(viewer PRIVATE NNP_PLATFORM_WINDOWS=1)
endif()

if(MSVC)
    target_compile_options(viewer PRIVATE /arch:AVX2 /EHsc
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:RelWithDebInfo>:/O2 /Zi>
    )
    target_link_options(viewer PRIVATE $<$<CONFIG:RelWithDebInfo>:/DEBUG>)
endif()
