cmake_minimum_required(VERSION 3.10)
project(navigation_engine)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find JNI headers manually to bypass AWT requirement
find_path(JNI_INCLUDE_DIR jni.h
    PATHS
    /usr/lib/jvm/java-17-openjdk-amd64/include
    /usr/lib/jvm/default-java/include
    /usr/lib/jvm/java-11-openjdk-amd64/include
    /usr/lib/jvm/java-8-openjdk-amd64/include
)

find_path(JNI_MD_INCLUDE_DIR jni_md.h
    PATHS
    /usr/lib/jvm/java-17-openjdk-amd64/include/linux
    /usr/lib/jvm/default-java/include/linux
    /usr/lib/jvm/java-11-openjdk-amd64/include/linux
    /usr/lib/jvm/java-8-openjdk-amd64/include/linux
)

if (NOT JNI_INCLUDE_DIR OR NOT JNI_MD_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find jni.h or jni_md.h. Please make sure JDK is installed.")
endif()

include_directories(
    src
    ${JNI_INCLUDE_DIR}
    ${JNI_MD_INCLUDE_DIR}
)

# Compile Navigation Engine Shared Library (so)
add_library(navigation_engine SHARED
    src/navigation_engine.cpp
    src/pointer_angle_detector.cpp
)

# Set compile definitions for shared library
target_compile_definitions(navigation_engine PRIVATE NAV_EXPORTS)
