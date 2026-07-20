cmake_minimum_required(VERSION 3.10)
project(MinimapTracker)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. OpenCV configuration
if (ANDROID_ABI)
    if (NOT OpenCV_DIR)
        set(OpenCV_DIR /home/diana/app/OpenCV-android-sdk/sdk/native/jni)
    endif()
    find_package(OpenCV REQUIRED)
else()
    # On host systems, try default path or standard find_package search
    if (NOT OpenCV_DIR)
        set(OpenCV_DIR /usr/share/opencv4)
    endif()
    find_package(OpenCV REQUIRED)
endif()

# 2. JNI configuration (required for Android bindings)
# Find JNI headers manually to bypass AWT requirement on host systems
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
    message(FATAL_ERROR "Could not find jni.h or jni_md.h. Make sure JDK is installed.")
endif()

# 3. NCNN configuration
# Use Android NCNN for mobile builds, and Ubuntu x86_64 NCNN for PC test builds
if (ANDROID_ABI)
    # Target path relative to app's main directory
    set(NCNN_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../app/src/main/ncnn/ncnn-20260113-android/${ANDROID_ABI})
else()
    # PC local path
    set(NCNN_ROOT /home/diana/fishing/tools/superpoint_superglue_ncnn/ncnn/ncnn-20260113-ubuntu-2204)
endif()

add_library(ncnn STATIC IMPORTED)
set_target_properties(
    ncnn
    PROPERTIES
    IMPORTED_LOCATION ${NCNN_ROOT}/lib/libncnn.a
)

# 4. Include directories
include_directories(
    src
    ${OpenCV_INCLUDE_DIRS}
    ${JNI_INCLUDE_DIR}
    ${JNI_MD_INCLUDE_DIR}
    ${NCNN_ROOT}/include/ncnn
)

# 5. Compile Shared Library
add_library(navigation_engine SHARED
    src/navigation_engine.cpp
    src/pointer_angle_detector.cpp
    src/map_sift_tracker.cpp
)

# Set compile definitions for shared library
target_compile_definitions(navigation_engine PRIVATE NAV_EXPORTS)

# Link OpenCV, NCNN and logging/thread libraries
if (ANDROID_ABI)
    find_library(log-lib log)
    find_library(jnigraphics-lib jnigraphics)
    target_link_libraries(navigation_engine PRIVATE
        ncnn
        ${log-lib}
        ${jnigraphics-lib}
        android
        -fopenmp
        -static-openmp
        ${OpenCV_LIBS}
    )
else()
    target_link_libraries(navigation_engine PRIVATE
        ncnn
        pthread
        -fopenmp
        ${OpenCV_LIBS}
    )
endif()
