# MinimapTracker

用于从 1280×720 游戏画面中计算玩家地图坐标和小地图指针角度的 C++ 动态库。

## 编译

依赖 CMake、OpenCV C++ 和 NCNN。

Linux：

```bash
cmake -S . -B build \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++
cmake --build build -j8
```

产物：`build/libfishing_native.so`

Android arm64-v8a：

```bash
NDK=/path/to/android-sdk/ndk/26.1.10909125

cmake -S . -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-29 \
  -DOpenCV_DIR=/path/to/OpenCV-android-sdk/sdk/native/jni \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-android -j8
```

产物：`build-android/libfishing_native.so`

NCNN 路径由 `CMakeLists.txt` 中的 `NCNN_ROOT` 配置。

## Python 使用

Python 只负责加载 `.so`，所有算法均在 Native 库中执行。

```bash
PYTHONPATH=python python3 your_script.py
```

```python
from minimap_tracker import NavigationEngine

with NavigationEngine(
    model_root="models",
    map_path="/path/to/big_map.png",
    cache_path="build/sift_cache.bin",
) as engine:
    result = engine.process(frame_bgr)  # uint8 BGR ndarray

    if result.located:
        print("position:", result.x, result.y)

    if result.pointer_detected:
        print("angle:", result.angle_deg)
```

## Android 使用

1. 将 `libfishing_native.so` 放入：

   ```text
   app/src/main/jniLibs/arm64-v8a/
   ```

2. 加载动态库：

   ```java
   System.loadLibrary("fishing_native");
   ```

3. 在 JNI 层包含 `src/navigation_engine.h`，调用：

   ```c
   navigation_engine_create(...)
   navigation_engine_process_bgr(...)
   navigation_engine_release(...)
   ```

`navigation_engine_process_bgr()` 接收连续的 `uint8 BGR` 图像。默认小地图区域为
`(1072, 25, 127, 127)`。
