# MinimapTracker

轻量、通用的实时小地图玩家位置追踪（Localization）与指针朝向检测（Pointer Orientation Detection）C++ 动态链接库。

本库已剔除了特定游戏的寻路及动作控制决策，只专注于提供高内聚的视觉特征定位与状态跟踪服务，适用于各种基于 2D/3D 小地图（Minimap）的位置跟踪应用场景。

---

## 🎨 系统架构手稿

![MinimapTracker 系统架构](minimap_tracker_architecture_excalidraw.svg)

本系统由两个核心处理模块构成：
1. **小地图朝向检测器 (Pointer Angle Detector)**：使用二值颜色过滤与 BFS 连通域连通性分析，结合图像几何矩求解亚像素级别的物理指针旋转角度。
2. **混合地图定位引擎 (Hybrid Tracker)**：在启动时使用 **SIFT 快速粗配准**（支持特征缓存秒级加载）锁定玩家初始坐标；定位成功后自动切入由 NCNN 驱动的 **SuperPoint 深度特征网络局部追踪**，实现高帧率、低延迟的亚像素高精锁定。

---

## ⚙️ 编译构建

### 依赖项
* **CMake** (>= 3.10)
* **OpenCV** (>= 4.0, 用于 SIFT 及矩阵变换)
* **NCNN** (用于运行 SuperPoint 深度神经网络推理)

### 本地编译 (Linux PC)
```bash
mkdir -p build
cmake -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ -S . -B build
cmake --build build
```

---

## 🔌 核心 API 接口说明

头文件位于 `src/navigation_engine.h`。

### 1. 指针朝向检测接口
```cpp
// 传入小地图指针区域像素，输出检测状态、角度(0~359度)和置信度
NAV_EXPORT bool detect_pointer_angle_c(const int32_t* pixels, int width, int height,
                                       float centerX, float centerY,
                                       bool* has_match, int* angle_deg, float* confidence);
```

### 2. SIFT + SuperPoint 定位跟踪接口
```cpp
// 1. 初始化定位器（自动载入大地图、特征缓存和 NCNN param/bin 权重）
NAV_EXPORT void* player_tracker_create(const char* map_path,
                                    const char* cache_path,
                                    const char* model_dir,
                                    int minimap_left,
                                    int minimap_top,
                                    int minimap_width,
                                    int minimap_height,
                                    int base_search_radius,
                                    int max_lost_frames,
                                    double clahe_limit,
                                    float match_ratio,
                                    int min_match_count,
                                    double ransac_threshold);

// 2. 输入当前帧整幅图像的 32 位 RGBA 像素数据，解算出在大地图上的精确像素坐标 (out_x, out_y)
NAV_EXPORT bool player_tracker_locate(void* handle, const int32_t* frame_pixels, int w, int h, 
                                    float* out_x, float* out_y, float* confidence, int* cost_ms);

// 3. 释放定位器实例，回收堆内存
NAV_EXPORT void player_tracker_release(void* handle);
```

---

## 🧪 离线视频仿真测试

项目中提供了标准的 Python ctypes 测试套件 `test/test_video.py`。
它会在无 JVM 依赖的条件下，加载真实游戏录像并调用编译出的 `.so` 库进行连续定位，将检测出的朝向角度和物理坐标实时以 HUD 雷达图层形式渲染并叠加在输出视频中。

**运行测试**：
```bash
python3 test/test_video.py <您的测试视频.mp4>
```
