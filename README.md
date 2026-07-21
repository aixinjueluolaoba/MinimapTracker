# MinimapTracker

轻量、通用的实时小地图玩家定位（Localization）与指针朝向检测（Pointer Orientation Detection）C++ 动态链接库。

本库已剔除特定游戏的寻路与动作决策，只提供视觉特征定位与状态跟踪，适用于各种基于 2D 小地图的位置跟踪场景。

---

## 🎬 效果演示

下面是在一段 1280×720、298 帧的真实游戏录像上跑出来的结果。

### 渲染结果（含定位与指针角度）

<video src="docs/media/rendered_result.mp4" poster="docs/hud_preview.png" controls muted playsinline width="100%"></video>

> 若上方播放器未渲染，直接打开 [`docs/media/rendered_result.mp4`](docs/media/rendered_result.mp4)。

![HUD 预览](docs/hud_preview.png)

HUD 各部分含义：

| 位置 | 内容 |
|---|---|
| 左上 | 状态面板：CNN 指针角度、定位模式（global / local）、解算出的全局坐标、单帧耗时 |
| 右侧 | 大地图局部贴图，红色十字为解算出的玩家位置，随移动同步滚动 |
| 右下 | **CNN 的真实输入**（32×32 patch 放大），黄色箭头是网络解算出的方向 |
| 左下 | 雷达盘，标准极坐标（0°=东，90°=北） |

右下角这块是刻意加的：把网络实际看到的像素和它输出的角度画在一起，就能直接核对角度对不对，而不是靠"看起来没问题"。

<img src="docs/pointer_zoom.png" width="320" alt="指针角度核对">

### 输入原始录像（未处理）

<video src="docs/media/input_capture.mp4" controls muted playsinline width="100%"></video>

> 直接打开：[`docs/media/input_capture.mp4`](docs/media/input_capture.mp4)

两个视频都是 H.264 / yuv420p，可直接在浏览器播放。仓库 `.gitignore` 默认忽略 `*.mp4`，这两个通过 `!docs/media/*.mp4` 例外保留。

### 实测数据

```
定位          292/298 帧成功 (98.0%)，0 次丢失后重配准
首次锁定      第 6 帧，全局 SIFT 配准，(5440.8, 2824.8)，耗时 29ms
              第 1 帧的误匹配被质量门禁正确拒绝
单帧耗时      局部 18-25ms，全局配准 29-93ms
指针角度      与另一套独立算法交叉验证，中位差 8.1°，93.1% 在 20° 以内
```

---

## 🎨 系统架构

![MinimapTracker 系统架构](minimap_tracker_architecture_excalidraw.svg)

两个核心模块：

1. **神经网络指针检测器**：NCNN 加载 CNN 做单次前向角度回归，输出 `[sin, cos]` 后 `atan2` 解角。相比传统的颜色过滤 + BFS 连通域方法，在光照渐变和指针对齐偏差下更稳。
2. **混合地图定位引擎**：启动或丢失时用 **SIFT 全局配准**（特征带缓存）锁定坐标；锁定后切 **SuperPoint** 局部深度特征追踪，把搜索限制在上一帧位置 ±150 px（即 300×300 窗口）。

---

## ⚙️ 编译构建

### 依赖
* CMake >= 3.10
* **OpenCV >= 4.0（C++ 开发包，不是 Python 的 `cv2`）** —— 用于 SIFT 与矩阵变换
* NCNN —— 用于 SuperPoint 与指针 CNN 推理

### Android（arm64-v8a）

已验证可用：

```bash
NDK=/path/to/android-sdk/ndk/26.1.10909125
cmake -S . -B build-android \
      -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-29 \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build-android -j8
```

### Linux PC

```bash
cmake -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
      -DCMAKE_CXX_STANDARD=17 -S . -B build
cmake --build build
```

两个注意点：

1. **必须先装 OpenCV C++ 开发包**（`libopencv-dev` 或等价物）。conda 里的 `cv2` 是 Python 绑定，不带 C++ 头文件和 CMake config，`find_package(OpenCV REQUIRED)` 找不到它。
2. **显式指定编译器**。如果 `PATH` 里的 `cc` 被别的东西占用（不是真编译器），CMake 探测阶段会报 `error: unknown option '-o'`。

---

## 🔌 核心 API

头文件 `src/navigation_engine.h`。

### 像素格式契约（重要）

所有 `const int32_t* frame_pixels` 都是 **ARGB packed int32**：

```
每个 int = (A<<24) | (R<<16) | (G<<8) | B
```

与 Android `Bitmap.getPixels(int[])` 一致。**小端机器上它的内存字节序是 `B,G,R,A`**，所以库内部按 BGRA 解释。

若你手上是 `AndroidBitmap_lockPixels` 拿到的 RGBA_8888 原始缓冲区（内存序真是 `R,G,B,A`），请走 `MapSiftTracker` 的 JNI 接口，不要用这里的 C API。

### 指针检测

```cpp
NAV_EXPORT void* pointer_detector_create(const char* model_dir);
NAV_EXPORT bool  pointer_detector_detect(void* handle, const int32_t* frame_pixels, int w, int h,
                                         bool* has_match, float* angle_deg, float* confidence);
NAV_EXPORT void  pointer_detector_release(void* handle);
```

> ⚠️ 该网络是纯角度回归，**没有拒识头**，输出向量已 L2 归一化（实测 `|[sin,cos]|` 恒为 1.0）。
> 所以 `has_match` 只表示"推理成功"，`confidence` 是固定占位常量 1.0，**不是概率，不要拿它做阈值**。

### 地图定位

```cpp
NAV_EXPORT void* player_tracker_create(const char* map_path, const char* cache_path,
                                       const char* model_dir,
                                       int minimap_left, int minimap_top,
                                       int minimap_width, int minimap_height,
                                       int base_search_radius, int max_lost_frames,
                                       double clahe_limit, float match_ratio,
                                       int min_match_count, double ransac_threshold);
NAV_EXPORT bool  player_tracker_locate(void* handle, const int32_t* frame_pixels, int w, int h,
                                       float* out_x, float* out_y, float* confidence, int* cost_ms);
NAV_EXPORT void  player_tracker_release(void* handle);
```

`base_search_radius` 是**半径**，不是窗口边长。`confidence` 目前只是 `found ? 1.0 : 0.0` 的二值量。

---

## 🧪 测试

### `test/test_video_pure_python.py` —— 纯 Python 参照实现

不依赖编译产物，用 `cv2` + `ncnn` 的 Python 绑定复现定位与测角流程，渲染出上面那段视频。

```bash
python3 test/test_video_pure_python.py [视频路径] [--no-play]
```

`--no-play` 跳过结尾的 mpv 播放，用于无人值守运行。大地图 SIFT 特征首次提取约 2.6 秒（139562 个特征），之后缓存在 `build/sift_cache_py.npz`。

**与生产 C++ 的已知差异，不要当等价物**：

* 本脚本**全程使用 SIFT**，局部模式只是缩小搜索范围，**没有 SuperPoint**，耗时数字不可与 C++ 对比。
* C++ 的全局搜索有 2 秒节流，本脚本没有。
* 没有移植 C++ 的 `smooth_position()`（EMA 平滑 + 跳变拒绝），输出的是原始解算坐标。

### `test/test_video.py` —— ctypes 集成测试

加载编译出的 `.so`，在无 JVM 依赖的条件下直接调 C API。这是**唯一能真正验证 C++ 实现**的手段，需要先完成上面的 Linux PC 构建。

```bash
python3 test/test_video.py [视频路径]
```

输出写到 `test_output_ctypes.mp4`（与纯 Python 版分开，避免互相覆盖）。

---

## ⚠️ 踩过的坑

三个都是同一类错误：**把非连续或非预期布局的缓冲区直接交给了推理框架**。改这类代码时，任何进 `from_pixels` / `ncnn.Mat` 的缓冲区都要先确认 stride 和排布。

| # | 问题 | 后果 |
|---|---|---|
| 1 | 整帧的 ROI 视图喂给不带 stride 的 `from_pixels` | 角度误差中位数 **141°** |
| 2 | ARGB packed int 当成 RGBA 解释（R/B 对调） | 角度误差中位数 **136°** |
| 3 | numpy 转置视图未转连续就交给 `ncnn.Mat` | 网络收到交错 BGR 当 planar，角度完全错 |

**坑 1**：`frame(mm_rect)(patch_rect)` 得到的是**视图**，行间距仍是整帧的 3840 字节，而 `from_pixels(data, type, w, h)` 按 `w*3 = 96` 字节紧凑读取 —— 只有第 0 行是对的。用带 stride 的重载，传 `patch.step`。

**坑 3**：`patch.transpose(2,0,1).astype(np.float32)` 看着像做了拷贝，但 `astype` 默认 `order='K'` 会**保留输入的内存布局**，转出来的数组 strides 是 `(4, 384, 12)`，内存里仍是 HWC 交错。必须显式 `np.ascontiguousarray(...)`。

---

## 📌 已知限制

* **SuperPoint 局部追踪路径尚未端到端验证** —— 需要 Android 设备或可用的主机构建。上面的实测数据全部来自纯 Python 的 SIFT 路径。
* CNN 没有拒识能力，上层若需要"当前帧是否有指针"的判断，得自己补。
* 与原 Rust 算法交叉验证时，约 6.6% 的帧差异较大（最大 173.8°），尚未逐帧归因。
* 演示录像的游戏画面顶部本身就显示玩家坐标，可作 ground truth 逐帧比对精度 —— 尚未做。
