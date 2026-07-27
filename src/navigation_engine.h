#pragma once

#include <cstdint>

// ==========================================
// 像素格式契约
// ==========================================
// 所有 `const int32_t* frame_pixels` 参数都是 **ARGB packed int32**，
// 即每个 int 的布局为 (A<<24)|(R<<16)|(G<<8)|B —— 与 Android
// Bitmap.getPixels(int[]) 和 test/test_video.py 的打包方式一致。
// 注意：在小端机器上它的内存字节序是 B,G,R,A，所以内部按 BGRA 解释。
// 若要传 AndroidBitmap_lockPixels 拿到的 RGBA_8888 原始缓冲区（内存序 R,G,B,A），
// 请改用 MapSiftTracker 的 JNI 接口，不要走这里的 C API。

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#define NAV_EXPORT __declspec(dllexport)
#else
#define NAV_EXPORT __attribute__((visibility("default")))
#endif

// ==========================================
// 1. Neural Pointer Detector API (神经网络指针检测)
// ==========================================

// 创建指针检测器实例（自动载入 NCNN 格式的模型 param 和 bin 权重）
NAV_EXPORT void* pointer_detector_create(const char* model_dir);

// 传入整帧的 ARGB packed 像素数据，解算出指针朝向角度 (0~359.9度)。
// 注意：该模型是纯角度回归、没有拒识头，has_match 只表示"推理成功"，
// confidence 是固定占位常量 1.0，不是概率，不要用它做阈值判断。
NAV_EXPORT bool pointer_detector_detect(void* handle, const int32_t* frame_pixels, int w, int h,
                                        bool* has_match, float* angle_deg, float* confidence);

// 释放指针检测器实例，回收堆内存
NAV_EXPORT void pointer_detector_release(void* handle);


// ==========================================
// 2. Hybrid Map Player Tracker API (地图定位器: SIFT + SuperPoint)
// ==========================================

// 初始化定位器（自动载入大地图、特征缓存和 NCNN param/bin 权重）
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

// 输入当前帧整幅图像的 ARGB packed 像素数据，解算出在大地图上的精确像素坐标 (out_x, out_y)。
// confidence 目前只是 found ? 1.0 : 0.0 的二值量，不是连续置信度。
NAV_EXPORT bool player_tracker_locate(void* handle, const int32_t* frame_pixels, int w, int h,
                                    float* out_x, float* out_y, float* confidence, int* cost_ms);

// 释放定位器实例，回收堆内存
NAV_EXPORT void player_tracker_release(void* handle);

// ==========================================
// 3. Simple Navigation Engine API (推荐给 Python / FFI 调用方)
// ==========================================

// 单帧聚合结果。located 和 pointer_detected 分别表示本帧是否得到定位与角度；
// navigation_engine_process_bgr() 返回 1 只表示调用成功，不保证两项均命中。
typedef struct NavigationEngineResult {
    int32_t located;
    float x;
    float y;
    int32_t locate_cost_ms;
    int32_t pointer_detected;
    float angle_deg;
} NavigationEngineResult;

// 使用生产默认参数一次性创建指针检测器和地图定位器。
// model_root_dir 下必须包含 pointer_model/ 和 superpoint_model/。
// cache_path 可传 NULL 或空字符串，此时不读写 SIFT 特征缓存。
NAV_EXPORT void* navigation_engine_create(const char* model_root_dir,
                                          const char* map_path,
                                          const char* cache_path);

// 直接接收 OpenCV 常用的 uint8 BGR 图像，允许每行带 padding。
// stride_bytes 必须不小于 width * 3。
NAV_EXPORT int navigation_engine_process_bgr(void* handle,
                                             const uint8_t* frame_bgr,
                                             int width,
                                             int height,
                                             int stride_bytes,
                                             NavigationEngineResult* out_result);

// 返回当前实例最近一次错误；创建失败时可传 NULL 获取本线程的创建错误。
// 返回指针由库持有，只保证在下一次 API 调用前有效。
NAV_EXPORT const char* navigation_engine_last_error(void* handle);

// 允许传 NULL。
NAV_EXPORT void navigation_engine_release(void* handle);

#ifdef __cplusplus
}
#endif
