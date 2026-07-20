#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#define NAV_EXPORT __declspec(dllexport)
#else
#define NAV_EXPORT __attribute__((visibility("default")))
#endif

// ==========================================
// 1. Pointer Angle Detector API (指针角度检测)
// ==========================================
NAV_EXPORT bool detect_pointer_angle_c(const int32_t* pixels, int width, int height,
                                       float centerX, float centerY,
                                       bool* has_match, int* angle_deg, float* confidence);

// ==========================================
// 2. Hybrid Map Player Tracker API (地图定位器: SIFT + SuperPoint)
// ==========================================

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

NAV_EXPORT bool player_tracker_locate(void* handle, const int32_t* frame_pixels, int w, int h,
                                    float* out_x, float* out_y, float* confidence, int* cost_ms);

NAV_EXPORT void player_tracker_release(void* handle);

#ifdef __cplusplus
}
#endif
