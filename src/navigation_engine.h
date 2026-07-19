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

// Action constants
constexpr int kActionWait = 0;
constexpr int kActionStop = 1;
constexpr int kActionStart = 2;
constexpr int kActionUpdate = 3;
constexpr int kActionComplete = 4;
constexpr int kActionError = 5;

// Status constants
constexpr int kStatusNone = 0;
constexpr int kStatusMoving = 2;
constexpr int kStatusComplete = 3;
constexpr int kStatusFailed = 4;

NAV_EXPORT void* navigation_engine_create();

NAV_EXPORT bool navigation_engine_start(void* handle,
                                        float target_x,
                                        float target_y,
                                        float initial_hold_angle_deg,
                                        int base_speed_percent,
                                        float arrival_distance_px,
                                        float slow_down_distance_px);

NAV_EXPORT bool navigation_engine_step(void* handle,
                                       float track_x,
                                       float track_y,
                                       bool track_found,
                                       bool manual_required,
                                       bool pointer_has_match,
                                       float pointer_angle_deg,
                                       bool pointer_frozen,
                                       int64_t pointer_age_ms,
                                       int64_t now_ms,
                                       int* action,
                                       int* status,
                                       int64_t* sleep_ms,
                                       float* hold_angle_deg,
                                       int* speed_percent,
                                       float* map_angle_deg,
                                       float* current_angle_deg,
                                       float* angle_delta_deg,
                                       float* signed_delta_deg,
                                       int* target_index,
                                       float* target_x,
                                       float* target_y,
                                       bool* using_cached_track,
                                       bool* jump_requested);

NAV_EXPORT void navigation_engine_release(void* handle);

NAV_EXPORT bool detect_pointer_angle_c(const int32_t* pixels, int width, int height,
                                       float centerX, float centerY,
                                       bool* has_match, int* angle_deg, float* confidence);

#ifdef __cplusplus
}
#endif
