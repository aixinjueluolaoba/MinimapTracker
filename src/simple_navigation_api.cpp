#include "navigation_engine.h"

#include <opencv2/imgproc.hpp>

#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

constexpr int kMinimapLeft = 1072;
constexpr int kMinimapTop = 25;
constexpr int kMinimapWidth = 127;
constexpr int kMinimapHeight = 127;
constexpr int kMinimumFrameWidth = 1200;
constexpr int kMinimumFrameHeight = 153;
constexpr int kBaseSearchRadius = 150;
constexpr int kMaxLostFrames = 4;
constexpr double kClaheLimit = 3.0;
constexpr float kMatchRatio = 0.9f;
constexpr int kMinMatchCount = 5;
constexpr double kRansacThreshold = 8.0;

struct SimpleNavigationEngine {
    void* pointer_detector = nullptr;
    void* player_tracker = nullptr;
    std::string last_error;

    ~SimpleNavigationEngine() {
        player_tracker_release(player_tracker);
        pointer_detector_release(pointer_detector);
    }
};

thread_local std::string g_last_create_error;

std::string join_path(const std::string& base, const char* child) {
    if (base.empty() || base.back() == '/') {
        return base + child;
    }
    return base + "/" + child;
}

bool is_readable_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    return input.good();
}

void require_file(const std::string& path, const char* description) {
    if (!is_readable_file(path)) {
        throw std::runtime_error(std::string(description) + " not found: " + path);
    }
}

void validate_resources(const std::string& model_root, const std::string& map_path) {
    if (model_root.empty()) {
        throw std::invalid_argument("model_root_dir is empty");
    }
    if (map_path.empty()) {
        throw std::invalid_argument("map_path is empty");
    }

    const std::string pointer_dir = join_path(model_root, "pointer_model");
    const std::string superpoint_dir = join_path(model_root, "superpoint_model");
    require_file(join_path(pointer_dir, "pointer_angle_cnn_v2_bgr_pool2.ncnn.param"),
                 "pointer param");
    require_file(join_path(pointer_dir, "pointer_angle_cnn_v2_bgr_pool2.ncnn.bin"),
                 "pointer model");
    require_file(map_path, "map");

    const bool has_fp32 =
            is_readable_file(join_path(superpoint_dir, "official_superpoint_dense_480x640.param"))
            && is_readable_file(join_path(superpoint_dir, "official_superpoint_dense_480x640.bin"));
    const bool has_int8 =
            is_readable_file(join_path(superpoint_dir, "official_superpoint_dense_480x640_int8.param"))
            && is_readable_file(join_path(superpoint_dir, "official_superpoint_dense_480x640_int8.bin"));
    if (!has_fp32 && !has_int8) {
        throw std::runtime_error("SuperPoint param/bin pair not found under: " + superpoint_dir);
    }
}

SimpleNavigationEngine* cast_handle(void* handle) {
    return static_cast<SimpleNavigationEngine*>(handle);
}

}  // namespace

extern "C" {

void* navigation_engine_create(const char* model_root_dir,
                               const char* map_path,
                               const char* cache_path) {
    g_last_create_error.clear();
    try {
        const std::string model_root = model_root_dir == nullptr ? "" : model_root_dir;
        const std::string map = map_path == nullptr ? "" : map_path;
        const std::string cache = cache_path == nullptr ? "" : cache_path;
        validate_resources(model_root, map);

        auto engine = std::make_unique<SimpleNavigationEngine>();
        const std::string pointer_dir = join_path(model_root, "pointer_model");
        const std::string superpoint_dir = join_path(model_root, "superpoint_model");

        engine->pointer_detector = pointer_detector_create(pointer_dir.c_str());
        if (engine->pointer_detector == nullptr) {
            throw std::runtime_error("failed to create pointer detector");
        }

        engine->player_tracker = player_tracker_create(
                map.c_str(),
                cache.c_str(),
                superpoint_dir.c_str(),
                kMinimapLeft,
                kMinimapTop,
                kMinimapWidth,
                kMinimapHeight,
                kBaseSearchRadius,
                kMaxLostFrames,
                kClaheLimit,
                kMatchRatio,
                kMinMatchCount,
                kRansacThreshold);
        if (engine->player_tracker == nullptr) {
            throw std::runtime_error("failed to create player tracker");
        }

        return engine.release();
    } catch (const std::exception& error) {
        g_last_create_error = error.what();
        return nullptr;
    } catch (...) {
        g_last_create_error = "unknown error while creating navigation engine";
        return nullptr;
    }
}

int navigation_engine_process_bgr(void* handle,
                                  const uint8_t* frame_bgr,
                                  int width,
                                  int height,
                                  int stride_bytes,
                                  NavigationEngineResult* out_result) {
    auto* engine = cast_handle(handle);
    if (out_result != nullptr) {
        std::memset(out_result, 0, sizeof(*out_result));
    }
    if (engine == nullptr) {
        g_last_create_error = "navigation engine handle is null";
        return 0;
    }

    engine->last_error.clear();
    if (frame_bgr == nullptr) {
        engine->last_error = "frame_bgr is null";
        return 0;
    }
    if (out_result == nullptr) {
        engine->last_error = "out_result is null";
        return 0;
    }
    if (width <= 0 || height <= 0) {
        engine->last_error = "frame dimensions must be positive";
        return 0;
    }
    if (stride_bytes < width * 3) {
        engine->last_error = "stride_bytes must be at least width * 3";
        return 0;
    }
    if (width < kMinimumFrameWidth || height < kMinimumFrameHeight) {
        engine->last_error = "frame is smaller than the configured minimap region";
        return 0;
    }

    try {
        cv::Mat bgr(height,
                    width,
                    CV_8UC3,
                    const_cast<uint8_t*>(frame_bgr),
                    static_cast<size_t>(stride_bytes));
        cv::Mat bgra;
        cv::cvtColor(bgr, bgra, cv::COLOR_BGR2BGRA);
        const auto* packed_argb = reinterpret_cast<const int32_t*>(bgra.data);

        bool pointer_detected = false;
        float angle_deg = 0.f;
        float ignored_pointer_confidence = 0.f;
        pointer_detector_detect(
                engine->pointer_detector,
                packed_argb,
                width,
                height,
                &pointer_detected,
                &angle_deg,
                &ignored_pointer_confidence);

        float x = 0.f;
        float y = 0.f;
        float ignored_location_confidence = 0.f;
        int locate_cost_ms = 0;
        const bool located = player_tracker_locate(
                engine->player_tracker,
                packed_argb,
                width,
                height,
                &x,
                &y,
                &ignored_location_confidence,
                &locate_cost_ms);

        out_result->located = located ? 1 : 0;
        out_result->x = x;
        out_result->y = y;
        out_result->locate_cost_ms = locate_cost_ms;
        out_result->pointer_detected = pointer_detected ? 1 : 0;
        out_result->angle_deg = angle_deg;
        return 1;
    } catch (const std::exception& error) {
        engine->last_error = error.what();
        return 0;
    } catch (...) {
        engine->last_error = "unknown error while processing BGR frame";
        return 0;
    }
}

const char* navigation_engine_last_error(void* handle) {
    auto* engine = cast_handle(handle);
    return engine == nullptr ? g_last_create_error.c_str() : engine->last_error.c_str();
}

void navigation_engine_release(void* handle) {
    delete cast_handle(handle);
}

}  // extern "C"
