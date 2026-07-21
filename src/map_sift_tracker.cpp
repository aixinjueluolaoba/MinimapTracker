// AndroidBitmap / android_log 只在 Android 上存在；jni.h 在主机上由 JDK 提供，
// 所以纯 C API（player_tracker_*）可以脱离 Android 编译，供 ctypes 测试使用。
#ifdef __ANDROID__
#include <android/bitmap.h>
#include <android/log.h>
#endif
#include <jni.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

#include <net.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#ifdef __ANDROID__
#define TRACK_LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "luoke", __VA_ARGS__)
#define TRACK_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "luoke", __VA_ARGS__)
#else
#include <cstdio>
#define TRACK_LOGD(...) do { printf("[tracker DEBUG] "); printf(__VA_ARGS__); printf("\n"); } while(0)
#define TRACK_LOGE(...) do { fprintf(stderr, "[tracker ERROR] "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#endif

namespace {

constexpr long long kGlobalSearchFallbackDelayMs = 2000;
constexpr uint32_t kFeatureCacheMagic = 0x53465443;  // SFTC
constexpr uint32_t kFeatureCacheVersion = 3;
constexpr uint32_t kLegacyFeatureCacheVersion = 2;
constexpr bool kEnableStepTiming = true;
constexpr int kTimingLogInterval = 120;
constexpr int kSlowLocalMatchLogMs = 250;
// 局部 SuperPoint 提取时喂给网络的输入尺寸。
// 曾经有 fast/precise/full 三档，但三者取值一直都是 192，从未真正区分过分辨率，
// 只是被借用来切换特征来源。特征来源的分支已删除，这里合并成单一常量。
constexpr int kLocalNetWidth = 192;
constexpr int kLocalNetHeight = 192;
constexpr int kGridStride = 8;
constexpr int kDescriptorDim = 256;
constexpr int kMaxKeypoints = 512;
constexpr float kSuperPointThreshold = 0.005f;
constexpr int kSuperPointNmsRadius = 4;
constexpr int kSuperPointBorder = 4;
constexpr int kMaxCandidatesBeforeNms = 6000;
constexpr int kSuperPointTileSize = 960;
constexpr int kSuperPointTileOverlap = 160;
constexpr float kSuperPointGlobalNmsRadius = 4.f;
constexpr int kMinLocalGoodMatches = 20;
constexpr int kMinLocalInliers = 15;
constexpr float kMinLocalInlierRatio = 0.45f;
constexpr int kMinGlobalGoodMatches = 30;
constexpr int kMinGlobalInliers = 10;
constexpr float kMinGlobalInlierRatio = 0.10f;

static inline long long elapsed_ms(const std::chrono::steady_clock::time_point& start,
                                   const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

struct StepTiming {
    long long roi_ms = 0;
    long long gray_ms = 0;
    long long sift_ms = 0;
    long long mini_sp_ms = 0;
    long long map_sp_ms = 0;
    long long subset_ms = 0;
    long long match_ms = 0;
    long long ratio_ms = 0;
    long long homo_ms = 0;
    long long xform_ms = 0;
    long long smooth_ms = 0;
    int mini_kp = 0;
    int local_kp = 0;
    int good_matches = 0;
    int inlier_matches = 0;
    float inlier_ratio = 0.f;
    float raw_x = -1.f;
    float raw_y = -1.f;
    int sp_width = 0;
    int sp_height = 0;
    bool global_search = false;
};

struct CacheHeader {
    uint32_t magic = kFeatureCacheMagic;
    uint32_t version = kFeatureCacheVersion;
    int64_t map_size = 0;
    int64_t map_mtime_sec = 0;
    int32_t map_width = 0;
    int32_t map_height = 0;
    int32_t descriptor_rows = 0;
    int32_t descriptor_cols = 0;
    int32_t descriptor_type = 0;
    int32_t keypoint_count = 0;
};

struct CacheKeyPoint {
    float x = 0.f;
    float y = 0.f;
    float size = 0.f;
    float angle = 0.f;
    float response = 0.f;
    int32_t octave = 0;
    int32_t class_id = 0;
};

struct TrackResult {
    int cost_ms = 0;
    float x = 0.f;
    float y = 0.f;
    bool found = false;
    bool inertial = false;
    int good_matches = 0;
    int search_radius = 0;
    int lost_frames = 0;
    bool manual_required = false;
    int inlier_matches = 0;
    float inlier_ratio = 0.f;
    bool global_search = false;
    float rotation_angle = 0.f;
};

float round_to_tenth(float value) {
    return std::round(value * 10.f) / 10.f;
}

struct FeatureSet {
    std::vector<cv::Point2f> net_points;
    std::vector<cv::Point2f> original_points;
    std::vector<float> scores;
    std::vector<float> descriptors;

    int size() const {
        return static_cast<int>(scores.size());
    }
};

struct Candidate {
    float score = 0.f;
    int x = 0;
    int y = 0;
    Candidate() = default;
    Candidate(float s, int px, int py) : score(s), x(px), y(py) {}
};

class NativeMapSiftTracker {
public:
    NativeMapSiftTracker(const std::string& map_path,
                         const std::string& cache_path,
                         const std::string& model_dir,
                         int minimap_left,
                         int minimap_top,
                         int minimap_width,
                         int minimap_height,
                         int base_search_radius,
                         int max_lost_frames,
                         double clahe_limit,
                         float match_ratio,
                         int min_match_count,
                         double ransac_threshold)
            : minimap_left_(minimap_left),
              minimap_top_(minimap_top),
              minimap_width_(minimap_width),
              minimap_height_(minimap_height),
              map_path_(map_path),
              cache_path_(cache_path),
              model_dir_(model_dir),
              base_search_radius_(base_search_radius),
              current_search_radius_(base_search_radius),
              max_lost_frames_(max_lost_frames),
              match_ratio_(match_ratio),
              min_match_count_(min_match_count),
              ransac_threshold_(ransac_threshold) {
        TRACK_LOGD("hybrid tracker init map=%s cache=%s models=%s",
                   map_path_.c_str(),
                   cache_path_.c_str(),
                   model_dir_.c_str());
        logic_map_bgr_ = cv::imread(map_path, cv::IMREAD_COLOR);
        if (logic_map_bgr_.empty()) {
            throw std::runtime_error("Failed to load logic map: " + map_path);
        }
        map_width_ = logic_map_bgr_.cols;
        map_height_ = logic_map_bgr_.rows;

        clahe_ = cv::createCLAHE(clahe_limit, cv::Size(8, 8));
        sift_ = cv::SIFT::create();
        orb_ = cv::ORB::create(250);
        flann_ = cv::makePtr<cv::FlannBasedMatcher>(
                cv::makePtr<cv::flann::KDTreeIndexParams>(5),
                cv::makePtr<cv::flann::SearchParams>(50));
        if (!load_feature_cache()) {
            TRACK_LOGD("feature cache unavailable; extracting full map SIFT features");
            cv::Mat full_map_gray = preprocess_gray(logic_map_bgr_);
            sift_->detectAndCompute(
                    full_map_gray,
                    cv::noArray(),
                    full_map_keypoints_,
                    full_map_descriptors_);
            save_feature_cache();
        }
        if (full_map_keypoints_.size() < 2 || full_map_descriptors_.empty()) {
            throw std::runtime_error("Failed to extract full map SIFT features");
        }

        load_ncnn_model(superpoint_net_,
                        resolve_model_path("official_superpoint_dense_480x640.param",
                                           "official_superpoint_dense_480x640_int8.param"),
                        resolve_model_path("official_superpoint_dense_480x640.bin",
                                           "official_superpoint_dense_480x640_int8.bin"),
                        "SuperPoint");

        reset_to_global_search();
        TRACK_LOGD("hybrid tracker ready map=%dx%d minimap=(%d,%d %dx%d) baseRadius=%d siftFeatures=%zu localSpCache=memory",
                   map_width_,
                   map_height_,
                   minimap_left_,
                   minimap_top_,
                   minimap_width_,
                   minimap_height_,
                   base_search_radius_,
                   full_map_keypoints_.size());
    }

#ifdef __ANDROID__
    // 以下两个入口直接消费 AndroidBitmap，只在 Android 上可用。
    // 主机侧请走 player_tracker_locate()，它接的是同一台状态机 track_minimap()。
    TrackResult track(JNIEnv* env, jobject bitmap) {
        const auto start = std::chrono::steady_clock::now();
        TrackResult result = make_status_result();
        StepTiming timing;
        track_count_ += 1;

        AndroidBitmapInfo info;
        if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS) {
            TRACK_LOGE("AndroidBitmap_getInfo failed");
            return result;
        }
        if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
            TRACK_LOGE("Unsupported bitmap format: %u", info.format);
            return result;
        }
        if (minimap_left_ < 0
                || minimap_top_ < 0
                || minimap_left_ + minimap_width_ > static_cast<int>(info.width)
                || minimap_top_ + minimap_height_ > static_cast<int>(info.height)) {
            TRACK_LOGE("Minimap ROI out of bounds frame=%ux%u roi=(%d,%d %dx%d)",
                       info.width,
                       info.height,
                       minimap_left_,
                       minimap_top_,
                       minimap_width_,
                       minimap_height_);
            return result;
        }

        void* pixels = nullptr;
        if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS || pixels == nullptr) {
            TRACK_LOGE("AndroidBitmap_lockPixels failed");
            return result;
        }

        const auto roi_start = std::chrono::steady_clock::now();
        cv::Mat rgba(static_cast<int>(info.height), static_cast<int>(info.width), CV_8UC4, pixels, info.stride);
        cv::Rect minimap_rect(minimap_left_, minimap_top_, minimap_width_, minimap_height_);
        cv::Mat mini_rgba = rgba(minimap_rect);
        cv::Mat mini_bgr;
        cv::cvtColor(mini_rgba, mini_bgr, cv::COLOR_RGBA2BGR);
        AndroidBitmap_unlockPixels(env, bitmap);
        const auto roi_end = std::chrono::steady_clock::now();
        timing.roi_ms = elapsed_ms(roi_start, roi_end);

        result = track_minimap(mini_bgr, timing);
        const auto end = std::chrono::steady_clock::now();
        result.cost_ms = static_cast<int>(elapsed_ms(start, end));
        log_timing_if_needed(result, timing);
        return result;
    }

    TrackResult track_superpoint_local(JNIEnv* env, jobject bitmap) {
        const auto start = std::chrono::steady_clock::now();
        TrackResult result = make_status_result();
        StepTiming timing;

        AndroidBitmapInfo info;
        if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS) {
            return result;
        }
        if (minimap_left_ < 0
                || minimap_top_ < 0
                || minimap_left_ + minimap_width_ > static_cast<int>(info.width)
                || minimap_top_ + minimap_height_ > static_cast<int>(info.height)) {
            TRACK_LOGE("SuperPoint local ROI out of bounds frame=%ux%u roi=(%d,%d %dx%d)",
                       info.width,
                       info.height,
                       minimap_left_,
                       minimap_top_,
                       minimap_width_,
                       minimap_height_);
            return result;
        }

        void* pixels = nullptr;
        if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS || pixels == nullptr) {
            return result;
        }

        cv::Mat rgba(static_cast<int>(info.height), static_cast<int>(info.width), CV_8UC4, pixels, info.stride);
        cv::Rect minimap_rect(minimap_left_, minimap_top_, minimap_width_, minimap_height_);
        cv::Mat mini_rgba = rgba(minimap_rect);
        cv::Mat mini_bgr;
        cv::cvtColor(mini_rgba, mini_bgr, cv::COLOR_RGBA2BGR);
        AndroidBitmap_unlockPixels(env, bitmap);

        result = track_minimap_superpoint_local(mini_bgr, timing);
        result.cost_ms = static_cast<int>(elapsed_ms(start, std::chrono::steady_clock::now()));
        return result;
    }
#endif  // __ANDROID__

    void set_manual_position(float x, float y) {
        smoothed_cx_ = round_to_tenth(std::clamp(x, 0.f, static_cast<float>(map_width_ - 1)));
        smoothed_cy_ = round_to_tenth(std::clamp(y, 0.f, static_cast<float>(map_height_ - 1)));
        last_x_ = clamp_to_map(std::lround(smoothed_cx_), map_width_);
        last_y_ = clamp_to_map(std::lround(smoothed_cy_), map_height_);
        has_position_ = true;
        has_smoothed_position_ = true;
        has_cached_local_sp_features_ = false;
        lost_frames_ = 0;
        has_local_failure_since_ = false;
        current_search_radius_ = base_search_radius_;
        manual_required_ = false;
        TRACK_LOGD("manual position set x=%.1f y=%.1f radius=%d", smoothed_cx_, smoothed_cy_, current_search_radius_);
    }

    void force_global_search() {
        TRACK_LOGD("force global search requested");
        reset_to_global_search();
    }

public:
    std::string resolve_model_path(const char* preferred_file_name,
                                   const char* fallback_file_name) const {
        std::string preferred = model_path(preferred_file_name);
        struct stat st {};
        if (stat(preferred.c_str(), &st) == 0) {
            return preferred;
        }
        return model_path(fallback_file_name);
    }

    std::string model_path(const char* file_name) const {
        return model_dir_ + "/" + file_name;
    }

    static void load_ncnn_model(ncnn::Net& net,
                                const std::string& param_path,
                                const std::string& bin_path,
                                const char* label) {
        net.opt.use_vulkan_compute = false;
        net.opt.num_threads = 2;
        net.opt.lightmode = true;
        net.opt.use_int8_inference = false;
        net.opt.use_bf16_storage = false;
        net.opt.use_fp16_packed = true;
        net.opt.use_fp16_storage = true;
        net.opt.use_fp16_arithmetic = true;
        if (net.load_param(param_path.c_str()) != 0) {
            throw std::runtime_error(std::string("Failed to load ") + label + " param: " + param_path);
        }
        if (net.load_model(bin_path.c_str()) != 0) {
            throw std::runtime_error(std::string("Failed to load ") + label + " bin: " + bin_path);
        }
        TRACK_LOGD("loaded %s model", label);
    }

    TrackResult track_minimap(const cv::Mat& minimap_bgr, StepTiming& timing) {
        if (minimap_bgr.empty()) {
            return make_status_result();
        }

        if (!has_position_) {
            return track_minimap_sift_global(minimap_bgr, timing);
        }
        if (has_local_failure_since_
                && local_failure_elapsed_ms() >= kGlobalSearchFallbackDelayMs) {
            TRACK_LOGD("local SuperPoint lost for %lldms/%d frames; fallback to global SIFT",
                       local_failure_elapsed_ms(),
                       lost_frames_);
            reset_to_global_search();
            return track_minimap_sift_global(minimap_bgr, timing);
        }
        return track_minimap_superpoint_local(minimap_bgr, timing);
    }

    TrackResult track_minimap_sift_global(const cv::Mat& minimap_bgr, StepTiming& timing) {
        const auto now = std::chrono::steady_clock::now();
        if (elapsed_ms(last_global_search_time_, now) < 2000) {
            // Throttling global search to once every 2 seconds to keep CPU cool when lost!
            return make_status_result();
        }
        last_global_search_time_ = now;
        timing.global_search = true;

        const auto gray_start = std::chrono::steady_clock::now();
        cv::Mat mini_gray = preprocess_gray(minimap_bgr);
        const auto gray_end = std::chrono::steady_clock::now();
        timing.gray_ms = elapsed_ms(gray_start, gray_end);

        std::vector<cv::KeyPoint> mini_keypoints;
        cv::Mat mini_descriptors;
        const auto sift_start = std::chrono::steady_clock::now();
        sift_->detectAndCompute(mini_gray, cv::noArray(), mini_keypoints, mini_descriptors);
        const auto sift_end = std::chrono::steady_clock::now();
        timing.sift_ms = elapsed_ms(sift_start, sift_end);
        timing.mini_kp = static_cast<int>(mini_keypoints.size());
        if (mini_keypoints.size() < 2 || mini_descriptors.empty()) {
            return handle_failure(0);
        }

        std::vector<cv::KeyPoint> local_keypoints;
        cv::Mat local_descriptors;
        const auto subset_start = std::chrono::steady_clock::now();
        build_search_feature_subset(compute_search_rect(), local_keypoints, local_descriptors);
        const auto subset_end = std::chrono::steady_clock::now();
        timing.subset_ms = elapsed_ms(subset_start, subset_end);
        timing.local_kp = static_cast<int>(local_keypoints.size());
        if (local_keypoints.size() < 2 || local_descriptors.empty()) {
            return handle_failure(0);
        }

        std::vector<std::vector<cv::DMatch>> matches;
        const auto match_start = std::chrono::steady_clock::now();
        flann_->knnMatch(mini_descriptors, local_descriptors, matches, 2);
        const auto match_end = std::chrono::steady_clock::now();
        timing.match_ms = elapsed_ms(match_start, match_end);

        std::vector<cv::DMatch> good_matches;
        good_matches.reserve(matches.size());
        const auto ratio_start = std::chrono::steady_clock::now();
        for (const auto& pair : matches) {
            if (pair.size() != 2) {
                continue;
            }
            const cv::DMatch& best = pair[0];
            const cv::DMatch& second = pair[1];
            if (best.distance < match_ratio_ * second.distance) {
                good_matches.push_back(best);
            }
        }
        const auto ratio_end = std::chrono::steady_clock::now();
        timing.ratio_ms = elapsed_ms(ratio_start, ratio_end);
        timing.good_matches = static_cast<int>(good_matches.size());
        if (timing.good_matches < min_match_count_) {
            return handle_failure(timing.good_matches);
        }

        std::vector<cv::Point2f> src_points;
        std::vector<cv::Point2f> dst_points;
        src_points.reserve(good_matches.size());
        dst_points.reserve(good_matches.size());
        for (const cv::DMatch& match : good_matches) {
            src_points.push_back(mini_keypoints[match.queryIdx].pt);
            dst_points.push_back(local_keypoints[match.trainIdx].pt);
        }

        return finish_matched_points(src_points, dst_points, timing.good_matches, true, timing);
    }

    TrackResult track_minimap_orb_local(const cv::Mat& minimap_bgr, StepTiming& timing) {
        const auto local_start = std::chrono::steady_clock::now();
        timing.global_search = false;

        cv::Rect search_rect = compute_search_rect(base_search_radius_);
        cv::Mat map_crop = logic_map_bgr_(search_rect);

        // Preprocess to grayscale with CLAHE
        const auto gray_start = std::chrono::steady_clock::now();
        cv::Mat mini_gray = preprocess_gray(minimap_bgr);
        cv::Mat map_gray = preprocess_gray(map_crop);
        const auto gray_end = std::chrono::steady_clock::now();
        timing.gray_ms = elapsed_ms(gray_start, gray_end);

        // Extract ORB
        std::vector<cv::KeyPoint> mini_kps, map_kps;
        cv::Mat mini_descs, map_descs;

        const auto orb_start = std::chrono::steady_clock::now();
        orb_->detectAndCompute(mini_gray, cv::noArray(), mini_kps, mini_descs);
        orb_->detectAndCompute(map_gray, cv::noArray(), map_kps, map_descs);
        const auto orb_end = std::chrono::steady_clock::now();
        timing.sift_ms = elapsed_ms(orb_start, orb_end);

        timing.mini_kp = static_cast<int>(mini_kps.size());
        timing.local_kp = static_cast<int>(map_kps.size());

        if (mini_kps.size() < 4 || map_kps.size() < 4 || mini_descs.empty() || map_descs.empty()) {
            return handle_failure(0, 0, 0.f, false);
        }

        // Match descriptors
        std::vector<std::vector<cv::DMatch>> matches;
        const auto match_start = std::chrono::steady_clock::now();
        cv::BFMatcher matcher(cv::NORM_HAMMING);
        matcher.knnMatch(mini_descs, map_descs, matches, 2);
        const auto match_end = std::chrono::steady_clock::now();
        timing.match_ms = elapsed_ms(match_start, match_end);

        std::vector<cv::DMatch> good_matches;
        for (const auto& pair : matches) {
            if (pair.size() == 2) {
                // Balanced Hamming distance threshold (< 64) and ratio test (< 0.85) for ORB binary features
                if (pair[0].distance < 64 && pair[0].distance < 0.85f * pair[1].distance) {
                    good_matches.push_back(pair[0]);
                }
            }
        }

        timing.good_matches = static_cast<int>(good_matches.size());
        if (good_matches.size() < 4) {
            return handle_failure(good_matches.size(), 0, 0.f, false);
        }

        // Map keypoints back to global coordinates
        std::vector<cv::Point2f> src_points;
        std::vector<cv::Point2f> dst_points;
        src_points.reserve(good_matches.size());
        dst_points.reserve(good_matches.size());
        for (const cv::DMatch& match : good_matches) {
            src_points.push_back(mini_kps[match.queryIdx].pt);
            cv::Point2f pt = map_kps[match.trainIdx].pt;
            pt.x += static_cast<float>(search_rect.x);
            pt.y += static_cast<float>(search_rect.y);
            dst_points.push_back(pt);
        }

        TrackResult res = finish_matched_points(src_points, dst_points, good_matches.size(), false, timing);
        const auto local_end = std::chrono::steady_clock::now();
        const int64_t local_ms = elapsed_ms(local_start, local_end);
        
        TRACK_LOGD("ORB局部匹配 total=%lldms good=%d inliers=%d ratio=%.2f result=(%.1f,%.1f) found=%d",
                   static_cast<long long>(local_ms),
                   timing.good_matches,
                   timing.inlier_matches,
                   timing.inlier_ratio,
                   res.x,
                   res.y,
                   res.found ? 1 : 0);
                   
        return res;
    }

    TrackResult track_minimap_superpoint_local(const cv::Mat& minimap_bgr, StepTiming& timing) {
        const auto local_start = std::chrono::steady_clock::now();
        timing.global_search = false;
        const int sp_width = kLocalNetWidth;
        const int sp_height = kLocalNetHeight;
        timing.sp_width = sp_width;
        timing.sp_height = sp_height;
        const char* map_feature_source = "none";
        auto finish_local = [&](TrackResult result, const char* reason) -> TrackResult {
            const auto local_end = std::chrono::steady_clock::now();
            const int64_t local_ms = elapsed_ms(local_start, local_end);
            if (false && local_ms >= kSlowLocalMatchLogMs) {
                TRACK_LOGD("局部匹配慢 total=%lldms reason=%s sp=%dx%d source=%s good=%d inliers=%d ratio=%.2f result=(%.1f,%.1f) found=%d lost=%d",
                           static_cast<long long>(local_ms),
                           reason,
                           timing.sp_width,
                           timing.sp_height,
                           map_feature_source,
                           timing.good_matches,
                           timing.inlier_matches,
                           timing.inlier_ratio,
                           result.x,
                           result.y,
                           result.found ? 1 : 0,
                           result.lost_frames);
            }
            return result;
        };
        auto fail_local = [&](int good_matches, int inlier_matches = 0, float inlier_ratio = 0.f) -> TrackResult {
            has_cached_local_sp_features_ = false;
            return handle_failure(good_matches, inlier_matches, inlier_ratio, false);
        };

        const auto mini_start = std::chrono::steady_clock::now();
        FeatureSet mini_features = extract_superpoint(minimap_bgr, cv::Point2f(0.f, 0.f), cv::Size2f(
                static_cast<float>(minimap_width_),
                static_cast<float>(minimap_height_)),
                sp_width,
                sp_height);
        const auto mini_end = std::chrono::steady_clock::now();
        timing.mini_sp_ms = elapsed_ms(mini_start, mini_end);
        timing.mini_kp = mini_features.size();
        if (mini_features.size() < 4) {
            return finish_local(fail_local(0), "mini_kp_low");
        }

        FeatureSet map_features;
        cv::Rect search_rect = compute_search_rect(base_search_radius_);
        if (has_cached_local_sp_features_
                && cached_local_sp_rect_ == search_rect
                && cached_local_sp_width_ == sp_width
                && cached_local_sp_height_ == sp_height) {
            const auto subset_start = std::chrono::steady_clock::now();
            map_features = cached_local_sp_features_;
            const auto subset_end = std::chrono::steady_clock::now();
            timing.subset_ms = elapsed_ms(subset_start, subset_end);
            map_feature_source = "memory-cache";
        } else {
            // 曾经这里会优先从全图 SuperPoint 特征缓存取子集。那份缓存是分块降采样提取的，
            // 全图仅约 7k 个特征，落到 300x300 窗口内平均只剩 ~17 个，实测 0/111 成功，
            // 已连同缓存本身一并删除。现在恒定对地图裁剪现场提取（实测 110/110）。
            {
                cv::Mat map_crop = logic_map_bgr_(search_rect);
                const auto map_start = std::chrono::steady_clock::now();
                map_features = extract_superpoint(
                        map_crop,
                        cv::Point2f(static_cast<float>(search_rect.x), static_cast<float>(search_rect.y)),
                        cv::Size2f(static_cast<float>(search_rect.width), static_cast<float>(search_rect.height)),
                        sp_width,
                        sp_height);
                const auto map_end = std::chrono::steady_clock::now();
                timing.map_sp_ms = elapsed_ms(map_start, map_end);
                map_feature_source = "crop-sp";
            }
            cached_local_sp_rect_ = search_rect;
            cached_local_sp_width_ = sp_width;
            cached_local_sp_height_ = sp_height;
            cached_local_sp_features_ = map_features;
            has_cached_local_sp_features_ = true;
        }
        timing.local_kp = map_features.size();
        if (map_features.size() < 4) {
            return finish_local(fail_local(0), "map_kp_low");
        }

        cv::Mat mini_descriptors = descriptors_to_mat(mini_features);
        cv::Mat map_descriptors = descriptors_to_mat(map_features);
        if (mini_descriptors.empty() || map_descriptors.empty()) {
            return finish_local(fail_local(0), "descriptor_empty");
        }

        std::vector<std::vector<cv::DMatch>> matches;
        const auto match_start = std::chrono::steady_clock::now();
        cv::BFMatcher matcher(cv::NORM_L2);
        matcher.knnMatch(mini_descriptors, map_descriptors, matches, 2);
        const auto match_end = std::chrono::steady_clock::now();
        timing.match_ms = elapsed_ms(match_start, match_end);

        std::vector<cv::DMatch> good_matches;
        good_matches.reserve(matches.size());
        const auto ratio_start = std::chrono::steady_clock::now();
        for (const auto& pair : matches) {
            if (pair.size() != 2) {
                continue;
            }
            const cv::DMatch& best = pair[0];
            const cv::DMatch& second = pair[1];
            if (best.distance < match_ratio_ * second.distance) {
                good_matches.push_back(best);
            }
        }
        const auto ratio_end = std::chrono::steady_clock::now();
        timing.ratio_ms = elapsed_ms(ratio_start, ratio_end);
        timing.good_matches = static_cast<int>(good_matches.size());
        if (timing.good_matches < min_match_count_) {
            return finish_local(fail_local(timing.good_matches), "good_matches_low");
        }

        std::vector<cv::Point2f> src_points;
        std::vector<cv::Point2f> dst_points;
        src_points.reserve(good_matches.size());
        dst_points.reserve(good_matches.size());
        for (const cv::DMatch& match : good_matches) {
            src_points.push_back(mini_features.original_points[match.queryIdx]);
            dst_points.push_back(map_features.original_points[match.trainIdx]);
        }

        TrackResult result = finish_matched_points(src_points, dst_points, timing.good_matches, false, timing);
        if (!result.found || result.inertial) {
            has_cached_local_sp_features_ = false;
        }
        return finish_local(result, result.found ? (result.inertial ? "inertial" : "ok") : "not_found");
    }

    TrackResult finish_matched_points(const std::vector<cv::Point2f>& src_points,
                                      const std::vector<cv::Point2f>& dst_points,
                                      int good_match_count,
                                      bool global_search,
                                      StepTiming& timing) {
        if (src_points.size() < 4 || dst_points.size() < 4 || src_points.size() != dst_points.size()) {
            return handle_failure(good_match_count, 0, 0.f, global_search);
        }

        const auto homo_start = std::chrono::steady_clock::now();
        cv::Mat inlier_mask;
        cv::Mat homography = cv::findHomography(src_points, dst_points, cv::RANSAC, ransac_threshold_, inlier_mask);
        const auto homo_end = std::chrono::steady_clock::now();
        timing.homo_ms = elapsed_ms(homo_start, homo_end);
        if (homography.empty()) {
            return handle_failure(good_match_count, 0, 0.f, global_search);
        }

        if (homography.cols >= 3 && homography.rows >= 3) {
            double h00 = homography.at<double>(0, 0);
            double h10 = homography.at<double>(1, 0);
            last_rotation_angle_ = static_cast<float>(std::atan2(h10, h00) * 180.0 / M_PI);
        }

        const int inlier_count = inlier_mask.empty() ? 0 : cv::countNonZero(inlier_mask);
        const float inlier_ratio = good_match_count <= 0
                ? 0.f
                : static_cast<float>(inlier_count) / static_cast<float>(good_match_count);
        timing.inlier_matches = inlier_count;
        timing.inlier_ratio = inlier_ratio;
        if (!is_match_quality_sufficient(global_search,
                                         good_match_count,
                                         inlier_count,
                                         inlier_ratio)) {
            return handle_failure(good_match_count, inlier_count, inlier_ratio, global_search);
        }

        std::vector<cv::Point2f> center_in(1, cv::Point2f(minimap_width_ / 2.0f, minimap_height_ / 2.0f));
        std::vector<cv::Point2f> center_out;
        const auto xform_start = std::chrono::steady_clock::now();
        cv::perspectiveTransform(center_in, center_out, homography);
        const auto xform_end = std::chrono::steady_clock::now();
        timing.xform_ms = elapsed_ms(xform_start, xform_end);
        if (center_out.empty()) {
            return handle_failure(good_match_count, inlier_count, inlier_ratio, global_search);
        }

        timing.raw_x = center_out[0].x;
        timing.raw_y = center_out[0].y;
        if (timing.raw_x < 0.f
                || timing.raw_x >= static_cast<float>(map_width_)
                || timing.raw_y < 0.f
                || timing.raw_y >= static_cast<float>(map_height_)) {
            return handle_failure(good_match_count, inlier_count, inlier_ratio, global_search);
        }

        const auto smooth_start = std::chrono::steady_clock::now();
        bool accepted = smooth_position(timing.raw_x, timing.raw_y, inlier_count, inlier_ratio);
        const auto smooth_end = std::chrono::steady_clock::now();
        timing.smooth_ms = elapsed_ms(smooth_start, smooth_end);
        if (!accepted) {
            TRACK_LOGD("detected large jump (dist >= 500px), discarding raw coordinate (%.1f, %.1f) and keeping last valid (%.1f, %.1f)",
                       timing.raw_x, timing.raw_y, smoothed_cx_, smoothed_cy_);
            last_x_ = clamp_to_map(std::lround(smoothed_cx_), map_width_);
            last_y_ = clamp_to_map(std::lround(smoothed_cy_), map_height_);
            has_position_ = true;
            lost_frames_ = 0;
            has_local_failure_since_ = false;
            current_search_radius_ = base_search_radius_;
            manual_required_ = false;

            TrackResult result = make_status_result();
            result.global_search = global_search;
            result.found = true;
            result.good_matches = good_match_count;
            result.inlier_matches = inlier_count;
            result.inlier_ratio = inlier_ratio;
            return result;
        }

        last_x_ = clamp_to_map(std::lround(smoothed_cx_), map_width_);
        last_y_ = clamp_to_map(std::lround(smoothed_cy_), map_height_);
        has_position_ = true;
        lost_frames_ = 0;
        has_local_failure_since_ = false;
        current_search_radius_ = base_search_radius_;
        manual_required_ = false;

        TrackResult result = make_status_result();
        result.global_search = global_search;
        result.found = true;
        result.good_matches = good_match_count;
        result.inlier_matches = inlier_count;
        result.inlier_ratio = inlier_ratio;
        return result;
    }

    cv::Mat descriptors_to_mat(const FeatureSet& features) const {
        if (features.size() <= 0
                || features.descriptors.size() < static_cast<size_t>(features.size() * kDescriptorDim)) {
            return cv::Mat();
        }
        cv::Mat descriptors(features.size(), kDescriptorDim, CV_32F);
        const int value_count = features.size() * kDescriptorDim;
        std::copy(features.descriptors.data(),
                  features.descriptors.data() + value_count,
                  descriptors.ptr<float>());
        return descriptors;
    }

    FeatureSet extract_superpoint(const cv::Mat& source_bgr,
                                  const cv::Point2f& origin,
                                  const cv::Size2f& original_size,
                                  int net_width = kLocalNetWidth,
                                  int net_height = kLocalNetHeight) {
        cv::Mat resized;
        cv::resize(source_bgr, resized, cv::Size(net_width, net_height), 0, 0, cv::INTER_AREA);
        cv::Mat gray;
        cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
        gray.convertTo(gray, CV_32FC1, 1.0 / 255.0);

        ncnn::Mat input(net_width, net_height, 1);
        for (int y = 0; y < net_height; y++) {
            const float* row = gray.ptr<float>(y);
            float* dst = static_cast<float*>(input.data) + y * net_width;
            std::copy(row, row + net_width, dst);
        }

        ncnn::Extractor extractor = superpoint_net_.create_extractor();
        if (extractor.input("in0", input) != 0) {
            return FeatureSet{};
        }

        ncnn::Mat scores;
        ncnn::Mat dense_descriptors;
        if (extractor.extract("out0", scores) != 0 || extractor.extract("out1", dense_descriptors) != 0) {
            return FeatureSet{};
        }

        std::vector<Candidate> candidates;
        candidates.reserve((net_width / kGridStride) * (net_height / kGridStride));
        for (int channel = 0; channel < scores.c; channel++) {
            const float* score_channel = scores.channel(channel);
            int offset_x = channel % kGridStride;
            int offset_y = channel / kGridStride;
            for (int gy = 0; gy < scores.h; gy++) {
                for (int gx = 0; gx < scores.w; gx++) {
                    float score = score_channel[gy * scores.w + gx];
                    int x = gx * kGridStride + offset_x;
                    int y = gy * kGridStride + offset_y;
                    
                    // Filter out features outside the inscribed circle of the minimap
                    float dx = static_cast<float>(x) - net_width / 2.0f;
                    float dy = static_cast<float>(y) - net_height / 2.0f;
                    float radius_limit = (net_width / 2.0f) * 0.98f; // 0.98 safely keeps inside the border
                    if (dx * dx + dy * dy > radius_limit * radius_limit) {
                        continue;
                    }

                    if (score < kSuperPointThreshold
                            || x < kSuperPointBorder
                            || y < kSuperPointBorder
                            || x >= net_width - kSuperPointBorder
                            || y >= net_height - kSuperPointBorder) {
                        continue;
                    }
                    candidates.push_back(Candidate{score, x, y});
                }
            }
        }

        std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            return a.score > b.score;
        });
        if (static_cast<int>(candidates.size()) > kMaxCandidatesBeforeNms) {
            candidates.resize(kMaxCandidatesBeforeNms);
        }

        FeatureSet features;
        features.net_points.reserve(kMaxKeypoints);
        features.original_points.reserve(kMaxKeypoints);
        features.scores.reserve(kMaxKeypoints);
        features.descriptors.reserve(kMaxKeypoints * kDescriptorDim);

        std::vector<uint8_t> occupied(net_width * net_height, 0);
        for (const Candidate& candidate : candidates) {
            if (features.size() >= kMaxKeypoints) {
                break;
            }
            bool suppressed = false;
            int left = std::max(0, candidate.x - kSuperPointNmsRadius);
            int right = std::min(net_width - 1, candidate.x + kSuperPointNmsRadius);
            int top = std::max(0, candidate.y - kSuperPointNmsRadius);
            int bottom = std::min(net_height - 1, candidate.y + kSuperPointNmsRadius);
            for (int y = top; y <= bottom && !suppressed; y++) {
                for (int x = left; x <= right; x++) {
                    if (occupied[y * net_width + x]) {
                        suppressed = true;
                        break;
                    }
                }
            }
            if (suppressed) {
                continue;
            }
            occupied[candidate.y * net_width + candidate.x] = 1;

            float net_x = static_cast<float>(candidate.x);
            float net_y = static_cast<float>(candidate.y);
            float original_x = origin.x + net_x * original_size.width / static_cast<float>(net_width);
            float original_y = origin.y + net_y * original_size.height / static_cast<float>(net_height);
            features.net_points.emplace_back(net_x, net_y);
            features.original_points.emplace_back(original_x, original_y);
            features.scores.push_back(candidate.score);
            append_descriptor(dense_descriptors, candidate.x, candidate.y, features.descriptors);
        }
        return features;
    }

    static void append_descriptor(const ncnn::Mat& dense_descriptors,
                                  int keypoint_x,
                                  int keypoint_y,
                                  std::vector<float>& out) {
        int gx = std::clamp(keypoint_x / kGridStride, 0, dense_descriptors.w - 1);
        int gy = std::clamp(keypoint_y / kGridStride, 0, dense_descriptors.h - 1);
        size_t offset = out.size();
        out.resize(offset + kDescriptorDim, 0.f);
        float norm_sq = 0.f;
        for (int d = 0; d < std::min(kDescriptorDim, dense_descriptors.c); d++) {
            float value = dense_descriptors.channel(d)[gy * dense_descriptors.w + gx];
            out[offset + d] = value;
            norm_sq += value * value;
        }
        float norm = std::sqrt(std::max(norm_sq, 1e-12f));
        for (int d = 0; d < kDescriptorDim; d++) {
            out[offset + d] /= norm;
        }
    }

    TrackResult handle_failure(int good_matches,
                               int inlier_matches = 0,
                               float inlier_ratio = 0.f,
                               bool expand_search_radius = true) {
        if (has_position_) {
            if (!has_local_failure_since_) {
                local_failure_since_ = std::chrono::steady_clock::now();
                has_local_failure_since_ = true;
            }
            lost_frames_ += 1;
            if (expand_search_radius && lost_frames_ == 1) {
                current_search_radius_ = std::min(current_search_radius_ + 300, std::max(map_width_, map_height_));
            }
        } else {
            manual_required_ = false;
        }

        TrackResult result;
        result.x = -1.f;
        result.y = -1.f;
        result.search_radius = has_position_
                ? current_search_radius_
                : std::max(map_width_, map_height_);
        result.lost_frames = has_position_ ? lost_frames_ : 0;
        result.manual_required = manual_required_;
        result.good_matches = good_matches;
        result.inlier_matches = inlier_matches;
        result.inlier_ratio = inlier_ratio;
        result.inertial = false;
        result.found = false;
        result.global_search = expand_search_radius;
        return result;
    }

    long long local_failure_elapsed_ms() const {
        if (!has_local_failure_since_) {
            return 0;
        }
        return elapsed_ms(local_failure_since_, std::chrono::steady_clock::now());
    }

    static bool is_match_quality_sufficient(bool global_search,
                                            int good_match_count,
                                            int inlier_count,
                                            float inlier_ratio) {
        if (global_search) {
            return good_match_count >= kMinGlobalGoodMatches
                    && inlier_count >= kMinGlobalInliers
                    && inlier_ratio >= kMinGlobalInlierRatio;
        }
        return good_match_count >= kMinLocalGoodMatches
                && inlier_count >= kMinLocalInliers
                && inlier_ratio >= kMinLocalInlierRatio;
    }

    bool smooth_position(float raw_x, float raw_y, int inlier_count, float inlier_ratio) {
        if (!has_smoothed_position_) {
            smoothed_cx_ = raw_x;
            smoothed_cy_ = raw_y;
            has_smoothed_position_ = true;
            return true;
        }
        // 以下三个常量原先随 precise_tracking_mode_ 二选一。该模式已固定为开启
        // （关闭时局部匹配走稀疏全图缓存，实测 0/111 成功），故直接取 precise 那一档：
        // 局部特征是现场提取的，质量高，可以卡更严的跳变阈值并更信任新解。
        float distance = std::hypot(raw_x - smoothed_cx_, raw_y - smoothed_cy_);
        if (distance >= 40.f) {
            return false;
        }
        float base_alpha = distance < 15.f ? 0.15f : 0.45f;
        float inlier_term = std::clamp((static_cast<float>(inlier_count) - 8.f) / 24.f, 0.f, 1.f);
        float ratio_term = std::clamp((inlier_ratio - 0.20f) / 0.50f, 0.f, 1.f);
        float confidence = std::max(inlier_term, ratio_term);
        float alpha = std::clamp(base_alpha + confidence * 0.30f, 0.35f, 0.85f);
        smoothed_cx_ = alpha * raw_x + (1.f - alpha) * smoothed_cx_;
        smoothed_cy_ = alpha * raw_y + (1.f - alpha) * smoothed_cy_;
        return true;
    }

    void reset_to_global_search() {
        has_position_ = false;
        has_smoothed_position_ = false;
        has_cached_local_sp_features_ = false;
        manual_required_ = false;
        lost_frames_ = 0;
        has_local_failure_since_ = false;
        current_search_radius_ = base_search_radius_;
        last_x_ = -1;
        last_y_ = -1;
        smoothed_cx_ = 0.f;
        smoothed_cy_ = 0.f;
        last_rotation_angle_ = 0.f;
    }

    TrackResult make_status_result() const {
        TrackResult result;
        result.x = has_position_ ? round_to_tenth(smoothed_cx_) : -1.f;
        result.y = has_position_ ? round_to_tenth(smoothed_cy_) : -1.f;
        result.search_radius = has_position_
                ? current_search_radius_
                : std::max(map_width_, map_height_);
        result.lost_frames = has_position_ ? lost_frames_ : 0;
        result.manual_required = manual_required_;
        result.rotation_angle = has_position_ ? last_rotation_angle_ : 0.f;
        return result;
    }

    cv::Rect compute_search_rect() const {
        return compute_search_rect(current_search_radius_);
    }

    cv::Rect compute_search_rect(int radius) const {
        if (!has_position_) {
            return cv::Rect(0, 0, map_width_, map_height_);
        }
        float center_x = has_smoothed_position_ ? smoothed_cx_ : static_cast<float>(last_x_);
        float center_y = has_smoothed_position_ ? smoothed_cy_ : static_cast<float>(last_y_);
        int left = std::max(0, static_cast<int>(std::floor(center_x - radius)));
        int top = std::max(0, static_cast<int>(std::floor(center_y - radius)));
        int right = std::min(map_width_, static_cast<int>(std::ceil(center_x + radius)));
        int bottom = std::min(map_height_, static_cast<int>(std::ceil(center_y + radius)));
        right = std::max(right, left + 1);
        bottom = std::max(bottom, top + 1);
        return cv::Rect(left, top, right - left, bottom - top);
    }

    cv::Mat preprocess_gray(const cv::Mat& source_bgr) {
        cv::Mat gray;
        cv::cvtColor(source_bgr, gray, cv::COLOR_BGR2GRAY);
        clahe_->apply(gray, gray);
        return gray;
    }

    void build_search_feature_subset(const cv::Rect& search_rect,
                                     std::vector<cv::KeyPoint>& keypoints,
                                     cv::Mat& descriptors) const {
        keypoints.clear();
        descriptors.release();

        if (search_rect.x <= 0
                && search_rect.y <= 0
                && search_rect.width >= map_width_
                && search_rect.height >= map_height_) {
            keypoints = full_map_keypoints_;
            descriptors = full_map_descriptors_;
            return;
        }

        keypoints.reserve(full_map_keypoints_.size() / 4);
        for (int index = 0; index < static_cast<int>(full_map_keypoints_.size()); index++) {
            const cv::KeyPoint& keypoint = full_map_keypoints_[index];
            if (keypoint.pt.x < static_cast<float>(search_rect.x)
                    || keypoint.pt.x >= static_cast<float>(search_rect.x + search_rect.width)
                    || keypoint.pt.y < static_cast<float>(search_rect.y)
                    || keypoint.pt.y >= static_cast<float>(search_rect.y + search_rect.height)) {
                continue;
            }
            keypoints.push_back(keypoint);
            descriptors.push_back(full_map_descriptors_.row(index));
        }
    }

    static int clamp_to_map(long value, int upper_bound) {
        return static_cast<int>(std::clamp(value, 0L, static_cast<long>(upper_bound - 1)));
    }

    static std::vector<int> make_tile_starts(int length) {
        const int tile_size = std::min(kSuperPointTileSize, std::max(1, length));
        const int stride = std::max(1, tile_size - kSuperPointTileOverlap);
        std::vector<int> starts;
        for (int value = 0; value < length; value += stride) {
            int start = std::min(value, std::max(0, length - tile_size));
            if (starts.empty() || starts.back() != start) {
                starts.push_back(start);
            }
            if (start + tile_size >= length) {
                break;
            }
        }
        return starts;
    }

    static long long superpoint_cell_key(int cell_x, int cell_y) {
        return (static_cast<long long>(cell_x) << 32)
                ^ static_cast<unsigned int>(cell_y);
    }

    static FeatureSet dedupe_superpoint_features(const FeatureSet& raw_features) {
        std::vector<int> order(raw_features.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&raw_features](int lhs, int rhs) {
            return raw_features.scores[lhs] > raw_features.scores[rhs];
        });

        constexpr float radius_sq = kSuperPointGlobalNmsRadius * kSuperPointGlobalNmsRadius;
        const int cell_size = std::max(1, static_cast<int>(std::ceil(kSuperPointGlobalNmsRadius)));
        std::unordered_map<long long, int> occupied;
        occupied.reserve(order.size());

        FeatureSet kept;
        kept.original_points.reserve(raw_features.original_points.size());
        kept.scores.reserve(raw_features.scores.size());
        kept.descriptors.reserve(raw_features.descriptors.size());

        for (int index : order) {
            const cv::Point2f& point = raw_features.original_points[index];
            int cell_x = static_cast<int>(std::floor(point.x / static_cast<float>(cell_size)));
            int cell_y = static_cast<int>(std::floor(point.y / static_cast<float>(cell_size)));
            bool duplicate = false;
            for (int dy = -1; dy <= 1 && !duplicate; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    auto found = occupied.find(superpoint_cell_key(cell_x + dx, cell_y + dy));
                    if (found == occupied.end()) {
                        continue;
                    }
                    const cv::Point2f& kept_point = kept.original_points[found->second];
                    float delta_x = point.x - kept_point.x;
                    float delta_y = point.y - kept_point.y;
                    if (delta_x * delta_x + delta_y * delta_y <= radius_sq) {
                        duplicate = true;
                        break;
                    }
                }
            }
            if (duplicate) {
                continue;
            }

            int kept_index = kept.size();
            kept.original_points.push_back(point);
            kept.scores.push_back(raw_features.scores[index]);
            const size_t offset = static_cast<size_t>(index) * kDescriptorDim;
            kept.descriptors.insert(kept.descriptors.end(),
                                    raw_features.descriptors.begin() + static_cast<ptrdiff_t>(offset),
                                    raw_features.descriptors.begin() + static_cast<ptrdiff_t>(offset + kDescriptorDim));
            occupied.emplace(superpoint_cell_key(cell_x, cell_y), kept_index);
        }
        return kept;
    }

    bool load_feature_cache() {
        if (cache_path_.empty()) {
            return false;
        }

        struct stat map_stat {};
        if (stat(map_path_.c_str(), &map_stat) != 0) {
            return false;
        }

        std::ifstream input(cache_path_, std::ios::binary);
        if (!input.is_open()) {
            return false;
        }

        CacheHeader header;
        input.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (!input
                || header.magic != kFeatureCacheMagic
                || (header.version != kFeatureCacheVersion && header.version != kLegacyFeatureCacheVersion)
                || header.map_size != static_cast<int64_t>(map_stat.st_size)
                || header.map_width != map_width_
                || header.map_height != map_height_
                || header.descriptor_rows < 2
                || header.descriptor_cols <= 0
                || header.keypoint_count != header.descriptor_rows) {
            TRACK_LOGD("feature cache rejected path=%s magic=%u version=%u size=%lld/%lld dims=%dx%d/%dx%d rows=%d keypoints=%d",
                       cache_path_.c_str(),
                       header.magic,
                       header.version,
                       static_cast<long long>(header.map_size),
                       static_cast<long long>(map_stat.st_size),
                       header.map_width,
                       header.map_height,
                       map_width_,
                       map_height_,
                       header.descriptor_rows,
                       header.keypoint_count);
            return false;
        }
        if (header.map_mtime_sec != static_cast<int64_t>(map_stat.st_mtime)) {
            TRACK_LOGD("feature cache accepted with map mtime mismatch path=%s cacheMtime=%lld mapMtime=%lld",
                       cache_path_.c_str(),
                       static_cast<long long>(header.map_mtime_sec),
                       static_cast<long long>(map_stat.st_mtime));
        }

        std::vector<cv::KeyPoint> keypoints;
        keypoints.reserve(header.keypoint_count);
        for (int index = 0; index < header.keypoint_count; index++) {
            CacheKeyPoint raw;
            input.read(reinterpret_cast<char*>(&raw), sizeof(raw));
            if (!input) {
                return false;
            }
            keypoints.emplace_back(
                    cv::Point2f(raw.x, raw.y),
                    raw.size,
                    raw.angle,
                    raw.response,
                    raw.octave,
                    raw.class_id);
        }

        cv::Mat descriptors(header.descriptor_rows, header.descriptor_cols, header.descriptor_type);
        const size_t byte_count = descriptors.total() * descriptors.elemSize();
        input.read(reinterpret_cast<char*>(descriptors.data), static_cast<std::streamsize>(byte_count));
        if (!input) {
            return false;
        }

        full_map_keypoints_ = std::move(keypoints);
        full_map_descriptors_ = descriptors;
        TRACK_LOGD("loaded full map SIFT features from cache path=%s count=%zu",
                   cache_path_.c_str(),
                   full_map_keypoints_.size());
        return true;
    }

    void save_feature_cache() const {
        if (cache_path_.empty() || full_map_keypoints_.empty() || full_map_descriptors_.empty()) {
            return;
        }

        struct stat map_stat {};
        if (stat(map_path_.c_str(), &map_stat) != 0) {
            return;
        }

        std::ofstream output(cache_path_, std::ios::binary | std::ios::trunc);
        if (!output.is_open()) {
            TRACK_LOGE("failed to open feature cache for write: %s", cache_path_.c_str());
            return;
        }

        CacheHeader header;
        header.map_size = static_cast<int64_t>(map_stat.st_size);
        header.map_mtime_sec = static_cast<int64_t>(map_stat.st_mtime);
        header.map_width = map_width_;
        header.map_height = map_height_;
        header.descriptor_rows = full_map_descriptors_.rows;
        header.descriptor_cols = full_map_descriptors_.cols;
        header.descriptor_type = full_map_descriptors_.type();
        header.keypoint_count = static_cast<int32_t>(full_map_keypoints_.size());
        output.write(reinterpret_cast<const char*>(&header), sizeof(header));

        for (const cv::KeyPoint& keypoint : full_map_keypoints_) {
            CacheKeyPoint raw;
            raw.x = keypoint.pt.x;
            raw.y = keypoint.pt.y;
            raw.size = keypoint.size;
            raw.angle = keypoint.angle;
            raw.response = keypoint.response;
            raw.octave = keypoint.octave;
            raw.class_id = keypoint.class_id;
            output.write(reinterpret_cast<const char*>(&raw), sizeof(raw));
        }

        cv::Mat continuous_descriptors = full_map_descriptors_.isContinuous()
                ? full_map_descriptors_
                : full_map_descriptors_.clone();
        const size_t byte_count = continuous_descriptors.total() * continuous_descriptors.elemSize();
        output.write(reinterpret_cast<const char*>(continuous_descriptors.data), static_cast<std::streamsize>(byte_count));
        if (!output.good()) {
            TRACK_LOGE("failed to save feature cache: %s", cache_path_.c_str());
            return;
        }

        TRACK_LOGD("saved full map SIFT features to cache path=%s count=%zu",
                   cache_path_.c_str(),
                   full_map_keypoints_.size());
    }

    void log_timing_if_needed(const TrackResult& result, const StepTiming& timing) const {
        if (!kEnableStepTiming) {
            return;
        }
        const bool should_log = (track_count_ % kTimingLogInterval) == 0
                || result.cost_ms >= 120
                || result.manual_required;
        if (!should_log) {
            return;
        }
        TRACK_LOGD("定位耗时 total=%dms mode=%s sp=%dx%d good=%d inliers=%d ratio=%.2f result=(%.1f,%.1f) found=%d R=%d L=%d",
                   result.cost_ms,
                   timing.global_search ? "sift-global" : "sp-local",
                   timing.sp_width,
                   timing.sp_height,
                   timing.good_matches,
                   timing.inlier_matches,
                   timing.inlier_ratio,
                   result.x,
                   result.y,
                   result.found ? 1 : 0,
                   result.search_radius,
                   result.lost_frames);
    }

    int track_count_ = 0;
    std::string map_path_;
    std::string cache_path_;
    std::string model_dir_;
    int minimap_left_;
    int minimap_top_;
    int minimap_width_;
    int minimap_height_;
    int base_search_radius_;
    int current_search_radius_;
    int max_lost_frames_;
    float match_ratio_;
    int min_match_count_;
    double ransac_threshold_;
    int map_width_ = 0;
    int map_height_ = 0;
    int last_x_ = 0;
    int last_y_ = 0;
    int lost_frames_ = 0;
    bool has_position_ = false;
    bool has_smoothed_position_ = false;
    bool manual_required_ = false;
    bool has_cached_local_sp_features_ = false;
    bool has_local_failure_since_ = false;
    float smoothed_cx_ = 0.f;
    float smoothed_cy_ = 0.f;
    std::chrono::steady_clock::time_point local_failure_since_;
    std::chrono::steady_clock::time_point last_global_search_time_ = std::chrono::steady_clock::now() - std::chrono::seconds(10);
    float last_rotation_angle_ = 0.f;

    cv::Mat logic_map_bgr_;
    std::vector<cv::KeyPoint> full_map_keypoints_;
    cv::Mat full_map_descriptors_;
    cv::Rect cached_local_sp_rect_;
    int cached_local_sp_width_ = 0;
    int cached_local_sp_height_ = 0;
    FeatureSet cached_local_sp_features_;
    cv::Ptr<cv::CLAHE> clahe_;
    cv::Ptr<cv::SIFT> sift_;
    cv::Ptr<cv::ORB> orb_;
    cv::Ptr<cv::DescriptorMatcher> flann_;
    ncnn::Net superpoint_net_;
};

#ifdef __ANDROID__
jfloatArray make_track_payload(JNIEnv* env, const TrackResult& result) {
    jfloat payload[30] = {0.f};
    payload[0] = static_cast<jfloat>(result.cost_ms);
    payload[1] = static_cast<jfloat>(result.x);
    payload[2] = static_cast<jfloat>(result.y);
    payload[3] = result.found ? 1.0f : 0.0f;
    payload[4] = result.inertial ? 1.0f : 0.0f;
    payload[5] = static_cast<jfloat>(result.good_matches);
    payload[6] = static_cast<jfloat>(result.search_radius);
    payload[7] = static_cast<jfloat>(result.lost_frames);
    payload[8] = result.manual_required ? 1.0f : 0.0f;
    payload[9] = static_cast<jfloat>(result.inlier_matches);
    payload[10] = static_cast<jfloat>(result.inlier_ratio);
    payload[11] = result.global_search ? 1.0f : 0.0f;
    payload[29] = static_cast<jfloat>(result.rotation_angle);

    jfloatArray out = env->NewFloatArray(30);
    env->SetFloatArrayRegion(out, 0, 30, payload);
    return out;
}
#endif  // __ANDROID__

template <typename T>
T* from_handle(jlong handle) {
    return reinterpret_cast<T*>(handle);
}

}  // namespace

// MapSiftTracker 的 JNI 绑定层只在 Android 上编译；主机构建只导出下方的纯 C API。
#ifdef __ANDROID__

extern "C"
JNIEXPORT jlong JNICALL
Java_com_example_myapplication_MapSiftTracker_nativeCreate(
        JNIEnv* env,
        jclass,
        jstring map_path,
        jstring cache_path,
        jstring model_dir,
        jint minimap_left,
        jint minimap_top,
        jint minimap_width,
        jint minimap_height,
        jint base_search_radius,
        jint max_lost_frames,
        jfloat clahe_limit,
        jfloat match_ratio,
        jint min_match_count,
        jfloat ransac_threshold) {
    if (map_path == nullptr || model_dir == nullptr) {
        return 0L;
    }

    const char* map_chars = env->GetStringUTFChars(map_path, nullptr);
    const char* cache_chars = cache_path == nullptr ? nullptr : env->GetStringUTFChars(cache_path, nullptr);
    const char* model_chars = env->GetStringUTFChars(model_dir, nullptr);
    if (map_chars == nullptr || model_chars == nullptr) {
        if (cache_chars != nullptr) {
            env->ReleaseStringUTFChars(cache_path, cache_chars);
        }
        if (map_chars != nullptr) {
            env->ReleaseStringUTFChars(map_path, map_chars);
        }
        if (model_chars != nullptr) {
            env->ReleaseStringUTFChars(model_dir, model_chars);
        }
        return 0L;
    }

    jlong handle = 0L;
    try {
        auto* tracker = new NativeMapSiftTracker(
                map_chars,
                cache_chars == nullptr ? "" : cache_chars,
                model_chars,
                minimap_left,
                minimap_top,
                minimap_width,
                minimap_height,
                base_search_radius,
                max_lost_frames,
                clahe_limit,
                match_ratio,
                min_match_count,
                ransac_threshold);
        handle = reinterpret_cast<jlong>(tracker);
    } catch (const std::exception& e) {
        TRACK_LOGE("nativeCreate failed: %s", e.what());
    }

    if (cache_chars != nullptr) {
        env->ReleaseStringUTFChars(cache_path, cache_chars);
    }
    env->ReleaseStringUTFChars(model_dir, model_chars);
    env->ReleaseStringUTFChars(map_path, map_chars);
    return handle;
}

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_example_myapplication_MapSiftTracker_nativeTrack(
        JNIEnv* env,
        jclass,
        jlong handle,
        jobject bitmap) {
    auto* tracker = from_handle<NativeMapSiftTracker>(handle);
    if (tracker == nullptr || bitmap == nullptr) {
        return make_track_payload(env, TrackResult{});
    }
    return make_track_payload(env, tracker->track(env, bitmap));
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_myapplication_MapSiftTracker_nativeSetManualPosition(
        JNIEnv*,
        jclass,
        jlong handle,
        jfloat x,
        jfloat y) {
    auto* tracker = from_handle<NativeMapSiftTracker>(handle);
    if (tracker == nullptr) {
        return;
    }
    tracker->set_manual_position(x, y);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_myapplication_MapSiftTracker_nativeForceGlobalSearch(
        JNIEnv*,
        jclass,
        jlong handle) {
    auto* tracker = from_handle<NativeMapSiftTracker>(handle);
    if (tracker == nullptr) {
        return;
    }
    tracker->force_global_search();
}

// nativeSetPreciseTracking / nativeSetRetryFullResolutionEnabled /
// nativeSetLocalSuperPointResolution 三个 setter 已随对应模式一并删除：
// 它们控制的是"局部匹配改用稀疏全图缓存"这条实测 0/111 成功的路径。
// 注意 app 侧编译的是 app/src/main/cpp/ 下的独立副本，不受此处影响。

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_example_myapplication_MapSiftTracker_nativeTrackSuperpointLocal(
        JNIEnv* env,
        jclass,
        jlong handle,
        jobject bitmap) {
    auto* tracker = from_handle<NativeMapSiftTracker>(handle);
    if (tracker == nullptr || bitmap == nullptr) {
        return make_track_payload(env, TrackResult{});
    }
    return make_track_payload(env, tracker->track_superpoint_local(env, bitmap));
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_myapplication_MapSiftTracker_nativeRelease(
        JNIEnv*,
        jclass,
        jlong handle) {
    delete from_handle<NativeMapSiftTracker>(handle);
}

#endif  // __ANDROID__

// --- C Export API for Universal Player Tracker ---

extern "C" {

void* player_tracker_create(const char* map_path,
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
                            double ransac_threshold) {
    try {
        return new NativeMapSiftTracker(
            map_path ? map_path : "",
            cache_path ? cache_path : "",
            model_dir ? model_dir : "",
            minimap_left, minimap_top, minimap_width, minimap_height,
            base_search_radius, max_lost_frames, clahe_limit,
            match_ratio, min_match_count, ransac_threshold
        );
    } catch (...) {
        return nullptr;
    }
}

bool player_tracker_locate(void* handle, const int32_t* frame_pixels, int w, int h,
                           float* out_x, float* out_y, float* confidence, int* cost_ms) {
    auto* tracker = from_handle<NativeMapSiftTracker>(reinterpret_cast<jlong>(handle));
    if (!tracker || !frame_pixels || w <= 0 || h <= 0) {
        return false;
    }

    try {
        const auto start = std::chrono::steady_clock::now();
        StepTiming timing;
        tracker->track_count_ += 1;

        // frame_pixels 是 ARGB packed int32，小端内存序为 B,G,R,A。
        cv::Mat bgra(h, w, CV_8UC4, const_cast<int32_t*>(frame_pixels));
        cv::Rect minimap_rect(tracker->minimap_left_, tracker->minimap_top_, tracker->minimap_width_, tracker->minimap_height_);
        
        if (minimap_rect.x < 0 || minimap_rect.y < 0 || 
            minimap_rect.x + minimap_rect.width > w || 
            minimap_rect.y + minimap_rect.height > h) {
            return false;
        }

        cv::Mat mini_bgra = bgra(minimap_rect);
        cv::Mat mini_bgr;
        cv::cvtColor(mini_bgra, mini_bgr, cv::COLOR_BGRA2BGR);

        TrackResult result = tracker->track_minimap(mini_bgr, timing);
        const auto end = std::chrono::steady_clock::now();
        int elapsed = static_cast<int>(std::chrono::duration<double, std::milli>(end - start).count());

        if (out_x) *out_x = result.x;
        if (out_y) *out_y = result.y;
        if (confidence) *confidence = result.found ? 1.0f : 0.0f;
        if (cost_ms) *cost_ms = elapsed;

        return result.found;
    } catch (...) {
        return false;
    }
}

void player_tracker_release(void* handle) {
    delete from_handle<NativeMapSiftTracker>(reinterpret_cast<jlong>(handle));
}

} // extern "C"
