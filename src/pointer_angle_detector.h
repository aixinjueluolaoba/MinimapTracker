#pragma once

#include <string>
#include <opencv2/opencv.hpp>
#include <net.h>

struct PointerDetectResult {
    bool has_match;
    float angle_deg;
    float confidence;
};

class PointerAngleDetector {
public:
    PointerAngleDetector(const std::string& model_dir);
    ~PointerAngleDetector() = default;

    // Detects pointer angle from full 1280x720 BGR frame
    PointerDetectResult detect(const cv::Mat& frame_bgr);

    // Detects pointer angle from pre-cropped 32x32 CV_8UC3 patch.
    // 允许传入非连续的 ROI 视图，内部按 Mat::step 读取。
    PointerDetectResult detect_patch(const cv::Mat& patch_bgr_32x32);

private:
    ncnn::Net net_;
    bool loaded_ = false;
};
