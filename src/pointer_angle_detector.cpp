#include "pointer_angle_detector.h"
#include <cmath>

// 见 detect_patch()：模型无拒识头，confidence 只是占位常量。
static constexpr float kNoRejectionConfidence = 1.0f;

#ifdef __ANDROID__
#include <android/log.h>
#define DET_LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "luoke_det", __VA_ARGS__)
#define DET_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "luoke_det", __VA_ARGS__)
#else
#include <cstdio>
#define DET_LOGD(...) do { printf("[detector DEBUG] "); printf(__VA_ARGS__); printf("\n"); } while(0)
#define DET_LOGE(...) do { fprintf(stderr, "[detector ERROR] "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#endif

PointerAngleDetector::PointerAngleDetector(const std::string& model_dir) {
    std::string param_path = model_dir + "/pointer_angle_cnn_v2_bgr_pool2.ncnn.param";
    std::string bin_path = model_dir + "/pointer_angle_cnn_v2_bgr_pool2.ncnn.bin";

    DET_LOGD("Loading neural pointer param: %s", param_path.c_str());
    DET_LOGD("Loading neural pointer model: %s", bin_path.c_str());

    net_.opt.use_vulkan_compute = false;
    net_.opt.num_threads = 2;

    int param_ret = net_.load_param(param_path.c_str());
    int model_ret = net_.load_model(bin_path.c_str());

    if (param_ret != 0 || model_ret != 0) {
        DET_LOGE("Failed to load neural pointer detector model! param=%d model=%d", param_ret, model_ret);
        loaded_ = false;
    } else {
        DET_LOGD("Neural pointer detector model loaded successfully.");
        loaded_ = true;
    }
}

PointerDetectResult PointerAngleDetector::detect(const cv::Mat& frame_bgr) {
    if (!loaded_) {
        return {false, 0.f, 0.f};
    }

    // Safety checks for 1280x720 frame
    // Small map ROI is MM_X=1072, MM_Y=25, MM_S=128
    if (frame_bgr.empty() || frame_bgr.cols < 1200 || frame_bgr.rows < 153) {
        return {false, 0.f, 0.f};
    }

    try {
        cv::Rect mm_rect(1072, 25, 128, 128);
        cv::Mat mm = frame_bgr(mm_rect);

        // Crop 32x32 patch centered at relative (63,63)
        // Center: (PTR_CX, PTR_CY) = (63, 63). Crop width = 32, half width = 16.
        // Start position = (63-16, 63-16) = (47, 47)
        cv::Rect patch_rect(47, 47, 32, 32);
        cv::Mat patch = mm(patch_rect);

        return detect_patch(patch);
    } catch (...) {
        return {false, 0.f, 0.f};
    }
}

PointerDetectResult PointerAngleDetector::detect_patch(const cv::Mat& patch_bgr_32x32) {
    if (!loaded_ || patch_bgr_32x32.empty() || patch_bgr_32x32.cols != 32 || patch_bgr_32x32.rows != 32
            || patch_bgr_32x32.type() != CV_8UC3) {
        return {false, 0.f, 0.f};
    }

    try {
        // Convert to NCNN Mat (3 channels, 32x32).
        // detect() 传进来的 patch 是整帧的 ROI 视图，行间距等于整帧的 step（1280x720 帧为 3840 字节），
        // 而不带 stride 的 from_pixels 会按 32*3=96 字节紧凑读取，只有第 0 行是对的。
        // 必须显式传 patch 的真实 step。
        ncnn::Mat in = ncnn::Mat::from_pixels(patch_bgr_32x32.data, ncnn::Mat::PIXEL_BGR, 32, 32,
                                              static_cast<int>(patch_bgr_32x32.step));

        // Normalize BGR channels by dividing 255.0
        const float norm[3] = {1/255.f, 1/255.f, 1/255.f};
        in.substract_mean_normalize(nullptr, norm);

        ncnn::Extractor ex = net_.create_extractor();
        ex.input("in0", in);

        ncnn::Mat out;
        int extract_ret = ex.extract("out0", out);
        if (extract_ret != 0) {
            return {false, 0.f, 0.f};
        }

        // Output: [sin, cos]
        float s = out[0];
        float c = out[1];

        // Solve angle: angle = (rad2deg(arctan2(sin, cos)) + 360) % 360
        float angle_rad = atan2f(s, c);
        float angle_deg = angle_rad * 180.f / 3.14159265f;
        if (angle_deg < 0) {
            angle_deg += 360.f;
        }

        // 该网络是纯角度回归、没有拒识头，输出向量已 L2 归一化（实测 |[s,c]| 恒为 1.0），
        // 因此无法从中导出真实置信度。这里的 has_match 只表示"推理成功"，
        // confidence 是占位常量，调用方不要把它当成概率使用。
        return {true, angle_deg, kNoRejectionConfidence};
    } catch (...) {
        return {false, 0.f, 0.f};
    }
}
