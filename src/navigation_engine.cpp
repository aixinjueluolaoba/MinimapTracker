#include "navigation_engine.h"
#include "pointer_angle_detector.h"

#include <jni.h>
#include <opencv2/opencv.hpp>

#ifdef __ANDROID__
#include <android/log.h>
#define PTR_LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "luoke_jni", __VA_ARGS__)
#define PTR_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "luoke_jni", __VA_ARGS__)
#else
#include <cstdio>
#define PTR_LOGD(...) do { printf("[jni DEBUG] "); printf(__VA_ARGS__); printf("\n"); } while(0)
#define PTR_LOGE(...) do { fprintf(stderr, "[jni ERROR] "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#endif

// Global static pointer detector singleton for Android JNI bindings
static PointerAngleDetector* g_pointer_detector = nullptr;

extern "C" {

// ==========================================
// 1. C-Export APIs Implementation
// ==========================================

void* pointer_detector_create(const char* model_dir) {
    try {
        return new PointerAngleDetector(model_dir ? model_dir : "");
    } catch (...) {
        return nullptr;
    }
}

bool pointer_detector_detect(void* handle, const int32_t* frame_pixels, int w, int h,
                             bool* has_match, float* angle_deg, float* confidence) {
    auto* detector = static_cast<PointerAngleDetector*>(handle);
    if (!detector || !frame_pixels || w <= 0 || h <= 0) {
        return false;
    }

    try {
        // frame_pixels 是 ARGB packed int32（等同 Android Bitmap.getPixels）。
        // 小端机器上其内存字节序为 B,G,R,A，因此这里必须按 BGRA 解释，不是 RGBA。
        cv::Mat bgra(h, w, CV_8UC4, const_cast<int32_t*>(frame_pixels));
        cv::Mat bgr;
        cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);

        PointerDetectResult res = detector->detect(bgr);
        
        if (has_match) *has_match = res.has_match;
        if (angle_deg) *angle_deg = res.angle_deg;
        if (confidence) *confidence = res.confidence;

        return res.has_match;
    } catch (...) {
        return false;
    }
}

void pointer_detector_release(void* handle) {
    auto* detector = static_cast<PointerAngleDetector*>(handle);
    delete detector;
}

} // extern "C"

// ==========================================
// 2. Android JNI Bindings for NativePointerDetector
// ==========================================

extern "C" JNIEXPORT void JNICALL
Java_com_example_myapplication_NativePointerDetector_nativeInit(
        JNIEnv* env, jclass clazz,
        jstring model_dir) {
    (void) clazz;
    if (model_dir == nullptr) return;

    const char* raw_dir = env->GetStringUTFChars(model_dir, nullptr);
    if (raw_dir != nullptr) {
        if (g_pointer_detector) {
            delete g_pointer_detector;
        }
        g_pointer_detector = new PointerAngleDetector(raw_dir);
        env->ReleaseStringUTFChars(model_dir, raw_dir);
    }
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_myapplication_NativePointerDetector_nativeDetect(
        JNIEnv* env, jclass clazz,
        jintArray pixels,
        jint width, jint height) {
    (void) clazz;

    jfloatArray result = env->NewFloatArray(3);
    if (result == nullptr) return nullptr;

    jfloat raw[3] = {0.f, 0.f, 0.f};

    if (pixels == nullptr || width <= 0 || height <= 0) {
        env->SetFloatArrayRegion(result, 0, 3, raw);
        return result;
    }

    if (g_pointer_detector == nullptr) {
        PTR_LOGE("JNI detect failed: Pointer detector has not been initialized. Please call nativeInit first.");
        env->SetFloatArrayRegion(result, 0, 3, raw);
        return result;
    }

    jint* raw_pixels = env->GetIntArrayElements(pixels, nullptr);
    if (raw_pixels == nullptr) {
        env->SetFloatArrayRegion(result, 0, 3, raw);
        return result;
    }

    try {
        // Java int[] 来自 Bitmap.getPixels()，是 ARGB packed int，小端内存序为 B,G,R,A。
        cv::Mat bgra(height, width, CV_8UC4, reinterpret_cast<void*>(raw_pixels));
        cv::Mat bgr;
        cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);

        PointerDetectResult res = g_pointer_detector->detect(bgr);
        raw[0] = res.has_match ? 1.f : 0.f;
        raw[1] = res.angle_deg;
        raw[2] = res.confidence;
    } catch (...) {
        PTR_LOGE("JNI detect encountered unexpected C++ exception.");
    }

    env->ReleaseIntArrayElements(pixels, raw_pixels, JNI_ABORT);
    env->SetFloatArrayRegion(result, 0, 3, raw);
    return result;
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_myapplication_NativePointerDetector_nativeRelease(
        JNIEnv* env, jclass clazz) {
    (void) env;
    (void) clazz;
    if (g_pointer_detector) {
        delete g_pointer_detector;
        g_pointer_detector = nullptr;
    }
}
