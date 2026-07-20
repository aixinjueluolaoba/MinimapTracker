#include "navigation_engine.h"
#include "pointer_angle_detector.h"

#include <jni.h>

#ifdef __ANDROID__
#include <android/log.h>
#define PTR_LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "luoke", __VA_ARGS__)
#define PTR_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "luoke", __VA_ARGS__)
#else
#include <cstdio>
#define PTR_LOGD(...) do { printf("[pointer DEBUG] "); printf(__VA_ARGS__); printf("\n"); } while(0)
#define PTR_LOGE(...) do { fprintf(stderr, "[pointer ERROR] "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#endif

extern "C" {

// C-Export Pointer Detector
bool detect_pointer_angle_c(const int32_t* pixels, int width, int height,
                               float centerX, float centerY,
                               bool* has_match, int* angle_deg, float* confidence) {
    DetectResult res = detectPointerAngle(pixels, width, height, centerX, centerY);
    if (has_match) *has_match = res.has_match;
    if (angle_deg) *angle_deg = res.angle_deg;
    if (confidence) *confidence = res.confidence;
    return true;
}

} // extern "C"

// Android JNI Pointer Angle Detector
extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_myapplication_PointerAngleDetector_nativeDetect(
        JNIEnv* env, jclass clazz,
        jintArray pixels,
        jint width, jint height,
        jfloat centerX, jfloat centerY) {
    (void) clazz;
    if (pixels == nullptr || width <= 0 || height <= 0) {
        jfloatArray empty = env->NewFloatArray(3);
        if (empty == nullptr) return nullptr;
        jfloat raw[3] = {0.f, 0.f, 0.f};
        env->SetFloatArrayRegion(empty, 0, 3, raw);
        return empty;
    }

    jint* raw_pixels = env->GetIntArrayElements(pixels, nullptr);
    if (raw_pixels == nullptr) {
        return nullptr;
    }

    DetectResult res = detectPointerAngle(reinterpret_cast<const int32_t*>(raw_pixels), width, height, centerX, centerY);
    env->ReleaseIntArrayElements(pixels, raw_pixels, JNI_ABORT);

    jfloatArray result = env->NewFloatArray(3);
    if (result == nullptr) return nullptr;

    jfloat raw[3];
    raw[0] = res.has_match ? 1.f : 0.f;
    raw[1] = static_cast<jfloat>(res.angle_deg);
    raw[2] = res.confidence;
    env->SetFloatArrayRegion(result, 0, 3, raw);
    return result;
}
