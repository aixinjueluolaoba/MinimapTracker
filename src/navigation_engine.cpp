#include "navigation_engine.h"
#include "pointer_angle_detector.h"

#include <jni.h>

#ifdef __ANDROID__
#include <android/log.h>
#define NAV_LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "luoke", __VA_ARGS__)
#define NAV_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "luoke", __VA_ARGS__)
#else
#include <cstdio>
#define NAV_LOGD(...) do { printf("[luoke DEBUG] "); printf(__VA_ARGS__); printf("\n"); } while(0)
#define NAV_LOGE(...) do { fprintf(stderr, "[luoke ERROR] "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>

#include "blind_nav_rl_policy_weights.h"

namespace {

using namespace blind_nav_rl_weights;

constexpr float kWorldSize = 2400.f;
constexpr float kDefaultArrivalDistancePx = 8.f;
constexpr float kDefaultSlowDownDistancePx = 30.f;
constexpr float kDeltaStickLimitDeg = 45.f;
constexpr float kJumpThreshold = 0.5f;
constexpr int64_t kStepSleepMs = 300L;
constexpr int64_t kJumpCooldownMs = 800L;
constexpr int kNearTargetForcedSpeedPercent = 15;
constexpr float kPi = 3.14159265358979323846f;
constexpr double kRadToDeg = 57.29577951308232;
constexpr double kDegToRad = 0.017453292519943295;

struct StepInput {
    float track_x = 0.f;
    float track_y = 0.f;
    bool track_found = false;
    bool manual_required = false;
    bool pointer_has_match = false;
    float pointer_angle_deg = 0.f;
    bool pointer_frozen = false;
    int64_t pointer_age_ms = 0L;
    int64_t now_ms = 0L;
};

struct StepOutput {
    int action = kActionWait;
    int status = kStatusNone;
    int64_t sleep_ms = 0L;
    float hold_angle_deg = 0.f;
    int speed_percent = 0;
    float map_angle_deg = std::numeric_limits<float>::quiet_NaN();
    float current_angle_deg = std::numeric_limits<float>::quiet_NaN();
    float angle_delta_deg = 0.f;
    float signed_delta_deg = 0.f;
    int target_index = 0;
    float target_x = -1.f;
    float target_y = -1.f;
    bool using_cached_track = false;
    bool jump_requested = false;
};

float clamp(float value, float low, float high) {
    return std::max(low, std::min(high, value));
}

float sigmoid(float value) {
    return 1.f / (1.f + std::exp(-value));
}

float normalizeAngle(float angle_deg) {
    float normalized = std::fmod(angle_deg, 360.f);
    if (normalized < 0.f) {
        normalized += 360.f;
    }
    return normalized;
}

float signedAngleDelta(float target_angle_deg, float current_angle_deg) {
    float delta = normalizeAngle(target_angle_deg) - normalizeAngle(current_angle_deg);
    if (delta > 180.f) {
        delta -= 360.f;
    } else if (delta < -180.f) {
        delta += 360.f;
    }
    return delta;
}

float shortestAngleDelta(float target_angle_deg, float current_angle_deg) {
    return std::fabs(signedAngleDelta(target_angle_deg, current_angle_deg));
}

float distance(float from_x, float from_y, float to_x, float to_y) {
    return std::hypot(to_x - from_x, to_y - from_y);
}

float computeMapAngle(float from_x, float from_y, float target_x, float target_y) {
    float dx = target_x - from_x;
    float dy = -(target_y - from_y);
    return normalizeAngle(static_cast<float>(std::atan2(dy, dx) * kRadToDeg));
}

template <int OUT, int IN>
void linear(const float* weight,
            const float* bias,
            const std::array<float, IN>& input,
            std::array<float, OUT>& output) {
    for (int row = 0; row < OUT; ++row) {
        float sum = bias[row];
        const float* row_weight = weight + row * IN;
        for (int col = 0; col < IN; ++col) {
            sum += row_weight[col] * input[col];
        }
        output[row] = sum;
    }
}

class NativeAutoNavigationEngine {
public:
    bool start(float target_x,
               float target_y,
               float initial_hold_angle_deg,
               int base_speed_percent,
               float arrival_distance_px,
               float slow_down_distance_px) {
        (void) initial_hold_angle_deg;
        (void) base_speed_percent;
        resetState();
        if (!std::isfinite(target_x) || !std::isfinite(target_y) || target_x < 0.f || target_y < 0.f) {
            return false;
        }
        target_started_ = true;
        target_x_ = target_x;
        target_y_ = target_y;
        arrival_distance_px_ = std::isfinite(arrival_distance_px) && arrival_distance_px > 0.f
                ? arrival_distance_px
                : kDefaultArrivalDistancePx;
        slow_down_distance_px_ = std::isfinite(slow_down_distance_px) && slow_down_distance_px > 0.f
                ? slow_down_distance_px
                : kDefaultSlowDownDistancePx;
        return true;
    }

    StepOutput step(const StepInput& input) {
        if (!target_started_ || input.pointer_frozen) {
            return buildErrorOutput();
        }
        if (!input.track_found || input.manual_required || !input.pointer_has_match) {
            return buildErrorOutput();
        }

        float current_distance = distance(input.track_x, input.track_y, target_x_, target_y_);
        float map_angle_deg = computeMapAngle(input.track_x, input.track_y, target_x_, target_y_);
        float current_angle_deg = normalizeAngle(input.pointer_angle_deg);
        if (current_distance <= arrival_distance_px_) {
            return buildCompleteOutput(map_angle_deg, current_angle_deg);
        }

        std::array<float, kObsDim> obs = buildObservation(input, current_distance, map_angle_deg, current_angle_deg);
        auto inference_started_at = std::chrono::steady_clock::now();
        std::array<float, kActionDim> action = predict(obs);
        auto inference_finished_at = std::chrono::steady_clock::now();
        double inference_ms = std::chrono::duration<double, std::milli>(
                inference_finished_at - inference_started_at).count();
        for (float& value : action) {
            value = clamp(value, -1.f, 1.f);
        }
        int64_t inference_interval_ms = last_inference_at_ms_ > 0L
                ? input.now_ms - last_inference_at_ms_
                : -1L;
        last_inference_at_ms_ = input.now_ms;
        NAV_LOGD("RL推理耗时=%.3fms 间隔=%lldms pointerAge=%lldms action=(%.3f,%.3f,%.3f)",
                 inference_ms,
                 static_cast<long long>(inference_interval_ms),
                 static_cast<long long>(input.pointer_age_ms),
                 action[0],
                 action[1],
                 action[2]);

        float delta_stick_deg = action[0] * kDeltaStickLimitDeg;
        float hold_base_angle_deg = has_last_hold_angle_ ? last_hold_angle_deg_ : current_angle_deg;
        float hold_angle_deg = normalizeAngle(hold_base_angle_deg + delta_stick_deg);
        float speed_norm = clamp(action[1], 0.f, 1.f);
        int speed_percent = static_cast<int>(std::round(50.f + speed_norm * 50.f));
        speed_percent = std::max(1, std::min(100, speed_percent));
        bool inside_slow_down_range = current_distance <= slow_down_distance_px_;
        if (inside_slow_down_range) {
            near_target_speed_latched_ = true;
        }
        bool near_target = near_target_speed_latched_;
        if (near_target) {
            speed_percent = kNearTargetForcedSpeedPercent;
            speed_norm = 0.f;
        }
        bool jump_requested = !near_target
                && action[2] > kJumpThreshold
                && input.now_ms >= jump_blocked_until_ms_;
        if (jump_requested) {
            jump_blocked_until_ms_ = input.now_ms + kJumpCooldownMs;
        }

        last_action_speed_ = speed_norm * 2.f - 1.f;
        last_action_jump_ = action[2];
        previous_track_x_ = input.track_x;
        previous_track_y_ = input.track_y;
        previous_distance_to_target_ = current_distance;
        previous_angle_error_rad_ = angleErrorRad(map_angle_deg, current_angle_deg);
        has_previous_track_ = true;
        last_hold_angle_deg_ = hold_angle_deg;
        has_last_hold_angle_ = true;

        StepOutput output;
        output.action = continuous_hold_active_ ? kActionUpdate : kActionStart;
        output.status = kStatusMoving;
        output.sleep_ms = kStepSleepMs;
        output.hold_angle_deg = hold_angle_deg;
        output.speed_percent = speed_percent;
        output.map_angle_deg = map_angle_deg;
        output.current_angle_deg = current_angle_deg;
        output.angle_delta_deg = shortestAngleDelta(map_angle_deg, current_angle_deg);
        output.signed_delta_deg = signedAngleDelta(map_angle_deg, current_angle_deg);
        output.target_index = 0;
        output.target_x = target_x_;
        output.target_y = target_y_;
        output.using_cached_track = false;
        output.jump_requested = jump_requested;
        continuous_hold_active_ = true;
        return output;
    }

    void resetState() {
        target_started_ = false;
        target_x_ = -1;
        target_y_ = -1;
        arrival_distance_px_ = kDefaultArrivalDistancePx;
        slow_down_distance_px_ = kDefaultSlowDownDistancePx;
        hidden_.fill(0.f);
        cell_.fill(0.f);
        last_action_speed_ = -1.f;
        last_action_jump_ = 0.f;
        previous_track_x_ = 0.f;
        previous_track_y_ = 0.f;
        previous_distance_to_target_ = std::numeric_limits<float>::quiet_NaN();
        previous_angle_error_rad_ = 0.f;
        has_previous_track_ = false;
        continuous_hold_active_ = false;
        jump_blocked_until_ms_ = 0L;
        near_target_speed_latched_ = false;
        last_hold_angle_deg_ = 0.f;
        has_last_hold_angle_ = false;
        last_inference_at_ms_ = -1L;
    }

private:
    std::array<float, kObsDim> buildObservation(const StepInput& input,
                                                float current_distance,
                                                float map_angle_deg,
                                                float current_angle_deg) {
        float dx = target_x_ - input.track_x;
        float dy = -(target_y_ - input.track_y);
        float angle_error = angleErrorRad(map_angle_deg, current_angle_deg);
        float previous_angle_error = has_previous_track_ ? previous_angle_error_rad_ : angle_error;
        float distance_progress = has_previous_track_
                ? previous_distance_to_target_ - current_distance
                : 0.f;
        float movement_dist = has_previous_track_
                ? distance(previous_track_x_, previous_track_y_, input.track_x, input.track_y)
                : 0.f;
        float delta_angle_error = previous_angle_error - angle_error;
        return {
                clamp(dx / kWorldSize, -1.f, 1.f),
                clamp(dy / kWorldSize, -1.f, 1.f),
                clamp(angle_error / kPi, -1.f, 1.f),
                clamp(delta_angle_error / kPi, -1.f, 1.f),
                clamp(distance_progress / 30.f, -1.f, 1.f),
                clamp(movement_dist / 40.f, 0.f, 1.f),
                clamp(last_action_speed_, -1.f, 1.f),
                clamp(last_action_jump_, -1.f, 1.f),
        };
    }

    float angleErrorRad(float map_angle_deg, float current_angle_deg) const {
        float delta_deg = signedAngleDelta(map_angle_deg, current_angle_deg);
        return static_cast<float>(delta_deg * kDegToRad);
    }

    std::array<float, kActionDim> predict(const std::array<float, kObsDim>& obs) {
        std::array<float, kHiddenDim> next_hidden{};
        std::array<float, kHiddenDim> next_cell{};
        for (int gate_index = 0; gate_index < kHiddenDim; ++gate_index) {
            float gates[4];
            for (int gate = 0; gate < 4; ++gate) {
                int row = gate * kHiddenDim + gate_index;
                float sum = kLstmBiasIh[row] + kLstmBiasHh[row];
                const float* ih = kLstmWeightIh + row * kObsDim;
                for (int col = 0; col < kObsDim; ++col) {
                    sum += ih[col] * obs[col];
                }
                const float* hh = kLstmWeightHh + row * kHiddenDim;
                for (int col = 0; col < kHiddenDim; ++col) {
                    sum += hh[col] * hidden_[col];
                }
                gates[gate] = sum;
            }
            float input_gate = sigmoid(gates[0]);
            float forget_gate = sigmoid(gates[1]);
            float cell_gate = std::tanh(gates[2]);
            float output_gate = sigmoid(gates[3]);
            next_cell[gate_index] = forget_gate * cell_[gate_index] + input_gate * cell_gate;
            next_hidden[gate_index] = output_gate * std::tanh(next_cell[gate_index]);
        }
        hidden_ = next_hidden;
        cell_ = next_cell;

        std::array<float, kPolicyHiddenDim> policy{};
        linear<kPolicyHiddenDim, kHiddenDim>(kPolicyWeight, kPolicyBias, hidden_, policy);
        for (float& value : policy) {
            value = std::tanh(value);
        }
        std::array<float, kActionDim> action{};
        linear<kActionDim, kPolicyHiddenDim>(kActionWeight, kActionBias, policy, action);
        return action;
    }

    StepOutput buildCompleteOutput(float map_angle_deg, float current_angle_deg) {
        StepOutput output;
        output.action = kActionComplete;
        output.status = kStatusComplete;
        output.map_angle_deg = map_angle_deg;
        output.current_angle_deg = current_angle_deg;
        output.target_index = 0;
        output.target_x = target_x_;
        output.target_y = target_y_;
        target_started_ = false;
        continuous_hold_active_ = false;
        return output;
    }

    StepOutput buildErrorOutput() {
        StepOutput output;
        output.action = continuous_hold_active_ ? kActionStop : kActionError;
        output.status = kStatusFailed;
        output.target_index = target_started_ ? 0 : -1;
        output.target_x = target_x_;
        output.target_y = target_y_;
        target_started_ = false;
        continuous_hold_active_ = false;
        return output;
    }

    bool target_started_ = false;
    float target_x_ = -1.f;
    float target_y_ = -1.f;
    float arrival_distance_px_ = kDefaultArrivalDistancePx;
    float slow_down_distance_px_ = kDefaultSlowDownDistancePx;
    std::array<float, kHiddenDim> hidden_{};
    std::array<float, kHiddenDim> cell_{};
    float last_action_speed_ = -1.f;
    float last_action_jump_ = 0.f;
    float previous_track_x_ = 0.f;
    float previous_track_y_ = 0.f;
    float previous_distance_to_target_ = std::numeric_limits<float>::quiet_NaN();
    float previous_angle_error_rad_ = 0.f;
    bool has_previous_track_ = false;
    bool continuous_hold_active_ = false;
    int64_t jump_blocked_until_ms_ = 0L;
    bool near_target_speed_latched_ = false;
    float last_hold_angle_deg_ = 0.f;
    bool has_last_hold_angle_ = false;
    int64_t last_inference_at_ms_ = -1L;
};

jintArray makeStartResult(JNIEnv* env, bool success) {
    jintArray result = env->NewIntArray(1);
    if (result == nullptr) {
        return nullptr;
    }
    jint raw[1];
    raw[0] = success ? 1 : 0;
    env->SetIntArrayRegion(result, 0, 1, raw);
    return result;
}

jfloatArray makeStepResult(JNIEnv* env, const StepOutput& output) {
    jfloatArray result = env->NewFloatArray(14);
    if (result == nullptr) {
        return nullptr;
    }
    jfloat raw[14];
    raw[0] = static_cast<jfloat>(output.action);
    raw[1] = static_cast<jfloat>(output.status);
    raw[2] = static_cast<jfloat>(output.sleep_ms);
    raw[3] = output.hold_angle_deg;
    raw[4] = static_cast<jfloat>(output.speed_percent);
    raw[5] = output.map_angle_deg;
    raw[6] = output.current_angle_deg;
    raw[7] = output.angle_delta_deg;
    raw[8] = output.signed_delta_deg;
    raw[9] = static_cast<jfloat>(output.target_index);
    raw[10] = static_cast<jfloat>(output.target_x);
    raw[11] = static_cast<jfloat>(output.target_y);
    raw[12] = output.using_cached_track ? 1.f : 0.f;
    raw[13] = output.jump_requested ? 1.f : 0.f;
    env->SetFloatArrayRegion(result, 0, 14, raw);
    return result;
}

template <typename T>
T* fromHandle(jlong handle) {
    return reinterpret_cast<T*>(static_cast<intptr_t>(handle));
}

} // namespace

// --- C Export API Implementation ---

extern "C" {

void* navigation_engine_create() {
    return new NativeAutoNavigationEngine();
}

bool navigation_engine_start(void* handle,
                                float target_x,
                                float target_y,
                                float initial_hold_angle_deg,
                                int base_speed_percent,
                                float arrival_distance_px,
                                float slow_down_distance_px) {
    auto* engine = static_cast<NativeAutoNavigationEngine*>(handle);
    if (engine == nullptr) {
        return false;
    }
    return engine->start(target_x, target_y, initial_hold_angle_deg,
                          base_speed_percent, arrival_distance_px, slow_down_distance_px);
}

bool navigation_engine_step(void* handle,
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
                               bool* jump_requested) {
    auto* engine = static_cast<NativeAutoNavigationEngine*>(handle);
    if (engine == nullptr) {
        return false;
    }

    StepInput input;
    input.track_x = track_x;
    input.track_y = track_y;
    input.track_found = track_found;
    input.manual_required = manual_required;
    input.pointer_has_match = pointer_has_match;
    input.pointer_angle_deg = pointer_angle_deg;
    input.pointer_frozen = pointer_frozen;
    input.pointer_age_ms = pointer_age_ms;
    input.now_ms = now_ms;

    StepOutput output = engine->step(input);

    if (action) *action = output.action;
    if (status) *status = output.status;
    if (sleep_ms) *sleep_ms = output.sleep_ms;
    if (hold_angle_deg) *hold_angle_deg = output.hold_angle_deg;
    if (speed_percent) *speed_percent = output.speed_percent;
    if (map_angle_deg) *map_angle_deg = output.map_angle_deg;
    if (current_angle_deg) *current_angle_deg = output.current_angle_deg;
    if (angle_delta_deg) *angle_delta_deg = output.angle_delta_deg;
    if (signed_delta_deg) *signed_delta_deg = output.signed_delta_deg;
    if (target_index) *target_index = output.target_index;
    if (target_x) *target_x = output.target_x;
    if (target_y) *target_y = output.target_y;
    if (using_cached_track) *using_cached_track = output.using_cached_track;
    if (jump_requested) *jump_requested = output.jump_requested;

    return true;
}

void navigation_engine_release(void* handle) {
    auto* engine = static_cast<NativeAutoNavigationEngine*>(handle);
    delete engine;
}

bool detect_pointer_angle_c(const int32_t* pixels, int width, int height,
                               float centerX, float centerY,
                               bool* has_match, int* angle_deg, float* confidence) {
    DetectResult res = detectPointerAngle(pixels, width, height, centerX, centerY);
    if (has_match) *has_match = res.has_match;
    if (angle_deg) *angle_deg = res.angle_deg;
    if (confidence) *confidence = res.confidence;
    return true;
}

bool navigation_engine_step_struct(void* handle, const struct NavInput* input, struct NavOutput* output) {
    if (!input || !output) return false;
    return navigation_engine_step(
        handle,
        input->track_x, input->track_y, input->track_found, input->manual_required,
        input->pointer_has_match, input->pointer_angle_deg, input->pointer_frozen,
        input->pointer_age_ms, input->now_ms,
        &output->action, &output->status, &output->sleep_ms, &output->hold_angle_deg, &output->speed_percent,
        &output->map_angle_deg, &output->current_angle_deg, &output->angle_delta_deg, &output->signed_delta_deg,
        &output->target_index, &output->target_x, &output->target_y, &output->using_cached_track, &output->jump_requested
    );
}

} // extern "C"

// --- Android JNI Export API Implementation ---

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_myapplication_AutoNavigationEngine_nativeCreate(JNIEnv* env, jclass clazz) {
    (void) env;
    (void) clazz;
    auto* engine = new NativeAutoNavigationEngine();
    return static_cast<jlong>(reinterpret_cast<intptr_t>(engine));
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_example_myapplication_AutoNavigationEngine_nativeStart(JNIEnv* env,
                                                                jclass clazz,
                                                                jlong handle,
                                                                jfloat target_x,
                                                                jfloat target_y,
                                                                jfloat initial_hold_angle_deg,
                                                                jint base_speed_percent,
                                                                jfloat arrival_distance_px,
                                                                jfloat slow_down_distance_px) {
    (void) clazz;
    auto* engine = fromHandle<NativeAutoNavigationEngine>(handle);
    if (engine == nullptr) {
        return makeStartResult(env, false);
    }

    bool success = false;
    try {
        success = engine->start(
                target_x,
                target_y,
                initial_hold_angle_deg,
                base_speed_percent,
                arrival_distance_px,
                slow_down_distance_px);
    } catch (...) {
        NAV_LOGE("nativeStart failed");
        success = false;
    }
    return makeStartResult(env, success);
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_myapplication_AutoNavigationEngine_nativeStep(JNIEnv* env,
                                                               jclass clazz,
                                                               jlong handle,
                                                               jfloat track_x,
                                                               jfloat track_y,
                                                               jboolean track_found,
                                                               jboolean manual_required,
                                                               jboolean pointer_has_match,
                                                               jint pointer_angle_deg,
                                                               jboolean pointer_frozen,
                                                               jlong pointer_age_ms,
                                                               jlong now_ms) {
    (void) clazz;
    auto* engine = fromHandle<NativeAutoNavigationEngine>(handle);
    if (engine == nullptr) {
        StepOutput output;
        output.action = kActionError;
        output.status = kStatusFailed;
        return makeStepResult(env, output);
    }

    StepInput input;
    input.track_x = track_x;
    input.track_y = track_y;
    input.track_found = track_found == JNI_TRUE;
    input.manual_required = manual_required == JNI_TRUE;
    input.pointer_has_match = pointer_has_match == JNI_TRUE;
    input.pointer_angle_deg = static_cast<float>(pointer_angle_deg);
    input.pointer_frozen = pointer_frozen == JNI_TRUE;
    input.pointer_age_ms = pointer_age_ms;
    input.now_ms = now_ms;

    try {
        return makeStepResult(env, engine->step(input));
    } catch (...) {
        NAV_LOGE("nativeStep failed");
        StepOutput output;
        output.action = kActionError;
        output.status = kStatusFailed;
        return makeStepResult(env, output);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_myapplication_AutoNavigationEngine_nativeRelease(JNIEnv* env,
                                                                  jclass clazz,
                                                                  jlong handle) {
    (void) env;
    (void) clazz;
    auto* engine = fromHandle<NativeAutoNavigationEngine>(handle);
    delete engine;
}

// PointerAngleDetector JNI
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
