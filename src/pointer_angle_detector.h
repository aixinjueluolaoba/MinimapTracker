#pragma once

#include <cstdint>

struct DetectResult {
    bool has_match = false;
    int angle_deg = 0;
    float confidence = 0.f;
};

DetectResult detectPointerAngle(const int32_t* pixels, int width, int height,
                                 float centerX, float centerY);
