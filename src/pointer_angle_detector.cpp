#include "pointer_angle_detector.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cstdio>

#ifdef __ANDROID__
#include <android/log.h>
#define PTR_LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "luoke", __VA_ARGS__)
#else
#define PTR_LOGD(...) do { printf("[pointer DEBUG] "); printf(__VA_ARGS__); printf("\n"); } while(0)
#endif

namespace {

constexpr int kTargetColorsRgb[][3] = {
        {235, 189, 60},
        {212, 144, 38}
};
constexpr int kTargetColorCount = 2;
constexpr float kColorTolerance = 42.f;
constexpr float kColorToleranceSq = kColorTolerance * kColorTolerance;
constexpr float kMinScore = 0.16f;
constexpr int kUpscale = 8;
constexpr int kNeighborhoodRadius = 2;
constexpr double kAngleOffsetDeg = 90.0;
constexpr double kRadToDeg = 57.29577951308232;

inline float colorDistanceSq(int r, int g, int b, int tr, int tg, int tb) {
    float dr = static_cast<float>(r - tr);
    float dg = static_cast<float>(g - tg);
    float db = static_cast<float>(b - tb);
    return dr * dr + dg * dg + db * db;
}

inline bool keepPixel(int r, int g, int b) {
    for (int i = 0; i < kTargetColorCount; ++i) {
        if (colorDistanceSq(r, g, b,
                            kTargetColorsRgb[i][0],
                            kTargetColorsRgb[i][1],
                            kTargetColorsRgb[i][2]) <= kColorToleranceSq) {
            return true;
        }
    }
    return false;
}

inline int matchedColorIndex(int r, int g, int b) {
    for (int i = 0; i < kTargetColorCount; ++i) {
        if (colorDistanceSq(r, g, b,
                            kTargetColorsRgb[i][0],
                            kTargetColorsRgb[i][1],
                            kTargetColorsRgb[i][2]) <= kColorToleranceSq) {
            return i;
        }
    }
    return -1;
}

inline float distanceF(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return std::sqrt(dx * dx + dy * dy);
}

} // namespace

DetectResult detectPointerAngle(const int32_t* pixels, int width, int height,
                                 float centerX, float centerY) {
    DetectResult result;
    int total = width * height;

    // Step 1: Color filter
    std::vector<bool> keep(total, false);
    std::vector<int> colorIndex(total, -1);
    int activeCount = 0;
    for (int i = 0; i < total; ++i) {
        int32_t argb = pixels[i];
        int r = (argb >> 16) & 0xff;
        int g = (argb >> 8) & 0xff;
        int b = argb & 0xff;
        if (keepPixel(r, g, b)) {
            keep[i] = true;
            colorIndex[i] = matchedColorIndex(r, g, b);
            activeCount++;
        }
    }
    if (activeCount < 2) {
        return result;
    }

    // Step 2: BFS largest connected component (8-connectivity)
    std::vector<int> labels(total, 0);
    std::vector<int> queue(total);
    int bestLabel = 0;
    int bestSize = 0;
    int label = 1;
    for (int i = 0; i < total; ++i) {
        if (!keep[i] || labels[i] != 0) continue;
        int head = 0, tail = 0;
        queue[tail++] = i;
        labels[i] = label;
        int size = 0;
        while (head < tail) {
            int idx = queue[head++];
            size++;
            int x = idx % width;
            int y = idx / width;
            for (int ny = std::max(0, y - 1); ny <= std::min(height - 1, y + 1); ++ny) {
                for (int nx = std::max(0, x - 1); nx <= std::min(width - 1, x + 1); ++nx) {
                    int next = ny * width + nx;
                    if (keep[next] && labels[next] == 0) {
                        labels[next] = label;
                        queue[tail++] = next;
                    }
                }
            }
        }
        if (size > bestSize) {
            bestSize = size;
            bestLabel = label;
        }
        label++;
    }

    // Build largest component mask
    std::vector<bool> largest(total, false);
    int largestCount = 0;
    for (int i = 0; i < total; ++i) {
        if (labels[i] == bestLabel) {
            largest[i] = true;
            largestCount++;
        }
    }
    if (largestCount < 2) {
        return result;
    }

    // Step 3: Upscale binary mask
    int outW = width * kUpscale;
    int outH = height * kUpscale;
    std::vector<bool> upscaled(outW * outH, false);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (!largest[y * width + x]) continue;
            for (int dy = 0; dy < kUpscale; ++dy) {
                int row = (y * kUpscale + dy) * outW;
                for (int dx = 0; dx < kUpscale; ++dx) {
                    upscaled[row + x * kUpscale + dx] = true;
                }
            }
        }
    }

    // Step 4: Find farthest point from center with color balance tie-breaking
    float scaledCenterX = centerX * kUpscale;
    float scaledCenterY = centerY * kUpscale;

    // First pass: find max distance
    float maxDistance = -1.f;
    for (int y = 0; y < outH; ++y) {
        for (int x = 0; x < outW; ++x) {
            if (!upscaled[y * outW + x]) continue;
            float d = distanceF(scaledCenterX, scaledCenterY,
                                static_cast<float>(x), static_cast<float>(y));
            if (d > maxDistance) maxDistance = d;
        }
    }
    if (maxDistance <= 0.f) {
        return result;
    }

    // Second pass: among candidates near max distance, pick best by color balance
    float distanceTolerance = std::max(static_cast<float>(kUpscale), maxDistance * 0.06f);
    float farX = -1.f, farY = -1.f, farDistance = -1.f;
    int bestBalancedColorCount = -1;
    int bestTotalColorCount = -1;

    for (int y = 0; y < outH; ++y) {
        for (int x = 0; x < outW; ++x) {
            if (!upscaled[y * outW + x]) continue;
            float candidateDistance = distanceF(scaledCenterX, scaledCenterY,
                                               static_cast<float>(x), static_cast<float>(y));
            if (maxDistance - candidateDistance > distanceTolerance) continue;

            // Count target colors in neighborhood (in original resolution)
            float pointX = static_cast<float>(x) / kUpscale;
            float pointY = static_cast<float>(y) / kUpscale;
            int cx = static_cast<int>(std::round(pointX));
            int cy = static_cast<int>(std::round(pointY));
            int counts[2] = {0, 0};
            for (int ny = std::max(0, cy - kNeighborhoodRadius);
                 ny <= std::min(height - 1, cy + kNeighborhoodRadius); ++ny) {
                for (int nx = std::max(0, cx - kNeighborhoodRadius);
                     nx <= std::min(width - 1, cx + kNeighborhoodRadius); ++nx) {
                    int idx = ny * width + nx;
                    if (largest[idx] && colorIndex[idx] >= 0) {
                        counts[colorIndex[idx]]++;
                    }
                }
            }
            int balancedColorCount = std::min(counts[0], counts[1]);
            int totalColorCount = counts[0] + counts[1];

            if (balancedColorCount > bestBalancedColorCount
                || (balancedColorCount == bestBalancedColorCount && totalColorCount > bestTotalColorCount)
                || (balancedColorCount == bestBalancedColorCount
                    && totalColorCount == bestTotalColorCount
                    && candidateDistance > farDistance)) {
                farDistance = candidateDistance;
                farX = static_cast<float>(x);
                farY = static_cast<float>(y);
                bestBalancedColorCount = balancedColorCount;
                bestTotalColorCount = totalColorCount;
            }
        }
    }
    if (farDistance <= 0.f) {
        return result;
    }

    // Step 5: Compute angle
    double dx = static_cast<double>(farX) - scaledCenterX;
    double dy = static_cast<double>(farY) - scaledCenterY;
    double angle = std::atan2(-dx, -dy) * kRadToDeg + kAngleOffsetDeg;
    angle = std::fmod(angle + 360.0, 360.0);
    int angleDeg = static_cast<int>(std::round(angle)) % 360;

    // Step 6: Confidence
    float maxRadius = std::sqrt(
            static_cast<float>(outW * outW + outH * outH));
    float score = farDistance / std::max(maxRadius, 1.f);

    result.angle_deg = angleDeg;
    result.confidence = score;
    result.has_match = score >= kMinScore;
    return result;
}
