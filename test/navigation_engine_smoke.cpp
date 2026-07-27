#include "navigation_engine.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 4 && argc != 6) {
        std::fprintf(
                stderr,
                "usage: %s MODEL_ROOT MAP_PATH CACHE_PATH [WIDTH HEIGHT < frames.bgr]\n",
                argv[0]);
        return 2;
    }

    void* engine = navigation_engine_create(argv[1], argv[2], argv[3]);
    if (engine == nullptr) {
        std::fprintf(stderr, "create failed: %s\n", navigation_engine_last_error(nullptr));
        return 3;
    }

    const int width = argc == 6 ? std::atoi(argv[4]) : 1280;
    const int height = argc == 6 ? std::atoi(argv[5]) : 720;
    if (width <= 0 || height <= 0) {
        std::fprintf(stderr, "invalid frame dimensions: %dx%d\n", width, height);
        navigation_engine_release(engine);
        return 4;
    }

    const size_t frame_size = static_cast<size_t>(width) * height * 3;
    std::vector<uint8_t> frame(frame_size, 0);
    int frame_count = 0;
    int located_count = 0;
    int pointer_count = 0;
    int first_fix_frame = 0;
    long long total_cost_ms = 0;

    while (true) {
        if (argc == 6) {
            size_t received = 0;
            while (received < frame_size) {
                const size_t count = std::fread(
                        frame.data() + received,
                        1,
                        frame_size - received,
                        stdin);
                if (count == 0) {
                    break;
                }
                received += count;
            }
            if (received == 0) {
                break;
            }
            if (received != frame_size) {
                std::fprintf(
                        stderr,
                        "truncated BGR frame: got %zu of %zu bytes\n",
                        received,
                        frame_size);
                navigation_engine_release(engine);
                return 5;
            }
        }

        NavigationEngineResult result{};
        const int ok = navigation_engine_process_bgr(
                engine,
                frame.data(),
                width,
                height,
                width * 3,
                &result);
        if (!ok) {
            std::fprintf(stderr, "process failed: %s\n", navigation_engine_last_error(engine));
            navigation_engine_release(engine);
            return 6;
        }

        frame_count += 1;
        located_count += result.located;
        pointer_count += result.pointer_detected;
        total_cost_ms += result.locate_cost_ms;
        if (result.located && first_fix_frame == 0) {
            first_fix_frame = frame_count;
        }
        if (argc == 6 && (frame_count == 1 || frame_count % 25 == 0)) {
            std::fprintf(
                    stderr,
                    "frame=%d located=%d pos=(%.2f,%.2f) cost_ms=%d pointer=%d angle=%.2f\n",
                    frame_count,
                    result.located,
                    result.x,
                    result.y,
                    result.locate_cost_ms,
                    result.pointer_detected,
                    result.angle_deg);
        }
        if (argc == 4) {
            break;
        }
    }

    const double average_cost_ms =
            frame_count == 0 ? 0.0 : static_cast<double>(total_cost_ms) / frame_count;
    std::printf(
            "smoke ok frames=%d located=%d pointer=%d first_fix=%d avg_cost_ms=%.2f\n",
            frame_count,
            located_count,
            pointer_count,
            first_fix_frame,
            average_cost_ms);
    navigation_engine_release(engine);
    return frame_count > 0 ? 0 : 7;
}
