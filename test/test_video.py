#!/usr/bin/env python3
import cv2
import numpy as np
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from minimap_tracker import NavigationEngine  # noqa: E402

# Minimap region in the 1280x720 frame (used by SIFT/SuperPoint hybrid tracker).
# 必须与生产端一致，见 app/src/main/java/com/example/myapplication/MapSiftTracker.java:8-11
MINIMAP_LEFT = 1072
MINIMAP_TOP = 25
MINIMAP_WIDTH = 127
MINIMAP_HEIGHT = 127

def main():
    # Use the official user-provided test video
    video_path = "/home/diana/screencap/file/cv_tools/record/rec_20260720_014710_032096.mp4"
    if len(sys.argv) > 1:
        video_path = sys.argv[1]

    print(f"Opening test video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.", file=sys.stderr)
        return -1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video Info: {width}x{height}, {fps} FPS, {total_frames} frames total.")

    # 先校验前置条件再建 VideoWriter：VideoWriter 一创建就会把目标文件截断，
    # 若之后才发现 .so 缺失并返回，就会留下一个 0 帧的空视频（并覆盖掉别人的产物）。
    so_path = REPO / "build/libfishing_native.so"
    if not so_path.exists():
        print(f"Error: Shared library not found at {so_path}.\n"
              "本机构建目前是跑不通的，原因有两个，都要先解决：\n"
              "  1. 宿主没有安装 OpenCV C++ 开发包（全盘找不到 OpenCVConfig.cmake），\n"
              "     find_package(OpenCV REQUIRED) 必然失败；\n"
              "  2. PATH 里的 `cc` 是一个 claude 启动脚本而不是编译器，cmake 探测编译器会报\n"
              "     \"unknown option '-o'\"，所以必须显式指定编译器。\n"
              "装好 OpenCV 之后用：\n"
              "  cmake -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \\\n"
              "        -DCMAKE_CXX_STANDARD=17 -S . -B build && cmake --build build\n"
              "Python 端只调用生产 .so，请先完成本地库构建。",
              file=sys.stderr)
        return -1

    print("Initializing native navigation engine...")
    try:
        engine = NavigationEngine(
            model_root=REPO / "models",
            map_path=REPO.parent / "app/src/main/assets/maps/big_map.png",
            cache_path=REPO / "build/sift_cache.bin",
            library_path=so_path,
        )
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        cap.release()
        return -1
    print("Native navigation engine initialized.")

    # ctypes 集成测试输出
    out_video_path = REPO / "test_output_ctypes.mp4"
    writer = cv2.VideoWriter(str(out_video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        print("Error: Could not open VideoWriter for saving output.", file=sys.stderr)
        engine.close()
        cap.release()
        return -1

    frame_idx = 0
    located_frames = 0
    first_fix_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        result = engine.process(frame)
        ptr_matched = result.pointer_detected
        ptr_angle = result.angle_deg
        loc_success = result.located
        loc_x = result.x
        loc_y = result.y
        loc_cost = result.locate_cost_ms

        if loc_success:
            located_frames += 1
            if first_fix_frame is None:
                first_fix_frame = frame_idx
                print(f"First fix on frame {frame_idx}: "
                      f"({loc_x:.1f}, {loc_y:.1f}) in {loc_cost}ms")

        # Print progress info every 10 frames
        if frame_idx % 10 == 0 or frame_idx == 1:
            print(f"Frame {frame_idx:03d} | CNN Pointer Match: {ptr_matched} Ang={ptr_angle:.2f}° "
                  f"| Locate Success: {loc_success} Pos=({loc_x:.1f}, {loc_y:.1f}) Cost={loc_cost}ms")

        # ----------------- Visualizations -----------------
        # 1. Draw pointer detection source area boundary on frame: [1072, 25, 128, 128]
        cv2.rectangle(frame, (1072, 25), (1072 + 128, 25 + 128), (0, 255, 0), 1)
        cv2.putText(frame, "Pointer Neural ROI (128x128)", (1060, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 2. Draw SIFT/SP search ROI boundary
        cv2.rectangle(frame, (MINIMAP_LEFT, MINIMAP_TOP), (MINIMAP_LEFT + MINIMAP_WIDTH, MINIMAP_TOP + MINIMAP_HEIGHT), (0, 165, 255), 1)
        cv2.putText(frame, "SIFT/SP Minimap ROI", (MINIMAP_LEFT, MINIMAP_TOP - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

        # 3. Draw HUD Radar Circle
        hud_cx, hud_cy, hud_r = 150, height - 150, 100
        cv2.circle(frame, (hud_cx, hud_cy), hud_r, (40, 40, 40), -1)
        cv2.circle(frame, (hud_cx, hud_cy), hud_r, (120, 120, 120), 2)

        # 4. Draw compass pointer angle line (using standards polar axis)
        if ptr_matched:
            rad = ptr_angle * np.pi / 180.0
            px = int(hud_cx + (hud_r - 20) * np.cos(rad))
            py = int(hud_cy - (hud_r - 20) * np.sin(rad)) # invert y for HUD rendering
            cv2.line(frame, (hud_cx, hud_cy), (px, py), (0, 255, 0), 3)
            cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"{ptr_angle:.1f} deg", (hud_cx - 30, hud_cy + hud_r + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 5. Text status HUD
        status_y = 50
        cv2.putText(frame, "MinimapTracker Status HUD", (50, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        
        status_y += 30
        cv2.putText(frame, f"CNN Pointer Detection: {'OK' if ptr_matched else 'LOST'}",
                    (50, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if ptr_matched else (0, 0, 255), 1)
        
        status_y += 25
        loc_color = (0, 255, 0) if loc_success else (0, 0, 255)
        cv2.putText(frame, f"SIFT/SP Localization: {'OK' if loc_success else 'LOST'}", 
                    (50, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, loc_color, 1)

        status_y += 25
        cv2.putText(frame, f"Computed Position: ({loc_x:.1f}, {loc_y:.1f})",
                    (50, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        status_y += 25
        cv2.putText(frame, f"Locate Time Cost: {loc_cost}ms",
                    (50, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        writer.write(frame)

    # Clean up
    cap.release()
    writer.release()
    engine.close()
    rate = located_frames / frame_idx * 100 if frame_idx else 0.0
    print(f"\nC++ pipeline complete: {located_frames}/{frame_idx} frames located ({rate:.1f}%), "
          f"first fix on frame {first_fix_frame}.")
    print(f"Output saved to: {out_video_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
