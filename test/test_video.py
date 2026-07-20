#!/usr/bin/env python3
import cv2
import numpy as np
import ctypes
import os
import sys

# Minimap region in the 1280x720 frame (used by SIFT/SuperPoint hybrid tracker)
MINIMAP_LEFT = 1100
MINIMAP_TOP = 50
MINIMAP_WIDTH = 150
MINIMAP_HEIGHT = 150

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

    out_video_path = "/home/diana/洛克导航/navigation_engine/test_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        print("Error: Could not open VideoWriter for saving output.", file=sys.stderr)
        return -1

    # Load Navigation Engine SO Library
    so_path = "/home/diana/洛克导航/navigation_engine/build/libfishing_native.so"
    if not os.path.exists(so_path):
        print(f"Error: Shared library not found at {so_path}. Please compile the project first.", file=sys.stderr)
        return -1

    lib = ctypes.CDLL(so_path)

    # 1. Declare Neural Pointer Detector API
    lib.pointer_detector_create.restype = ctypes.c_void_p
    lib.pointer_detector_create.argtypes = [ctypes.c_char_p]

    lib.pointer_detector_detect.restype = ctypes.c_bool
    lib.pointer_detector_detect.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)
    ]

    lib.pointer_detector_release.restype = None
    lib.pointer_detector_release.argtypes = [ctypes.c_void_p]

    # 2. Declare Hybrid Map Tracker API (SIFT + SuperPoint)
    lib.player_tracker_create.restype = ctypes.c_void_p
    lib.player_tracker_create.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_float, ctypes.c_int, ctypes.c_double
    ]

    lib.player_tracker_locate.restype = ctypes.c_bool
    lib.player_tracker_locate.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int)
    ]

    lib.player_tracker_release.restype = None
    lib.player_tracker_release.argtypes = [ctypes.c_void_p]

    # 3. Define paths inside the sub-repository
    pointer_model_dir = "/home/diana/洛克导航/navigation_engine/models/pointer_model".encode('utf-8')
    superpoint_model_dir = "/home/diana/洛克导航/navigation_engine/models/superpoint_model".encode('utf-8')
    map_path = "/home/diana/洛克导航/app/src/main/assets/maps/big_map.png".encode('utf-8')
    
    cache_dir = "/home/diana/洛克导航/navigation_engine/build"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "sift_cache.bin").encode('utf-8')

    # 4. Initialize Neural Pointer Detector
    print("Initializing Neural CNN Pointer Detector...")
    pointer_detector = lib.pointer_detector_create(pointer_model_dir)
    if not pointer_detector:
        print("Error: Failed to initialize neural pointer detector.", file=sys.stderr)
        return -1
    print("Pointer detector initialization successful.")

    # 5. Initialize Hybrid SIFT+SuperPoint Tracker
    print("Initializing Hybrid SIFT+SuperPoint Tracker...")
    tracker = lib.player_tracker_create(
        map_path,
        cache_path,
        superpoint_model_dir,
        MINIMAP_LEFT, MINIMAP_TOP, MINIMAP_WIDTH, MINIMAP_HEIGHT,
        150,     # base_search_radius
        10,      # max_lost_frames
        40.0,    # clahe_limit
        0.8,     # match_ratio
        4,       # min_match_count
        8.0      # ransac_threshold
    )

    if not tracker:
        print("Error: Failed to initialize hybrid player tracker.", file=sys.stderr)
        lib.pointer_detector_release(pointer_detector)
        return -1
    print("Tracker initialization successful.")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Convert full frame to 32-bit RGBA (A=255) for NCNN CNN and SIFT/SP trackers
        rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        rgba_data = rgba_frame.astype(np.int32)
        packed_frame = (255 << 24) | (rgba_data[:, :, 0] << 16) | (rgba_data[:, :, 1] << 8) | rgba_data[:, :, 2]
        frame_pixels_flat = packed_frame.flatten()
        frame_ptr = frame_pixels_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

        # ----------------- Neural Pointer Detection -----------------
        ptr_matched = ctypes.c_bool(False)
        ptr_angle = ctypes.c_float(0.0)
        ptr_conf = ctypes.c_float(0.0)
        
        lib.pointer_detector_detect(
            pointer_detector, frame_ptr, width, height,
            ctypes.byref(ptr_matched), ctypes.byref(ptr_angle), ctypes.byref(ptr_conf)
        )

        # ----------------- Player SIFT/SuperPoint Positioning -----------------
        loc_x = ctypes.c_float(0.0)
        loc_y = ctypes.c_float(0.0)
        loc_conf = ctypes.c_float(0.0)
        loc_cost = ctypes.c_int(0)

        loc_success = lib.player_tracker_locate(
            tracker, frame_ptr, width, height,
            ctypes.byref(loc_x), ctypes.byref(loc_y), ctypes.byref(loc_conf), ctypes.byref(loc_cost)
        )

        # Print progress info every 10 frames
        if frame_idx % 10 == 0 or frame_idx == 1:
            print(f"Frame {frame_idx:03d} | CNN Pointer Match: {ptr_matched.value} Ang={ptr_angle.value:.2f}° "
                  f"| Locate Success: {loc_success} Pos=({loc_x.value:.1f}, {loc_y.value:.1f}) Cost={loc_cost.value}ms")

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
        if ptr_matched.value:
            rad = ptr_angle.value * np.pi / 180.0
            px = int(hud_cx + (hud_r - 20) * np.cos(rad))
            py = int(hud_cy - (hud_r - 20) * np.sin(rad)) # invert y for HUD rendering
            cv2.line(frame, (hud_cx, hud_cy), (px, py), (0, 255, 0), 3)
            cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"{ptr_angle.value:.1f} deg", (hud_cx - 30, hud_cy + hud_r + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 5. Text status HUD
        status_y = 50
        cv2.putText(frame, "MinimapTracker Status HUD", (50, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        
        status_y += 30
        cv2.putText(frame, f"CNN Pointer Detection: {'OK' if ptr_matched.value else 'LOST'}", 
                    (50, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if ptr_matched.value else (0, 0, 255), 1)
        
        status_y += 25
        loc_color = (0, 255, 0) if loc_success else (0, 0, 255)
        cv2.putText(frame, f"SIFT/SP Localization: {'OK' if loc_success else 'LOST'}", 
                    (50, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, loc_color, 1)

        status_y += 25
        cv2.putText(frame, f"Computed Position: ({loc_x.value:.1f}, {loc_y.value:.1f})", 
                    (50, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        status_y += 25
        cv2.putText(frame, f"Locate Time Cost: {loc_cost.value}ms", 
                    (50, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        writer.write(frame)

    # Clean up
    cap.release()
    writer.release()
    lib.pointer_detector_release(pointer_detector)
    lib.player_tracker_release(tracker)
    print(f"Video processing complete. Output saved to: {out_video_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
