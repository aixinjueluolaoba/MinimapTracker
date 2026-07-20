#!/usr/bin/env python3
import cv2
import numpy as np
import ctypes
import os
import sys

# Define ROI parameters for Minimap Pointer Angle Detector
ROI_X = 1124
ROI_Y = 77
ROI_W = 23
ROI_H = 23
CENTER_X = 11.63
CENTER_Y = 11.80

# Minimap region in the 1280x720 frame (used by SIFT/SuperPoint hybrid tracker)
MINIMAP_LEFT = 1100
MINIMAP_TOP = 50
MINIMAP_WIDTH = 150
MINIMAP_HEIGHT = 150

def main():
    video_path = "/home/diana/fishing/video_20s.mp4"
    if len(sys.argv) > 1:
        video_path = sys.argv[1]

    print(f"Opening video file: {video_path}")
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
    so_path = "/home/diana/洛克导航/navigation_engine/build/libnavigation_engine.so"
    if not os.path.exists(so_path):
        print(f"Error: Shared library not found at {so_path}. Please compile the project first.", file=sys.stderr)
        return -1

    lib = ctypes.CDLL(so_path)

    # 1. Declare Pointer Angle Detector API
    lib.detect_pointer_angle_c.restype = ctypes.c_bool
    lib.detect_pointer_angle_c.argtypes = [
        ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float,
        ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float)
    ]

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

    # 3. Define paths for initialization
    map_path = "/home/diana/洛克导航/app/src/main/assets/maps/big_map.png".encode('utf-8')
    cache_dir = "/home/diana/洛克导航/navigation_engine/build"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "sift_cache.bin").encode('utf-8')
    model_dir = "/home/diana/洛克导航/app/src/main/assets/models/superpoint_superglue_ncnn".encode('utf-8')

    print("Initializing Hybrid SIFT+SuperPoint Tracker...")
    # Initialize tracker with ROI and matching params
    tracker = lib.player_tracker_create(
        map_path,
        cache_path,
        model_dir,
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
        return -1
    print("Tracker initialization successful.")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # ----------------- Pointer Angle Detection -----------------
        crop_x = min(ROI_X, width - ROI_W)
        crop_y = min(ROI_Y, height - ROI_H)
        roi = frame[crop_y:crop_y+ROI_H, crop_x:crop_x+ROI_W]

        # Convert ROI to 32-bit ARGB representation for C++ detector
        b = roi[:, :, 0].astype(np.int32)
        g = roi[:, :, 1].astype(np.int32)
        r = roi[:, :, 2].astype(np.int32)
        a = np.ones_like(b) * 255
        argb = (a << 24) | (r << 16) | (g << 8) | b
        pixels = argb.flatten()
        pixels_ptr = pixels.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

        ptr_matched = ctypes.c_bool(False)
        ptr_angle = ctypes.c_int(0)
        ptr_conf = ctypes.c_float(0.0)
        
        lib.detect_pointer_angle_c(
            pixels_ptr, ROI_W, ROI_H, CENTER_X, CENTER_Y,
            ctypes.byref(ptr_matched), ctypes.byref(ptr_angle), ctypes.byref(ptr_conf)
        )

        # ----------------- Player SIFT/SuperPoint Positioning -----------------
        # Convert full frame to 32-bit RGBA for tracker (A=255)
        rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        rgba_data = rgba_frame.astype(np.int32)
        packed_frame = (255 << 24) | (rgba_data[:, :, 0] << 16) | (rgba_data[:, :, 1] << 8) | rgba_data[:, :, 2]
        frame_pixels_flat = packed_frame.flatten()
        frame_ptr = frame_pixels_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

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
            print(f"Frame {frame_idx:03d} | Pointer Match: {ptr_matched.value} Ang={ptr_angle.value}° "
                  f"| Locate Success: {loc_success} Pos=({loc_x.value:.1f}, {loc_y.value:.1f}) Cost={loc_cost.value}ms")

        # ----------------- Visualizations -----------------
        # 1. Minimap and Pointer ROI Boxes
        cv2.rectangle(frame, (crop_x, crop_y), (crop_x + ROI_W, crop_y + ROI_H), (0, 255, 0), 1)
        cv2.rectangle(frame, (MINIMAP_LEFT, MINIMAP_TOP), (MINIMAP_LEFT + MINIMAP_WIDTH, MINIMAP_TOP + MINIMAP_HEIGHT), (0, 165, 255), 1)
        cv2.putText(frame, "Pointer ROI", (crop_x - 10, crop_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(frame, "SIFT/SP Minimap ROI", (MINIMAP_LEFT, MINIMAP_TOP - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

        # 2. Draw HUD HUD Radar Circle
        hud_cx, hud_cy, hud_r = 150, height - 150, 100
        cv2.circle(frame, (hud_cx, hud_cy), hud_r, (40, 40, 40), -1)
        cv2.circle(frame, (hud_cx, hud_cy), hud_r, (120, 120, 120), 2)

        # 3. Draw compass pointer angle line
        if ptr_matched.value:
            rad = ptr_angle.value * np.pi / 180.0
            px = int(hud_cx + (hud_r - 20) * np.cos(rad))
            py = int(hud_cy - (hud_r - 20) * np.sin(rad)) # standard coordinate system offset
            cv2.line(frame, (hud_cx, hud_cy), (px, py), (0, 255, 0), 3)
            cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"{ptr_angle.value} deg", (hud_cx - 30, hud_cy + hud_r + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 4. Text status HUD
        status_y = 50
        cv2.putText(frame, "MinimapTracker Status HUD", (50, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        
        status_y += 30
        cv2.putText(frame, f"Pointer Detection: {'OK' if ptr_matched.value else 'LOST'}", 
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
    lib.player_tracker_release(tracker)
    print(f"Video processing complete. Output saved to: {out_video_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
