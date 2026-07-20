#!/usr/bin/env python3
import cv2
import numpy as np
import ctypes
import os
import sys

# Define ROI parameters (aligned with C++ test_video.cpp)
ROI_X = 1124
ROI_Y = 77
ROI_W = 23
ROI_H = 23
CENTER_X = 11.63
CENTER_Y = 11.80

class NavInput(ctypes.Structure):
    _fields_ = [
        ("track_x", ctypes.c_float),
        ("track_y", ctypes.c_float),
        ("track_found", ctypes.c_bool),
        ("manual_required", ctypes.c_bool),
        ("pointer_has_match", ctypes.c_bool),
        ("pointer_angle_deg", ctypes.c_float),
        ("pointer_frozen", ctypes.c_bool),
        ("pointer_age_ms", ctypes.c_int64),
        ("now_ms", ctypes.c_int64),
    ]

class NavOutput(ctypes.Structure):
    _fields_ = [
        ("action", ctypes.c_int),
        ("status", ctypes.c_int),
        ("sleep_ms", ctypes.c_int64),
        ("hold_angle_deg", ctypes.c_float),
        ("speed_percent", ctypes.c_int),
        ("map_angle_deg", ctypes.c_float),
        ("current_angle_deg", ctypes.c_float),
        ("angle_delta_deg", ctypes.c_float),
        ("signed_delta_deg", ctypes.c_float),
        ("target_index", ctypes.c_int),
        ("target_x", ctypes.c_float),
        ("target_y", ctypes.c_float),
        ("using_cached_track", ctypes.c_bool),
        ("jump_requested", ctypes.c_bool),
    ]

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
    # Use mp4v or XVID codec to output test_output.mp4
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

    # Declare JNI-like C-export APIs
    lib.navigation_engine_create.restype = ctypes.c_void_p
    lib.navigation_engine_create.argtypes = []

    lib.navigation_engine_start.restype = ctypes.c_bool
    lib.navigation_engine_start.argtypes = [
        ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_float, ctypes.c_float
    ]

    lib.navigation_engine_step_struct.restype = ctypes.c_bool
    lib.navigation_engine_step_struct.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(NavInput), ctypes.POINTER(NavOutput)
    ]

    lib.navigation_engine_release.restype = None
    lib.navigation_engine_release.argtypes = [ctypes.c_void_p]

    lib.detect_pointer_angle_c.restype = ctypes.c_bool
    lib.detect_pointer_angle_c.argtypes = [
        ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float,
        ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float)
    ]

    # Initialize Engine
    engine = lib.navigation_engine_create()
    if not engine:
        print("Error: Failed to create navigation engine.", file=sys.stderr)
        return -1

    # Simulation state variables
    sim_x = 1200.0
    sim_y = 1000.0
    target_x = 1350.0
    target_y = 850.0
    max_speed_per_step = 2.5

    print(f"Initializing navigation target to: ({target_x}, {target_y})")
    started = lib.navigation_engine_start(engine, target_x, target_y, 0.0, 100, 8.0, 30.0)
    if not started:
        print("Error: Failed to start navigation engine.", file=sys.stderr)
        lib.navigation_engine_release(engine)
        return -1

    frame_idx = 0
    kStatusMoving = 2
    kStatusComplete = 3
    kStatusFailed = 4

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Crop Minimap Pointer ROI
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

        # Detect pointer angle using the C++ detector inside the SO
        has_match = ctypes.c_bool(False)
        angle_deg = ctypes.c_int(0)
        confidence = ctypes.c_float(0.0)
        
        lib.detect_pointer_angle_c(
            pixels_ptr, ROI_W, ROI_H, CENTER_X, CENTER_Y,
            ctypes.byref(has_match), ctypes.byref(angle_deg), ctypes.byref(confidence)
        )

        action = ctypes.c_int(0)
        status = ctypes.c_int(0)
        sleep_ms = ctypes.c_int64(0)
        hold_angle_deg = ctypes.c_float(0.0)
        speed_percent = ctypes.c_int(0)
        map_angle_deg = ctypes.c_float(0.0)
        current_angle_deg = ctypes.c_float(0.0)
        angle_delta_deg = ctypes.c_float(0.0)
        signed_delta_deg = ctypes.c_float(0.0)
        target_index = ctypes.c_int(0)
        out_target_x = ctypes.c_float(0.0)
        out_target_y = ctypes.c_float(0.0)
        using_cached_track = ctypes.c_bool(False)
        jump_requested = ctypes.c_bool(False)

        # Run navigation step if pointer matches
        if has_match.value:
            now_ms = frame_idx * 300
            nav_input = NavInput(
                track_x=float(sim_x),
                track_y=float(sim_y),
                track_found=True,
                manual_required=False,
                pointer_has_match=has_match.value,
                pointer_angle_deg=float(angle_deg.value),
                pointer_frozen=False,
                pointer_age_ms=50,
                now_ms=now_ms
            )
            nav_output = NavOutput()
            lib.navigation_engine_step_struct(engine, ctypes.byref(nav_input), ctypes.byref(nav_output))

            # Map struct output back to existing ctypes variables to keep visualization unchanged
            action.value = nav_output.action
            status.value = nav_output.status
            sleep_ms.value = nav_output.sleep_ms
            hold_angle_deg.value = nav_output.hold_angle_deg
            speed_percent.value = nav_output.speed_percent
            map_angle_deg.value = nav_output.map_angle_deg
            current_angle_deg.value = nav_output.current_angle_deg
            angle_delta_deg.value = nav_output.angle_delta_deg
            signed_delta_deg.value = nav_output.signed_delta_deg
            target_index.value = nav_output.target_index
            out_target_x.value = nav_output.target_x
            out_target_y.value = nav_output.target_y
            using_cached_track.value = nav_output.using_cached_track
            jump_requested.value = nav_output.jump_requested

            # Update simulated position based on control commands
            if status.value == kStatusMoving:
                speed = (speed_percent.value / 100.0) * max_speed_per_step
                angle_rad = hold_angle_deg.value * np.pi / 180.0
                sim_x += speed * np.cos(angle_rad)
                sim_y -= speed * np.sin(angle_rad)

        # --- Visualizations ---

        # 1. Pointer ROI Highlight
        cv2.rectangle(frame, (crop_x, crop_y), (crop_x + ROI_W, crop_y + ROI_H), (0, 255, 0), 1)
        cv2.putText(frame, "Minimap Pointer ROI", (crop_x - 120, crop_y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 2. Draw HUD HUD Radar Circle
        hud_cx, hud_cy, hud_r = 150, height - 150, 100
        cv2.circle(frame, (hud_cx, hud_cy), hud_r, (40, 40, 40), -1)
        cv2.circle(frame, (hud_cx, hud_cy), hud_r, (180, 180, 180), 2)

        # Draw Goal Point relative to virtual player on Radar
        rel_x = target_x - sim_x
        rel_y = target_y - sim_y
        dist = np.hypot(rel_x, rel_y)
        scale = 0.5

        target_draw_x = int(hud_cx + rel_x * scale)
        target_draw_y = int(hud_cy + rel_y * scale)

        draw_dist = np.hypot(target_draw_x - hud_cx, target_draw_y - hud_cy)
        if draw_dist > hud_r - 10:
            ratio = (hud_r - 10) / draw_dist
            target_draw_x = int(hud_cx + (target_draw_x - hud_cx) * ratio)
            target_draw_y = int(hud_cy + (target_draw_y - hud_cy) * ratio)

        cv2.circle(frame, (target_draw_x, target_draw_y), 6, (0, 165, 255), -1)
        cv2.circle(frame, (target_draw_x, target_draw_y), 7, (255, 255, 255), 1)
        cv2.putText(frame, "Goal", (target_draw_x + 8, target_draw_y + 4), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

        # Draw Player at the center
        cv2.circle(frame, (hud_cx, hud_cy), 5, (255, 0, 0), -1)

        # Draw Heading Direction
        if has_match.value:
            pointer_rad = angle_deg.value * np.pi / 180.0
            px = int(hud_cx + 30 * np.cos(pointer_rad))
            py = int(hud_cy - 30 * np.sin(pointer_rad))
            cv2.arrowedLine(frame, (hud_cx, hud_cy), (px, py), (0, 255, 255), 2)

        # Draw Joystick Control Direction
        if has_match.value and status.value == kStatusMoving:
            stick_rad = hold_angle_deg.value * np.pi / 180.0
            sx = int(hud_cx + 60 * (speed_percent.value / 100.0) * np.cos(stick_rad))
            sy = int(hud_cy - 60 * (speed_percent.value / 100.0) * np.sin(stick_rad))
            cv2.arrowedLine(frame, (hud_cx, hud_cy), (sx, sy), (0, 0, 255), 3)

        # HUD text info
        cv2.rectangle(frame, (10, 10), (380, 180), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (380, 180), (180, 180, 180), 1)

        cv2.putText(frame, "NAVIGATION ENGINE OFFLINE TEST (PYTHON)", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        pos_str = f"Pos: ({sim_x:.1f}, {sim_y:.1f})"
        tgt_str = f"Goal: ({target_x:.1f}, {target_y:.1f}) Dist: {dist:.1f}"
        pointer_str = f"Heading Det: {angle_deg.value} deg" if has_match.value else "Heading Det: None"
        
        status_str = "IDLE"
        if status.value == kStatusMoving:
            status_str = "MOVING"
        elif status.value == kStatusComplete:
            status_str = "ARRIVED (SUCCESS)"
        elif status.value == kStatusFailed:
            status_str = "FAILED"
        engine_status = f"Engine Status: {status_str}"
        
        ctrl_str = f"Joystick Output: Ang={int(hold_angle_deg.value)} Speed={speed_percent.value}%"

        cv2.putText(frame, pos_str, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, tgt_str, (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, pointer_str, (20, 101), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, ctrl_str, (20, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1, cv2.LINE_AA)
        
        text_color = (0, 255, 0) if status.value == kStatusComplete else (0, 165, 255)
        cv2.putText(frame, engine_status, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

        if jump_requested.value:
            cv2.circle(frame, (hud_cx, hud_cy), hud_r + 20, (0, 255, 255), 3)
            cv2.putText(frame, "JUMP!", (hud_cx - 20, hud_cy - hud_r - 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        writer.write(frame)

        if status.value == kStatusComplete:
            print("SUCCESS: Simulated agent arrived at target destination!")
            break

    print(f"Video processing complete. Output saved to: {out_video_path}")
    cap.release()
    writer.release()
    lib.navigation_engine_release(engine)

if __name__ == "__main__":
    main()
