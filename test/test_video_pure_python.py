#!/usr/bin/env python3
"""纯 Python 的定位 + 测角离线仿真。

这是 C++ 定位路径 (navigation_engine/src/map_sift_tracker.cpp) 的参照实现：
小地图 ROI、CLAHE 预处理、搜索半径、匹配参数全部取自生产值
(app/src/main/java/com/example/myapplication/MapSiftTracker.java)。

与生产 C++ 的已知差异，不要当成等价物：
  * C++ 在首次锁定后会切到 SuperPoint 局部深度特征追踪；本脚本全程使用 SIFT，
    只是把搜索范围收缩到局部窗口。这里没有 SuperPoint，因此耗时数字不可与 C++ 对比。
  * C++ 的全局搜索有 2 秒节流；本脚本不节流，丢失即重配准。

用法：
    python3 test/test_video_pure_python.py [视频路径] [--no-play]
"""
import os
import subprocess
import sys
import time

import cv2
import ncnn
import numpy as np

# 小地图 ROI —— 与 MapSiftTracker.java:8-11 一致
MINIMAP_LEFT = 1072
MINIMAP_TOP = 25
MINIMAP_WIDTH = 127
MINIMAP_HEIGHT = 127

# 指针 CNN 的 ROI 是另一套常量，见 pointer_angle_detector.cpp:48-54
MM_X, MM_Y, MM_S = 1072, 25, 128
PTR_CX, PTR_CY = 63, 63

# 跟踪参数 —— 与 MapSiftTracker.java:12-17 一致
BASE_SEARCH_RADIUS = 150     # 半径；搜索窗口因此是 300x300
MAX_LOST_FRAMES = 4
CLAHE_LIMIT = 3.0
MATCH_RATIO = 0.9
MIN_MATCH_COUNT = 5
RANSAC_THRESHOLD = 8.0

# 匹配质量门禁 —— 取自 map_sift_tracker.cpp:63-65 的 kMinGlobal* 三个常量。
# 注意 C++ 里那两套阈值是按特征类型分的，不是按搜索范围分的：
#   kMinGlobal* (30/10/0.10) 只被 track_minimap_sift_global() 使用   -> SIFT 的阈值
#   kMinLocal*  (20/15/0.45) 只被 superpoint/orb 的局部路径使用      -> 深度特征的阈值
# 本脚本全程是 SIFT（局部模式只是缩小了搜索范围，特征类型没变），所以两种模式都用 SIFT 的阈值。
# 套用 kMinLocal* 会把定位成功率从 98% 压到 30%，那是拿 SuperPoint 的标准去卡 SIFT。
MIN_GOOD_MATCHES, MIN_INLIERS, MIN_INLIER_RATIO = 30, 10, 0.10

REPO = "/home/diana/洛克导航"
MAP_PATH = f"{REPO}/app/src/main/assets/maps/big_map.png"
MODEL_DIR = f"{REPO}/navigation_engine/models/pointer_model"
OUT_VIDEO = f"{REPO}/navigation_engine/test_output.mp4"
# 与 C++ 的 sift_cache.bin 格式不同，故用独立文件名，避免互相覆盖
FEATURE_CACHE = f"{REPO}/navigation_engine/build/sift_cache_py.npz"


def preprocess_gray(source_bgr, clahe):
    """对应 C++ preprocess_gray()：BGR2GRAY 后套 CLAHE。"""
    gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY)
    return clahe.apply(gray)


def load_map_features(clahe):
    """整张大地图的 SIFT 特征，带磁盘缓存。

    对应 C++ 构造函数里的 full_map_keypoints_ / full_map_descriptors_ 与 sift_cache.bin。
    下游只用到 keypoint 的坐标，所以缓存里只存 (N,2) 的 pt 和 (N,128) 的描述子。
    """
    if os.path.exists(FEATURE_CACHE):
        t0 = time.time()
        data = np.load(FEATURE_CACHE)
        pts, des = data["pts"], data["des"]
        print(f"Loaded map SIFT cache: {len(pts)} features in {time.time() - t0:.2f}s "
              f"({FEATURE_CACHE})")
        return pts, des

    print("No feature cache; extracting full-map SIFT (one-time cost)...")
    t0 = time.time()
    big_map_gray = cv2.imread(MAP_PATH, cv2.IMREAD_GRAYSCALE)
    if big_map_gray is None:
        raise RuntimeError(f"Could not read big map from {MAP_PATH}")
    keypoints, des = cv2.SIFT_create().detectAndCompute(clahe.apply(big_map_gray), None)
    pts = np.float32([kp.pt for kp in keypoints])
    print(f"Extracted {len(pts)} map features in {time.time() - t0:.2f}s")

    os.makedirs(os.path.dirname(FEATURE_CACHE), exist_ok=True)
    np.savez(FEATURE_CACHE, pts=pts, des=des)
    print(f"Saved cache to {FEATURE_CACHE}")
    return pts, des


def ratio_filter(matches):
    """对应 C++ 第 526-536 行的 ratio test。

    近邻不足 2 个时必须跳过，否则解包会抛 ValueError（C++: if (pair.size() != 2) continue）。
    """
    good = []
    for pair in matches:
        if len(pair) != 2:
            continue
        best, second = pair
        if best.distance < MATCH_RATIO * second.distance:
            good.append(best)
    return good


def solve_center(mini_keypoints, good, dst_points, map_shape):
    """RANSAC 单应性 + 质量门禁 + 小地图中心点投影，对应 C++ finish_matched_points()。

    没有这道门禁的话，全局搜索会接受明显的误匹配（实测首帧会偏出约 900px）。
    """
    src = np.float32([mini_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = dst_points.reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESHOLD)
    if homography is None:
        return None

    inliers = int(mask.sum()) if mask is not None else 0
    ratio = inliers / len(good) if good else 0.0
    if len(good) < MIN_GOOD_MATCHES or inliers < MIN_INLIERS or ratio < MIN_INLIER_RATIO:
        return None

    center = np.float32([[[MINIMAP_WIDTH / 2.0, MINIMAP_HEIGHT / 2.0]]])
    out = cv2.perspectiveTransform(center, homography)
    x, y = float(out[0][0][0]), float(out[0][0][1])
    if not (0.0 <= x < map_shape[1] and 0.0 <= y < map_shape[0]):
        return None
    return x, y


class SiftLocalizer:
    """未定位（或丢失过久）时全图配准，已定位时在半径 150 的窗口内匹配。

    对应 C++ compute_search_rect() + build_search_feature_subset() 的两种模式。
    """

    def __init__(self, map_pts, map_des, map_shape):
        self.map_pts = map_pts
        self.map_des = map_des
        self.map_shape = map_shape
        self.has_position = False
        self.last_x = 0.0
        self.last_y = 0.0
        self.lost_frames = 0
        self.global_relocalizations = 0

        index_params = dict(algorithm=1, trees=5)   # FLANN_INDEX_KDTREE
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        # 全图匹配器只训练一次；局部子集每帧变化，用一次性的 matcher
        self.global_flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.global_flann.add([map_des])
        self.global_flann.train()

    def _subset(self):
        radius = BASE_SEARCH_RADIUS
        inside = ((np.abs(self.map_pts[:, 0] - self.last_x) <= radius)
                  & (np.abs(self.map_pts[:, 1] - self.last_y) <= radius))
        return np.nonzero(inside)[0]

    def locate(self, mini_keypoints, mini_des):
        """返回 (x, y, mode)；失败时返回 (None, None, mode)。"""
        if mini_des is None or len(mini_des) < 2:
            return self._fail("skip")

        use_global = not self.has_position or self.lost_frames > MAX_LOST_FRAMES
        if use_global:
            matches = self.global_flann.knnMatch(mini_des, k=2)
            good = ratio_filter(matches)
            if len(good) < MIN_MATCH_COUNT:
                return self._fail("global")
            dst = np.float32([self.map_pts[m.trainIdx] for m in good])
        else:
            indices = self._subset()
            if len(indices) < 2:
                return self._fail("local")
            matches = self.flann.knnMatch(mini_des, self.map_des[indices], k=2)
            good = ratio_filter(matches)
            if len(good) < MIN_MATCH_COUNT:
                return self._fail("local")
            dst = np.float32([self.map_pts[indices[m.trainIdx]] for m in good])

        mode = "global" if use_global else "local"
        solved = solve_center(mini_keypoints, good, dst, self.map_shape)
        if solved is None:
            return self._fail(mode)

        if use_global and self.has_position:
            self.global_relocalizations += 1
        self.last_x, self.last_y = solved
        self.has_position = True
        self.lost_frames = 0
        return solved[0], solved[1], mode

    def _fail(self, mode):
        self.lost_frames += 1
        return None, None, mode


def detect_pointer_angle(net, frame):
    """32x32 patch 单次前向，输出 [sin, cos] 后 atan2 解角。与 C++ detect_patch() 等价。

    连 patch 一起返回，HUD 里回显它，好让人直接对照"网络看到了什么"和"它算出了什么"。
    """
    minimap = frame[MM_Y:MM_Y + MM_S, MM_X:MM_X + MM_S]
    patch = minimap[PTR_CY - 16:PTR_CY + 16, PTR_CX - 16:PTR_CX + 16].copy()

    # ascontiguousarray 不能省。astype 默认 order='K' 会保留输入的内存布局，
    # 所以 transpose 出来的视图转完在内存里仍是 HWC（交错 BGR），而 ncnn.Mat
    # 按 C 连续读裸缓冲，会把交错数据当成 planar CHW —— 网络收到的就是乱码。
    # C++ 侧 from_pixels(..., PIXEL_BGR, ...) 产出的是 planar，这里必须对齐。
    patch_float = np.ascontiguousarray(patch.transpose(2, 0, 1)).astype(np.float32) / 255.0

    extractor = net.create_extractor()
    extractor.input("in0", ncnn.Mat(patch_float))
    _, out = extractor.extract("out0")
    s, c = np.array(out).reshape(-1)[:2]
    return float((np.rad2deg(np.arctan2(s, c)) + 360) % 360), patch


def dim_panel(frame, x, y, w, h, alpha=0.6):
    """在游戏画面上压一层半透明暗底，否则 HUD 文字盖在花哨的游戏 UI 上根本读不清。"""
    x, y = max(0, x), max(0, y)
    roi = frame[y:y + h, x:x + w]
    if roi.size:
        cv2.addWeighted(np.zeros_like(roi), alpha, roi, 1 - alpha, 0, roi)


def draw_pointer_inset(frame, patch, angle):
    """把 CNN 的真实输入（32x32 patch）放大回显，并叠上解算出的方向箭头。

    这样一眼就能核对角度是否和 patch 里的指针一致，不用靠"看起来没问题"。
    """
    size, x, y = 128, 1090, 410
    inset = cv2.resize(patch, (size, size), interpolation=cv2.INTER_NEAREST)
    cx = cy = size // 2
    rad = angle * np.pi / 180.0
    tip = (int(cx + (size // 2 - 12) * np.cos(rad)), int(cy - (size // 2 - 12) * np.sin(rad)))
    cv2.arrowedLine(inset, (cx, cy), tip, (0, 255, 255), 2, tipLength=0.25)
    frame[y:y + size, x:x + size] = inset
    cv2.rectangle(frame, (x, y), (x + size, y + size), (0, 255, 255), 2)
    cv2.putText(frame, "CNN input 32x32", (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)


def draw_hud(frame, big_map_gray, loc_x, loc_y, located, mode, angle, cost_ms, height):
    """状态 HUD：右侧贴大地图局部、左下角雷达盘、左上角文字。

    这里会画矩形和圆作为 HUD 边框，原始小地图区域不受影响（ROI 已在此之前取完）。
    """
    hud_map_x, hud_map_y, hud_map_s = 1090, 240, 150
    half = hud_map_s // 2

    patch_ok = False
    if located:
        lx, ly = int(loc_x), int(loc_y)
        top, bottom = max(0, ly - half), min(big_map_gray.shape[0], ly + half)
        left, right = max(0, lx - half), min(big_map_gray.shape[1], lx + half)
        map_patch = big_map_gray[top:bottom, left:right]
        if map_patch.shape[:2] == (hud_map_s, hud_map_s):
            patch_bgr = cv2.cvtColor(map_patch, cv2.COLOR_GRAY2BGR)
            cv2.drawMarker(patch_bgr, (half, half), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            frame[hud_map_y:hud_map_y + hud_map_s, hud_map_x:hud_map_x + hud_map_s] = patch_bgr
            cv2.rectangle(frame, (hud_map_x, hud_map_y),
                          (hud_map_x + hud_map_s, hud_map_y + hud_map_s), (0, 255, 0), 2)
            cv2.putText(frame, f"BigMap {mode} ({lx},{ly})", (hud_map_x, hud_map_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            patch_ok = True

    if not patch_ok:
        placeholder = np.zeros((hud_map_s, hud_map_s, 3), dtype=np.uint8)
        cv2.line(placeholder, (0, 0), (hud_map_s, hud_map_s), (0, 0, 255), 2)
        cv2.line(placeholder, (0, hud_map_s), (hud_map_s, 0), (0, 0, 255), 2)
        cv2.putText(placeholder, "LOC LOST", (28, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        frame[hud_map_y:hud_map_y + hud_map_s, hud_map_x:hud_map_x + hud_map_s] = placeholder
        cv2.rectangle(frame, (hud_map_x, hud_map_y),
                      (hud_map_x + hud_map_s, hud_map_y + hud_map_s), (0, 0, 255), 2)
        cv2.putText(frame, "BigMap (Lost)", (hud_map_x, hud_map_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    # 雷达盘：标准极坐标，逆时针
    hud_cx, hud_cy, hud_r = 150, height - 150, 100
    cv2.circle(frame, (hud_cx, hud_cy), hud_r, (40, 40, 40), -1)
    cv2.circle(frame, (hud_cx, hud_cy), hud_r, (120, 120, 120), 2)
    rad = angle * np.pi / 180.0
    px = int(hud_cx + (hud_r - 20) * np.cos(rad))
    py = int(hud_cy - (hud_r - 20) * np.sin(rad))   # HUD 渲染需要翻转 y
    cv2.line(frame, (hud_cx, hud_cy), (px, py), (0, 255, 0), 3)
    cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)
    cv2.putText(frame, f"{angle:.1f} deg", (hud_cx - 30, hud_cy + hud_r + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    lines = [
        ("MinimapTracker Status HUD", (0, 200, 255)),
        (f"CNN Pointer Angle: {angle:.1f} deg", (0, 255, 0)),
        (f"SIFT Localization [{mode}]: {'OK' if located else 'LOST'}",
         (0, 255, 0) if located else (0, 0, 255)),
        (f"Computed Position: ({loc_x:.1f}, {loc_y:.1f})", (255, 255, 255)),
        (f"Time Cost: {cost_ms}ms", (255, 255, 255)),
    ]
    dim_panel(frame, 40, 28, 350, 27 * len(lines) + 12)
    y = 50
    for text, color in lines:
        cv2.putText(frame, text, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += 27


def main():
    args = [a for a in sys.argv[1:] if a != "--no-play"]
    play = "--no-play" not in sys.argv[1:]
    video_path = args[0] if args else \
        "/home/diana/screencap/file/cv_tools/record/rec_20260720_014710_032096.mp4"

    print(f"Opening test video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.", file=sys.stderr)
        return -1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Specs: {width}x{height}, {fps} FPS, {total_frames} frames.")

    writer = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        print("Error: Could not open VideoWriter.", file=sys.stderr)
        return -1

    clahe = cv2.createCLAHE(CLAHE_LIMIT, (8, 8))
    map_pts, map_des = load_map_features(clahe)
    big_map_gray = cv2.imread(MAP_PATH, cv2.IMREAD_GRAYSCALE)
    localizer = SiftLocalizer(map_pts, map_des, big_map_gray.shape)
    sift = cv2.SIFT_create()

    print("Loading neural NCNN pointer detector...")
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = 2
    net.load_param(os.path.join(MODEL_DIR, "pointer_angle_cnn_v2_bgr_pool2.ncnn.param"))
    net.load_model(os.path.join(MODEL_DIR, "pointer_angle_cnn_v2_bgr_pool2.ncnn.bin"))
    print("Neural pointer detector loaded successfully.")

    frame_idx = 0
    located_frames = 0
    first_fix_ms = None
    last_x, last_y = 0.0, 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        start_time = time.time()

        angle, patch = detect_pointer_angle(net, frame)

        mini_bgr = frame[MINIMAP_TOP:MINIMAP_TOP + MINIMAP_HEIGHT,
                         MINIMAP_LEFT:MINIMAP_LEFT + MINIMAP_WIDTH]
        mini_keypoints, mini_des = sift.detectAndCompute(preprocess_gray(mini_bgr, clahe), None)
        loc_x, loc_y, mode = localizer.locate(mini_keypoints, mini_des)

        located = loc_x is not None
        if located:
            last_x, last_y = loc_x, loc_y
            located_frames += 1
        cost_ms = int((time.time() - start_time) * 1000)
        if located and first_fix_ms is None:
            first_fix_ms = cost_ms
            print(f"First fix on frame {frame_idx} via {mode} search: "
                  f"({loc_x:.1f}, {loc_y:.1f}) in {cost_ms}ms")

        if frame_idx % 10 == 0 or frame_idx == 1:
            pos = f"({loc_x:.1f}, {loc_y:.1f})" if located else "(lost)"
            print(f"Frame {frame_idx:03d}/{total_frames} | CNN Pointer Angle: {angle:6.2f}deg "
                  f"| Localization[{mode}]: {located} Pos={pos} Cost={cost_ms}ms")

        draw_hud(frame, big_map_gray, last_x, last_y, located, mode, angle, cost_ms, height)
        draw_pointer_inset(frame, patch, angle)
        writer.write(frame)

    cap.release()
    writer.release()

    rate = located_frames / frame_idx * 100 if frame_idx else 0.0
    print(f"\nSimulation complete: {located_frames}/{frame_idx} frames located ({rate:.1f}%), "
          f"{localizer.global_relocalizations} global re-localizations after loss.")
    print(f"Output saved to: {OUT_VIDEO}")

    if play:
        print("Launching mpv to play the output video locally...")
        subprocess.run(["mpv", OUT_VIDEO], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
