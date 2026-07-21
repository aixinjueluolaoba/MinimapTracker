#!/usr/bin/env python3
"""生成 MinimapTracker 架构图的 .excalidraw 场景（中文）。

图中所有参数与数字都取自已验证的实现（navigation_engine/src/），不含设计意图。

用法：
    python3 images/generate_architecture.py images/minimap_tracker_architecture.excalidraw
再用 render_excalidraw.js 出 SVG，最后跑 embed_cjk_font.py 把中文字体子集嵌进去。
"""
import json
import sys

E = []
_seed = [1000]


def nid(p):
    _seed[0] += 1
    return f"{p}{_seed[0]}"


def is_cjk(ch):
    o = ord(ch)
    return (0x3000 <= o <= 0x9FFF) or (0xFF00 <= o <= 0xFFEF)


def text_width(s, size):
    """CJK 为全角，ASCII 约 0.53 字宽。中英混排必须分开算，否则居中会偏。"""
    return sum(size * (1.0 if is_cjk(c) else 0.53) for c in s)


def base(t, x, y, w, h, **kw):
    _seed[0] += 1
    d = dict(id=nid(t[:2]), type=t, x=x, y=y, width=w, height=h, angle=0,
             strokeColor="#1e1e1e", backgroundColor="transparent", fillStyle="solid",
             strokeWidth=2, strokeStyle="solid", roughness=1, opacity=100,
             seed=_seed[0], version=1, versionNonce=_seed[0], isDeleted=False,
             groupIds=[], frameId=None, roundness={"type": 3}, boundElements=[],
             updated=1, link=None, locked=False)
    d.update(kw)
    return d


def text(s, x, y, size=16, color="#1e1e1e"):
    w = text_width(s, size)
    e = base("text", x - w / 2, y, w, size * 1.25,
             roundness=None, strokeColor=color, strokeWidth=1)
    e.update(text=s, fontSize=size, fontFamily=1, textAlign="center",
             verticalAlign="top", containerId=None, originalText=s, lineHeight=1.25)
    E.append(e)
    return e


def box(x, y, w, h, lines, bg="#ffffff", size=16, title_size=None):
    E.append(base("rectangle", x, y, w, h, backgroundColor=bg))
    cx = x + w / 2
    ts = title_size or size
    total = ts * 1.25 + (len(lines) - 1) * size * 1.4
    ty = y + (h - total) / 2
    text(lines[0], cx, ty, ts)
    ty += ts * 1.25 + 6
    for ln in lines[1:]:
        text(ln, cx, ty, size, color="#495057")
        ty += size * 1.4


def arrow(x1, y1, x2, y2, label=None, dashed=False):
    e = base("arrow", x1, y1, x2 - x1, y2 - y1, roundness={"type": 2},
             strokeStyle="dashed" if dashed else "solid")
    e.update(points=[[0, 0], [x2 - x1, y2 - y1]], lastCommittedPoint=None,
             startBinding=None, endBinding=None,
             startArrowhead=None, endArrowhead="arrow")
    E.append(e)
    if label:
        text(label, (x1 + x2) / 2, (y1 + y2) / 2 - 24, 13, color="#1971c2")


BLUE, GREEN, YELLOW, PURPLE, GRAY = "#a5d8ff", "#b2f2bb", "#ffec99", "#d0bfff", "#e9ecef"

text("MinimapTracker 感知流水线", 700, 22, 27)
text("指针朝向检测  +  玩家定位     （298 帧真实录像实测）", 700, 64, 14, color="#868e96")

# ---- 输入与分叉 ----
box(500, 102, 400, 66, ["1280x720 帧", "ARGB packed int32"], BLUE)
text("指针朝向", 300, 118, 18, color="#e8590c")
text("玩家定位", 1100, 118, 18, color="#e8590c")

arrow(700, 168, 700, 190)
E.append(base("line", 350, 190, 700, 0, roundness=None,
              points=[[0, 0], [700, 0]], lastCommittedPoint=None,
              startBinding=None, endBinding=None,
              startArrowhead=None, endArrowhead=None))
arrow(350, 190, 350, 216)
arrow(1050, 190, 1050, 216)

# ---- 左：指针朝向 ----
box(160, 216, 380, 62, ["小地图 ROI  [1072, 25, 128x128]"], GRAY, size=15)
arrow(350, 278, 350, 308)
box(160, 314, 380, 70, ["中心 32x32 patch @ (47,47)", "BGR / 255，planar CHW 连续内存"], GRAY, size=13, title_size=15)
arrow(350, 384, 350, 414)
box(160, 420, 380, 74, ["指针 CNN（NCNN）", "in0  →  out0 = [sin, cos]"], YELLOW, size=14, title_size=16)
arrow(350, 494, 350, 524)
box(160, 530, 380, 62, ["angle = atan2(sin, cos)"], GRAY, size=15)
arrow(350, 592, 350, 622)
box(160, 628, 380, 62, ["指针角度  0 - 360 度"], GREEN)
text("网络无拒识头，输出向量已 L2 归一化，", 350, 708, 13, color="#868e96")
text("模长恒为 1.0，不携带任何置信度信息", 350, 728, 13, color="#868e96")

# ---- 右：玩家定位 ----
box(860, 216, 380, 62, ["小地图 ROI  [1072, 25, 127x127]"], GRAY, size=15)
arrow(1050, 278, 1050, 308)
box(860, 314, 380, 70, ["灰度  +  CLAHE(3.0, 8x8)"], GRAY, size=15)
arrow(1050, 384, 1050, 414)

E.append(base("rectangle", 810, 414, 480, 248, backgroundColor="#f8f9fa",
              strokeStyle="dashed", strokeWidth=1))
text("全自动状态机  track_minimap()", 1050, 424, 13, color="#868e96")
box(832, 454, 192, 102, ["全局 SIFT", "139,709 缓存特征", "门禁 30/10/0.10"], PURPLE, size=12, title_size=15)
box(1080, 454, 192, 102, ["SuperPoint 局部", "现场提取 300x300", "门禁 20/15/0.45"], PURPLE, size=12, title_size=15)
arrow(1024, 480, 1080, 480, "锁定")
arrow(1080, 532, 1024, 532, "丢失 2 秒", dashed=True)
text("未定位时从全局开始，全局搜索每 2 秒最多一次", 1050, 592, 13, color="#868e96")
text("搜索窗口 = 上一帧位置 ±150 px", 1050, 614, 13, color="#868e96")
text("局部特征每帧从地图裁剪现场提取", 1050, 636, 13, color="#868e96")

arrow(1050, 662, 1050, 692)
box(860, 698, 380, 74, ["RANSAC 单应性求解", "把小地图中心投影到大地图"], YELLOW, size=14, title_size=16)
arrow(1050, 772, 1050, 802)
box(860, 808, 380, 62, ["玩家坐标 (X, Y) 于 6144x5888 地图"], GREEN, size=15)

# ---- 页脚：实测数据 ----
E.append(base("rectangle", 160, 906, 1080, 96, backgroundColor="#fff9db",
              strokeStyle="dashed", strokeWidth=1))
text("经 test/test_video.py 端到端实测（ctypes 调用 libfishing_native.so）", 700, 922, 15)
text("295 / 298 帧定位成功（99.0%）　首次锁定第 4 帧　局部匹配 18-25 ms　首次全局配准 2.4 s",
     700, 952, 13, color="#495057")
text("指针角度与原颜色 + BFS 解算器交叉验证：中位差 8.1 度", 700, 976, 13, color="#495057")

doc = {"type": "excalidraw", "version": 2, "source": "navigation_engine",
       "elements": E, "appState": {"viewBackgroundColor": "#ffffff", "gridSize": None},
       "files": {}}
out = sys.argv[1] if len(sys.argv) > 1 else "minimap_tracker_architecture.excalidraw"
with open(out, "w", encoding="utf-8") as f:
    json.dump(doc, f, ensure_ascii=False, indent=1)
print(f"wrote {out}  ({len(E)} elements)")
