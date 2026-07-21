#!/usr/bin/env python3
"""把中文字体的子集嵌进 render_excalidraw.js 产出的 SVG。

为什么需要这一步：excalidraw 内嵌的 Virgil 是拉丁手绘字体，没有 CJK 字形。
不处理的话中文会回退到查看端的系统字体 —— 在没装中文字体的环境里直接变豆腐块，
在装了的环境里也各不相同。这里把用到的那几十个汉字子集化后以 woff2 内联进 SVG，
让图到哪都长一样。图形部分仍由 roughjs 生成，保持手绘感。

用法：
    python3 images/embed_cjk_font.py images/xxx.svg
就地改写该 SVG。
"""
import base64
import io
import re
import sys
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont, TTCollection

FONT_CANDIDATES = [
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "Noto Sans CJK SC"),
    ("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", "WenQuanYi Micro Hei"),
    ("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", "Droid Sans Fallback"),
]
EMBED_NAME = "NavCJK"


def load_font(path):
    if path.endswith(".ttc"):
        coll = TTCollection(path)
        for f in coll.fonts:                      # .ttc 里多个地区变体，取含简体的那个
            names = {n.toUnicode() for n in f["name"].names if n.nameID == 1}
            if any("SC" in n or "Micro Hei" in n for n in names):
                return f
        return coll.fonts[0]
    return TTFont(path)


def collect_cjk(svg_text):
    chars = set()
    for run in re.findall(r">([^<]*)<", svg_text):
        for ch in run:
            o = ord(ch)
            if o > 0x2000:                        # 汉字、全角标点、箭头、± 等
                chars.add(ch)
    return chars


def main(svg_path):
    svg_path = Path(svg_path)
    svg = svg_path.read_text(encoding="utf-8")

    if EMBED_NAME in svg:
        print(f"{svg_path} 已内嵌 {EMBED_NAME}，跳过")
        return 0

    chars = collect_cjk(svg)
    if not chars:
        print("SVG 中没有需要嵌入的非 ASCII 字符")
        return 0

    src = next((p for p, _ in FONT_CANDIDATES if Path(p).exists()), None)
    if src is None:
        print("找不到可用的中文字体", file=sys.stderr)
        return 1

    font = load_font(src)
    opts = subset.Options()
    opts.flavor = "woff2"
    opts.desubroutinize = True
    opts.drop_tables += ["GSUB", "GPOS", "DSIG"]
    subsetter = subset.Subsetter(options=opts)
    subsetter.populate(text="".join(sorted(chars)))
    subsetter.subset(font)

    buf = io.BytesIO()
    font.flavor = "woff2"
    font.save(buf)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    face = (f"@font-face{{font-family:'{EMBED_NAME}';"
            f"src:url(data:font/woff2;base64,{b64}) format('woff2');"
            f"font-weight:normal;font-style:normal;}}")

    # 插进已有的 <style>，并让 CJK 字体排在 Virgil 之后（拉丁仍走 Virgil 手绘体）
    if "<style>" in svg:
        svg = svg.replace("<style>", "<style>" + face, 1)
    else:
        svg = re.sub(r"(<svg[^>]*>)", r"\1<style>" + face + "</style>", svg, count=1)
    svg = svg.replace('font-family="Virgil, Segoe UI Emoji, cursive"',
                      f'font-family="Virgil, {EMBED_NAME}, Segoe UI Emoji, cursive"')

    svg_path.write_text(svg, encoding="utf-8")
    print(f"已嵌入 {len(chars)} 个字形（{src}），woff2 {len(b64) * 3 // 4 // 1024} KB "
          f"-> {svg_path} 现为 {svg_path.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "minimap_tracker_architecture.svg"))
