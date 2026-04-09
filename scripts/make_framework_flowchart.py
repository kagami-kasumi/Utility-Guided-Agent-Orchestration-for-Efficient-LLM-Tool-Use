#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont


TITLE = "Overview of the Utility-Guided Agent Framework"

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"
PNG_PATH = OUT_DIR / "utility_guided_agent_framework.png"
SVG_PATH = OUT_DIR / "utility_guided_agent_framework.svg"

WIDTH = 1680
HEIGHT = 980

BG = "#FBF8F1"
PRIMARY = "#22333B"
ACCENT = "#8C5E3C"
BOX_FILL = "#F5EEE0"
PANEL_FILL = "#EFE3CB"
WHITE = "#FFFFFF"
LINE = "#5E6B73"
TEXT = "#1E2529"
MUTED = "#59656C"
BLUE_FILL = "#E7F0F6"
BLUE_STROKE = "#537A8A"
GREEN_FILL = "#E7F3EC"
GREEN_STROKE = "#4F7A5E"


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_TITLE = _load_font(40, bold=True)
FONT_PANEL = _load_font(24, bold=True)
FONT_BOX = _load_font(22, bold=True)
FONT_ITEM = _load_font(18, bold=False)
FONT_SMALL = _load_font(16, bold=False)
FONT_LOOP = _load_font(18, bold=True)


def draw_round_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    fill: str,
    outline: str,
    *,
    radius: int = 22,
    width: int = 3,
) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
    *,
    spacing: int = 4,
) -> None:
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="center")
    x = box[0] + ((box[2] - box[0]) - (bbox[2] - bbox[0])) / 2
    y = box[1] + ((box[3] - box[1]) - (bbox[3] - bbox[1])) / 2 - 1
    draw.multiline_text((x, y), text, font=font, fill=fill, spacing=spacing, align="center")


def draw_down_arrow(draw: ImageDraw.ImageDraw, x: int, y1: int, y2: int) -> None:
    draw.line((x, y1, x, y2), fill=LINE, width=5)
    draw.polygon([(x, y2), (x - 9, y2 - 14), (x + 9, y2 - 14)], fill=LINE)


def draw_right_arrow(draw: ImageDraw.ImageDraw, x1: int, y: int, x2: int, *, color: str = LINE) -> None:
    draw.line((x1, y, x2, y), fill=color, width=4)
    draw.polygon([(x2, y), (x2 - 13, y - 8), (x2 - 13, y + 8)], fill=color)


def draw_loop_path(draw: ImageDraw.ImageDraw, x: int, y_bottom: int, y_top: int, x_to: int) -> None:
    draw.line((x, y_bottom, x, y_top), fill=ACCENT, width=5)
    draw.line((x, y_top, x_to, y_top), fill=ACCENT, width=5)
    draw.polygon([(x_to, y_top), (x_to - 14, y_top - 9), (x_to - 14, y_top + 9)], fill=ACCENT)


def pill(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    label: str,
    *,
    fill: str = WHITE,
    outline: str = PRIMARY,
    font: ImageFont.ImageFont = FONT_ITEM,
) -> None:
    draw_round_box(draw, xy, fill=fill, outline=outline, radius=18, width=2)
    centered_text(draw, xy, label, font, TEXT)


def make_png() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)

    draw_round_box(draw, (26, 16, WIDTH - 26, HEIGHT - 24), BG, "#D9CCB3", radius=30, width=3)
    centered_text(draw, (80, 18, WIDTH - 80, 60), TITLE, FONT_TITLE, PRIMARY)

    left_panel = (56, 62, 1096, 926)
    right_panel = (1130, 62, 1624, 926)
    draw_round_box(draw, left_panel, "#FAF4E8", "#D9CCB3", radius=26, width=2)
    draw_round_box(draw, right_panel, "#F8F2E7", "#D9CCB3", radius=26, width=2)
    centered_text(draw, (110, 72, 1042, 102), "Agent Loop", FONT_PANEL, PRIMARY)
    centered_text(draw, (1170, 72, 1588, 102), "Method Variants and Outputs", FONT_PANEL, PRIMARY)

    query = (184, 116, 968, 176)
    state = (174, 200, 978, 316)
    utility = (130, 354, 1022, 506)
    action = (130, 542, 1022, 682)
    observe = (184, 718, 968, 796)
    update = (184, 832, 968, 880)

    for box in (query, state, observe, update):
        draw_round_box(draw, box, BOX_FILL, PRIMARY)
    for box in (utility, action):
        draw_round_box(draw, box, PANEL_FILL, ACCENT)

    centered_text(draw, query, "User Query", FONT_BOX, TEXT)

    centered_text(draw, (220, 212, 934, 240), "State Builder", FONT_BOX, TEXT)
    for rect, label in [
        ((202, 252, 414, 294), "history"),
        ((472, 252, 684, 294), "evidence"),
        ((742, 252, 954, 294), "tool traces"),
    ]:
        pill(draw, rect, label)

    centered_text(draw, (170, 366, 982, 394), "Utility Scorer", FONT_BOX, TEXT)
    for rect, label in [
        ((164, 416, 372, 462), "Gain"),
        ((392, 416, 600, 462), "Step Cost"),
        ((620, 416, 828, 462), "Uncertainty"),
        ((392, 468, 600, 494), "Redundancy"),
        ((620, 468, 828, 494), "Budget Aware"),
    ]:
        pill(draw, rect, label, outline=ACCENT)

    centered_text(draw, (170, 554, 982, 582), "Action Selector", FONT_BOX, TEXT)
    for rect, label in [
        ((162, 606, 356, 650), "respond"),
        ((376, 606, 570, 650), "retrieve"),
        ((590, 606, 784, 650), "tool_call"),
        ((804, 606, 988, 650), "verify / stop"),
    ]:
        pill(draw, rect, label)

    centered_text(draw, (220, 732, 934, 756), "Environment / Tools", FONT_BOX, TEXT)
    for rect, label in [
        ((220, 764, 414, 788), "retriever"),
        ((480, 764, 674, 788), "search"),
        ((740, 764, 934, 788), "verifier"),
    ]:
        pill(draw, rect, label, font=FONT_SMALL)
    centered_text(draw, update, "State Update", FONT_BOX, TEXT)

    method_box = (1188, 124, 1566, 376)
    eval_box = (1188, 414, 1566, 696)
    draw_round_box(draw, method_box, BLUE_FILL, BLUE_STROKE, radius=24, width=3)
    draw_round_box(draw, eval_box, GREEN_FILL, GREEN_STROKE, radius=24, width=3)
    centered_text(draw, (1210, 138, 1544, 166), "Method Variants", FONT_BOX, PRIMARY)
    for rect, label in [
        ((1220, 172, 1376, 216), "direct"),
        ((1400, 172, 1556, 216), "workflow"),
        ((1220, 232, 1376, 276), "react"),
        ((1400, 232, 1556, 276), "threshold"),
        ((1310, 292, 1466, 336), "policy"),
    ]:
        pill(draw, rect, label, fill=WHITE, outline=BLUE_STROKE)

    centered_text(draw, (1210, 430, 1544, 458), "Evaluation Outputs", FONT_BOX, PRIMARY)
    for rect, label in [
        ((1220, 474, 1390, 524), "pilot runs"),
        ((1410, 474, 1556, 524), "expanded runs"),
        ((1220, 542, 1390, 592), "Pareto figures"),
        ((1410, 542, 1556, 592), "tables"),
        ((1310, 610, 1466, 660), "traces"),
    ]:
        pill(draw, rect, label, fill=WHITE, outline=GREEN_STROKE)

    center_x = 576
    draw_down_arrow(draw, center_x, query[3], state[1] - 10)
    draw_down_arrow(draw, center_x, state[3], utility[1] - 10)
    draw_down_arrow(draw, center_x, utility[3], action[1] - 10)
    draw_down_arrow(draw, center_x, action[3], observe[1] - 8)
    draw_down_arrow(draw, center_x, observe[3], update[1] - 12)

    draw_loop_path(draw, 92, update[1] + 24, state[1] + 12, 160)
    centered_text(draw, (34, 428, 112, 612), "feedback\nloop", FONT_LOOP, ACCENT)

    draw_right_arrow(draw, 1038, 430, 1170, color=ACCENT)
    draw_right_arrow(draw, 1038, 612, 1170, color=GREEN_STROKE)

    img.save(PNG_PATH, format="PNG")


def _svg_box(x1: int, y1: int, x2: int, y2: int, fill: str, stroke: str, *, rx: int = 22, stroke_width: int = 3) -> str:
    return f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" rx="{rx}" ry="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'


def _svg_text(x: int, y: int, text: str, size: int, *, weight: str = "600", color: str = TEXT, anchor: str = "middle") -> str:
    lines = text.split("\n")
    if len(lines) == 1:
        return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="DejaVu Sans, Arial, sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}">{escape(text)}</text>'
    line_h = int(size * 1.28)
    start_y = y - ((len(lines) - 1) * line_h) / 2
    spans = []
    for i, line in enumerate(lines):
        dy = 0 if i == 0 else line_h
        spans.append(f'<tspan x="{x}" dy="{dy if i else 0}">{escape(line)}</tspan>')
    return f'<text x="{x}" y="{int(start_y)}" text-anchor="{anchor}" font-family="DejaVu Sans, Arial, sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}">{"".join(spans)}</text>'


def _svg_down_arrow(x: int, y1: int, y2: int) -> str:
    return (
        f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{LINE}" stroke-width="5"/>'
        f'<polygon points="{x},{y2} {x-9},{y2-14} {x+9},{y2-14}" fill="{LINE}"/>'
    )


def _svg_right_arrow(x1: int, y: int, x2: int, *, color: str = LINE) -> str:
    return (
        f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{color}" stroke-width="4"/>'
        f'<polygon points="{x2},{y} {x2-13},{y-8} {x2-13},{y+8}" fill="{color}"/>'
    )


def _svg_loop(x: int, y_bottom: int, y_top: int, x_to: int) -> str:
    return (
        f'<line x1="{x}" y1="{y_bottom}" x2="{x}" y2="{y_top}" stroke="{ACCENT}" stroke-width="5"/>'
        f'<line x1="{x}" y1="{y_top}" x2="{x_to}" y2="{y_top}" stroke="{ACCENT}" stroke-width="5"/>'
        f'<polygon points="{x_to},{y_top} {x_to-14},{y_top-9} {x_to-14},{y_top+9}" fill="{ACCENT}"/>'
    )


def _svg_pill(x1: int, y1: int, x2: int, y2: int, label: str, *, fill: str = WHITE, outline: str = PRIMARY, size: int = 18) -> str:
    return _svg_box(x1, y1, x2, y2, fill, outline, rx=18, stroke_width=2) + _svg_text((x1 + x2) // 2, y1 + (y2 - y1) // 2 + 6, label, size, weight="400")


def make_svg() -> None:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        f'<rect width="100%" height="100%" fill="{BG}"/>',
        _svg_box(26, 16, WIDTH - 26, HEIGHT - 24, BG, "#D9CCB3", rx=30, stroke_width=3),
        _svg_text(WIDTH // 2, 52, TITLE, 40, weight="700", color=PRIMARY),
        _svg_box(56, 62, 1096, 926, "#FAF4E8", "#D9CCB3", rx=26, stroke_width=2),
        _svg_box(1130, 62, 1624, 926, "#F8F2E7", "#D9CCB3", rx=26, stroke_width=2),
        _svg_text(576, 92, "Agent Loop", 24, weight="700", color=PRIMARY),
        _svg_text(1378, 92, "Method Variants and Outputs", 24, weight="700", color=PRIMARY),
    ]

    for box in [
        _svg_box(184, 116, 968, 176, BOX_FILL, PRIMARY),
        _svg_box(174, 200, 978, 316, BOX_FILL, PRIMARY),
        _svg_box(130, 354, 1022, 506, PANEL_FILL, ACCENT),
        _svg_box(130, 542, 1022, 682, PANEL_FILL, ACCENT),
        _svg_box(184, 718, 968, 796, BOX_FILL, PRIMARY),
        _svg_box(184, 832, 968, 880, BOX_FILL, PRIMARY),
    ]:
        parts.append(box)

    parts.extend(
        [
            _svg_text(576, 152, "User Query", 22, color=TEXT),
            _svg_text(576, 228, "State Builder", 22, color=TEXT),
            _svg_text(576, 380, "Utility Scorer", 22, color=TEXT),
            _svg_text(576, 568, "Action Selector", 22, color=TEXT),
            _svg_text(576, 746, "Environment / Tools", 22, color=TEXT),
            _svg_text(576, 862, "State Update", 22, color=TEXT),
        ]
    )

    for spec in [
        (202, 252, 414, 294, "history"),
        (472, 252, 684, 294, "evidence"),
        (742, 252, 954, 294, "tool traces"),
        (164, 416, 372, 462, "Gain"),
        (392, 416, 600, 462, "Step Cost"),
        (620, 416, 828, 462, "Uncertainty"),
        (392, 468, 600, 494, "Redundancy"),
        (620, 468, 828, 494, "Budget Aware"),
        (162, 606, 356, 650, "respond"),
        (376, 606, 570, 650, "retrieve"),
        (590, 606, 784, 650, "tool_call"),
        (804, 606, 988, 650, "verify / stop"),
        (220, 764, 414, 788, "retriever"),
        (480, 764, 674, 788, "search"),
        (740, 764, 934, 788, "verifier"),
    ]:
        outline = ACCENT if spec[4] in {"Gain", "Step Cost", "Uncertainty", "Redundancy", "Budget Aware"} else PRIMARY
        parts.append(_svg_pill(*spec, outline=outline))

    for spec in [
        (1188, 184, 1566, 454, "", "", "", ""),
        (1188, 500, 1566, 790, "", "", "", ""),
    ]:
        pass

    parts.append(_svg_box(1188, 124, 1566, 376, BLUE_FILL, BLUE_STROKE, rx=24, stroke_width=3))
    parts.append(_svg_box(1188, 414, 1566, 696, GREEN_FILL, GREEN_STROKE, rx=24, stroke_width=3))
    parts.append(_svg_text(1377, 150, "Method Variants", 22, color=PRIMARY))
    parts.append(_svg_text(1377, 440, "Evaluation Outputs", 22, color=PRIMARY))

    for spec in [
        (1220, 172, 1376, 216, "direct"),
        (1400, 172, 1556, 216, "workflow"),
        (1220, 232, 1376, 276, "react"),
        (1400, 232, 1556, 276, "threshold"),
        (1310, 292, 1466, 336, "policy"),
    ]:
        parts.append(_svg_pill(*spec, fill=WHITE, outline=BLUE_STROKE))

    for spec in [
        (1220, 474, 1390, 524, "pilot runs"),
        (1410, 474, 1556, 524, "expanded runs"),
        (1220, 542, 1390, 592, "Pareto figures"),
        (1410, 542, 1556, 592, "tables"),
        (1310, 610, 1466, 660, "traces"),
    ]:
        parts.append(_svg_pill(*spec, fill=WHITE, outline=GREEN_STROKE))

    for spec in [
        (248, 762, 458, 788, "direct / workflow"),
        (480, 762, 690, 788, "react / threshold"),
        (712, 762, 922, 788, "policy"),
    ]:
        parts.append(_svg_pill(*spec, fill=BLUE_FILL, outline=BLUE_STROKE, size=16))

    parts.extend(
        [
            _svg_down_arrow(576, 176, 190),
            _svg_down_arrow(576, 316, 342),
            _svg_down_arrow(576, 506, 530),
            _svg_down_arrow(576, 682, 708),
            _svg_down_arrow(576, 796, 820),
            _svg_loop(92, 856, 212, 160),
            _svg_text(72, 510, "feedback\nloop", 18, weight="700", color=ACCENT),
            _svg_right_arrow(1038, 430, 1170, color=ACCENT),
            _svg_right_arrow(1038, 612, 1170, color=GREEN_STROKE),
        ]
    )

    parts.append("</svg>")
    SVG_PATH.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    make_png()
    make_svg()
    print(PNG_PATH)
    print(SVG_PATH)


if __name__ == "__main__":
    main()
