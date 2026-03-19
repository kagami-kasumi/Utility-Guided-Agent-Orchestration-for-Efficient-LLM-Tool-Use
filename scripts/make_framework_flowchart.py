#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont


TITLE = "Overview of the Utility-Guided Agent Orchestration Framework"

OUT_DIR = Path("/home/inspur-02/lby/outputs/figures")
PNG_PATH = OUT_DIR / "utility_guided_agent_framework.png"
SVG_PATH = OUT_DIR / "utility_guided_agent_framework.svg"

WIDTH = 1600
HEIGHT = 1900

BG = "#FBF8F1"
PRIMARY = "#22333B"
ACCENT = "#8C5E3C"
BOX_FILL = "#F5EEE0"
PANEL_FILL = "#EFE3CB"
WHITE = "#FFFFFF"
LINE = "#5E6B73"
TEXT = "#1E2529"
MUTED = "#59656C"


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


FONT_TITLE = _load_font(54, bold=True)
FONT_BOX = _load_font(34, bold=True)
FONT_ITEM = _load_font(28, bold=False)
FONT_SMALL = _load_font(24, bold=False)


def draw_round_box(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], fill: str, outline: str, radius: int = 28, width: int = 4) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def centered_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font: ImageFont.ImageFont, fill: str) -> None:
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=6, align="center")
    x = box[0] + ((box[2] - box[0]) - (bbox[2] - bbox[0])) / 2
    y = box[1] + ((box[3] - box[1]) - (bbox[3] - bbox[1])) / 2 - 4
    draw.multiline_text((x, y), text, font=font, fill=fill, spacing=6, align="center")


def draw_arrow(draw: ImageDraw.ImageDraw, x: int, y1: int, y2: int) -> None:
    draw.line((x, y1, x, y2), fill=LINE, width=6)
    draw.polygon([(x, y2), (x - 14, y2 - 22), (x + 14, y2 - 22)], fill=LINE)


def pill(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], label: str, fill: str = WHITE, outline: str = ACCENT) -> None:
    draw_round_box(draw, xy, fill=fill, outline=outline, radius=22, width=3)
    centered_text(draw, xy, label, FONT_ITEM, TEXT)


def make_png() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle((70, 70, WIDTH - 70, HEIGHT - 70), radius=40, outline="#D9CCB3", width=3, fill=BG)

    title_box = (130, 110, WIDTH - 130, 220)
    centered_text(draw, title_box, TITLE, FONT_TITLE, PRIMARY)

    center_x = WIDTH // 2

    boxes = {
        "query": (450, 280, 1150, 390),
        "state_builder": (450, 450, 1150, 560),
        "utility": (300, 640, 1300, 980),
        "selector": (300, 1060, 1300, 1405),
        "observation": (360, 1480, 1240, 1600),
        "state_update": (450, 1670, 1150, 1780),
    }

    draw_round_box(draw, boxes["query"], BOX_FILL, PRIMARY)
    draw_round_box(draw, boxes["state_builder"], BOX_FILL, PRIMARY)
    draw_round_box(draw, boxes["utility"], PANEL_FILL, ACCENT)
    draw_round_box(draw, boxes["selector"], PANEL_FILL, ACCENT)
    draw_round_box(draw, boxes["observation"], BOX_FILL, PRIMARY)
    draw_round_box(draw, boxes["state_update"], BOX_FILL, PRIMARY)

    centered_text(draw, boxes["query"], "User Query", FONT_BOX, TEXT)
    centered_text(draw, boxes["state_builder"], "State Builder", FONT_BOX, TEXT)
    centered_text(draw, (boxes["utility"][0], boxes["utility"][1] + 15, boxes["utility"][2], boxes["utility"][1] + 90), "Utility Scorer", FONT_BOX, TEXT)
    centered_text(draw, (boxes["selector"][0], boxes["selector"][1] + 15, boxes["selector"][2], boxes["selector"][1] + 90), "Action Selector", FONT_BOX, TEXT)
    centered_text(draw, boxes["observation"], "Environment / Tool Observation", FONT_BOX, TEXT)
    centered_text(draw, boxes["state_update"], "State Update", FONT_BOX, TEXT)

    utility_y = 745
    pill_w = 340
    gap = 40
    left = 350
    for idx, label in enumerate(["Gain", "StepCost", "Uncertainty", "Redundancy"]):
        row = idx // 2
        col = idx % 2
        x1 = left + col * (pill_w + gap)
        y1 = utility_y + row * 110
        pill(draw, (x1, y1, x1 + pill_w, y1 + 78), label)

    selector_labels = ["respond", "retrieve", "tool_call", "verify", "stop"]
    selector_positions = [
        (360, 1170, 640, 1248),
        (680, 1170, 960, 1248),
        (1000, 1170, 1280, 1248),
        (520, 1290, 800, 1368),
        (840, 1290, 1120, 1368),
    ]
    for pos, label in zip(selector_positions, selector_labels):
        pill(draw, pos, label, fill=WHITE, outline=PRIMARY)

    draw_arrow(draw, center_x, boxes["query"][3], boxes["state_builder"][1] - 18)
    draw_arrow(draw, center_x, boxes["state_builder"][3], boxes["utility"][1] - 18)
    draw_arrow(draw, center_x, boxes["utility"][3], boxes["selector"][1] - 18)
    draw_arrow(draw, center_x, boxes["selector"][3], boxes["observation"][1] - 18)
    draw_arrow(draw, center_x, boxes["observation"][3], boxes["state_update"][1] - 18)

    loop_left = 180
    draw.line((loop_left, boxes["state_update"][1] + 55, loop_left, boxes["utility"][1] + 70), fill=ACCENT, width=6)
    draw.line((loop_left, boxes["utility"][1] + 70, 290, boxes["utility"][1] + 70), fill=ACCENT, width=6)
    draw.polygon([(290, boxes["utility"][1] + 70), (268, boxes["utility"][1] + 56), (268, boxes["utility"][1] + 84)], fill=ACCENT)
    draw.text((loop_left - 22, boxes["state_update"][1] + 15), "↺", font=_load_font(42, bold=True), fill=ACCENT)
    draw.text((95, 1210), "Iterative\nstate feedback", font=FONT_SMALL, fill=MUTED, spacing=4, align="center")

    img.save(PNG_PATH, format="PNG")


def _svg_box(x1: int, y1: int, x2: int, y2: int, fill: str, stroke: str, rx: int = 28, stroke_width: int = 4) -> str:
    return f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" rx="{rx}" ry="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'


def _svg_text(x: int, y: int, text: str, size: int, weight: str = "600", color: str = TEXT, anchor: str = "middle") -> str:
    lines = text.split("\n")
    if len(lines) == 1:
        return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="DejaVu Sans, Arial, sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}">{escape(text)}</text>'
    tspan = []
    line_h = int(size * 1.25)
    start_y = y - ((len(lines) - 1) * line_h) / 2
    for i, line in enumerate(lines):
        dy = 0 if i == 0 else line_h
        tspan.append(f'<tspan x="{x}" dy="{dy if i else 0}">{escape(line)}</tspan>')
    return f'<text x="{x}" y="{int(start_y)}" text-anchor="{anchor}" font-family="DejaVu Sans, Arial, sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}">{"".join(tspan)}</text>'


def _svg_arrow(x: int, y1: int, y2: int) -> str:
    return (
        f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{LINE}" stroke-width="6"/>'
        f'<polygon points="{x},{y2} {x-14},{y2-22} {x+14},{y2-22}" fill="{LINE}"/>'
    )


def make_svg() -> None:
    loop_label = _svg_text(112, 1220, "Iterative\nstate feedback", 24, weight="400", color=MUTED)
    loop_icon = _svg_text(160, 1708, "↺", 38, weight="700", color=ACCENT)
    utility_pills = []
    pill_specs = [
        (350, 745, 690, 823, "Gain"),
        (730, 745, 1070, 823, "StepCost"),
        (350, 855, 690, 933, "Uncertainty"),
        (730, 855, 1070, 933, "Redundancy"),
    ]
    for x1, y1, x2, y2, label in pill_specs:
        utility_pills.append(_svg_box(x1, y1, x2, y2, WHITE, ACCENT, rx=22, stroke_width=3))
        utility_pills.append(_svg_text((x1 + x2) // 2, y1 + 48, label, 28, weight="400"))

    selector_pills = []
    selector_specs = [
        (360, 1170, 640, 1248, "respond"),
        (680, 1170, 960, 1248, "retrieve"),
        (1000, 1170, 1280, 1248, "tool_call"),
        (520, 1290, 800, 1368, "verify"),
        (840, 1290, 1120, 1368, "stop"),
    ]
    for x1, y1, x2, y2, label in selector_specs:
        selector_pills.append(_svg_box(x1, y1, x2, y2, WHITE, PRIMARY, rx=22, stroke_width=3))
        selector_pills.append(_svg_text((x1 + x2) // 2, y1 + 48, label, 28, weight="400"))

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
  <rect width="100%" height="100%" fill="{BG}"/>
  <rect x="70" y="70" width="{WIDTH-140}" height="{HEIGHT-140}" rx="40" ry="40" fill="{BG}" stroke="#D9CCB3" stroke-width="3"/>
  {_svg_text(WIDTH//2, 175, TITLE, 40, weight="700", color=PRIMARY)}
  {_svg_box(450, 280, 1150, 390, BOX_FILL, PRIMARY)}
  {_svg_box(450, 450, 1150, 560, BOX_FILL, PRIMARY)}
  {_svg_box(300, 640, 1300, 980, PANEL_FILL, ACCENT)}
  {_svg_box(300, 1060, 1300, 1405, PANEL_FILL, ACCENT)}
  {_svg_box(360, 1480, 1240, 1600, BOX_FILL, PRIMARY)}
  {_svg_box(450, 1670, 1150, 1780, BOX_FILL, PRIMARY)}
  {_svg_text(800, 348, "User Query", 34, color=TEXT)}
  {_svg_text(800, 518, "State Builder", 34, color=TEXT)}
  {_svg_text(800, 720, "Utility Scorer", 34, color=TEXT)}
  {_svg_text(800, 1140, "Action Selector", 34, color=TEXT)}
  {_svg_text(800, 1550, "Environment / Tool Observation", 34, color=TEXT)}
  {_svg_text(800, 1738, "State Update", 34, color=TEXT)}
  {''.join(utility_pills)}
  {''.join(selector_pills)}
  {_svg_arrow(800, 390, 432)}
  {_svg_arrow(800, 560, 622)}
  {_svg_arrow(800, 980, 1042)}
  {_svg_arrow(800, 1405, 1462)}
  {_svg_arrow(800, 1600, 1652)}
  <line x1="180" y1="1725" x2="180" y2="710" stroke="{ACCENT}" stroke-width="6"/>
  <line x1="180" y1="710" x2="290" y2="710" stroke="{ACCENT}" stroke-width="6"/>
  <polygon points="290,710 268,696 268,724" fill="{ACCENT}"/>
  {loop_label}
  {loop_icon}
</svg>
"""
    SVG_PATH.write_text(svg, encoding="utf-8")


def main() -> None:
    make_png()
    make_svg()
    print(PNG_PATH)
    print(SVG_PATH)


if __name__ == "__main__":
    main()
