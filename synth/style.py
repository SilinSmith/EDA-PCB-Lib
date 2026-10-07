"""Palettes, fonts and the supersampled painter used by the schematic synthesizer.

Everything the synthesizer draws goes through :class:`Painter`, which owns a canvas scaled by
``ss`` (supersampling factor).  All public coordinates are in *final image* pixels with the
origin at the top-left -- the same convention the YOLO labels use.  Callers never see the
supersampled space; ``finish()`` downsamples with LANCZOS so the output has the soft,
antialiased look of the real raster drawings instead of hard aliased edges.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_DIR = Path(r"C:\Windows\Fonts")

# The real drawings are anti-aliased rasters of vector CAD output, so a plain TTF is a good
# stand-in; these three cover monospace (pin numbers / values) and proportional (sheet titles).
FONT_FILES = {
    "mono": FONT_DIR / "consola.ttf",
    "mono_bold": FONT_DIR / "consolab.ttf",
    "sans": FONT_DIR / "arial.ttf",
    "sans_bold": FONT_DIR / "arialbd.ttf",
    "serif": FONT_DIR / "times.ttf",
}


@dataclass
class Palette:
    """Colours sampled from the real sheets (see ``work/survey_images.py`` for the measurements)."""

    name: str
    bg: tuple[int, int, int]
    wire: tuple[int, int, int]
    pin: tuple[int, int, int]
    text: tuple[int, int, int]
    body_fill: tuple[int, int, int] | None
    body_line: tuple[int, int, int]
    junction: tuple[int, int, int]
    value_text: tuple[int, int, int] | None = None
    grid: tuple[int, int, int] | None = None
    sheet_border: tuple[int, int, int] | None = None

    def value_colour(self) -> tuple[int, int, int]:
        return self.value_text or self.text


# KiCad's default eeschema theme, measured off `200_train_cases/0001-KiCad.png`:
#   bg rgb(245,244,239) 83.1%, highlight rgb(255,255,194) 7.2%,
#   wire rgb(0,100,100), pin rgb(132,0,0), text rgb(0,0,132)
PALETTES = {
    "kicad": Palette("kicad", (245, 244, 239), (0, 100, 100), (132, 0, 0), (0, 0, 132),
                     (255, 255, 194), (132, 0, 0), (0, 100, 100), grid=(228, 226, 220)),
    "altium": Palette("altium", (255, 255, 255), (0, 0, 176), (160, 0, 0), (0, 0, 0),
                      (255, 255, 214), (128, 0, 0), (0, 0, 176), value_text=(96, 0, 96),
                      grid=(238, 238, 238), sheet_border=(150, 150, 150)),
    "jlc": Palette("jlc", (255, 255, 255), (0, 128, 0), (128, 0, 0), (0, 0, 0),
                   None, (192, 0, 0), (0, 128, 0), value_text=(0, 0, 128),
                   grid=(240, 240, 240)),
    # `0002-Datasheet.png` is 256-colour greyscale: any colour-based segmentation dies here,
    # so the synthesizer has to be able to produce this style too.
    "datasheet": Palette("datasheet", (255, 255, 255), (40, 40, 40), (40, 40, 40), (20, 20, 20),
                         None, (40, 40, 40), (40, 40, 40)),
}

_font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def get_font(kind: str, size: int, ss: int) -> ImageFont.FreeTypeFont:
    """Return a TTF at ``size`` final pixels, rendered at ``size * ss`` device pixels."""
    key = (f"{kind}@{ss}", size)
    if key not in _font_cache:
        path = FONT_FILES.get(kind) or FONT_FILES["mono"]
        if not path.exists():
            path = FONT_FILES["mono"] if FONT_FILES["mono"].exists() else FONT_FILES["sans"]
        _font_cache[key] = ImageFont.truetype(str(path), max(6, int(round(size * ss))))
    return _font_cache[key]


@dataclass
class Painter:
    """Draw in final-image coordinates onto a ``ss``-times-larger canvas."""

    w: int
    h: int
    palette: Palette
    ss: int = 3
    img: Image.Image = field(init=False)
    d: ImageDraw.ImageDraw = field(init=False)
    touches: list[tuple[int, int, int, int]] = field(default_factory=list)
    # Sink for TextItems.  Kept untyped so style.py need not import symbols.py (which imports
    # style.py); the scene sets this to its own list before drawing anything.
    ctx_text: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.img = Image.new("RGB", (self.w * self.ss, self.h * self.ss), self.palette.bg)
        self.d = ImageDraw.Draw(self.img)

    # -- primitives ------------------------------------------------------------------
    def line(self, pts, colour=None, width: float = 1.0) -> None:
        s = self.ss
        self.d.line([(x * s, y * s) for x, y in pts], fill=colour or self.palette.wire,
                    width=max(1, int(round(width * s))), joint="curve")
        for x, y in pts:
            self.touches.append((int(x), int(y), int(x), int(y)))

    def rect(self, box, outline=None, width: float = 1.0, fill=None) -> None:
        s = self.ss
        x1, y1, x2, y2 = box
        self.d.rectangle([x1 * s, y1 * s, x2 * s, y2 * s], outline=outline,
                         fill=fill, width=max(1, int(round(width * s))))
        self.touches.append((int(x1), int(y1), int(x2), int(y2)))

    def ellipse(self, box, outline=None, width: float = 1.0, fill=None) -> None:
        s = self.ss
        x1, y1, x2, y2 = box
        self.d.ellipse([x1 * s, y1 * s, x2 * s, y2 * s], outline=outline,
                       fill=fill, width=max(1, int(round(width * s))))
        self.touches.append((int(x1), int(y1), int(x2), int(y2)))

    def dot(self, x: float, y: float, r: float = 2.6, colour=None) -> None:
        s = self.ss
        self.d.ellipse([(x - r) * s, (y - r) * s, (x + r) * s, (y + r) * s],
                       fill=colour or self.palette.junction)

    def arc(self, box, start: float, end: float, colour=None, width: float = 1.0) -> None:
        s = self.ss
        x1, y1, x2, y2 = box
        self.d.arc([x1 * s, y1 * s, x2 * s, y2 * s], start, end,
                   fill=colour or self.palette.pin, width=max(1, int(round(width * s))))
        self.touches.append((int(x1), int(y1), int(x2), int(y2)))

    def text(self, x: float, y: float, s: str, kind: str = "mono", size: int = 11,
             colour=None, anchor: str = "lm") -> tuple[float, float, float, float]:
        """Draw ``s`` and return its tight bounding box in final-image coordinates."""
        if not s:
            return (x, y, x, y)
        ss = self.ss
        font = get_font(kind, size, ss)
        col = colour if colour is not None else self.palette.text
        dx, dy = x * ss, y * ss
        self.d.text((dx, dy), s, font=font, fill=col, anchor=anchor)
        x1, y1, x2, y2 = self.d.textbbox((dx, dy), s, font=font, anchor=anchor)
        self.touches.append((int(x1 / ss), int(y1 / ss), int(x2 / ss), int(y2 / ss)))
        return (x1 / ss, y1 / ss, x2 / ss, y2 / ss)

    def text_width(self, s: str, kind: str = "mono", size: int = 11) -> float:
        if not s:
            return 0.0
        font = get_font(kind, size, self.ss)
        return self.d.textlength(s, font=font) / self.ss

    # -- output ----------------------------------------------------------------------
    def finish(self, noise: float = 0.0, jpeg: int | None = None) -> Image.Image:
        out = self.img.resize((self.w, self.h), Image.LANCZOS)
        if noise > 0:
            px = out.load()
            rnd = random.Random(0xC0FFEE)
            for y in range(self.h):
                for x in range(self.w):
                    if rnd.random() < noise:
                        r, g, b = px[x, y]
                        d = rnd.randint(-14, 14)
                        px[x, y] = (max(0, min(255, r + d)), max(0, min(255, g + d)),
                                    max(0, min(255, b + d)))
        return out


def ink_bbox(img: Image.Image, bg: tuple[int, int, int], tol: int = 26):
    """Bounding box of everything that differs from ``bg`` -- used to crop the sheet margin."""
    g = img.convert("L")
    bgl = int(0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2])
    mask = g.point(lambda v: 255 if abs(v - bgl) > tol else 0)
    return mask.getbbox()
