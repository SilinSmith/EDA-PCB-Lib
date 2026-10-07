"""Procedural schematic symbols.

Each ``draw_*`` function paints one component at a requested centre and returns a
:class:`Comp` describing exactly what it drew: the outer bounding box (which, per the problem
statement's appendix, *includes* the pin stubs -- the pin tip sits on the border), and one
:class:`Pin` per pin with both its ``tip`` (where a wire attaches) and its ``base`` (where it
meets the body).  Text that a recogniser would need to read is registered as a :class:`TextItem`
with the role that produced it, which is what makes the synthesizer useful for training a
*designator* detector and not just a blob detector.

The rendering is deliberately plain vector work: the real corpus is antialiased raster CAD
output, so a rectangle-and-line drawing downsampled from 3x is close enough at this stage.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .style import Painter

# ---------------------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------------------


@dataclass
class Pin:
    number: str
    name: str
    tip: tuple[float, float]
    base: tuple[float, float]
    side: str  # 'L' | 'R' | 'T' | 'B'


@dataclass
class TextItem:
    content: str
    role: str  # designator | value | pin_name | pin_number | net_label | note
    bbox: tuple[float, float, float, float]
    comp: str | None = None


@dataclass
class Comp:
    key: str
    ctype: str
    value: str | None
    bbox: tuple[float, float, float, float]
    pins: list[Pin] = field(default_factory=list)
    name: str | None = None
    angle: int = 0

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2

    def inflated(self, m: float):
        x1, y1, x2, y2 = self.bbox
        return (x1 - m, y1 - m, x2 + m, y2 + m)


# ---------------------------------------------------------------------------------------
# designator prefixes and part-number banks
# ---------------------------------------------------------------------------------------

PREFIX = {
    "box": "U", "r": "R", "c": "C", "l": "L", "d": "D", "led": "LED", "switch": "SW",
    "crystal_2pin": "X", "crystal_3pin": "Y", "crystal_4pin": "Y", "crystal": "X",
    "testpoint": "TP", "mosfet": "Q", "mosfet_pnp": "Q", "mosfet_npn": "Q", "bjt": "Q",
    "bjt_pnp": "Q", "bjt_npn": "Q", "fuse": "F", "buzzer": "BZ", "motor": "M", "ant": "E",
    "jumper": "JP", "battary": "BT", "block": "A", "pin": "P", "opt": "U", "dc": "J",
    "seg": "DS", "circle_header": "J", "esd": "D", "amp": "U", "amp_3pin": "U",
    "amp_5pin": "U", "transfomer": "T", "switch_2": "SW",
    # power / ground symbols are drawn without a designator: the corpus spells their key as a
    # nameless "\u2205<n>" placeholder, which is the single largest source of unmatchable keys
    # (4214 of 9609).  Reproducing that here keeps the synthesizer honest about the corpus.
    "v": "\u2205", "gnd": "\u2205", "net_input": "\u2205", "net_output": "\u2205",
    "net_short": "\u2205", "net_bidirection": "\u2205", "m3\u87ba\u4e1d": "\u2205",
    "other": "\u2205",
}

IC_NAMES = [
    "RP2350B", "BCM2835", "TPS7B770x-Q1", "TPS7B633x-Q1", "S-8252 Series", "STM32F103C8T6",
    "STM32F031K6U6", "ESP32-S3-WROOM-1", "CH340C", "AMS1117-3.3", "MP1584EN", "NE555",
    "LM358", "LM324", "TL431", "ATmega328P-AU", "GD32F103C8T6", "TP4056", "SY8113BADC",
    "AP3216C", "BMI160", "W25Q128JVSIQ", "MT3608", "XL1509-5.0E1", "OB2263",
]

PIN_NAME_BANK = [
    "VDD", "VSS", "GND", "VCC", "VDDA", "VDDIO", "VIN", "VOUT", "EN", "FB", "SW", "BST",
    "COMP", "SS", "RT", "CS", "SCLK", "SDIO", "SDA", "SCL", "MOSI", "MISO", "SCK", "RST",
    "NRST", "BOOT0", "XTALN", "XTALP", "XIN", "XOUT", "TX", "RX", "CTS", "RTS", "DTR",
    "DSR", "D+", "D-", "USB_DM", "USB_DP", "VBUS", "SWDIO", "SWCLK", "SWO", "RUN",
    "IOVDD", "DVDD", "AVDD", "AGND", "PGND", "EPAD", "NC", "GPIO0", "GPIO1", "GPIO2",
    "GPIO3", "GPIO4", "GPIO5", "GPIO6", "GPIO7", "PA0", "PA1", "PA2", "PB0", "PB1",
    "PC13", "AIN0", "AIN1", "REF", "OSC1", "OSC2", "CLK", "CSN", "IRQ", "INT", "ADC0",
    "ADC1", "PWM0", "PWM1", "LIN", "LOUT", "ROUT", "VREF", "ISENSE", "TEMP", "SENSE_EN",
]

VALUE_BANK = {
    "r": ["100", "220", "330", "470", "1k", "2.2k", "4.7k", "10k", "12k", "20k", "47k",
          "100k", "200k", "360k", "1M", "0.1R", "0.5R", "10mR"],
    "c": ["10pF", "22pF", "27pF", "100nF", "1uF", "2.2uF", "4.7uF", "10uF", "22uF",
          "47uF", "100uF", "220uF", "1nF", "470uF", "1000uF"],
    "l": ["1uH", "2.2uH", "4.7uH", "10uH", "22uH", "47uH", "100uH", "1mH"],
    "v": ["3.3V", "5V", "+1V8", "1.2V", "2.5V", "VBUS", "VCC", "12V", "24V"],
    "gnd": ["GND", "AGND", "PGND", "DGND", "Earth"],
    "fuse": ["0.5A", "1A", "2A", "3A", "F1", "5A"],
    "testpoint": ["TP", "1x1", "PAD"],
    "switch": ["SW_Push", "SW_SPST", "SW_DIP"],
}

NET_LABEL_BANK = [
    "VCC", "VDD", "VDD_3V3", "5V", "VBUS", "GND", "PLL_VDD", "PLL_1V8", "IOVDD", "DVDD",
    "SDA", "SCL", "UART_TX", "UART_RX", "SPI_CLK", "SPI_MOSI", "SPI_MISO", "CS_N", "RESET",
    "XTAL_IN", "XTAL_OUT", "USB_DP", "USB_DM", "ADC_IN", "VREF", "ENABLE", "ALERT",
    "1010_SCK", "MISO", "MOSI", "SCK", "RST", "SWDIO", "SWCLK", "CAN_H", "CAN_L",
    "AUX1", "AUX2", "SIG_A", "SIG_B", "NRF_CE", "NRF_CSN", "LCD_RS", "LCD_CS",
]


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------


class Namer:
    """Allocates the designators a real sheet would carry (R1, R2, ..., U1, C1, ...)."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.count: dict[str, int] = {}

    def __call__(self, ctype: str) -> str:
        p = PREFIX.get(ctype, "U")
        if p == "\u2205":
            n = self.count.setdefault(p, 0) + 1
            self.count[p] = n
            return f"\u2205{100 + n}"
        n = self.count.setdefault(p, 0) + 1
        self.count[p] = n
        return f"{p}{n}"


def _pin_num(rng: random.Random) -> str:
    style = rng.random()
    if style < 0.72:
        return str(rng.randint(1, 64))
    if style < 0.86:
        return f"{rng.choice('ABCD')}{rng.randint(1, 8)}"
    return f"P{rng.randint(1, 24)}"


def _emit(texts: list[TextItem], bbox, content: str, role: str, comp: str | None) -> None:
    if content and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
        texts.append(TextItem(content, role, tuple(float(v) for v in bbox), comp))


# ---------------------------------------------------------------------------------------
# symbol renderers
# ---------------------------------------------------------------------------------------
# Every renderer receives (painter, rng, key, cx, cy, value, name) and returns a Comp.
# The painter carries `ctx_text` (the scene's TextItem sink) as a side attribute.


def draw_ic(p: Painter, rng, key: str, cx: float, cy: float, value, name) -> Comp:
    # Sized against the corpus: a `box` has a median bbox of 59x60 px and a median of 5 pins
    # (143 at the extreme), so most ICs here must be small regulators, not 48-pin MCUs.
    small = rng.random() < 0.62
    n_left = rng.randint(1, 3) if small else rng.randint(3, 8)
    n_right = rng.randint(1, 3) if small else rng.randint(3, 8)
    pitch = rng.choice([10, 12, 14])
    pin_len = rng.choice([7, 9, 11])
    rows = max(n_left, n_right)
    body_h = rows * pitch + 10

    used: set[str] = set()

    def pick_name() -> str:
        for _ in range(20):
            nm = rng.choice(PIN_NAME_BANK)
            if nm not in used:
                used.add(nm)
                return nm
        nm = f"IO{len(used)}"
        used.add(nm)
        return nm

    left = [(pick_name(), _pin_num(rng)) for _ in range(n_left)]
    right = [(pick_name(), _pin_num(rng)) for _ in range(n_right)]
    # the body has to be wide enough for both pin-name columns, or the two would collide
    lw = max(p.text_width(nm, "mono", 8) for nm, _ in left) + 6
    rw = max(p.text_width(nm, "mono", 8) for nm, _ in right) + 6
    body_w = max(34.0, lw + rw + 16)

    x1, y1 = cx - body_w / 2, cy - body_h / 2
    x2, y2 = cx + body_w / 2, cy + body_h / 2
    p.rect((x1, y1, x2, y2), outline=p.palette.body_line,
           fill=p.palette.body_fill, width=1.2)

    pins: list[Pin] = []

    for i, (nm, num) in enumerate(left):
        yy = y1 + 6 + i * pitch + (body_h - 12 - n_left * pitch) / 2
        tip, base = (x1 - pin_len, yy), (x1, yy)
        pins.append(Pin(num, nm, tip, base, "L"))
        p.line([base, tip], colour=p.palette.pin)
        bb = p.text(x1 - 3, yy - 4, num, "mono", 7, colour=p.palette.pin, anchor="rs")
        _emit(p.ctx_text, bb, num, "pin_number", key)
        bb = p.text(x1 + 4, yy, nm, "mono", 8, colour=p.palette.value_colour(), anchor="lm")
        _emit(p.ctx_text, bb, nm, "pin_name", key)

    for i, (nm, num) in enumerate(right):
        yy = y1 + 6 + i * pitch + (body_h - 12 - n_right * pitch) / 2
        tip, base = (x2 + pin_len, yy), (x2, yy)
        pins.append(Pin(num, nm, tip, base, "R"))
        p.line([base, tip], colour=p.palette.pin)
        bb = p.text(x2 + 3, yy - 4, num, "mono", 7, colour=p.palette.pin, anchor="ls")
        _emit(p.ctx_text, bb, num, "pin_number", key)
        bb = p.text(x2 - 4, yy, nm, "mono", 8, colour=p.palette.value_colour(), anchor="rm")
        _emit(p.ctx_text, bb, nm, "pin_name", key)

    # the part number sits above the body, the designator above that (KiCad's default layering)
    top = y1
    bb = p.text(cx, top - 5, value or name or "", "sans", 10,
                colour=p.palette.text, anchor="ms")
    _emit(p.ctx_text, bb, value or name or "", "value", key)
    bb = p.text(cx, top - 16, key, "mono", 10, colour=p.palette.text, anchor="ms")
    _emit(p.ctx_text, bb, key, "designator", key)

    bbox = (x1, y1, x2, y2)
    return Comp(key, "box", value, bbox, pins, name=name or value)


SHORT_AXIS = {"r": 4.0, "c": 8.0, "l": 3.0, "d": 7.0, "led": 7.0, "esd": 7.0, "fuse": 5.0}
LONG_AXIS = {"r": 20.0, "c": 20.0, "l": 34.0, "d": 20.0, "led": 20.0, "esd": 20.0, "fuse": 20.0}


def draw_passive(p: Painter, rng, key: str, cx: float, cy: float, value, ctype: str) -> Comp:
    """Resistor / capacitor / inductor / diode / LED / fuse, drawn on the vertical axis.

    Dimensions come straight from the corpus medians (r 8x20, c 16x20, l 6x34, led 14x20), which
    are far smaller than they look -- a real passive is about the same size as its own label.
    """
    vertical = rng.random() < 0.78
    body_l = LONG_AXIS.get(ctype, 20.0) * rng.uniform(0.88, 1.15)
    horizon = SHORT_AXIS.get(ctype, 6.0) * rng.uniform(0.9, 1.1)
    pin_len = rng.choice([6, 8, 10])
    half = body_l / 2

    if vertical:
        bx1, by1, bx2, by2 = cx - horizon, cy - half, cx + horizon, cy + half
        tips = [(cx, by1 - pin_len), (cx, by2 + pin_len)]
        bases = [(cx, by1), (cx, by2)]
        sides = ["T", "B"]
    else:
        bx1, by1, bx2, by2 = cx - half, cy - horizon, cx + half, cy + horizon
        tips = [(bx1 - pin_len, cy), (bx2 + pin_len, cy)]
        bases = [(bx1, cy), (bx2, cy)]
        sides = ["L", "R"]
    pins = [Pin("1", "", tips[0], bases[0], sides[0]),
            Pin("2", "", tips[1], bases[1], sides[1])]

    stroke = p.palette.pin
    if ctype == "r":
        if rng.random() < 0.5:  # IEC box
            p.rect((bx1, by1, bx2, by2), outline=stroke, fill=p.palette.bg, width=1.2)
        else:  # ANSI zig-zag
            n = 6
            pts = []
            for i in range(n * 2 + 1):
                t = i / (n * 2)
                off = 0 if i == 0 or i == n * 2 else (bx2 - cx if i % 2 else bx1 - cx)
                if vertical:
                    pts.append((cx + off, by1 + t * body_l))
                else:
                    pts.append((bx1 + t * body_l, cy + off))
            p.line(pts, colour=stroke)
    elif ctype == "c":
        gap = 3.0
        if vertical:
            p.line([(bx1, cy - gap), (bx2, cy - gap)], colour=stroke, width=1.6)
            p.line([(bx1, cy + gap), (bx2, cy + gap)], colour=stroke, width=1.6)
            if rng.random() < 0.25:  # polarized
                p.arc((bx1, cy + gap, bx2, cy + gap + 10), 0, 180, colour=stroke, width=1.6)
        else:
            p.line([(cx - gap, by1), (cx - gap, by2)], colour=stroke, width=1.6)
            p.line([(cx + gap, by1), (cx + gap, by2)], colour=stroke, width=1.6)
    elif ctype == "l":
        for i in range(4):
            r = body_l / 10
            if vertical:
                yy = by1 + i * body_l / 4
                p.arc((cx - r * 1.6, yy - r, cx + r * 1.6, yy + r), 0, 180, colour=stroke, width=1.3)
            else:
                xx = bx1 + i * body_l / 4
                p.arc((xx - r, cy - r * 1.6, xx + r, cy + r * 1.6), 270, 90, colour=stroke, width=1.3)
    elif ctype in ("d", "led", "esd"):
        tri = 7
        if vertical:
            p.line([(cx - tri, cy + tri), (cx + tri, cy + tri), (cx, cy - tri), (cx - tri, cy + tri)],
                   colour=stroke)
            p.line([(cx - tri, cy - tri), (cx + tri, cy - tri)], colour=stroke, width=1.6)
        else:
            p.line([(bx1, cy - tri), (bx1, cy + tri), (bx1 + 2 * tri, cy), (bx1, cy - tri)],
                   colour=stroke)
            p.line([(bx1 + 2 * tri, cy - tri), (bx1 + 2 * tri, cy + tri)], colour=stroke, width=1.6)
        if ctype == "led":
            for k in (-1, 1):
                ox = cx + 9 if vertical else bx1 + 3
                oy = cy - 9 if vertical else cy + k * 9
                p.line([(ox, oy), (ox + 7, oy - 7 if vertical else oy)],
                       colour=stroke)
    elif ctype == "fuse":
        p.rect((bx1, by1, bx2, by2), outline=stroke, fill=p.palette.bg, width=1.2)
        if vertical:
            p.line([(cx, by1), (cx, by2)], colour=stroke)
        else:
            p.line([(bx1, cy), (bx2, cy)], colour=stroke)

    labels = [item for item in (value,) if item]
    if vertical:
        bb = p.text(cx + horizon + 4, cy, value or "", "mono", 9,
                    colour=p.palette.value_colour(), anchor="lm")
    else:
        bb = p.text(cx, cy - horizon - 4, value or "", "mono", 9,
                    colour=p.palette.value_colour(), anchor="ms")
    if labels:
        _emit(p.ctx_text, bb, value, "value", key)
    dbb = p.text(cx, cy - half - 13 if vertical else cy + horizon + 15, key, "mono", 9,
                 colour=p.palette.text, anchor="ms")
    _emit(p.ctx_text, dbb, key, "designator", key)

    return Comp(key, ctype, value, (bx1, by1, bx2, by2), pins)


def draw_power(p: Painter, rng, ctype: str, key: str, cx: float, cy: float, value) -> Comp:
    """One-pin power flag: ``v`` (rail, bar/arrow/circle) or ``gnd`` (three shrinking bars).

    Corpus medians: gnd is 20x19, v is 10x10.  Both carry exactly one pin (221 of 2229 gnd
    symbols have one pin and no more), so these are flags rather than components.
    """
    stroke = p.palette.pin
    if ctype == "gnd":
        y = cy
        tip = (cx, y - 12)
        base = (cx, y)
        p.line([base, tip], colour=stroke)
        for i, wpx in enumerate((20, 13, 6)):
            p.line([(cx - wpx / 2, y + i * 3.5), (cx + wpx / 2, y + i * 3.5)],
                   colour=stroke, width=1.5)
        bbox = (cx - 10, y - 12, cx + 10, y + 7)
        pins = [Pin("1", "GND", tip, base, "T")]
        bb = p.text(cx, y + 16, value or "GND", "mono", 9,
                    colour=p.palette.value_colour(), anchor="ms")
        _emit(p.ctx_text, bb, value or "GND", "value", key)
        return Comp(key, ctype, value, bbox, pins)

    pole = rng.choice([9, 11, 13])
    tip = (cx, cy + pole / 2 + 8)
    base = (cx, cy + pole / 2)
    p.line([base, tip], colour=stroke)
    top = cy - pole / 2
    style = rng.random()
    if style < 0.4:  # arrow
        p.line([(cx - 5, top + 7), (cx, top), (cx + 5, top + 7)], colour=stroke, width=1.5)
        p.line([(cx - 5, top + 7), (cx + 5, top + 7)], colour=stroke, width=1.5)
    elif style < 0.75:  # bar
        p.line([(cx - 6, top), (cx + 6, top)], colour=stroke, width=1.6)
    else:  # circle
        p.ellipse((cx - 5, top - 5, cx + 5, top + 5), outline=stroke, width=1.4)
    label = value or rng.choice(["3.3V", "5V", "VCC"])
    bb = p.text(cx, top - 8, label, "mono", 9, colour=p.palette.value_colour(), anchor="ms")
    _emit(p.ctx_text, bb, label, "value", key)
    bbox = (cx - 6, top - 2, cx + 6, base[1])
    return Comp(key, ctype, value or label, bbox, [Pin("1", label, tip, base, "B")])


def draw_net_flag(p: Painter, rng, ctype: str, key: str, cx: float, cy: float, value) -> Comp:
    """The arrow-shaped net_input / net_output / net_short / net_bidirection markers.

    Corpus: net_input / net_output are 30x10 boxes with one pin; net_short is 18x7 with two.
    """
    stroke = p.palette.pin
    short = [l for l in NET_LABEL_BANK if len(l) <= 8] or NET_LABEL_BANK
    label = value or rng.choice(short)
    h = 7.0 if ctype == "net_short" else 11.0
    w = max(18.0, p.text_width(label, "mono", 8) + 10)
    right = ctype in ("net_output", "net_bidirection")
    left = ctype in ("net_input", "net_bidirection")
    pts = [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2), (cx + w / 2, cy + h / 2),
           (cx - w / 2, cy + h / 2)]
    p.line(pts + [pts[0]], colour=stroke)
    if right:
        p.line([(cx + w / 2, cy - h / 2), (cx + w / 2 + 6, cy), (cx + w / 2, cy + h / 2)],
               colour=stroke)
    if left:
        p.line([(cx - w / 2, cy - h / 2), (cx - w / 2 - 6, cy), (cx - w / 2, cy + h / 2)],
               colour=stroke)
    bb = p.text(cx, cy, label, "mono", 8, colour=p.palette.text, anchor="mm")
    _emit(p.ctx_text, bb, label, "net_label", key)
    tip = (cx - w / 2 - 6, cy) if left else (cx + w / 2 + 6, cy)
    base = (cx - w / 2, cy) if left else (cx + w / 2, cy)
    bbox = (cx - w / 2 - 6, cy - h / 2, cx + w / 2 + 6, cy + h / 2)
    pins = [Pin("1", label, tip, base, "L" if left else "R")]
    if ctype == "net_short" and rng.random() < 0.5:
        other = (cx + w / 2, cy) if left else (cx - w / 2, cy)
        pins.append(Pin("2", "", other, other, "R" if left else "L"))
    return Comp(key, ctype, label, bbox, pins)


def draw_generic(p: Painter, rng, ctype: str, key: str, cx: float, cy: float, value) -> Comp:
    """Two-pin fallback for the long tail (testpoint, switch, crystal, jumper, battary, ...)."""
    stroke = p.palette.pin
    vertical = rng.random() < 0.6
    body_l = rng.choice([14, 18, 22]) * rng.uniform(0.9, 1.1)
    pin_len = rng.choice([6, 8])
    half = body_l / 2
    horizon = rng.uniform(6.0, 8.5)

    if vertical:
        bx1, by1, bx2, by2 = cx - horizon, cy - half, cx + horizon, cy + half
        tips = [(cx, by1 - pin_len), (cx, by2 + pin_len)]
        bases = [(cx, by1), (cx, by2)]
        sides = ["T", "B"]
    else:
        bx1, by1, bx2, by2 = cx - half, cy - horizon, cx + half, cy + horizon
        tips = [(bx1 - pin_len, cy), (bx2 + pin_len, cy)]
        bases = [(bx1, cy), (bx2, cy)]
        sides = ["L", "R"]
    pins = [Pin("1", "", tips[0], bases[0], sides[0]),
            Pin("2", "", tips[1], bases[1], sides[1])]
    for pin in pins:
        p.line([pin.base, pin.tip], colour=stroke)

    if ctype == "testpoint":
        p.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), outline=stroke, width=1.4)
        pins = pins[:1]
    elif ctype == "switch":
        p.ellipse((bases[0][0] - 2.5, bases[0][1] - 2.5, bases[0][0] + 2.5, bases[0][1] + 2.5),
                  fill=stroke)
        p.ellipse((bases[1][0] - 2.5, bases[1][1] - 2.5, bases[1][0] + 2.5, bases[1][1] + 2.5),
                  fill=stroke)
        if vertical:
            p.line([bases[0], (bases[1][0] + 10, bases[1][1] - 4)], colour=stroke, width=1.4)
        else:
            p.line([bases[0], (bases[1][0] - 4, bases[1][1] - 10)], colour=stroke, width=1.4)
    elif ctype.startswith("crystal"):
        p.rect((bx1 + 3, by1, bx2 - 3, by2), outline=stroke, fill=p.palette.bg, width=1.3)
        p.line([(bx1, by1 + 3), (bx1, by2 - 3)], colour=stroke, width=1.6)
        p.line([(bx2, by1 + 3), (bx2, by2 - 3)], colour=stroke, width=1.6)
    elif ctype == "battary":
        p.line([(bx1, by1 + 4), (bx2, by1 + 4)], colour=stroke, width=1.8)
        p.line([(bx1 + 5, by1 + 10), (bx2 - 5, by1 + 10)], colour=stroke, width=1.2)
    elif ctype in ("mosfet", "mosfet_npn", "mosfet_pnp", "bjt", "bjt_npn", "bjt_pnp"):
        p.line([(cx, by1), (cx, by2)], colour=stroke, width=1.8)
        p.line([(cx, cy - 5), (bx1, cy - 5)], colour=stroke)
        p.line([(cx, cy + 5), (bx2, cy + 5)], colour=stroke)
        p.line([(cx, cy), (bx2, cy + 5)], colour=stroke)
        p.line([(cx - 6, cy - 2), (cx - 6, cy + 2)], colour=stroke, width=1.6)
    elif ctype == "ant":
        p.line([(cx, by2), (cx, cy)], colour=stroke)
        p.line([(cx - 10, cy - 10), (cx, cy), (cx + 10, cy - 10)], colour=stroke)
    elif ctype in ("buzzer", "motor"):
        p.ellipse((bx1, by1, bx2, by2), outline=stroke, width=1.4)
        p.text(cx, cy, "M" if ctype == "motor" else "BZ", "mono", 9, colour=stroke, anchor="mm")
    elif ctype == "m3\u87ba\u4e1d":
        p.ellipse((bx1, by1, bx2, by2), outline=stroke, width=1.4)
        p.line([(bx1 + 4, cy), (bx2 - 4, cy)], colour=stroke)
    elif ctype == "circle_header":
        p.ellipse((bx1, by1, bx2, by2), outline=stroke, width=1.4)
    elif ctype == "seg":
        for dy in (-7, 0, 7):
            p.line([(bx1 + 2, cy + dy), (bx2 - 2, cy + dy)], colour=stroke, width=1.5)
    elif ctype == "opt":
        p.line([(bx1, by1), (bx1, by2), (bx2, cy), (bx1, by1)], colour=stroke)
        p.line([(bx2, by1), (bx2, by2)], colour=stroke, width=1.5)
    elif ctype == "jumper":
        p.ellipse((bx1, by1, bx2, by2), outline=stroke, width=1.3)
        p.line([(cx, by1), (cx, by2)], colour=stroke)
    elif ctype == "transfomer":
        p.line([(bx1, by1), (bx1, by2)], colour=stroke, width=1.8)
        p.line([(bx2, by1), (bx2, by2)], colour=stroke, width=1.8)
        p.line([(bx1 - 6, by1 + 4), (bx1 - 6, by2 - 4)], colour=stroke)
    elif ctype in ("amp", "amp_3pin", "amp_5pin"):
        p.line([(bx1, by1), (bx1, by2), (bx2, cy), (bx1, by1)], colour=stroke)
    else:
        p.rect((bx1, by1, bx2, by2), outline=stroke, fill=p.palette.bg, width=1.3)

    label = value or ctype
    bb = p.text(cx, by2 + 4 if vertical else cy - horizon - 4, label, "mono", 8,
                colour=p.palette.value_colour(), anchor="ms")
    _emit(p.ctx_text, bb, label, "value", key)
    dbb = p.text(cx, by1 - 12 if vertical else cy + horizon + 13, key, "mono", 9,
                 colour=p.palette.text, anchor="ms")
    _emit(p.ctx_text, dbb, key, "designator", key)

    return Comp(key, ctype, label, (bx1, by1, bx2, by2), pins)


def draw_block(p: Painter, rng, ctype: str, key: str, cx: float, cy: float, value, name) -> Comp:
    """Hierarchical sheet / functional block: a labelled rectangle with a few pins."""
    w = rng.choice([50, 66, 84, 104])
    h = rng.choice([32, 42, 54])
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    p.rect((x1, y1, x2, y2), outline=p.palette.body_line, fill=p.palette.body_fill, width=1.3)
    p.line([(x1, y1 + 11), (x2, y1 + 11)], colour=p.palette.body_line)
    label = name or value or "Block"
    bb = p.text(x1 + 3, y1 + 5, label, "sans", 8, colour=p.palette.text, anchor="lm")
    _emit(p.ctx_text, bb, label, "value", key)
    dbb = p.text(x1 + 3, y1 - 5, key, "mono", 9, colour=p.palette.text, anchor="lm")
    _emit(p.ctx_text, dbb, key, "designator", key)
    pins: list[Pin] = []
    tips_x = [x1 - 10, x2 + 10]
    for i in range(rng.randint(1, 2)):
        yy = y1 + 20 + i * 12
        pins.append(Pin(str(i + 1), "", (tips_x[0], yy), (x1, yy), "L"))
        p.line([(x1, yy), (tips_x[0], yy)], colour=p.palette.pin)
    for i in range(rng.randint(1, 2)):
        yy = y2 - 14 - i * 12
        pins.append(Pin(str(len(pins) + 1), "", (tips_x[1], yy), (x2, yy), "R"))
        p.line([(x2, yy), (tips_x[1], yy)], colour=p.palette.pin)
    return Comp(key, ctype, label, (x1, y1, x2, y2), pins, name=label)


# ---------------------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------------------

IC_TYPES = {"box"}
POWER_TYPES = {"v", "gnd"}
FLAG_TYPES = {"net_input", "net_output", "net_short", "net_bidirection"}
PASSIVE_TYPES = {"r", "c", "l", "d", "led", "esd", "fuse"}
BLOCK_TYPES = {"block"}


def draw_component(p: Painter, rng: random.Random, ctype: str, key: str,
                   cx: float, cy: float, value, name, grid: float = 0.0) -> Comp:
    if ctype in IC_TYPES:
        comp = draw_ic(p, rng, key, cx, cy, value, name)
    elif ctype in POWER_TYPES:
        comp = draw_power(p, rng, ctype, key, cx, cy, value)
    elif ctype in FLAG_TYPES:
        comp = draw_net_flag(p, rng, ctype, key, cx, cy, value)
    elif ctype in PASSIVE_TYPES:
        comp = draw_passive(p, rng, key, cx, cy, value, ctype)
    elif ctype in BLOCK_TYPES:
        comp = draw_block(p, rng, ctype, key, cx, cy, value, name)
    else:
        comp = draw_generic(p, rng, ctype, key, cx, cy, value)
    if grid:
        # Snap every pin tip onto the routing lattice and re-draw the stub to match, so a wire
        # that ends on a lattice node still lands exactly on the pin tip.
        for pin in comp.pins:
            tip = (round(pin.tip[0] / grid) * grid, round(pin.tip[1] / grid) * grid)
            if tip != pin.tip:
                p.line([pin.base, tip], colour=p.palette.pin)
                pin.tip = tip
    return comp


def pick_value(rng: random.Random, ctype: str) -> str | None:
    if ctype in VALUE_BANK:
        return rng.choice(VALUE_BANK[ctype])
    if ctype == "box":
        return rng.choice(IC_NAMES)
    if ctype == "block":
        return rng.choice(["Power Supply", "MCU System", "USB", "Debug", "Radio",
                           "Display", "Sensor", "Audio", "Reset", "Clock"])
    return None
