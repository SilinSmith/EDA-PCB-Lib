"""Synthetic schematic pages: placement, orthogonal wiring, sheet furniture.

Everything produced here is in final-image pixels with a **top-left** origin -- the same frame
the YOLO labels want.  ``emit.py`` performs the single flip into the competition's bottom-left
origin JSON.

Wiring is a genuine maze router (A* over a lattice that treats component bodies as obstacles),
so a wire never crosses a symbol body and every polyline ends exactly on a pin tip.  That is
what makes the component / pin / net labels free and correct.
"""

from __future__ import annotations

import heapq
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .style import PALETTES, Palette, Painter
from .symbols import (FLAG_TYPES, NET_LABEL_BANK, PREFIX, Comp, TextItem,
                      draw_component, pick_value)

# ---------------------------------------------------------------------------------------
# naming


EMPTY = "\u2205"  # the corpus' unreproducible designator prefix (a bare circle-slash)


class Namer:
    """Designators that look like the real corpus, `EMPTY`-prefixed power/ground keys included."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.count: Counter = Counter()

    def next(self, ctype: str) -> str:
        prefix = PREFIX.get(ctype, "X")
        if prefix == EMPTY:
            self.count[EMPTY] += 1
            return EMPTY + str(100 + self.count[EMPTY])
        self.count[prefix] += 1
        return prefix + str(self.count[prefix])


# Corpus frequencies, measured over all 220 real cases: 43.7 components and 48.8 nets per sheet.
# Matching the mix matters more than matching the count -- a sheet that is 60% resistors teaches
# the detector the wrong prior.  Weight is "expected components per sheet".
TYPE_MIX = [
    ("gnd", 10.1), ("c", 7.6), ("r", 7.4), ("v", 5.2), ("box", 4.5), ("net_input", 1.5),
    ("d", 0.90), ("led", 0.85), ("pin", 0.80), ("net_short", 0.74), ("switch", 0.72),
    ("l", 0.52), ("net_output", 0.37), ("other", 0.31), ("mosfet", 0.30),
    ("m3\u87ba\u4e1d", 0.26), ("bjt", 0.23), ("crystal_2pin", 0.20), ("testpoint", 0.20),
    ("battary", 0.11), ("buzzer", 0.10), ("fuse", 0.09), ("amp_5pin", 0.05),
    ("amp_3pin", 0.05), ("crystal_4pin", 0.04), ("jumper", 0.04), ("motor", 0.04),
    ("mosfet_npn", 0.03), ("crystal_3pin", 0.03), ("esd", 0.03), ("dc", 0.03),
    ("transfomer", 0.02), ("circle_header", 0.02), ("ant", 0.02), ("bjt_npn", 0.01),
    ("bjt_pnp", 0.01), ("opt", 0.01), ("seg", 0.005), ("crystal", 0.005),
    ("mosfet_pnp", 0.005), ("amp", 0.05), ("block", 0.06),
]
_MIX_TYPES = [t for t, _ in TYPE_MIX]
_MIX_CUM: list[float] = []
_acc = 0.0
for _t, _w in TYPE_MIX:
    _acc += _w
    _MIX_CUM.append(_acc)


def _pick_type(rng: random.Random) -> str:
    r = rng.random() * _MIX_CUM[-1]
    for t, c in zip(_MIX_TYPES, _MIX_CUM):
        if r <= c:
            return t
    return _MIX_TYPES[-1]


from .profiles import SHEET_PROFILES

# Canvas, and the component count that really goes with it, both come from `profiles.py`, which
# is a dump of the measured corpus.  Size and density are NOT independent in real sheets, so we
# sample a whole (w, h, n) triple.  Kept as a name for callers that only want the geometry.
CANVAS_SIZES = [(w, h) for w, h, _ in SHEET_PROFILES]


# ---------------------------------------------------------------------------------------
# data model


@dataclass
class Net:
    key: str
    label: str | None
    pins: list[tuple[str, str]] = field(default_factory=list)
    edges: list[list[tuple[float, float]]] = field(default_factory=list)

    @property
    def hypergraph(self) -> str:
        return "(" + ",".join(f"{c}.{p}" for c, p in self.pins) + ")"


@dataclass
class Scene:
    w: int
    h: int
    palette_name: str
    comps: list[Comp]
    texts: list[TextItem]
    nets: list[Net]
    image: Any = None

    @property
    def palette(self) -> Palette:
        return PALETTES[self.palette_name]

    def comp(self, key: str) -> Comp | None:
        for c in self.comps:
            if c.key == key:
                return c
        return None


@dataclass
class GenConfig:
    width: int | None = None
    height: int | None = None
    palette: str | None = None
    seed: int | None = None
    ss: int = 3
    grid: float = 6.0
    # None => derive the component count from sheet area at the corpus' 50.1 components/Mpx.
    # A fixed count makes the large canvases come out visibly empty.
    n_comp: tuple[int, int] | None = None
    sheet_border: bool = True
    title_block: bool = True
    noise: float = 0.0


# ---------------------------------------------------------------------------------------
# maze router


class Router:
    """A* on a lattice; component bodies are obstacles, a pin tip is always a valid target."""

    DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))

    def __init__(self, w: float, h: float, step: float, blocked: set[tuple[int, int]]):
        self.step = step
        self.nx = int(w // step) + 1
        self.ny = int(h // step) + 1
        self.blocked = blocked
        self.used: Counter = Counter()

    def node(self, pt) -> tuple[int, int]:
        return (int(round(pt[0] / self.step)), int(round(pt[1] / self.step)))

    def point(self, n) -> tuple[float, float]:
        return (n[0] * self.step, n[1] * self.step)

    def free(self, n) -> bool:
        return 0 <= n[0] < self.nx and 0 <= n[1] < self.ny and n not in self.blocked

    def _h(self, n, goal) -> float:
        return abs(n[0] - goal[0]) + abs(n[1] - goal[1])

    def route(self, start, goals) -> list[tuple[float, float]] | None:
        cleaned = set()
        for g in goals:
            if self.free(g):
                cleaned.add(g)
            else:
                nf = self._nearest_free(g)
                if nf is not None:
                    cleaned.add(nf)
        goals = cleaned
        if not goals:
            return None
        start = self.node(start)
        if not self.free(start):
            start = self._nearest_free(start)
            if start is None:
                return None
        if start in goals:
            return [self.point(start)]
        anchor = min(goals, key=lambda g: self._h(start, g))

        dist = {(start, (0, 0)): 0.0}
        prev: dict = {}
        heap = [(self._h(start, anchor), 0.0, start, (0, 0))]
        found = None
        pops = 0
        while heap:
            _, g, n, d = heapq.heappop(heap)
            pops += 1
            if pops > 120000:
                break
            if g > dist.get((n, d), 1e18):
                continue
            if n in goals and (n, d) != (start, (0, 0)):
                found = (n, d)
                break
            for nd in self.DIRS:
                m = (n[0] + nd[0], n[1] + nd[1])
                if not self.free(m):
                    continue
                ng = g + 1.0
                if d != (0, 0) and nd != d:
                    ng += 1.8
                ng += 0.7 * self.used[m]
                st = (m, nd)
                if ng < dist.get(st, 1e18):
                    dist[st] = ng
                    prev[st] = (n, d)
                    heapq.heappush(heap, (ng + self._h(m, anchor), ng, m, nd))
        if found is None:
            return None

        path = [found]
        while path[-1] != (start, (0, 0)):
            path.append(prev[path[-1]])
        nodes = [p[0] for p in reversed(path)]
        for n in nodes:
            self.used[n] += 1
        # drop collinear intermediates so each edge is a clean polyline
        pts = [self.point(nodes[0])]
        for i in range(1, len(nodes) - 1):
            a, b, c = nodes[i - 1], nodes[i], nodes[i + 1]
            if (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1]):
                pts.append(self.point(b))
        pts.append(self.point(nodes[-1]))
        return pts

    def _nearest_free(self, n, radius: int = 6):
        for r in range(1, radius + 1):
            ring = []
            for dx in range(-r, r + 1):
                ring += [(n[0] + dx, n[1] - r), (n[0] + dx, n[1] + r)]
            for dy in range(-r + 1, r):
                ring += [(n[0] - r, n[1] + dy), (n[0] + r, n[1] + dy)]
            for cand in ring:
                if self.free(cand):
                    return cand
        return None


# ---------------------------------------------------------------------------------------
# placement


def _comp_rng(seed, ctype: str, key: str) -> random.Random:
    """A component's own random stream, derived from its designator.

    Sizing and drawing MUST replay this same stream.  Symbol geometry -- orientation, pin count,
    pin pitch, stub length, body size -- is drawn from it, so measuring on the global stream and
    then drawing later on an advanced one produced a symbol rotated and resized relative to the
    space reserved for it (an `r` measured 7x18 and drawn 23x8).  Saving and restoring the global
    stream does not help: the draws in between still move it.
    """
    return random.Random(f"{seed}|{ctype}|{key}")


def _measure(rng: random.Random, palette: Palette, ctype: str, key: str, value, name,
             grid: float) -> tuple[float, float, float, float]:
    """Footprint a symbol will occupy: its body plus every caption it draws.

    The reserved box must cover the captions too -- a designator is ink like any other, and
    reserving only the body is what let symbols and their labels land on top of each other.
    `rng` is the component's own stream (see `_comp_rng`), so nothing needs restoring here.
    """
    scratch = Painter(900, 900, palette, ss=1)
    sink: list[TextItem] = []
    scratch.ctx_text = sink
    comp = draw_component(scratch, rng, ctype, key, 450, 450, value, name, grid=grid)
    x1, y1, x2, y2 = comp.bbox
    for t in sink:
        x1 = min(x1, t.bbox[0])
        y1 = min(y1, t.bbox[1])
        x2 = max(x2, t.bbox[2])
        y2 = max(y2, t.bbox[3])
    return (x1 - 450, y1 - 450, x2 - 450, y2 - 450)


def _mk_specs(rng: random.Random, n: int) -> list[tuple[str, str]]:
    """`n` (ctype, value) pairs sampled from the corpus mix. ICs are stably sorted to the front
    so they get the roomy spots before the sheet fills up with two-pin parts."""
    specs: list[tuple[str, str]] = [(_pick_type(rng), None) for _ in range(n)]
    specs.sort(key=lambda s: 0 if s[0] in ("box", "block") else 1)
    return specs


def _place(rng: random.Random, palette: Palette, specs, cfg: GenConfig, w: int, h: int,
           namer: Namer, extra=()):
    """Random scatter on a coarse lattice, rejecting overlaps. Returns (ctype, key, value, name, cx, cy)."""
    lane = max(cfg.grid * 5, 30.0)
    margin = 30.0
    cols = max(1, int((w - 2 * margin) // lane))
    rows = max(1, int((h - 2 * margin) // lane))
    occupied = list(extra)
    anchors: list[tuple[float, float]] = []
    out = []
    for ctype, _ in specs:
        key = namer.next(ctype)
        value = pick_value(rng, ctype)
        name = value if ctype == "box" else None
        rw = _measure(_comp_rng(cfg.seed, ctype, key), palette, ctype, key, value, name, cfg.grid)
        sw, sh = rw[2] - rw[0], rw[3] - rw[1]
        cells = [(c, r) for c in range(cols) for r in range(rows)]
        if anchors and rng.random() < 0.68:
            # Real sheets are clusters of parts joined by short wires, not an even stipple across
            # the page -- and an even stipple forces every net to cross the sheet.  Bias most
            # placements towards a part that is already down, keeping the cell order random so
            # the cluster does not grow in one direction.
            ax, ay = anchors[rng.randrange(len(anchors))]
            cells.sort(key=lambda cr: abs(margin + cr[0] * lane + lane / 2 - ax)
                                      + abs(margin + cr[1] * lane + lane / 2 - ay))
            head = cells[:110]
            rng.shuffle(head)
            cells = head + cells[110:]
        else:
            rng.shuffle(cells)
        best = None
        for c, r in cells[:400]:
            cx = margin + c * lane + lane / 2
            cy = margin + r * lane + lane / 2
            bx1, by1 = cx + rw[0], cy + rw[1]
            bx2, by2 = cx + rw[2], cy + rw[3]
            if bx1 < 12 or by1 < 12 or bx2 > w - 12 or by2 > h - 12:
                continue
            # the reserved box already covers the captions, so only a small breath between
            # symbols is needed here
            pad = (6.0, 6.0, 6.0, 6.0)
            if any(not (bx2 + pad[2] < p[0] or bx1 - pad[0] > p[2] or
                        by2 + pad[3] < p[1] or by1 - pad[1] > p[3]) for p in occupied):
                continue
            best = (cx, cy, bx1, by1, bx2, by2)
            break
        if best is None:
            continue
        cx, cy, bx1, by1, bx2, by2 = best
        occupied.append((bx1, by1, bx2, by2))
        anchors.append((cx, cy))
        out.append((ctype, key, value, name, cx, cy))
    return out


# ---------------------------------------------------------------------------------------
# nets


def _build_nets(rng: random.Random, comps: list[Comp]) -> list[Net]:
    tips = {(c.key, p.number): p.tip for c in comps for p in c.pins}
    gnds = [c for c in comps if c.ctype == "gnd"]
    powers = [c for c in comps if c.ctype == "v"]
    flags = [c for c in comps if c.ctype in FLAG_TYPES]
    body = [c for c in comps if c.ctype not in ("gnd", "v") and c.ctype not in FLAG_TYPES]

    # Only body pins go in the shared pool: a power / ground / flag symbol's own pin is already
    # the anchor of its own net, and letting `nearest` hand it out again would put one pin in two
    # nets (which is how the pin count came out above 100% once).
    pool: list[tuple[str, str]] = [(c.key, p.number) for c in body for p in c.pins]
    rng.shuffle(pool)
    nets: list[Net] = []

    def nearest(anchor: tuple[str, str], k: int) -> list[tuple[str, str]]:
        """Pull up to k pins off the pool, biased towards the ones near `anchor`.

        Real sheets are drawn so that connected parts sit together -- the corpus' median wire
        segment is 22 px.  Taking pins at random instead makes wires cross the whole page,
        which is the single most obvious way this generator can look wrong.
        """
        if not pool or k <= 0:
            return []
        ax, ay = tips.get(anchor, (0.0, 0.0))

        def cost(i: int) -> float:
            x, y = tips[pool[i]]
            return (abs(x - ax) + abs(y - ay)) * (0.55 + rng.random() * 0.9)

        order = sorted(range(len(pool)), key=cost)
        chosen = [pool[i] for i in order[:k]]
        for i in sorted(order[:k], reverse=True):
            del pool[i]
        return chosen

    # Corpus: gnd-only nets average 1.38 edges and 33.5% of all nets hold a single pin, so a
    # ground symbol is its own little net with zero to two neighbours -- not one GND spanning
    # the sheet.
    for i, g in enumerate(gnds):
        anchor = (g.key, g.pins[0].number)
        members = [anchor] + nearest(anchor, rng.choice([0, 0, 1, 1, 2]))
        nets.append(Net("GND" if i == 0 else f"net_g{i}", "GND" if i == 0 else None, members))

    for v in powers:
        anchor = (v.key, v.pins[0].number)
        nets.append(Net(v.value or "VCC", v.value,
                        [anchor] + nearest(anchor, rng.choice([0, 1, 1, 2]))))

    for f in flags:
        anchor = (f.key, f.pins[0].number)
        nets.append(Net(f.value or "NET", f.value,
                        [anchor] + nearest(anchor, rng.choice([0, 1, 2]))))

    while len(pool) >= 2:
        # pop the anchor out first: leaving it in the pool lets `nearest` hand it straight back,
        # which is how one pin ended up listed in a net twice
        anchor = pool.pop(0)
        members = [anchor] + nearest(anchor, rng.randint(1, min(3, len(pool))))
        if len(members) < 2:
            break
        nets.append(Net(rng.choice(NET_LABEL_BANK), None, members))

    nets = [n for n in nets if n.pins]

    # give anonymous nets the corpus' `net_N` shape so the JSON keys look right
    used: set[str] = set()
    for i, n in enumerate(nets, 1):
        if n.label and n.label not in used:
            n.key = n.label
        elif n.key in used or not n.key:
            n.key = f"net_{i}"
        used.add(n.key)
    return nets


def _segments(path: list[tuple[float, float]]) -> list[list[tuple[float, float]]]:
    """Corpus fact: all 33037 edges in all 220 real cases are 2-point segments. A routed
    polyline therefore has to be cut at every corner rather than stored whole, or the JSON
    shape -- and everything net_line_f1 measures on it -- is wrong."""
    return [[path[i], path[i + 1]] for i in range(len(path) - 1) if path[i] != path[i + 1]]


def _route_nets(nets: list[Net], comps: list[Comp], w: int, h: int,
                cfg: GenConfig, title_box) -> None:
    blocked: set[tuple[int, int]] = set()
    step = cfg.grid
    nx, ny = int(w // step) + 1, int(h // step) + 1
    for c in comps:
        x1, y1, x2, y2 = c.inflated(5)
        for ix in range(int(x1 // step), int(x2 // step) + 2):
            for iy in range(int(y1 // step), int(y2 // step) + 2):
                blocked.add((ix, iy))
    if title_box:
        x1, y1, x2, y2 = title_box
        for ix in range(int(x1 // step) - 1, int(x2 // step) + 2):
            for iy in range(int(y1 // step) - 1, int(y2 // step) + 2):
                blocked.add((ix, iy))
    for ix in range(nx):
        for iy in range(0, 2):
            blocked.add((ix, iy))
            blocked.add((ix, ny - 1 - iy))
    for iy in range(ny):
        for ix in range(0, 2):
            blocked.add((ix, iy))
            blocked.add((nx - 1 - ix, iy))

    tips = {(c.key, p.number): p.tip for c in comps for p in c.pins}
    bases = {(c.key, p.number): p.base for c in comps for p in c.pins}
    router = Router(w, h, step, blocked)

    def stub(key: tuple[str, str]) -> list[list[tuple[float, float]]]:
        """The short segment a pin draws when its net needs no travel: away from the body."""
        tip = tips[key]
        bx, by = bases.get(key, tip)
        dx, dy = tip[0] - bx, tip[1] - by
        n = (dx * dx + dy * dy) ** 0.5 or 1.0
        return _segments([tip, (tip[0] + dx / n * 14.0, tip[1] + dy / n * 14.0)])

    for net in nets:
        pts = [tips.get(k) for k in net.pins if k in tips]
        pts = [p for p in pts if p is not None]
        anchored = [k for k in net.pins if k in tips]
        if not pts:
            net.edges = []
            continue
        if len(pts) == 1:
            # A one-pin net still draws: in the corpus every net has at least one edge and 33.5%
            # of nets hold exactly one pin, whose edge is the stub leaving the pin.
            net.edges = stub(anchored[0])
            continue
        order = [pts[0]]
        rest = pts[1:]
        while rest:
            last = order[-1]
            rest.sort(key=lambda p: abs(p[0] - last[0]) + abs(p[1] - last[1]))
            order.append(rest.pop(0))
        tree = {router.node(order[0])}
        for target in order[1:]:
            tn = router.node(target)
            path = router.route(target, tree)
            if path is None:
                continue
            net.edges.extend(_segments(path))
            for p in path[1:-1]:
                tree.add(router.node(p))
            tree.add(tn)
        if not net.edges:
            # either every pin snapped onto one lattice node, or the whole cluster sat inside an
            # obstacle; either way the net still owns one edge in the corpus
            net.edges = stub(anchored[0])


def junction_points(nets: list[Net]) -> list[tuple[float, float]]:
    """A dot belongs wherever three or more wire ends meet.

    Two cases, and a real sheet draws a dot for both: an end shared by >= 3 segments, and an end
    that lands on the *middle* of another wire (a T).  Counting only shared ends misses the
    second, which is why the synthetic sheets came out under-dotted against the corpus
    (6.5/sheet vs 20.1/sheet).
    """
    segs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    ends: Counter = Counter()
    for net in nets:
        for edge in net.edges:
            for a, b in zip(edge, edge[1:]):
                segs.append((a, b))
                ends[(round(a[0]), round(a[1]))] += 1
                ends[(round(b[0]), round(b[1]))] += 1

    def is_t(pt: tuple[float, float], a, b) -> bool:
        px, py = pt
        ax, ay = a
        bx, by = b
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        if abs(cross) > 0.8:
            return False
        if not (min(ax, bx) - 0.5 <= px <= max(ax, bx) + 0.5):
            return False
        if not (min(ay, by) - 0.5 <= py <= max(ay, by) + 0.5):
            return False
        return (abs(px - ax) > 1.0 or abs(py - ay) > 1.0) and \
               (abs(px - bx) > 1.0 or abs(py - by) > 1.0)

    out: list[tuple[float, float]] = []
    for (x, y), d in ends.items():
        pt = (float(x), float(y))
        if d >= 3 or any(is_t(pt, a, b) for a, b in segs):
            out.append(pt)
    return out


# ---------------------------------------------------------------------------------------
# sheet furniture


def _draw_sheet(p: Painter, w: int, h: int, rng: random.Random, cfg: GenConfig):
    pal = p.palette
    if not cfg.sheet_border:
        return None
    line = pal.sheet_border or pal.grid or pal.body_line
    m = 10
    p.rect((m, m, w - m, h - m), outline=line, width=1.0)
    p.rect((m + 4, m + 4, w - m - 4, h - m - 4), outline=line, width=0.8)
    if not cfg.title_block:
        return None
    bw, bh = 300, 96
    x1, y1 = w - m - 4 - bw, h - m - 4 - bh
    p.rect((x1, y1, w - m - 4, h - m - 4), outline=line, width=1.0)
    for i in (1, 2, 3):
        p.line([(x1, y1 + i * bh / 4), (x1 + bw, y1 + i * bh / 4)], colour=line, width=0.8)
    p.line([(x1 + bw * 0.55, y1), (x1 + bw * 0.55, y1 + bh)], colour=line, width=0.8)
    rows = [("Title", "Schematic"), ("Size", rng.choice(["A4", "A3", "A4"])),
            ("Date", f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"),
            ("Rev", f"V{rng.randint(1, 3)}.{rng.randint(0, 9)}")]
    for i, (k, v) in enumerate(rows):
        yy = y1 + i * bh / 4 + bh / 8
        p.text(x1 + 5, yy, k, "sans", 8, colour=line, anchor="lm")
        p.text(x1 + bw * 0.57, yy, v, "sans", 8, colour=line, anchor="lm")
    return (x1, y1, w - m - 4, h - m - 4)


# ---------------------------------------------------------------------------------------
# entry point


def generate(cfg: GenConfig | None = None) -> Scene:
    cfg = cfg or GenConfig()
    rng = random.Random(cfg.seed)
    if cfg.width and cfg.height:
        w, h = cfg.width, cfg.height
        profile_n = None
    else:
        w, h, profile_n = rng.choice(SHEET_PROFILES)
    pname = cfg.palette or rng.choice(list(PALETTES))
    palette = PALETTES[pname]

    namer = Namer(rng)
    # Component count.  When a measured profile was drawn, use *its* count -- width, height and
    # density are correlated in the corpus (a 5100x3296 datasheet scan carries only ~66
    # components because most of the page is a figure, while a 1190x845 CAD sheet carries ~56).
    # Only an explicitly sized sheet falls back to the ~50.1 components per megapixel average.
    if cfg.n_comp is not None:
        n_target = rng.randint(*cfg.n_comp)
    elif profile_n is not None:
        n_target = max(6, round(profile_n * rng.uniform(0.82, 1.20)))
    else:
        n_target = max(10, round((w * h) / 1e6 * 50.1 * rng.uniform(0.78, 1.28)))
    specs = _mk_specs(rng, n_target)

    p = Painter(w, h, palette, ss=cfg.ss)
    texts: list[TextItem] = []
    p.ctx_text = texts
    title_box = _draw_sheet(p, w, h, rng, cfg)

    # background grid, as most of the real sheets carry one
    if palette.grid and rng.random() < 0.7:
        g = 24
        for x in range(g, w, g):
            p.line([(x, 0), (x, h)], colour=palette.grid, width=0.4)
        for y in range(g, h, g):
            p.line([(0, y), (w, y)], colour=palette.grid, width=0.4)

    spots = _place(rng, palette, specs, cfg, w, h, namer, extra=[title_box] if title_box else [])

    comps: list[Comp] = []
    for ctype, key, value, name, cx, cy in spots:
        # replay exactly the stream that sized the reserved footprint in _place
        comps.append(draw_component(p, _comp_rng(cfg.seed, ctype, key), ctype, key, cx, cy,
                                    value, name, grid=cfg.grid))

    nets = _build_nets(rng, comps)
    _route_nets(nets, comps, w, h, cfg, title_box)

    for net in nets:
        for edge in net.edges:
            if len(edge) >= 2:
                p.line(edge, colour=palette.wire, width=1.2)
    for x, y in junction_points(nets):
        p.dot(x, y, colour=palette.wire)

    sc = Scene(w, h, pname, comps, texts, nets)
    sc.image = p.finish(noise=cfg.noise)
    return sc
