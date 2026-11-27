"""Обработка спектра звезды и запрос данных в SIMBAD (без Streamlit — удобно тестировать)."""
import re
import urllib.parse
import urllib.request

import numpy as np

# Справочные линии, нм (в воздухе)
LINES = {
    "Hα": 656.3, "Hβ": 486.1, "Hγ": 434.0, "Hδ": 410.2, "Hε": 397.0,
    "He I 447": 447.1, "He I 587": 587.6, "He II 468": 468.6,
    "Ca II K": 393.4, "Ca II H": 396.8, "Ca I 422": 422.7,
    "G (CH)": 430.8, "Mg b": 517.3, "Fe 527": 527.0, "Na D": 589.3,
    "TiO 476": 476.1, "TiO 496": 495.5, "TiO 517": 516.7, "TiO 545": 544.8,
    "TiO 559": 558.6, "TiO 616": 615.9, "TiO 671": 670.5, "TiO 705": 705.4,
    "O₂ B": 686.7, "O₂ A": 759.4,          # атмосфера Земли
}
CLASS_LINES = {
    "O": ["He II 468", "He I 447", "Hβ", "Hγ", "Hδ"],
    "B": ["He I 447", "He I 587", "Hα", "Hβ", "Hγ", "Hδ", "Hε"],
    "A": ["Hα", "Hβ", "Hγ", "Hδ", "Hε", "Ca II K"],
    "F": ["Hα", "Hβ", "Hγ", "Hδ", "Ca II K", "Ca II H", "G (CH)"],
    "G": ["Ca II K", "Ca II H", "G (CH)", "Hβ", "Mg b", "Na D", "Hα"],
    "K": ["Ca II K", "Ca II H", "Ca I 422", "G (CH)", "Mg b", "Na D", "Fe 527"],
    "M": ["TiO 476", "TiO 496", "TiO 517", "TiO 545", "TiO 559", "TiO 616",
          "TiO 671", "TiO 705", "Na D", "Ca I 422"],
}
TELLURIC = ["O₂ B", "O₂ A"]
TEFF = {"O": (45000, 40000), "B": (30000, 15500), "A": (9700, 8200),
        "F": (7200, 6500), "G": (5900, 5600), "K": (5200, 4400), "M": (3850, 3100)}


def expected_lines(sp_type):
    cls = (sp_type or " ").strip()[:1].upper()
    return CLASS_LINES.get(cls, ["Hα", "Hβ", "Hγ", "Na D"]) + TELLURIC


def teff_from_sptype(sp):
    """Грубая оценка Teff по спектральному классу, напр. 'A0Va', 'M1-M2Ia-Iab'."""
    m = re.match(r"\s*([OBAFGKM])\s*(\d(?:\.\d)?)?", sp or "")
    if not m:
        return None
    cls, sub = m.group(1), float(m.group(2) or 5)
    t0, t5 = TEFF[cls]
    if sub <= 5:
        return t0 + (t5 - t0) * sub / 5
    order = "OBAFGKM"
    t10 = TEFF[order[order.index(cls) + 1]][0] if cls != "M" else 2400
    return t5 + (t10 - t5) * (sub - 5) / 5


def planck(nm, T):
    lam = np.asarray(nm, dtype=float) * 1e-9
    h, c, k = 6.626e-34, 2.998e8, 1.381e-23
    return 1.0 / (lam ** 5 * (np.exp(np.clip(h * c / (lam * k * T), None, 700)) - 1))


def query_simbad(name, timeout=15):
    """Спектральный класс, V, координаты из SIMBAD (прямой HTTP-запрос, без astroquery)."""
    script = (
        "output console=off script=off\n"
        'format object "%IDLIST(1)|%SP(S)|%FLUXLIST(V;F)|%COO(d;A D)"\n'
        f"query id {name}\n"
    )
    url = "https://simbad.cds.unistra.fr/simbad/sim-script?" + \
        urllib.parse.urlencode({"script": script})
    req = urllib.request.Request(url, headers={"User-Agent": "star-spectrum-app"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        text = r.read().decode("utf-8", "replace")
    line = next((l for l in text.splitlines() if l.count("|") >= 3), None)
    if not line:
        raise LookupError(f"SIMBAD не нашёл объект «{name}»")
    ident, sp, vmag, coo = [p.strip() for p in line.split("|")[:4]]
    ra, dec = (coo.split() + [None, None])[:2]
    try:
        vmag = float(vmag)
    except ValueError:
        vmag = None
    return {"name": ident, "sp_type": sp, "vmag": vmag, "ra": ra, "dec": dec}


def to_gray(img):
    img = np.asarray(img, dtype=np.float32)
    return img.mean(axis=2) if img.ndim == 3 else img


def find_band_row(gray):
    """Строка с максимальной суммарной яркостью (сглаженная по вертикали)."""
    prof = np.asarray(gray, dtype=np.float32).sum(axis=1)
    prof = np.convolve(prof, np.ones(9) / 9, mode="same")
    return int(np.argmax(prof))


def extract_spectrum(gray, row, half=4):
    """Профиль вдоль полосы минус фон неба (медиана строк выше и ниже полосы)."""
    gray = np.asarray(gray, dtype=np.float32)
    h = gray.shape[0]
    r0, r1 = max(0, row - half), min(h, row + half + 1)
    band = gray[r0:r1].sum(axis=0)
    gap = 3 * half
    bg = [gray[max(0, r0 - gap - half):max(0, r0 - gap)],
          gray[min(h, r1 + gap):min(h, r1 + gap + half)]]
    bg = [b for b in bg if b.size]
    if bg:
        band = band - np.median(np.vstack(bg), axis=0) * (r1 - r0)
    return np.clip(band, 0, None)


class Calibration:
    """nm = a * px + b. Либо нулевой порядок + дисперсия, либо две известные линии."""

    def __init__(self, zero=None, disp=None, points=None):
        self.a = self.b = None
        if points and len(points) == 2:
            (p1, l1), (p2, l2) = points
            if p1 != p2:
                self.a = (l2 - l1) / (p2 - p1)
                self.b = l1 - self.a * p1
        elif zero is not None and disp:
            self.a, self.b = disp, -disp * zero

    @property
    def ok(self):
        return self.a is not None

    def to_nm(self, px):
        return self.a * np.asarray(px) + self.b if self.ok else np.asarray(px)


def process(gray, row, half, cal, wl_range=(370, 780)):
    """Возвращает (x, y_норм) — x в нм, если есть калибровка, иначе в пикселях."""
    spec = extract_spectrum(gray, row, half)
    x = cal.to_nm(np.arange(spec.size))
    if cal.ok:
        m = (x >= wl_range[0]) & (x <= wl_range[1])
        x, spec = x[m], spec[m]
    y = spec / spec.max() if spec.size and spec.max() > 0 else spec
    return x, y
