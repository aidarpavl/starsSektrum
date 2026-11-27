"""База известных звёзд и модельные спектры-шаблоны.

Шаблон — это кривая Планка для Teff звезды, умноженная на профили линий и полос,
сила которых зависит от спектрального класса, а ширина — от класса светимости.
Это модель для подбора класса и наглядного сравнения, а не наблюдённый спектр.
Чтобы подставить реальный спектр, положите файл data/observed/<id>.csv
(столбцы wavelength_nm,flux) — он будет использоваться вместо модели.
"""
import csv
import os
import re

import numpy as np

import spectrum_core as sc

HERE = os.path.dirname(os.path.abspath(__file__))
CATALOG = os.path.join(HERE, "data", "stars.csv")
OBSERVED_DIR = os.path.join(HERE, "data", "observed")
GRID = np.arange(370.0, 780.01, 0.25)          # нм

ORDER = "OBAFGKM"
LUM_GRAVITY = {"V": 1.0, "IV": 0.8, "III": 0.5, "II": 0.3, "I": 0.15}


def load_catalog(path=CATALOG):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("teff_k", "radius_rsun", "vmag", "dist_ly"):
            try:
                r[k] = float(r[k])
            except (ValueError, TypeError):
                r[k] = None
    return rows


def find_star(catalog, name):
    key = (name or "").strip().lower()
    for r in catalog:
        if key in {r["id"].lower(), r["name"].lower(), r["name_ru"].lower(),
                   (r["simbad_id"] or "").lower()}:
            return r
    return None


def parse_sptype(sp):
    """'M1-M2Ia-Iab' -> ('M', 1.0, 'I'); класс светимости по умолчанию V."""
    m = re.match(r"\s*([OBAFGKM])\s*(\d(?:\.\d)?)?", sp or "")
    if not m:
        return None
    cls, sub = m.group(1), float(m.group(2) or 5)
    rest = sp[m.end():]
    lm = re.search(r"(IV|V|I{1,3})", rest)
    lum = lm.group(1) if lm else "V"
    return cls, sub, lum


def _gauss(wl, c, s):
    return np.exp(-0.5 * ((wl - c) / s) ** 2)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _band(wl, head, length):
    """Молекулярная полоса: резкий край со стороны синего, плавное ослабление к красному."""
    t = (wl - head) / length
    shape = np.where(t >= 0, np.clip(1 - t, 0, 1), np.exp(-0.5 * (t * length / 0.6) ** 2))
    return shape


def synthetic_spectrum(sp_type, teff=None, wl=GRID):
    """Модельный спектр, нормированный на максимум 1."""
    parsed = parse_sptype(sp_type)
    if parsed is None:
        raise ValueError(f"не разобран спектральный класс: {sp_type!r}")
    cls, sub, lum = parsed
    teff = teff or sc.teff_from_sptype(sp_type)
    p = ORDER.index(cls) + sub / 10.0          # 0 (O0) … 6.9 (M9)
    g = LUM_GRAVITY.get(lum, 1.0)

    flux = sc.planck(wl, teff)
    flux = flux / flux.max()
    absorb = np.ones_like(wl)

    def line(name, depth, sigma):
        nonlocal absorb
        if depth > 0.003:
            absorb *= 1 - min(depth, 0.95) * _gauss(wl, sc.LINES[name], sigma)

    # водород: максимум у A0, шире у карликов
    h_depth = 0.6 * np.exp(-((p - 2.0) / 1.3) ** 2)
    h_sigma = 0.35 + 1.6 * g
    for n in ("Hα", "Hβ", "Hγ", "Hδ", "Hε"):
        line(n, h_depth, h_sigma)
    # гелий: He II у O, He I у ранних B
    line("He II 468", 0.25 * _sigmoid((0.8 - p) / 0.2), 0.4)
    he1 = 0.25 * np.exp(-((p - 1.2) / 0.5) ** 2)
    line("He I 447", he1, 0.4)
    line("He I 587", he1 * 0.8, 0.4)
    # Ca II H, K — растут от F к K
    cahk = 0.8 * _sigmoid((p - 3.0) / 0.5)
    line("Ca II K", cahk, 0.6 + 0.4 * g)
    line("Ca II H", cahk * 0.9, 0.6 + 0.4 * g)
    # металлы — с поздних F
    met = _sigmoid((p - 4.0) / 0.6)
    line("G (CH)", 0.35 * met * np.exp(-((p - 5.0) / 1.5) ** 2) / 0.8, 0.5)
    line("Mg b", 0.35 * met, 0.5)
    line("Fe 527", 0.2 * met, 0.3)
    line("Na D", 0.25 * met + 0.25 * _sigmoid((p - 6.0) / 0.4), 0.45)
    line("Ca I 422", 0.3 * _sigmoid((p - 5.0) / 0.5), 0.35)
    # TiO — только у M, сила растёт с подклассом
    if cls == "M" or p > 5.8:
        tio = _sigmoid((p - 6.0) / 0.25) * min(0.85, 0.35 + 0.08 * (sub if cls == "M" else 0))
        for name, length in (("TiO 476", 10), ("TiO 496", 12), ("TiO 517", 14),
                             ("TiO 545", 12), ("TiO 559", 15), ("TiO 616", 25),
                             ("TiO 671", 25), ("TiO 705", 35)):
            absorb *= 1 - tio * _band(wl, sc.LINES[name], length)
    return flux * absorb


def observed_spectrum(star_id, wl=GRID):
    path = os.path.join(OBSERVED_DIR, f"{star_id}.csv")
    if not os.path.exists(path):
        return None
    data = np.genfromtxt(path, delimiter=",", names=True)
    names = data.dtype.names
    y = np.interp(wl, data[names[0]], data[names[1]], left=np.nan, right=np.nan)
    return y / np.nanmax(y)


def template(star, wl=GRID):
    """(спектр, 'наблюдение' | 'модель') для записи каталога."""
    obs = observed_spectrum(star["id"], wl)
    if obs is not None:
        return obs, "наблюдение"
    return synthetic_spectrum(star["sp_type"], star["teff_k"], wl), "модель"


# ---------------- Сравнение ----------------
def smooth(y, step_nm, fwhm_nm):
    """Гауссово сглаживание на равномерной сетке до разрешения fwhm_nm."""
    if fwhm_nm <= 0 or step_nm <= 0:
        return y
    s = fwhm_nm / 2.355 / step_nm
    if s < 0.5:
        return y
    k = np.arange(-int(4 * s) - 1, int(4 * s) + 2)
    ker = np.exp(-0.5 * (k / s) ** 2)
    ker /= ker.sum()
    pad = len(k) // 2
    yp = np.pad(y, pad, mode="edge")
    return np.convolve(yp, ker, mode="same")[pad:-pad]


def lines_only(y, step_nm, width_nm=40):
    """Убираем континуум (на него сильно влияет чувствительность камеры) — остаются линии."""
    cont = smooth(y, step_nm, width_nm)
    return y / np.where(cont > 1e-6, cont, 1e-6) - 1


def match(x, y, catalog, fwhm_nm=2.0):
    """Корреляция линий измеренного спектра с каждым шаблоном. Возвращает отсортированный список."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 20:
        return []
    step = float(np.median(np.diff(x)))
    ym = lines_only(y, step)
    out = []
    for star in catalog:
        t, kind = template(star, x)
        t = smooth(t, step, fwhm_nm)
        tm = lines_only(t, step)
        ok = np.isfinite(tm) & np.isfinite(ym)
        if ok.sum() < 20 or np.std(tm[ok]) == 0 or np.std(ym[ok]) == 0:
            continue
        r = float(np.corrcoef(ym[ok], tm[ok])[0, 1])
        out.append({"star": star, "r": r, "kind": kind})
    return sorted(out, key=lambda d: -d["r"])
