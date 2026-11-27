"""Видимый цвет звезды: спектр -> CIE XYZ -> sRGB.

Кривые чувствительности глаза CIE 1931 взяты в виде аналитической аппроксимации
Wyman, Sloan & Shirley (2013), поэтому таблицы не нужны.
"""
import numpy as np


def _g(x, mu, s1, s2):
    s = np.where(x < mu, s1, s2)
    return np.exp(-0.5 * ((x - mu) / s) ** 2)


def cmf(wl):
    x = 1.056 * _g(wl, 599.8, 37.9, 31.0) + 0.362 * _g(wl, 442.0, 16.0, 26.7) \
        - 0.065 * _g(wl, 501.1, 20.4, 26.2)
    y = 0.821 * _g(wl, 568.8, 46.9, 40.5) + 0.286 * _g(wl, 530.9, 16.3, 31.1)
    z = 1.217 * _g(wl, 437.0, 11.8, 36.0) + 0.681 * _g(wl, 459.0, 26.0, 13.8)
    return x, y, z


_M = np.array([[3.2406, -1.5372, -0.4986],
               [-0.9689, 1.8758, 0.0415],
               [0.0557, -0.2040, 1.0570]])


def _gamma(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def xyz_to_srgb(X, Y, Z):
    rgb = _M @ np.array([X, Y, Z], dtype=float)
    rgb = rgb - min(0.0, rgb.min())          # вне гаммы sRGB — сдвиг к белому
    rgb = rgb / rgb.max() if rgb.max() > 0 else rgb
    return _gamma(rgb)


def spectrum_to_rgb(wl, flux):
    """Цвет света со спектром flux(wl), яркость нормирована (самый яркий канал = 1)."""
    wl = np.asarray(wl, float)
    flux = np.asarray(flux, float)
    x, y, z = cmf(wl)
    X, Y, Z = (np.trapezoid(flux * c, wl) for c in (x, y, z))
    return xyz_to_srgb(X, Y, Z)


def to_hex(rgb):
    return "#" + "".join(f"{int(round(float(c) * 255)):02x}" for c in rgb)


def wavelength_colors(wl):
    """Цвет каждой длины волны (для «радуги» спектра), массив N×3 в 0..1."""
    x, y, z = cmf(np.asarray(wl, float))
    xyz = np.vstack([x, y, z])
    rgb = _M @ xyz
    rgb = np.clip(rgb, 0, None)
    rgb = rgb / rgb.max(axis=0, keepdims=True).clip(1e-9)
    # яркость по краям видимого диапазона падает, как и чувствительность глаза
    lum = np.clip(y / y.max(), 0, 1) ** 0.5
    return _gamma(rgb * lum).T
