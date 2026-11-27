"""Пересобирает data/templates.csv: длина волны + спектр каждой звезды из data/stars.csv.

    python build_database.py
"""
import os

import numpy as np

import star_db

catalog = star_db.load_catalog()
cols, header = [star_db.GRID], ["wavelength_nm"]
for star in catalog:
    y, kind = star_db.template(star)
    cols.append(y)
    header.append(star["id"])
    print(f"{star['name_ru']:<22} {star['sp_type']:<14} {kind}")

out = os.path.join(star_db.HERE, "data", "templates.csv")
np.savetxt(out, np.column_stack(cols), delimiter=",", header=",".join(header),
           comments="", fmt="%.4f")
print(f"\nГотово: {out} ({len(catalog)} звёзд, {star_db.GRID.size} точек)")
