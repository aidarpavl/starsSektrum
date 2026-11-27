"""Каталог известных звёзд и их спектры-шаблоны."""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import spectrum_core as sc
import star_color
import star_db

st.set_page_config(page_title="База звёзд", page_icon="📚", layout="wide")
st.title("📚 База спектров известных звёзд")


@st.cache_data
def load():
    cat = star_db.load_catalog()
    rows = []
    for s in cat:
        y, kind = star_db.template(s)
        rows.append({
            "Звезда": s["name_ru"], "Name": s["name"], "Класс": s["sp_type"],
            "Teff, K": s["teff_k"], "R, R☉": s["radius_rsun"], "V": s["vmag"],
            "Расст., св. лет": s["dist_ly"], "Созвездие": s["constellation"],
            "Цвет": star_color.to_hex(star_color.spectrum_to_rgb(star_db.GRID, y)),
            "Спектр": kind, "Примечание": s["note"], "id": s["id"],
        })
    return cat, pd.DataFrame(rows)


cat, df = load()
st.dataframe(df.drop(columns=["id"]), hide_index=True, use_container_width=True,
             column_config={"R, R☉": st.column_config.NumberColumn(format="%g"),
                            "Расст., св. лет": st.column_config.NumberColumn(format="%g")})
st.caption("Параметры приблизительные (по литературе); у переменных звёзд V — типичное значение. "
           "Спектры — модельные шаблоны: Планк для Teff + линии и полосы, характерные для класса.")

pick = st.multiselect("Показать спектры", df["Звезда"].tolist(),
                      default=["Наос", "Ригель", "Вега", "Процион", "Солнце",
                               "Арктур", "Бетельгейзе", "Мира"])
fwhm = st.slider("Разрешение, нм (веб-камера + Star Analyser ≈ 2–5 нм)", 0.0, 10.0, 0.0, 0.5)
offset = st.checkbox("Разнести по вертикали", value=True)

if pick:
    fig, ax = plt.subplots(figsize=(11, 1.2 + 0.75 * len(pick) if offset else 5))
    step = star_db.GRID[1] - star_db.GRID[0]
    for i, name in enumerate(pick):
        s = next(r for r in cat if r["name_ru"] == name)
        y, _ = star_db.template(s)
        y = star_db.smooth(y, step, fwhm)
        y = y / y.max()
        color = star_color.to_hex(star_color.spectrum_to_rgb(star_db.GRID, y))
        shift = i * 0.8 if offset else 0
        ax.plot(star_db.GRID, y + shift, lw=1, color=color)
        ax.text(782, shift + y[-1], f"{s['name_ru']} {s['sp_type']}", fontsize=8,
                va="center", color=color)
    for name in ("Hα", "Hβ", "Hγ", "Hδ", "Ca II K", "Na D", "Mg b"):
        ax.axvline(sc.LINES[name], color="gray", lw=0.5, alpha=0.4)
        ax.text(sc.LINES[name], ax.get_ylim()[1], name, fontsize=7, rotation=90,
                ha="center", va="bottom", color="gray")
    ax.set_facecolor("#0e1117")
    ax.set_xlim(370, 830)
    ax.set_xlabel("Длина волны, нм")
    ax.set_yticks([])
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with open(star_db.CATALOG, "rb") as f:
    cat_bytes = f.read()
tbl = io.StringIO()
cols = {"wavelength_nm": star_db.GRID}
for s in cat:
    cols[s["id"]] = np.round(star_db.template(s)[0], 5)
pd.DataFrame(cols).to_csv(tbl, index=False)
c1, c2 = st.columns(2)
c1.download_button("⬇ Каталог (stars.csv)", cat_bytes, "stars.csv", "text/csv")
c2.download_button("⬇ Все спектры (templates.csv)", tbl.getvalue(), "templates.csv", "text/csv")

with st.expander("Как добавить звезду или реальный спектр"):
    st.markdown("""
- **Новая звезда:** добавьте строку в `data/stars.csv` (класс, Teff, радиус …), шаблон построится сам.
- **Реальный спектр:** положите `data/observed/<id>.csv` со столбцами `wavelength_nm,flux`.
  Он заменит модель везде, в том числе при подборе класса. Например, сохраните CSV с главной
  страницы для хорошо снятой звезды. Или скачайте спектры из атласов MILES, ELODIE
  или каталога Pickles (1998).
""")
