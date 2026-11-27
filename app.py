"""Streamlit-сайт: спектр звезды с камеры + данные SIMBAD."""
import io
import tempfile

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import spectrum_core as sc
import star_db

st.set_page_config(page_title="Спектр звезды", page_icon="🔭", layout="wide")
st.title("🔭 Спектр звезды с веб-камеры")
st.caption("Сфотографируйте спектр (камера + телескоп + дифракционная решётка) "
           "или загрузите снимки/видео. Данные о звезде берутся из SIMBAD.")


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def simbad(name):
    return sc.query_simbad(name)


@st.cache_data
def catalog():
    return star_db.load_catalog()


CAT = catalog()


def decode_image(data):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("не удалось прочитать изображение")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def video_frames(data, max_frames=200):
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        f.write(data)
        f.flush()
        cap = cv2.VideoCapture(f.name)
        frames = []
        while len(frames) < max_frames:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()
    return frames


# ---------------- Боковая панель ----------------
with st.sidebar:
    st.header("Звезда")
    star = st.text_input("Имя (как в SIMBAD)", "Vega",
                         help="Vega, Betelgeuse, Sirius, HD 172167, alf Lyr …")

    st.header("Изображение")
    rotate = st.selectbox("Поворот", ["0°", "90° ↻", "180°", "90° ↺"],
                          help="Спектр должен идти горизонтально, синим концом влево.")
    flip = st.checkbox("Отразить по горизонтали")
    half = st.slider("Полуширина полосы, пикс.", 1, 30, 4)

    st.header("Калибровка")
    mode = st.radio("Способ", ["Нет (пиксели)", "Нулевой порядок + дисперсия",
                               "Две известные линии"])
    cal = sc.Calibration()
    if mode == "Нулевой порядок + дисперсия":
        zero = st.number_input("Пиксель звезды (нулевой порядок)", value=100.0, step=1.0)
        disp = st.number_input("Дисперсия, нм/пикс.", value=1.85, step=0.05, format="%.3f")
        cal = sc.Calibration(zero=zero, disp=disp)
    elif mode == "Две известные линии":
        c1, c2 = st.columns(2)
        p1 = c1.number_input("Пиксель 1", value=410.0)
        l1 = c2.number_input("λ1, нм", value=486.1)
        p2 = c1.number_input("Пиксель 2", value=590.0)
        l2 = c2.number_input("λ2, нм", value=656.3)
        cal = sc.Calibration(points=[(p1, l1), (p2, l2)])
        if not cal.ok:
            st.warning("Пиксели должны различаться")
    show_bb = st.checkbox("Кривая Планка по спектр. классу", value=True)

    st.header("Сравнение с базой")
    ref_names = ["— не показывать —"] + [f"{s['name_ru']} ({s['sp_type']})" for s in CAT]
    in_db = star_db.find_star(CAT, star)
    ref_default = (ref_names.index(f"{in_db['name_ru']} ({in_db['sp_type']})")
                   if in_db else 0)
    ref_pick = st.selectbox("Эталон на графике", ref_names, index=ref_default)
    fwhm = st.slider("Разрешение спектроскопа, нм", 0.5, 10.0, 3.0, 0.5,
                     help="Шаблон сглаживается до этого разрешения, чтобы сравнение было честным.")

# ---------------- Данные о звезде ----------------
info = {"name": star, "sp_type": "", "vmag": None, "ra": None, "dec": None}
if star.strip():
    try:
        with st.spinner("Запрос SIMBAD…"):
            info = simbad(star.strip())
    except Exception as e:
        if in_db:   # нет связи с SIMBAD — берём данные из своей базы
            info = {"name": in_db["name"], "sp_type": in_db["sp_type"], "vmag": in_db["vmag"],
                    "ra": None, "dec": None}
            st.sidebar.info("SIMBAD недоступен — данные из встроенной базы")
        else:
            st.sidebar.error(f"SIMBAD: {e}")
teff = (in_db["teff_k"] if in_db and in_db["teff_k"] else None) or \
    sc.teff_from_sptype(info["sp_type"])
ref_star = None if ref_pick.startswith("—") else CAT[ref_names.index(ref_pick) - 1]

cols = st.columns(5)
cols[0].metric("Объект", info["name"] or "—")
cols[1].metric("Спектр. класс", info["sp_type"] or "—")
cols[2].metric("V, зв. вел.", f"{info['vmag']:.2f}" if info["vmag"] is not None else "—")
cols[3].metric("Teff (оценка)", f"{int(round(teff, -2))} K" if teff else "—")
cols[4].metric("RA / Dec, °", f"{info['ra'] or '—'} / {info['dec'] or '—'}")

# ---------------- Источник кадров ----------------
tab_cam, tab_img, tab_vid = st.tabs(["📷 Камера", "🖼 Снимки", "🎞 Видео"])
frames = []
with tab_cam:
    shot = st.camera_input("Снимок с камеры")
    if shot:
        frames = [decode_image(shot.getvalue())]
with tab_img:
    ups = st.file_uploader("Один или несколько снимков (будут усреднены)",
                           type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
                           accept_multiple_files=True)
    if ups and not frames:
        frames = [decode_image(u.getvalue()) for u in ups]
with tab_vid:
    vid = st.file_uploader("Видео (усредняется до 200 кадров)",
                           type=["mp4", "avi", "mov", "mkv"])
    if vid and not frames:
        with st.spinner("Читаю кадры…"):
            frames = video_frames(vid.getvalue())

if not frames:
    st.info("Сделайте снимок или загрузите файл, чтобы построить спектр.")
    st.stop()

# одинаковый размер — иначе нельзя усреднить
shape = frames[0].shape
frames = [f for f in frames if f.shape == shape]
img = np.mean(np.stack(frames).astype(np.float32), axis=0)

rot = {"90° ↻": cv2.ROTATE_90_CLOCKWISE, "180°": cv2.ROTATE_180,
       "90° ↺": cv2.ROTATE_90_COUNTERCLOCKWISE}.get(rotate)
if rot is not None:
    img = cv2.rotate(img, rot)
if flip:
    img = img[:, ::-1]

gray = sc.to_gray(img)
auto_row = sc.find_band_row(gray)
row = st.slider("Строка спектра", 0, gray.shape[0] - 1, auto_row,
                help=f"Найдена автоматически: {auto_row}. Поправьте, если промахнулась.")

x, y = sc.process(gray, row, half, cal)

# ---------------- Вывод ----------------
left, right = st.columns([1, 2])
with left:
    vis = np.clip(img, 0, 255).astype(np.uint8).copy()
    cv2.rectangle(vis, (0, max(0, row - half)), (vis.shape[1] - 1, row + half),
                  (0, 255, 0), max(1, vis.shape[0] // 300))
    st.image(vis, caption=f"Кадров усреднено: {len(frames)}; полоса — зелёная рамка",
             use_container_width=True)
    st.caption("Подсказка: позиция пикселя по X = номер столбца на графике "
               "без калибровки. По нему задаются точки калибровки.")

with right:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(x, y, lw=1.2, label="измерено")
    if cal.ok and x.size:
        if show_bb and teff:
            bb = sc.planck(x, teff)
            ax.plot(x, bb / bb.max(), "--", lw=1, alpha=0.6,
                    label=f"Планк, {int(round(teff, -2))} K")
        if ref_star:
            step = float(np.median(np.diff(x)))
            t, kind = star_db.template(ref_star, x)
            t = star_db.smooth(t, step, fwhm)
            ax.plot(x, t / np.nanmax(t), lw=1, alpha=0.8, color="tab:orange",
                    label=f"{ref_star['name_ru']} {ref_star['sp_type']} ({kind})")
        for name in sc.expected_lines(info["sp_type"]):
            nm = sc.LINES[name]
            if x.min() <= nm <= x.max():
                color = "tab:blue" if name in sc.TELLURIC else "tab:red"
                ax.axvline(nm, color=color, lw=0.7, alpha=0.6)
                ax.text(nm, 1.02, name, rotation=90, fontsize=7,
                        ha="center", va="bottom", color=color)
    ax.set_xlabel("Длина волны, нм" if cal.ok else "Пиксель")
    ax.set_ylabel("Интенсивность (отн.)")
    ax.set_ylim(0, 1.25)
    if x.size:
        ax.set_xlim(x.min(), x.max())
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"{info['name']}  {info['sp_type']}")
    fig.tight_layout()
    st.pyplot(fig)

    png = io.BytesIO()
    fig.savefig(png, format="png", dpi=150)
    csv = "wavelength_nm,intensity\n" if cal.ok else "pixel,intensity\n"
    csv += "\n".join(f"{a:.3f},{b:.5f}" for a, b in zip(x, y))
    safe = "".join(ch if ch.isalnum() else "_" for ch in star) or "star"
    d1, d2 = st.columns(2)
    d1.download_button("⬇ CSV", csv, f"{safe}_spectrum.csv", "text/csv")
    d2.download_button("⬇ PNG", png.getvalue(), f"{safe}_spectrum.png", "image/png")
    plt.close(fig)

st.subheader("Подбор спектрального класса по базе")
if not cal.ok:
    st.info("Нужна калибровка в нанометрах — тогда спектр сравнится со всеми звёздами базы.")
else:
    res = star_db.match(x, y, CAT, fwhm)
    if not res:
        st.warning("Слишком короткий участок спектра для сравнения.")
    else:
        st.dataframe(
            [{"Звезда": r["star"]["name_ru"], "Класс": r["star"]["sp_type"],
              "Teff, K": r["star"]["teff_k"], "Сходство линий": round(r["r"], 3),
              "Эталон": r["kind"]} for r in res[:5]],
            hide_index=True, use_container_width=True)
        best = res[0]["star"]
        st.caption(f"Ближе всего: **{best['name_ru']} ({best['sp_type']})**. Сравниваются только "
                   "линии: континуум убран, потому что его искажает чувствительность камеры. "
                   "При разрешении в несколько нм соседние классы (B8 и A0, G и K) легко спутать.")

with st.expander("Как получить хороший спектр"):
    st.markdown("""
- Нужен **телескоп и решётка** (например, Star Analyser 100) перед камерой: самодельный спектроскоп из DVD-диска для звёзд слишком тусклый.
- Звезда (нулевой порядок) и её спектр должны попасть в кадр. Спектр должен идти **горизонтально**, синим концом влево. Если нет, используйте поворот или отражение.
- Отключите автоэкспозицию в настройках камеры. Браузер этого не умеет, нужна программа камеры.
- Чем больше кадров усреднено, тем меньше шум. Лучше всего загрузить короткое видео.
- Калибровка «две линии»: у звезды класса A найдите провалы Hβ (486.1 нм) и Hα (656.3 нм) и введите их пиксели.
""")
