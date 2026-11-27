"""3D-вид звезды в её настоящем цвете + «радуга» её спектра."""
import json

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

import star_color
import star_db

st.set_page_config(page_title="3D звезда", page_icon="🌟", layout="wide")
st.title("🌟 Звезда в 3D: её настоящий цвет")


@st.cache_data
def catalog():
    return star_db.load_catalog()


cat = catalog()
labels = {f"{s['name_ru']} ({s['name']}) — {s['sp_type']}": s for s in cat}
default = next(i for i, k in enumerate(labels) if k.startswith("Бетельгейзе"))

c1, c2, c3 = st.columns([3, 1, 1])
choice = c1.selectbox("Звезда", list(labels), index=default)
show_sun = c2.checkbox("Солнце для масштаба", value=True)
spin = c3.slider("Вращение", 0.0, 3.0, 0.6, 0.1)
star = labels[choice]

# --- физика цвета ---
wl = star_db.GRID
flux, kind = star_db.template(star, wl)
rgb = star_color.spectrum_to_rgb(wl, flux)
hex_color = star_color.to_hex(rgb)
sun = star_db.find_star(cat, "sun")
sun_rgb = star_color.spectrum_to_rgb(wl, star_db.template(sun, wl)[0])

# «радуга»: цвет длины волны × яркость спектра (каждый 4-й отсчёт достаточно)
wl_s = wl[::4]
strip = star_color.wavelength_colors(wl_s) * (flux[::4] / flux.max())[:, None] ** 0.8
strip = (np.clip(strip, 0, 1) * 255).round().astype(int).tolist()

radius = star["radius_rsun"] or 1.0
payload = {
    "color": [float(c) for c in rgb],
    "sunColor": [float(c) for c in sun_rgb],
    "radius": radius,
    "showSun": bool(show_sun and star["id"] != "sun"),
    "spin": spin,
    "teff": star["teff_k"],
    "name": f"{star['name_ru']} · {star['sp_type']}",
    "strip": strip,
    "wl0": float(wl_s[0]), "wl1": float(wl_s[-1]),
}

HTML = r"""
<div id="wrap" style="position:relative;width:100%;height:560px;background:#000;border-radius:10px;overflow:hidden">
  <div id="label" style="position:absolute;left:14px;top:10px;color:#ddd;font:14px system-ui;z-index:2"></div>
  <div id="sunlabel" style="position:absolute;color:#bbb;font:12px system-ui;z-index:2;pointer-events:none"></div>
  <div style="position:absolute;right:14px;bottom:10px;color:#777;font:11px system-ui;z-index:2">мышь: вращать · колесо: приблизить</div>
</div>
<canvas id="strip" style="width:100%;height:70px;margin-top:8px;border-radius:6px;background:#000"></canvas>
<div style="display:flex;justify-content:space-between;color:#999;font:11px system-ui"><span id="w0"></span><span>длина волны, нм</span><span id="w1"></span></div>

<script type="importmap">
{ "imports": {
  "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
}}
</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
const P = __PAYLOAD__;

const wrap = document.getElementById("wrap");
const W = () => wrap.clientWidth, H = () => wrap.clientHeight;
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(W(), H());
wrap.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, W() / H(), 0.0001, 1000);
camera.position.set(0, 0.6, 4.2);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.minDistance = 1.15; controls.maxDistance = 40;
controls.autoRotate = P.spin > 0; controls.autoRotateSpeed = P.spin;

// звёздный фон
{
  const n = 4000, pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const v = new THREE.Vector3().randomDirection().multiplyScalar(200 + Math.random() * 200);
    pos.set([v.x, v.y, v.z], i * 3);
  }
  const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  scene.add(new THREE.Points(g, new THREE.PointsMaterial({ color: 0x8899aa, size: 0.6, sizeAttenuation: true })));
}

// поверхность: потемнение к краю + анимированная грануляция
const noise = `
vec3 h3(vec3 p){p=vec3(dot(p,vec3(127.1,311.7,74.7)),dot(p,vec3(269.5,183.3,246.1)),dot(p,vec3(113.5,271.9,124.6)));return -1.0+2.0*fract(sin(p)*43758.5453);}
float n3(vec3 p){vec3 i=floor(p),f=fract(p),u=f*f*(3.0-2.0*f);
 return mix(mix(mix(dot(h3(i),f),dot(h3(i+vec3(1,0,0)),f-vec3(1,0,0)),u.x),mix(dot(h3(i+vec3(0,1,0)),f-vec3(0,1,0)),dot(h3(i+vec3(1,1,0)),f-vec3(1,1,0)),u.x),u.y),
            mix(mix(dot(h3(i+vec3(0,0,1)),f-vec3(0,0,1)),dot(h3(i+vec3(1,0,1)),f-vec3(1,0,1)),u.x),mix(dot(h3(i+vec3(0,1,1)),f-vec3(0,1,1)),dot(h3(i+vec3(1,1,1)),f-vec3(1,1,1)),u.x),u.y),u.z);}`;

function starMaterial(rgb, cool) {
  return new THREE.ShaderMaterial({
    uniforms: { col: { value: new THREE.Color(...rgb) }, t: { value: 0 }, cool: { value: cool } },
    vertexShader: `varying vec3 vN; varying vec3 vP; varying vec3 vV;
      void main(){ vN=normalize(normalMatrix*normal); vP=position;
        vec4 mv=modelViewMatrix*vec4(position,1.0); vV=normalize(-mv.xyz); gl_Position=projectionMatrix*mv; }`,
    fragmentShader: noise + `
      uniform vec3 col; uniform float t; uniform float cool; varying vec3 vN; varying vec3 vP; varying vec3 vV;
      void main(){
        float mu = clamp(dot(vN, vV), 0.0, 1.0);
        float limb = 1.0 - 0.6*(1.0 - mu);                      // потемнение к краю
        // у холодных сверхгигантов ячейки конвекции огромные — меньше частота, сильнее контраст
        float sc = mix(9.0, 3.0, cool);
        float g = n3(vP*sc + t*0.15)*0.6 + n3(vP*sc*2.3 - t*0.1)*0.3 + n3(vP*sc*6.0 + t*0.3)*0.1;
        float gran = 1.0 + g*mix(0.25, 0.55, cool);
        vec3 c = col * limb * gran;
        c = mix(c, vec3(1.0), 0.18*pow(mu, 3.0));               // яркий центр диска
        gl_FragColor = vec4(c, 1.0);
      }`
  });
}

function glowSprite(rgb, scale) {
  const cv = document.createElement("canvas"); cv.width = cv.height = 256;
  const ctx = cv.getContext("2d");
  const [r, g, b] = rgb.map(v => Math.round(v * 255));
  const gr = ctx.createRadialGradient(128, 128, 0, 128, 128, 128);
  gr.addColorStop(0.0, `rgba(${r},${g},${b},0.9)`);
  gr.addColorStop(0.35, `rgba(${r},${g},${b},0.35)`);
  gr.addColorStop(1.0, `rgba(${r},${g},${b},0)`);
  ctx.fillStyle = gr; ctx.fillRect(0, 0, 256, 256);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({
    map: new THREE.CanvasTexture(cv), blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
  sp.scale.set(scale, scale, 1);
  return sp;
}

const cool = Math.min(1, Math.max(0, (6000 - (P.teff || 5800)) / 3000));
const mat = starMaterial(P.color, cool);
const starMesh = new THREE.Mesh(new THREE.SphereGeometry(1, 128, 128), mat);
scene.add(starMesh);
scene.add(glowSprite(P.color, 3.4));

// Солнце в том же масштабе (радиус звезды = 1)
let sunMesh = null;
const rs = 1 / P.radius;
if (P.showSun) {
  sunMesh = new THREE.Mesh(new THREE.SphereGeometry(Math.max(rs, 0.002), 48, 48), starMaterial(P.sunColor, 0));
  sunMesh.position.set(1.6 + rs, 0, 0);
  scene.add(sunMesh);
  if (rs < 0.03) {   // слишком мало — добавим маркер
    const m = glowSprite(P.sunColor, 0.06); m.position.copy(sunMesh.position); scene.add(m);
  }
}
document.getElementById("label").innerHTML =
  `<b style="font-size:16px">${P.name}</b><br>Teff ≈ ${P.teff ? Math.round(P.teff) : "—"} K · ` +
  `R ≈ ${P.radius} R☉ · цвет <span style="display:inline-block;width:12px;height:12px;border-radius:50%;` +
  `vertical-align:-2px;background:rgb(${P.color.map(v=>Math.round(v*255)).join(",")})"></span>`;

const sunLabel = document.getElementById("sunlabel");
const clock = new THREE.Clock();
function loop() {
  const t = clock.getElapsedTime();
  mat.uniforms.t.value = t;
  if (sunMesh) {
    sunMesh.material.uniforms.t.value = t;
    const p = sunMesh.position.clone().project(camera);
    sunLabel.style.display = p.z < 1 ? "block" : "none";
    sunLabel.style.left = ((p.x + 1) / 2 * W() + 8) + "px";
    sunLabel.style.top = ((1 - p.y) / 2 * H() - 18) + "px";
    sunLabel.textContent = P.radius > 1.01 ? `☉ Солнце (в ${Math.round(P.radius)} раз меньше)` : "☉ Солнце";
  }
  starMesh.rotation.y = t * 0.03;
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(loop);
}
loop();
window.addEventListener("resize", () => {
  renderer.setSize(W(), H()); camera.aspect = W() / H(); camera.updateProjectionMatrix();
});

// радуга спектра
const cv = document.getElementById("strip");
cv.width = P.strip.length; cv.height = 1;
const ctx = cv.getContext("2d"), img = ctx.createImageData(P.strip.length, 1);
P.strip.forEach((c, i) => img.data.set([c[0], c[1], c[2], 255], i * 4));
ctx.putImageData(img, 0, 0);
cv.style.imageRendering = "auto";
document.getElementById("w0").textContent = Math.round(P.wl0);
document.getElementById("w1").textContent = Math.round(P.wl1);
</script>
"""

components.html(HTML.replace("__PAYLOAD__", json.dumps(payload)), height=660)

a, b, c, d = st.columns(4)
a.metric("Цвет (sRGB)", hex_color)
b.metric("Температура", f"{star['teff_k']:.0f} K" if star["teff_k"] else "—")
c.metric("Радиус", f"{radius:g} R☉")
d.metric("Расстояние", f"{star['dist_ly']:g} св. лет" if star["dist_ly"] else "—")
if star["note"]:
    st.caption(star["note"])

st.markdown(f"""
**Как получен цвет.** Спектр звезды ({kind}) умножается на кривые чувствительности
глаза CIE 1931, интегрируется в координаты XYZ и переводится в sRGB. Так рассчитывают
цвет, который увидел бы глаз над атмосферой Земли. Яркость нормирована: реальный
блеск звёзд отличается в миллиарды раз, на экране это не показать.

**Полоса под звездой** — та же «радуга», что даёт решётка перед камерой: каждый
цвет имеет яркость спектра звезды на своей длине волны, тёмные провалы — линии поглощения.
Размер звезды и Солнца показан в одном масштабе, грануляция поверхности условная.
""")
