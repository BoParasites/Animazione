// Ciclo vitale di Ascaris lumbricoides su anatomia 3D reale (BodyParts3D, CC BY 4.0).
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import ROUTE from './route3d.json';

const TAU = Math.PI * 2;
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const smooth = x => x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x);
const lerp = (a, b, t) => a + (b - a) * t;
const $ = id => document.getElementById(id);
const params = new URLSearchParams(location.search);
const CAPTURE = params.has('capture');

// ---------------- Fasi ----------------
const STAGES = [
  { k: 'ingest', title: 'Ingestione', time: 'giorno 0', dur: 6, glyph: 'eggE',
    text: 'Le uova embrionate, che contengono la larva infettante, vengono ingerite con cibo, acqua o mani sporche di terra.' },
  { k: 'hatch', title: 'Schiusa nel tenue', time: 'duodeno', dur: 4, glyph: 'hatch',
    text: 'Nell’intestino tenue le larve escono dall’uovo e attraversano la parete intestinale.' },
  { k: 'liver', title: 'Vena porta e fegato', time: '', dur: 5, glyph: 'larva',
    text: 'Entrano nel sangue venoso: vene mesenteriche, vena porta, fegato, poi vena cava inferiore.' },
  { k: 'heart', title: 'Cuore destro', time: '', dur: 4, glyph: 'larva',
    text: 'Dalla vena cava inferiore passano nell’atrio e nel ventricolo destro, poi nell’arteria polmonare.' },
  { k: 'lungs', title: 'Polmoni', time: '10–14 giorni', dur: 7, ease: 'arrive', glyph: 'larva',
    text: 'Nei polmoni le larve maturano per 10–14 giorni, poi perforano la parete alveolare. In questa fase possono comparire tosse ed eosinofilia (sindrome di Löffler).' },
  { k: 'ascent', title: 'Risalita', time: '', dur: 5, glyph: 'larva',
    text: 'Risalgono bronchioli, bronchi e trachea fino alla faringe.' },
  { k: 'swallow', title: 'Deglutizione', time: '', dur: 6, glyph: 'larva',
    text: 'Vengono deglutite: secondo passaggio per esofago e stomaco, di nuovo fino all’intestino tenue.' },
  { k: 'adult', title: 'Vermi adulti', time: '2–3 mesi', dur: 9, glyph: 'grow',
    text: 'Nel digiuno diventano adulti: femmine 20–35 cm, maschi 15–30 cm con coda ricurva (qui in scala reale). Dall’ingestione alla deposizione delle uova passano 2–3 mesi; gli adulti vivono 1–2 anni.' },
  { k: 'eggs', title: 'Uova nelle feci', time: '≈200.000/giorno', dur: 7, glyph: 'eggs',
    text: 'Una femmina può deporre circa 200.000 uova al giorno. Attraversano il colon ed escono con le feci, non ancora embrionate e quindi non infettanti.' },
  { k: 'soil', title: 'Embrionazione nel suolo', time: '18 gg – settimane', dur: 7, ease: 'arrive', glyph: 'eggSoil',
    text: 'In un terreno caldo, umido e ombreggiato l’uovo fecondato sviluppa la larva infettante in 18 giorni o alcune settimane. Le uova non fecondate (quella allungata) non diventano infettanti.' },
  { k: 'contagion', title: 'Contagio', time: '', dur: 6, glyph: 'eggE',
    text: 'Terra, verdure crude, acqua e mani contaminate riportano le uova embrionate alla bocca. Il ciclo ricomincia.' },
];

// camera per fase: distanza, quota, punto guardato, quanto segue il parassita
const CAM = {
  ingest:    { d: 1.05, el: 0.06, at: [0, 1.36, 0.02], follow: 0.45 },
  hatch:     { d: 0.95, el: 0.05, at: [0, 1.12, 0.04], follow: 0.5 },
  liver:     { d: 1.00, el: 0.06, at: [-0.02, 1.17, 0.02], follow: 0.5 },
  heart:     { d: 0.90, el: 0.05, at: [0, 1.29, 0.02], follow: 0.5 },
  lungs:     { d: 1.05, el: 0.05, at: [-0.03, 1.27, 0.0], follow: 0.45 },
  ascent:    { d: 1.10, el: 0.04, at: [0, 1.33, 0.0], follow: 0.45 },
  swallow:   { d: 1.20, el: 0.05, at: [0, 1.27, 0.02], follow: 0.5 },
  adult:     { d: 0.95, el: 0.08, at: [0, 1.03, 0.04], follow: 0.35 },
  eggs:      { d: 1.15, el: 0.06, at: [0, 0.98, 0.02], follow: 0.45 },
  soil:      { d: 2.50, el: 0.12, at: [0.16, 0.55, 0.10], follow: 0.25 },
  contagion: { d: 2.10, el: 0.07, at: [0.10, 1.00, 0.08], follow: 0.40 },
};

// ---------------- Percorso ----------------
const ctrl = [], startIdx = [];
for (const st of STAGES) { startIdx.push(ctrl.length); for (const p of ROUTE[st.k]) ctrl.push(new THREE.Vector3(...p)); }
const curve = new THREE.CatmullRomCurve3(ctrl, true, 'centripetal');
const ARC = curve.getLengths(ctrl.length * 40);           // lunghezze cumulative sul parametro t
const LEN = ARC[ARC.length - 1];
const sAtCtrl = i => ARC[Math.round(i / ctrl.length * (ARC.length - 1))];
let T = 0;
STAGES.forEach((st, i) => { st.s0 = sAtCtrl(startIdx[i]); st.t0 = T; T += st.dur; });
STAGES.forEach((st, i) => { st.s1 = i < STAGES.length - 1 ? STAGES[i + 1].s0 : LEN; });

function locate(t) {
  t = ((t % T) + T) % T;
  let i = 0; while (i < STAGES.length - 1 && t >= STAGES[i + 1].t0) i++;
  const st = STAGES[i], u = clamp((t - st.t0) / st.dur, 0, 1);
  const e = st.ease === 'arrive' ? (u < 0.5 ? smooth(u / 0.5) : 1) : smooth(u) * 0.25 + u * 0.75;
  return { i, u, s: st.s0 + (st.s1 - st.s0) * e };
}
const pointAtS = (s, out = new THREE.Vector3()) => curve.getPointAt(clamp(s / LEN, 0, 1), out);

// ---------------- Renderer ----------------
const canvas = $('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: CAPTURE });
renderer.setPixelRatio(Math.min(2, devicePixelRatio || 1));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.setClearColor(0x04060b, 1);
const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x04060b, 4.5, 9);
const camera = new THREE.PerspectiveCamera(28, 1, 0.02, 30);

scene.add(new THREE.HemisphereLight(0xbfd8ff, 0x201410, 0.9));
const key = new THREE.DirectionalLight(0xffffff, 1.6); key.position.set(-1.2, 2.4, 2.2); scene.add(key);
const rim = new THREE.DirectionalLight(0x7fb4ff, 1.1); rim.position.set(1.5, 1.2, -2.5); scene.add(rim);

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(512, 512), 0.42, 0.28, 0.92);
composer.addPass(bloom);
composer.addPass(new OutputPass());

// sfondo: alone radiale
{
  const c = document.createElement('canvas'); c.width = c.height = 512; const g = c.getContext('2d');
  const gr = g.createRadialGradient(256, 230, 10, 256, 256, 360);
  gr.addColorStop(0, '#13233d'); gr.addColorStop(0.55, '#070b14'); gr.addColorStop(1, '#04060b');
  g.fillStyle = gr; g.fillRect(0, 0, 512, 512);
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
  scene.background = tex;
}

// ---------------- Materiali "a raggi X" ----------------
function xray({ color, opacity, rim = 0.6, rimColor = color, power = 2.2, rimAlpha = 0.25, rough = 0.55, side = THREE.FrontSide, emissive = 0 }) {
  const m = new THREE.MeshStandardMaterial({ color, transparent: true, opacity, depthWrite: false, roughness: rough, metalness: 0.0, side });
  if (emissive) { m.emissive = new THREE.Color(color); m.emissiveIntensity = emissive; }
  const rc = new THREE.Color(rimColor);
  m.onBeforeCompile = sh => {
    sh.fragmentShader = sh.fragmentShader.replace('#include <opaque_fragment>', `#include <opaque_fragment>
      { float fr = pow(1.0 - abs(dot(normalize(normal), normalize(vViewPosition))), ${power.toFixed(2)});
        gl_FragColor.rgb += vec3(${rc.r.toFixed(3)}, ${rc.g.toFixed(3)}, ${rc.b.toFixed(3)}) * fr * ${rim.toFixed(3)};
        gl_FragColor.a = clamp(gl_FragColor.a + fr * ${rimAlpha.toFixed(3)}, 0.0, 1.0); }`);
  };
  m.customProgramCacheKey = () => [color, opacity, rim, power, rimAlpha].join('|');
  return m;
}
const BONE = xray({ color: 0xe9e2cc, opacity: 0.03, rim: 0.16, rimAlpha: 0.10, power: 2.6 });
const MAT = {
  skin: xray({ color: 0x8fbfff, opacity: 0.025, rim: 0.55, rimColor: 0x6fa8ff, rimAlpha: 0.30, power: 3.0 }),
  brain: xray({ color: 0xe8c7c7, opacity: 0.05, rim: 0.15, rimAlpha: 0.10 }),
  eye: xray({ color: 0xffffff, opacity: 0.08, rim: 0.2, rimAlpha: 0.15 }),
  lung: xray({ color: 0xe98d9c, opacity: 0.16, rim: 0.45, rimColor: 0xffb3c0, rimAlpha: 0.22, power: 2.0 }),
  heart: xray({ color: 0xc8303f, opacity: 0.42, rim: 0.4, rimColor: 0xff7080, rimAlpha: 0.25 }),
  liver: xray({ color: 0x8a3426, opacity: 0.38, rim: 0.35, rimColor: 0xd06a50, rimAlpha: 0.22 }),
  gall: xray({ color: 0x5f8f4a, opacity: 0.45, rim: 0.3 }),
  stomach: xray({ color: 0xe39a72, opacity: 0.30, rim: 0.4, rimColor: 0xffc09a, rimAlpha: 0.25 }),
  spleen: xray({ color: 0x8a3b5a, opacity: 0.38, rim: 0.3 }),
  pancreas: xray({ color: 0xe8c27a, opacity: 0.30, rim: 0.3 }),
  si: xray({ color: 0xeea3ad, opacity: 0.26, rim: 0.45, rimColor: 0xffc6cf, rimAlpha: 0.22 }),
  colon: xray({ color: 0xcfae7c, opacity: 0.28, rim: 0.45, rimColor: 0xffdca8, rimAlpha: 0.22 }),
  eso: xray({ color: 0xe08f80, opacity: 0.40, rim: 0.4 }),
  air: xray({ color: 0xbfe3f0, opacity: 0.35, rim: 0.45, rimColor: 0xd8f4ff }),
  pharynx: xray({ color: 0xe0a090, opacity: 0.10, rim: 0.2, rimAlpha: 0.12 }),
  kidney: xray({ color: 0xb5524a, opacity: 0.32, rim: 0.3 }),
  bladder: xray({ color: 0xe9d27a, opacity: 0.22, rim: 0.3 }),
  artery: xray({ color: 0xd83a3a, opacity: 0.32, rim: 0.3 }),
  vein: xray({ color: 0x3f66d8, opacity: 0.36, rim: 0.35, rimColor: 0x7fa0ff }),
  pulm: xray({ color: 0x8a6fe0, opacity: 0.20, rim: 0.3 }),
  diaphragm: xray({ color: 0xc07070, opacity: 0.035, rim: 0.12, rimAlpha: 0.06 }),
};
const ASSIGN = {
  skin: ['skin', 9], brain: ['brain', 2], eye_l: ['eye', 2], eye_r: ['eye', 2],
  lung_l: ['lung', 3], lung_r: ['lung', 3], heart: ['heart', 3], liver: ['liver', 3], gallbladder: ['gall', 3],
  stomach: ['stomach', 3], spleen: ['spleen', 3], pancreas: ['pancreas', 3], small_intestine: ['si', 3],
  colon: ['colon', 3], appendix: ['colon', 3], esophagus: ['eso', 3], tongue: ['pharynx', 3], pharynx: ['pharynx', 3],
  trachea: ['air', 3], bronchi: ['air', 3], kidney_l: ['kidney', 2], kidney_r: ['kidney', 2], adrenal_l: ['pancreas', 2], adrenal_r: ['pancreas', 2],
  ureter_l: ['bladder', 2], ureter_r: ['bladder', 2], bladder: ['bladder', 2], aorta: ['artery', 2],
  vena_cava_inferior: ['vein', 2], vena_cava_superior: ['vein', 2], veins_abdomen: ['vein', 2], pulmonary_vessels: ['pulm', 2], diaphragm: ['diaphragm', 2],
};

// ---------------- Tubo luminoso del percorso ----------------
const SEG = 4000, RAD = 8;
const tubeGeo = new THREE.TubeGeometry(curve, SEG, 0.0011, RAD, true);
const guideGeo = new THREE.TubeGeometry(curve, SEG, 0.0007, 6, true);
const litMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(1.7, 1.15, 0.42), toneMapped: false, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
const ghostMat = litMat.clone();
const guideMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(0.9, 0.62, 0.3), transparent: true, opacity: 0.22, depthWrite: false, blending: THREE.AdditiveBlending });
const ghostGeo = tubeGeo.clone();
const lit = new THREE.Mesh(tubeGeo, litMat), ghost = new THREE.Mesh(ghostGeo, ghostMat), guide = new THREE.Mesh(guideGeo, guideMat);
for (const [o, r] of [[guide, 5], [lit, 6], [ghost, 6]]) { o.renderOrder = r; o.frustumCulled = false; scene.add(o); }

function glowTexture(inner = 'rgba(255,240,200,1)', mid = 'rgba(255,190,80,0.35)') {
  const c = document.createElement('canvas'); c.width = c.height = 128; const g = c.getContext('2d');
  const gr = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  gr.addColorStop(0, inner); gr.addColorStop(0.25, mid); gr.addColorStop(1, 'rgba(255,170,60,0)');
  g.fillStyle = gr; g.fillRect(0, 0, 128, 128);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}
const GLOW = glowTexture();
const headGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: GLOW, color: new THREE.Color(1.2, 0.95, 0.6), blending: THREE.AdditiveBlending, depthTest: false, depthWrite: false, toneMapped: false }));
headGlow.renderOrder = 10; scene.add(headGlow);

const NSP = 90;
const spGeo = new THREE.BufferGeometry();
spGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(NSP * 3), 3));
spGeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(NSP * 3), 3));
const sparks = new THREE.Points(spGeo, new THREE.PointsMaterial({ size: 0.011, map: GLOW, vertexColors: true, transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending, toneMapped: false }));
sparks.renderOrder = 9; sparks.frustumCulled = false; scene.add(sparks);

// ---------------- Glifi del parassita (sprite disegnati in 2D) ----------------
function drawEgg(c, emb, fertile = true) {
  c.save(); c.translate(64, 64); c.scale(5.2, 5.2);
  const g = c.createRadialGradient(0, 0, 0, 0, 0, 12);
  g.addColorStop(0, 'rgba(255,190,90,0.35)'); g.addColorStop(1, 'rgba(255,190,90,0)');
  c.fillStyle = g; c.beginPath(); c.arc(0, 0, 12, 0, TAU); c.fill();
  if (fertile) {
    c.fillStyle = 'rgba(176,122,52,0.95)'; c.beginPath(); c.ellipse(0, 0, 7.4, 6, 0, 0, TAU); c.fill();
    c.fillStyle = 'rgba(206,150,72,1)';
    for (let i = 0; i < 22; i++) { const q = i / 22 * TAU; c.beginPath(); c.arc(Math.cos(q) * 7.2, Math.sin(q) * 5.8, 1.25, 0, TAU); c.fill(); }
    c.fillStyle = 'rgba(56,36,18,1)'; c.beginPath(); c.ellipse(0, 0, 5.6, 4.4, 0, 0, TAU); c.fill();
    c.lineWidth = 0.5; c.strokeStyle = 'rgba(240,210,150,0.7)'; c.stroke();
    if (emb < 0.55) {
      const q = emb / 0.55, n = q < 0.25 ? 1 : q < 0.5 ? 2 : q < 0.75 ? 4 : 8;
      c.fillStyle = 'rgba(238,208,150,0.95)';
      const cells = n === 1 ? [[0, 0, 3.2]] : n === 2 ? [[-1.5, 0, 2.1], [1.5, 0, 2.1]]
        : n === 4 ? [[-1.4, -1.1, 1.5], [1.4, -1.1, 1.5], [-1.4, 1.1, 1.5], [1.4, 1.1, 1.5]]
        : Array.from({ length: 8 }, (_, i) => [Math.cos(i / 8 * TAU) * 2.4, Math.sin(i / 8 * TAU) * 1.7, 1.05]);
      for (const [x, y, r] of cells) { c.beginPath(); c.arc(x, y, r, 0, TAU); c.fill(); }
    } else {
      c.globalAlpha = smooth((emb - 0.55) / 0.25);
      c.beginPath(); c.moveTo(-3.6, 1.2);
      c.bezierCurveTo(-3.9, -3.1, 3.9, -3.1, 3.7, 0); c.bezierCurveTo(3.5, 2.7, -1.9, 2.9, -1.7, 0.3); c.bezierCurveTo(-1.5, -1.3, 1.4, -1.3, 1.6, 0.3);
      c.lineCap = 'round'; c.lineWidth = 1; c.strokeStyle = 'rgba(255,238,205,1)'; c.stroke();
    }
  } else {
    c.fillStyle = 'rgba(160,112,50,0.9)'; c.beginPath(); c.ellipse(0, 0, 8.8, 4.6, 0, 0, TAU); c.fill();
    c.fillStyle = 'rgba(70,46,22,1)'; c.beginPath(); c.ellipse(0, 0, 7.6, 3.5, 0, 0, TAU); c.fill();
    c.fillStyle = 'rgba(215,180,120,0.85)';
    for (let i = 0; i < 11; i++) { c.beginPath(); c.arc(-6 + i * 1.2, Math.sin(i * 2.1) * 1.6, 0.75, 0, TAU); c.fill(); }
  }
  c.restore();
}
function drawLarva(c, ph, len = 0.8) {
  c.save(); c.translate(64, 64);
  const L = 100 * len;
  c.beginPath();
  for (let i = 0; i <= 30; i++) { const f = i / 30, x = -L / 2 + L * f, y = 7 * Math.sin(f * TAU * 1.2 - ph) * (0.35 + 0.65 * (1 - f)); i ? c.lineTo(x, y) : c.moveTo(x, y); }
  c.lineCap = 'round'; c.lineJoin = 'round';
  c.lineWidth = 16; c.strokeStyle = 'rgba(255,210,140,0.20)'; c.stroke();
  c.lineWidth = 6; c.strokeStyle = 'rgba(255,244,225,1)'; c.stroke();
  c.restore();
}
function makeSprite(size) {
  const cv = document.createElement('canvas'); cv.width = cv.height = 128;
  const tex = new THREE.CanvasTexture(cv); tex.colorSpace = THREE.SRGBColorSpace;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false, depthWrite: false }));
  sp.scale.set(size, size, 1); sp.renderOrder = 11; sp.userData = { cv, ctx: cv.getContext('2d'), tex };
  return sp;
}
const organism = makeSprite(0.03); scene.add(organism);
const trail = Array.from({ length: 5 }, () => { const s = makeSprite(0.024); scene.add(s); return s; });
const EGG_TEX = [0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1].map(e => { const cv = document.createElement('canvas'); cv.width = cv.height = 128; drawEgg(cv.getContext('2d'), e); const t = new THREE.CanvasTexture(cv); t.colorSpace = THREE.SRGBColorSpace; return t; });
const UNFERT_TEX = (() => { const cv = document.createElement('canvas'); cv.width = cv.height = 128; drawEgg(cv.getContext('2d'), 0, false); const t = new THREE.CanvasTexture(cv); t.colorSpace = THREE.SRGBColorSpace; return t; })();
const eggTex = e => EGG_TEX[Math.round(clamp(e, 0, 1) * (EGG_TEX.length - 1))];

// ---------------- Suolo ----------------
{
  const N = 1024, c = document.createElement('canvas'); c.width = c.height = N; const g = c.getContext('2d');
  let seed = 7; const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
  g.fillStyle = 'rgb(30,21,14)'; g.fillRect(0, 0, N, N);
  for (let i = 0; i < 30000; i++) {
    const x = rnd() * N, y = rnd() * N, r = 0.5 + rnd() * rnd() * 3.5;
    const v = rnd(); g.fillStyle = `rgba(${70 + v * 70 | 0},${50 + v * 45 | 0},${30 + v * 30 | 0},${0.10 + rnd() * 0.25})`;
    g.beginPath(); g.ellipse(x, y, r * 1.3, r, rnd() * 3, 0, TAU); g.fill();
  }
  const fade = g.createRadialGradient(N / 2, N / 2, N * 0.12, N / 2, N / 2, N / 2);
  fade.addColorStop(0, 'rgba(0,0,0,0)'); fade.addColorStop(1, 'rgba(0,0,0,1)');
  g.globalCompositeOperation = 'destination-out'; g.fillStyle = fade; g.fillRect(0, 0, N, N);
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(2.6, 2.6), new THREE.MeshStandardMaterial({ map: tex, transparent: true, roughness: 1, depthWrite: false }));
  ground.rotation.x = -Math.PI / 2; ground.position.set(0.15, 0, 0.12); ground.renderOrder = 0; scene.add(ground);
}
const soil = STAGES[9];
const soilEggs = [];
for (let s = soil.s0 + (soil.s1 - soil.s0) * 0.45, j = 0; s < soil.s1 - 0.02; s += 0.035, j++) {
  const sp = makeSprite(0.026); const p = pointAtS(s);
  p.x += (j % 2 ? 1 : -1) * 0.012; p.z += ((j * 7) % 5 - 2) * 0.006; p.y = 0.012;
  sp.position.copy(p); sp.userData.s = s; sp.userData.fertile = j !== 3; scene.add(sp); soilEggs.push(sp);
}

// ---------------- Vermi adulti (tubi deformabili) ----------------
const coil = new THREE.CatmullRomCurve3(ROUTE.adult.map(p => new THREE.Vector3(...p)), false, 'centripetal');
const COIL_LEN = coil.getLength();
function makeWorm(len, r0, male, phase) {
  const NS = 110, NR = 10, pos = new Float32Array((NS + 1) * NR * 3), idx = [];
  for (let i = 0; i < NS; i++) for (let j = 0; j < NR; j++) {
    const a = i * NR + j, b = i * NR + (j + 1) % NR, c = (i + 1) * NR + j, d = (i + 1) * NR + (j + 1) % NR;
    idx.push(a, c, b, b, c, d);
  }
  const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(pos, 3)); geo.setIndex(idx);
  const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: 0xf7d2c0, roughness: 0.4, transparent: true, opacity: 0.95, emissive: 0x5a2418, emissiveIntensity: 0.7 }));
  mesh.renderOrder = 7; mesh.frustumCulled = false; mesh.userData = { NS, NR, len, r0, male, phase };
  scene.add(mesh); return mesh;
}
const _p = new THREE.Vector3(), _t = new THREE.Vector3(), _n = new THREE.Vector3(), _b = new THREE.Vector3(), UP = new THREE.Vector3(0, 0, 1);
function updateWorm(w, start, clock, alpha) {
  w.visible = alpha > 0.01; if (!w.visible) return;
  w.material.opacity = 0.95 * alpha;
  const { NS, NR, len, r0, male, phase } = w.userData, side = male ? -1 : 1, pos = w.geometry.attributes.position.array;
  for (let i = 0; i <= NS; i++) {
    const f = i / NS, u = clamp((start + f * len) / COIL_LEN, 0, 1);
    coil.getPointAt(u, _p); coil.getTangentAt(u, _t);
    _n.crossVectors(_t, UP).normalize(); _b.crossVectors(_t, _n).normalize();
    let off = side * 0.0085 + 0.0035 * Math.sin(f * len * 70 - clock * 1.8 + phase);
    let offB = 0.003 * Math.cos(f * len * 45 - clock * 1.3 + phase);
    if (male && f > 0.88) { const q = (f - 0.88) / 0.12; off += q * q * 0.012; }
    const r = Math.max(0.0005, r0 * Math.pow(Math.sin(Math.PI * clamp(f, 0.001, 0.999)), 0.45));
    for (let j = 0; j < NR; j++) {
      const a = j / NR * TAU, ca = Math.cos(a) * r, sa = Math.sin(a) * r, k = (i * NR + j) * 3;
      pos[k] = _p.x + _n.x * (off + ca) + _b.x * (offB + sa);
      pos[k + 1] = _p.y + _n.y * (off + ca) + _b.y * (offB + sa);
      pos[k + 2] = _p.z + _n.z * (off + ca) + _b.z * (offB + sa);
    }
  }
  w.geometry.attributes.position.needsUpdate = true; w.geometry.computeVertexNormals();
}
const female = makeWorm(0.27, 0.0026, false, 0);
const male = makeWorm(0.21, 0.0021, true, 1.7);

// ---------------- Anatomia ----------------
let ready = false;
const loader = new GLTFLoader(); loader.setMeshoptDecoder(MeshoptDecoder);
fetch('models/anatomy-ascaris.glb.b64.txt').then(r => { if (!r.ok) throw new Error(r.status); return r.text(); }).then(b64 => {
  const bin = atob(b64.trim()), buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  loader.parse(buf.buffer, '', onModel, onErr);
}).catch(onErr);
function onErr(err) { $('loading').textContent = 'Impossibile caricare il modello anatomico.'; console.error(err); }
function onModel(gltf) {
  gltf.scene.traverse(o => {
    if (!o.isMesh) return;
    const a = ASSIGN[o.name] || ASSIGN[o.parent?.name];
    if (!o.geometry.attributes.normal) o.geometry.computeVertexNormals();
    if (a) { o.material = MAT[a[0]]; o.renderOrder = a[1]; } else { o.material = BONE; o.renderOrder = 1; }
  });
  scene.add(gltf.scene);
  ready = true; $('loading').hidden = true; window.__ready = true;
}

// ---------------- Camera ----------------
const camAt = new THREE.Vector3(), camPos = new THREE.Vector3(), tmpV = new THREE.Vector3();
function camParams(t) {
  const L = locate(t), st = STAGES[L.i], prev = STAGES[(L.i + STAGES.length - 1) % STAGES.length];
  const a = CAM[prev.k], b = CAM[st.k], blend = smooth((((t % T) + T) % T - st.t0) / Math.min(2.2, st.dur * 0.5));
  return { d: lerp(a.d, b.d, blend), el: lerp(a.el, b.el, blend), follow: lerp(a.follow, b.follow, blend),
    at: [0, 1, 2].map(k => lerp(a.at[k], b.at[k], blend)) };
}
function updateCamera(t) {
  const p = camParams(t);
  const head = new THREE.Vector3();
  for (let k = 0; k < 6; k++) head.add(pointAtS(locate(t - k * 0.25).s, tmpV));   // posizione mediata: movimento morbido
  head.multiplyScalar(1 / 6);
  camAt.set(...p.at).lerp(head, p.follow);
  const az = 0.32 * Math.sin(((t % T) / T) * TAU * 2) + 0.12;
  const narrow = Math.min(2.6, Math.max(1, 1.35 / camera.aspect));
  camPos.set(Math.sin(az) * Math.cos(p.el), Math.sin(p.el), Math.cos(az) * Math.cos(p.el)).multiplyScalar(p.d * narrow).add(camAt);
  camera.position.copy(camPos); camera.lookAt(camAt);
}

// ---------------- Stato e rendering ----------------
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
let t = reduced ? T - 0.001 : 0, clock = 0, ghostA = 0, speed = 1, paused = reduced, last = null, curStage = -1;
const head = new THREE.Vector3(), proj1 = new THREE.Vector3(), proj2 = new THREE.Vector3();

function screenAngle(s) {
  pointAtS(s, proj1).project(camera); pointAtS(s + 0.01, proj2).project(camera);
  return Math.atan2((proj2.y - proj1.y) * innerHeight, (proj2.x - proj1.x) * innerWidth);
}

function render() {
  const L = locate(t), st = STAGES[L.i], u = L.u, sH = L.s;
  updateCamera(t);
  const frac = sH / LEN, segs = Math.floor(frac * SEG);
  tubeGeo.setDrawRange(0, segs * RAD * 6);
  litMat.opacity = 0.9 + 0.1 * Math.sin(clock * 2.4);
  ghost.visible = ghostA > 0.01;
  if (ghost.visible) { ghostMat.opacity = ghostA; ghostGeo.setDrawRange(segs * RAD * 6, (SEG - segs) * RAD * 6); }
  pointAtS(sH, head);
  headGlow.position.copy(head); headGlow.scale.setScalar(0.035 + 0.004 * Math.sin(clock * 5));

  // scintille lungo la linea accesa
  const pa = spGeo.attributes.position.array, ca = spGeo.attributes.color.array;
  for (let n = 0; n < NSP; n++) {
    const s = (n / NSP * LEN + clock * 0.05) % LEN;
    let a = s <= sH ? 1 : ghostA; a *= 0.5 + 0.5 * Math.sin(clock * 5 + n * 2.3);
    pointAtS(s, tmpV); pa[n * 3] = tmpV.x; pa[n * 3 + 1] = tmpV.y; pa[n * 3 + 2] = tmpV.z;
    ca[n * 3] = 1.6 * a; ca[n * 3 + 1] = 1.25 * a; ca[n * 3 + 2] = 0.7 * a;
  }
  spGeo.attributes.position.needsUpdate = true; spGeo.attributes.color.needsUpdate = true;

  // glifo
  const o = organism, oc = o.userData.ctx;
  o.position.copy(head); o.visible = true; o.material.rotation = 0;
  for (const tr of trail) tr.visible = false;
  oc.clearRect(0, 0, 128, 128);
  const ang = screenAngle(sH);
  switch (st.glyph) {
    case 'eggE': o.material.map = eggTex(1); break;
    case 'eggSoil': o.material.map = eggTex(smooth((u - 0.15) / 0.75)); break;
    case 'hatch': {
      const fe = 1 - smooth((u - 0.35) / 0.2);
      if (fe > 0.5) { o.material.map = eggTex(1); o.material.rotation = Math.sin(clock * 14) * 0.15 * smooth(u / 0.35); }
      else { drawLarva(oc, clock * 9); o.userData.tex.needsUpdate = true; o.material.map = o.userData.tex; o.material.rotation = ang; }
      break;
    }
    case 'larva': drawLarva(oc, clock * 9); o.userData.tex.needsUpdate = true; o.material.map = o.userData.tex; o.material.rotation = ang; break;
    case 'grow': {
      const a = 1 - smooth((u - 0.6) / 0.3);
      drawLarva(oc, clock * 9, 0.8 + 0.2 * u); o.userData.tex.needsUpdate = true; o.material.map = o.userData.tex; o.material.rotation = ang;
      o.material.opacity = a; break;
    }
    case 'eggs':
      o.material.map = eggTex(0);
      trail.forEach((tr, j) => { const s = sH - (j + 1) * 0.014; if (s < st.s0) return; tr.visible = true; pointAtS(s, tr.position); tr.material.map = eggTex(0); tr.material.opacity = 1 - j * 0.15; });
      break;
  }
  if (st.glyph !== 'grow') o.material.opacity = 1;

  // vermi adulti
  const wA = L.i > 7 ? 1 : L.i === 7 ? smooth((u - 0.2) / 0.5) : ghostA;
  const slide = Math.sin(clock * 0.3) * 0.02;
  updateWorm(female, COIL_LEN * 0.08 + slide, clock, wA);
  updateWorm(male, COIL_LEN * 0.55 - slide, clock, wA);

  // uova nel suolo
  for (const e of soilEggs) {
    let a = L.i === 9 ? smooth((sH - e.userData.s) / 0.02) : L.i === 10 ? 1 : ghostA;
    e.visible = a > 0.01; if (!e.visible) continue;
    e.material.opacity = a;
    e.material.map = e.userData.fertile ? eggTex(L.i === 9 ? smooth((u - 0.15) / 0.75) : 1) : UNFERT_TEX;
  }

  composer.render();
  if (L.i !== curStage) setStage(L.i);
  $('prog').style.width = (((t % T) + T) % T / T * 100).toFixed(2) + '%';
}

// ---------------- Layout ----------------
const wideMQ = matchMedia('(min-width: 760px)');
function layout() {
  const W = innerWidth, H = innerHeight;
  renderer.setSize(W, H, false); composer.setSize(W, H);
  bloom.resolution.set(W / 2, H / 2);
  camera.aspect = W / H;
  const pr = $('panel').getBoundingClientRect();
  if (CAPTURE || $('panel').hidden) camera.clearViewOffset();
  else if (wideMQ.matches) camera.setViewOffset(W, H, (W - pr.left + 8) / 2, 0, W, H);
  else camera.setViewOffset(W, H, 0, (H - pr.top + 4) / 2, W, H);
  camera.updateProjectionMatrix();
}

// ---------------- Pannello ----------------
const list = $('list');
STAGES.forEach((st, i) => {
  const li = document.createElement('li');
  li.innerHTML = `<button type="button"><span class="n">${String(i + 1).padStart(2, '0')}</span><span class="t">${st.title}</span><span class="d">${st.time}</span></button>`;
  li.firstChild.addEventListener('click', () => jump(i));
  list.appendChild(li);
});
function setStage(i) {
  curStage = i; const st = STAGES[i];
  $('nowNum').textContent = `${String(i + 1).padStart(2, '0')} / ${STAGES.length}`;
  $('nowTitle').textContent = st.title; $('nowText').textContent = st.text;
  $('nowTime').textContent = st.time; $('nowTime').hidden = !st.time;
  [...list.children].forEach((li, j) => { li.classList.toggle('active', j === i); li.classList.toggle('done', j < i); });
  if (CAPTURE) { $('capTitle').textContent = st.title; $('capTime').textContent = st.time; $('capTime').hidden = !st.time; $('capNum').textContent = `${String(i + 1).padStart(2, '0')} / ${STAGES.length}`; }
}
function jump(i) { i = (i + STAGES.length) % STAGES.length; t = STAGES[i].t0 + 0.001; ghostA = 0; }
function setPaused(p) { paused = p; $('play').textContent = p ? '▶ Riproduci' : '❚❚ Pausa'; }
$('play').addEventListener('click', () => setPaused(!paused));
$('prev').addEventListener('click', () => { const L = locate(t); jump(L.u > 0.15 ? L.i : L.i - 1); });
$('next').addEventListener('click', () => jump(locate(t).i + 1));
document.querySelectorAll('.speed button').forEach(b => b.addEventListener('click', () => {
  speed = parseFloat(b.dataset.s);
  document.querySelectorAll('.speed button').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
}));
addEventListener('keydown', e => {
  if (e.code === 'Space' && !(e.target instanceof HTMLButtonElement)) { e.preventDefault(); setPaused(!paused); }
  if (e.key === 'ArrowRight') jump(locate(t).i + 1);
  if (e.key === 'ArrowLeft') jump(locate(t).i - 1);
});
setPaused(paused);
if (CAPTURE) { $('panel').hidden = true; $('caption').hidden = false; }

// ---------------- Ciclo ----------------
function frame(now) {
  const dt = last == null ? 0 : Math.min(0.05, (now - last) / 1000); last = now;
  if (!paused) { t += dt * speed; if (t >= T) { t -= T; ghostA = 1; } }
  clock += dt * (paused ? 0.3 : speed);
  ghostA = Math.max(0, ghostA - dt / 2.5);
  if (ready) render();
  requestAnimationFrame(frame);
}
// esportazione video: renderAt(secondi) disegna un fotogramma deterministico
window.renderAt = sec => { paused = true; last = null; t = ((sec % T) + T) % T; clock = sec; ghostA = sec >= T && t < 2.5 ? 1 - t / 2.5 : 0; render(); };
window.CYCLE_SECONDS = T;
window.__dbg = { female, male, lit, ghost, guide, sparks, scene, camera };
addEventListener('resize', layout);
wideMQ.addEventListener?.('change', layout);
layout();
if (!CAPTURE) requestAnimationFrame(frame);
