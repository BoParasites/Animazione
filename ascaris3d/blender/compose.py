# Post-produzione: anatomia (Cycles) + livello linea/parassita + bagliore + didascalie + titoli e crediti.
# Uso: python compose.py --state state.json --anat DIR --over DIR --out DIR [--frames a:b:s] [--outro]
import argparse, json, os, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument('--state'); ap.add_argument('--anat'); ap.add_argument('--over'); ap.add_argument('--out')
ap.add_argument('--fonts', default=os.path.join(os.path.dirname(__file__), 'fonts'))
ap.add_argument('--frames', default='0:1980:1'); ap.add_argument('--anat-name', default='b_{:05d}.png')
ap.add_argument('--outro', action='store_true'); ap.add_argument('--anat-offset', type=int, default=0)
A = ap.parse_args()
S = json.load(open(A.state)); FR, ST = S['frames'], S['stages']
W, H, FPS = 1920, 1080, 30
os.makedirs(A.out, exist_ok=True)

F = lambda n, s: ImageFont.truetype(os.path.join(A.fonts, n), s)
f_species = F('instrument-serif-latin-400-italic.woff', 34)
f_title = F('instrument-sans-latin-600-normal.woff', 54)
f_mono = F('ibm-plex-mono-latin-600-normal.woff', 24)
f_text = F('instrument-sans-latin-400-normal.woff', 26)
f_big = F('instrument-serif-latin-400-italic.woff', 110)
f_sub = F('ibm-plex-mono-latin-400-normal.woff', 26)
f_cred = F('instrument-sans-latin-400-normal.woff', 25)
f_credh = F('instrument-sans-latin-600-normal.woff', 25)
GOLD = (255, 200, 90); INK = (227, 232, 242); MUTED = (150, 160, 180)

def clean(s): return s.replace('≈ ', '').replace('≈', '').replace('’', "'")
TIMES = {'liver': '3° giorno circa', 'lungs': '7° giorno circa', 'swallow': '10° giorno circa'}

def anat_frame(i):
    p = os.path.join(A.anat, A.anat_name.format(i + A.anat_offset))
    if not os.path.exists(p):  # prove: usa il fotogramma più vicino disponibile
        for d in range(1, 60):
            for j in (i - d, i + d):
                q = os.path.join(A.anat, A.anat_name.format(j))
                if os.path.exists(q): p = q; break
            else: continue
            break
    im = Image.open(p).convert('RGB')
    if im.size != (W, H):
        im = im.resize((W, H), Image.LANCZOS).filter(ImageFilter.UnsharpMask(radius=1.6, percent=55, threshold=2))
    return np.asarray(im).astype(np.float32) / 255

def blur(a, r):
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    return np.asarray(im.filter(ImageFilter.GaussianBlur(r))).astype(np.float32) / 255

VIG = None
def vignette():
    global VIG
    if VIG is None:
        y, x = np.mgrid[0:H, 0:W]; d = np.sqrt(((x - W / 2) / (W / 2)) ** 2 + ((y - H / 2) / (H / 2)) ** 2)
        VIG = (1 - 0.35 * np.clip(d - 0.55, 0, 1) ** 1.5)[..., None].astype(np.float32)
    return VIG

def chip(d, x, y, text, font):
    tw = d.textlength(text, font=font); h = font.size + 14
    d.rounded_rectangle((x, y, x + tw + 26, y + h), radius=h // 2, fill=(255, 200, 90, 40))
    d.text((x + 13, y + 6), text, font=font, fill=GOLD + (255,))
    return x + tw + 26

def caption(img, i):
    fr = FR[i]; k = fr['stage']; st = ST[k]
    # dissolvenza a ogni cambio di fase
    t_in = 0
    j = i
    while j > 0 and FR[j - 1]['stage'] == k and t_in < 12: j -= 1; t_in += 1
    a = min(1.0, t_in / 12) if i > 0 else 1.0
    layer = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(layer)
    x0, y0 = 72, H - 232
    # velo scuro dietro al testo per la leggibilità
    sh = Image.new('L', (W, H), 0); ImageDraw.Draw(sh).rectangle((0, H - 300, 980, H), fill=150)
    sh = sh.filter(ImageFilter.GaussianBlur(70))
    d.text((x0, y0), 'Ascaris lumbricoides', font=f_species, fill=MUTED + (255,))
    xx = x0; yy = y0 + 50
    d.text((xx, yy + 4), f'{k + 1:02d} / {len(ST)}', font=f_mono, fill=GOLD + (int(255 * a),))
    tm = TIMES.get(st['k'], clean(st['time']))
    if tm: chip(d, xx + 130, yy, tm, f_mono)
    d.text((x0, yy + 50), clean(st['title']), font=f_title, fill=INK + (int(255 * a),))
    # barra di avanzamento del ciclo
    prog = i / len(FR); bw = 520
    d.rectangle((x0, H - 52, x0 + bw, H - 50), fill=(150, 170, 210, 70))
    d.rectangle((x0, H - 52, x0 + int(bw * prog), H - 50), fill=GOLD + (230,))
    base = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).convert('RGBA')
    base = Image.composite(Image.new('RGBA', (W, H), (2, 4, 9, 255)), base, sh.point(lambda v: v * 0.55))
    base.alpha_composite(layer)
    return base

def intro(img_rgba, i):
    t = i / FPS
    a = 1.0 if t < 2.2 else max(0.0, 1 - (t - 2.2) / 0.8)
    if a <= 0: return img_rgba
    veil = Image.new('L', (W, H), 0); ImageDraw.Draw(veil).ellipse((W * 0.22, H * 0.18, W * 0.78, H * 0.58), fill=int(170 * a))
    img_rgba = Image.composite(Image.new('RGBA', (W, H), (2, 4, 9, 255)), img_rgba, veil.filter(ImageFilter.GaussianBlur(90)))
    layer = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(layer)
    s1, s2 = 'Ascaris lumbricoides', 'CICLO BIOLOGICO'
    w1 = d.textlength(s1, font=f_big); w2 = d.textlength(s2, font=f_sub)
    d.text(((W - w2) / 2 + 0, H * 0.30), s2, font=f_sub, fill=GOLD + (int(255 * a),))
    d.text(((W - w1) / 2, H * 0.30 + 46), s1, font=f_big, fill=INK + (int(255 * a),))
    img_rgba.alpha_composite(layer); return img_rgba

def frame(i):
    base = anat_frame(i)
    ovi = Image.open(os.path.join(A.over, f'b_{i:05d}.png')).convert('RGBA')
    ov = np.asarray(ovi.filter(ImageFilter.MedianFilter(3))).astype(np.float32) / 255  # il livello è senza denoiser
    a = ov[..., 3:4]; rgb = ov[..., :3]
    out = base * (1 - a) + rgb * a
    lum = np.clip((rgb.min(axis=2, keepdims=True) - 0.80) / 0.2, 0, 1) * a  # solo le parti emissive (quasi bianche)
    gold = np.array([1.0, 0.66, 0.24], np.float32)
    glow = 0.9 * blur(np.repeat(lum, 3, 2), 5) + 0.75 * blur(np.repeat(lum, 3, 2), 18) + 0.35 * blur(np.repeat(lum, 3, 2), 48)
    out = out + glow * gold
    out = 1 - (1 - np.clip(out, 0, 4)) * 1.0 if False else out
    out = out / (1 + 0.08 * out)  # compressione morbida delle alte luci
    out = out * vignette()
    img = caption(out, i)
    return img  # titolo iniziale rimosso su richiesta

def outro(last):
    """5 s finali: ultimo fotogramma attenuato e crediti."""
    base = Image.open(last).convert('RGBA')
    dark = Image.new('RGBA', (W, H), (3, 5, 10, 255))
    lines = [
        ('h', 'Fonti'),
        ('t', 'Holland CV (2021). Parasitology 148:1–9. doi:10.1017/S0031182021000366'),
        ('t', 'Roepstorff A et al. (1997). Parasitology 115:443–452. doi:10.1017/s0031182097001480'),
        ('t', 'Phuphisut O et al. (2022). Parasites & Vectors 15:477. doi:10.1186/s13071-022-05602-2'),
        ('t', 'Xie TH et al. (2024). Frontiers in Medicine 11:1525301. doi:10.3389/fmed.2024.1525301'),
        ('s', ''),
        ('t', 'Tempi della migrazione da infezioni sperimentali con A. suum nel maiale. Uova e larve ingrandite; vermi adulti in scala reale.'),
        ('s', ''),
        ('h', 'Anatomia'),
        ('t', 'BodyParts3D © The Database Center for Life Science (CC BY 4.0), via open-anatomy-3d (CC BY 4.0).'),
        ('t', 'Modificata: selezione delle strutture e materiali. Render: Blender / Cycles.'),
    ]
    frames = []
    for n in range(5 * FPS):
        t = n / FPS; k = min(1, t / 0.8)
        im = Image.blend(base, dark, 0.78 * k)
        layer = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(layer)
        al = int(255 * min(1, max(0, (t - 0.3) / 0.7)))
        y = 250
        d.text((180, y - 110), 'Ascaris lumbricoides', font=F('instrument-serif-latin-400-italic.woff', 64), fill=INK + (al,))
        for kind, s in lines:
            if kind == 'h': d.text((180, y), s.upper(), font=f_sub, fill=GOLD + (al,)); y += 44
            elif kind == 't': d.text((180, y), s, font=f_cred, fill=(205, 212, 226, al)); y += 38
            else: y += 22
        im.alpha_composite(layer); frames.append(im)
    return frames

a, b, s = map(int, A.frames.split(':'))
for i in range(a, b, s):
    p = os.path.join(A.out, f'c_{i:05d}.png')
    if os.path.exists(p): continue
    frame(i).convert('RGB').save(p)
    print('composed', i, flush=True)
if A.outro:
    last = os.path.join(A.out, f'c_{len(FR) - 1:05d}.png')
    for n, im in enumerate(outro(last)):
        im.convert('RGB').save(os.path.join(A.out, f'c_{len(FR) + n:05d}.png'))
    print('outro ok')
