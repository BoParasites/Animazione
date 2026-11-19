#!/usr/bin/env python3
"""
Cercarie di Schistosoma in acqua dolce, in controluce (luce trasmessa).

Rendering "ottico semplificato" invece che disegnato:
  * ogni oggetto ha uno spessore 3D (ellissoidi, tubi) e un assorbimento;
  * il contrasto nasce da assorbimento + rifrazione: il termine di fase è
    approssimato con l'equazione del trasporto d'intensità (contrasto ∝
    defocus × laplaciano dello spessore), per cui gli aloni chiari/scuri si
    invertono passando attraverso il piano di fuoco, come in campo chiaro;
  * profondità di campo: sfocatura gaussiana proporzionale alla distanza dal
    piano di fuoco, con leggera aberrazione cromatica assiale;
  * otturatore della camera (4 ms) con campionamento temporale della coda,
    rumore del sensore, vignettatura, polvere sull'ottica.

Le dimensioni (µm) sono indicative per S. mansoni; frequenza di battito e
velocità sono stimate, non misurate.

Uso:
  python3 render_realistico.py --preview 0,90,240 --outdir anteprime
  python3 render_realistico.py --video cercarie_realistico.mp4 --seconds 20
"""
import argparse
import math
import os
import subprocess
import sys
import time
import multiprocessing as mp

import numpy as np
import cv2

cv2.setNumThreads(1)

# ---------------------------------------------------------------- parametri
W, H = 1920, 1080
FPS = 30
PX = 1.0                      # px per µm -> campo ~1.9 x 1.1 mm
FW, FH = W / PX, H / PX       # campo in µm
EXPOSURE = 1 / 250            # s
TAIL_SAMPLES = 10             # campioni temporali della coda durante l'esposizione

# morfologia (µm, indicativa)
BODY_A = 88                   # semiasse lungo del corpo (lunghezza ~175)
BODY_R1, BODY_R2 = 33, 27     # semiassi trasversali
TAIL_L = 215
TAIL_R0, TAIL_R1 = 17, 10
FURCA_L = 72
FURCA_R0, FURCA_R1 = 7.5, 3.0

# ottica
SIGMA0 = 0.95                 # px, risoluzione/PSF a fuoco
KDOF = 0.021                  # px di sfocatura per µm di defocus
DZ0 = -90.0                   # µm, leggero defocus intrinseco (condensatore)
KAPPA = 0.042                 # guadagno del termine di fase
CHROMA = np.array([-12.0, 0.0, 18.0], np.float32)   # defocus extra per canale R,G,B
SCAT = 0.0004                 # oscuramento per µm di spessore (diffusione)

ABS_CERC = np.array([0.80, 0.95, 1.30], np.float32)   # tinta bruno-ambrata tenue
ABS_ALGA = np.array([1.05, 0.72, 1.30], np.float32)   # verde clorofilla
ABS_DIATOM = np.array([0.75, 0.95, 1.55], np.float32) # bruno-dorato
ABS_MINERAL = np.array([1.0, 1.05, 1.15], np.float32)
ABS_ORGANIC = np.array([0.85, 0.95, 1.25], np.float32)

N_CERC = 12
THRASH = False   # cercarie che si agitano sul posto: disattivate, il movimento non era convincente


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


# ---------------------------------------------------------------- simulazione
class Cerc:
    pass


def new_cerc(rng, z):
    c = Cerc()
    c.pos = np.array([rng.uniform(-100, FW + 100), rng.uniform(0.08, 0.95) * FH, z], float)
    c.e = unit(np.array([rng.normal(0, 0.4), -1.0, rng.normal(0, 0.3)]))
    c.roll = rng.uniform(0, 2 * math.pi)
    c.roll_rate = 0.0
    c.swim = rng.random() < 0.45
    c.timer = rng.uniform(0.2, 2.0 if c.swim else 4.0)
    c.dir = unit(np.array([rng.normal(0, 0.5), -1.0, rng.normal(0, 0.25)]))
    c.rest_dir = unit(np.array([rng.normal(0, 0.3), -1.0, rng.normal(0, 0.3)]))
    c.sink_to = rng.uniform(0.3, 0.9) * FH
    c.freq = rng.uniform(17, 24)
    c.phase = rng.uniform(0, 2 * math.pi)
    c.dphase = 0.0
    c.amp = 1.0 if c.swim else 0.1
    c.spread = 1.05 if c.swim else 0.12
    c.pulse = rng.uniform(0, 2 * math.pi)
    c.psi = rng.uniform(0, 2 * math.pi)
    c.speed = 650.0 if c.swim else 0.0
    c.twitch = 0.0
    c.bend = rng.normal(0, 0.06)
    c.bend_target = c.bend
    c.drift = np.array([rng.normal(0, 10), 0.0, rng.normal(0, 4)])
    c.v = np.zeros(3)
    c.size = rng.normal(1.0, 0.05)
    c.vmax = rng.uniform(560, 760)          # µm/s, stimata
    c.fwd = rng.random() < 0.5          # fase passiva: nuoto a testa avanti (True) o solo affondamento
    c.head_first = False
    c.arc = 0.0
    c.thrasher = False
    c.elong, c.girth, c.taper, c.bbend = 1.0, 1.0, 0.0, 0.0
    c.static = make_static(rng)
    return c


def make_static(rng):
    """Strutture interne fisse dell'individuo, in coordinate locali normalizzate del corpo
    (u lungo l'asse, +1 = posteriore verso la coda; v, w trasversali)."""
    s = {}
    glands = []
    # 2 paia preacetabolari (granuli grossi) + 3 paia postacetabolari (granuli fini)
    for side in (-1, 1):
        for k, u in enumerate((-0.12, 0.06, 0.32, 0.48, 0.62)):
            coarse = k < 2
            gv = side * rng.uniform(0.30, 0.46)
            gw = rng.normal(0, 0.15)
            r = rng.uniform(9, 12) if coarse else rng.uniform(8, 10.5)
            n = 55 if coarse else 85
            pts = rng.normal(0, 0.45, size=(n, 3))
            glands.append(dict(u=u + rng.normal(0, 0.03), v=gv, w=gw, r=r, coarse=coarse, pts=pts,
                               gs=rng.uniform(1.6, 2.4) if coarse else rng.uniform(0.9, 1.3)))
    s['glands'] = glands
    # dotti: dal gruppo ghiandolare all'apice
    ducts = []
    for side in (-1, 1):
        for j in range(4):
            ducts.append((side, rng.uniform(0.12, 0.38), rng.uniform(-0.1, 0.45), rng.normal(0, 0.1)))
    s['ducts'] = ducts
    # granulazione diffusa del parenchima
    g = rng.normal(0, 0.5, size=(950, 3))
    g = g[np.sum(g ** 2, axis=1) < 0.8]
    s['paren'] = g
    # nuclei lungo la coda
    s['tailnuc'] = np.column_stack([rng.uniform(0.03, 0.98, 70), rng.normal(0, 0.42, 70), rng.normal(0, 0.42, 70)])
    return s


def start_swim(c, rng):
    c.swim = True
    c.timer = rng.uniform(0.6, 2.0)
    zb = -np.sign(c.pos[2]) * min(abs(c.pos[2]) / 1600, 1) * 0.25
    c.dir = unit(np.array([np.clip(rng.normal(0, 0.55), -1.1, 1.1), -1.0, rng.normal(zb, 0.22)]))
    c.freq = rng.uniform(17, 24)
    c.roll_rate = rng.choice([-1, 1]) * 2 * math.pi * rng.uniform(0.6, 1.5)


def start_rest(c, rng):
    c.swim = False
    c.timer = rng.uniform(1.5, 5.0)
    c.rest_dir = unit(np.array([rng.normal(0, 0.28), -1.0, rng.normal(0, 0.28)]))
    c.sink_to = rng.uniform(0.25, 0.9) * FH
    c.drift = np.array([rng.normal(0, 10), 0.0, rng.normal(0, 4)])
    c.bend_target = rng.normal(0, 0.07)
    c.head_first = c.fwd and rng.random() < 0.85
    if c.head_first:
        # nuoto corpo-avanti verso il basso, con traiettoria ad arco (Panzner et al. 2025, fig. 3)
        c.timer = rng.uniform(2.6, 3.8)
        c.rest_dir = unit(np.array([rng.normal(0, 0.45), -1.0, rng.normal(0, 0.25)]))
        c.arc = rng.choice([-1, 1]) * rng.uniform(0.55, 0.9)


def start_tmode(c, rng):
    """Ciclo osservato nel video di riferimento: allungamento della testa -> contrazione ->
    colpi laterali rapidissimi quasi sul posto -> (pausa) -> di nuovo."""
    order = {'calm': 'extend', 'extend': 'contract', 'contract': 'flick'}
    if c.tmode == 'flick':
        c.tmode = 'calm' if rng.random() < 0.35 else 'extend'
    else:
        c.tmode = order[c.tmode]
    c.ttimer = {'calm': rng.uniform(0.6, 1.4), 'extend': rng.uniform(0.45, 0.8),
                'contract': rng.uniform(0.2, 0.35), 'flick': rng.uniform(0.7, 1.5)}[c.tmode]
    c.bend_sign = rng.choice([-1, 1])
    c.jerk = 0.0


def step_thrash(c, dt, t, rng):
    c.ttimer -= dt
    if c.ttimer <= 0:
        start_tmode(c, rng)
    m = c.tmode
    if m == 'extend':
        tg = (1.75, 0.60, 0.85, 0.35 * c.bend_sign, 0.25, 6.0, 0.3)
        k = 6.0
    elif m == 'contract':
        tg = (0.72, 1.32, 0.0, 0.0, 0.3, 6.0, 0.3)
        k = 11.0
    elif m == 'flick':
        osc = math.sin(2 * math.pi * 6.0 * t + c.psi)
        tg = (1.15 + 0.25 * math.sin(2 * math.pi * 3.1 * t), 0.88, 0.35, 0.65 * osc, 1.0, c.freq, 1.0)
        k = 16.0
    else:
        tg = (1.0, 1.0, 0.0, 0.0, 0.08, 0.5, 0.3)
        k = 4.0
    el, gi, ta, bb, am, f, spr = tg
    g = 1 - math.exp(-dt * k)
    c.elong += (el - c.elong) * g
    c.girth += (gi - c.girth) * g
    c.taper += (ta - c.taper) * g
    c.bbend += (bb - c.bbend) * g
    c.amp += (am - c.amp) * (1 - math.exp(-dt * 12))
    c.spread += (spr - c.spread) * (1 - math.exp(-dt * 6))
    c.dphase = 2 * math.pi * f
    c.phase += c.dphase * dt
    c.pulse = 0.0
    if m == 'flick':
        c.jerk -= dt
        if c.jerk <= 0:          # scatti bruschi di direzione
            c.jerk = rng.uniform(0.07, 0.18)
            ang = rng.choice([-1, 1]) * rng.uniform(0.5, 1.3)
            ca, sa = math.cos(ang), math.sin(ang)
            e = c.e
            c.rest_dir = unit(np.array([e[0] * ca - e[1] * sa, e[0] * sa + e[1] * ca, e[2] * 0.5 + rng.normal(0, 0.15)]))
            c.roll_rate = rng.choice([-1, 1]) * 2 * math.pi * rng.uniform(1.0, 2.5)
        rate = 14.0
    else:
        c.roll_rate *= math.exp(-dt * 3)
        rate = 3.0
    c.roll += c.roll_rate * dt
    c.e = unit(c.e + (c.rest_dir - c.e) * (1 - math.exp(-dt * rate)))
    c.anchor = c.anchor + np.array([4.0 * math.sin(t * 0.2), 6.0, 0.0]) * dt
    v = (c.anchor - c.pos) * 2.5
    if m == 'flick':
        v = v + rng.normal(0, 140, 3) * np.array([1, 1, 0.2])
    c.v = v
    c.pos = c.pos + v * dt


def step_cerc(c, dt, t, rng):
    if c.thrasher:
        step_thrash(c, dt, t, rng)
        return
    c.timer -= dt
    if c.swim:
        if c.timer <= 0 or c.pos[1] < -40:
            start_rest(c, rng)
    else:
        if (c.timer <= 0 and c.pos[1] >= c.sink_to) or c.pos[1] > FH + 80:
            start_swim(c, rng)
    if not c.swim and c.twitch <= 0 and rng.random() < dt * 0.15:
        c.twitch = rng.uniform(0.12, 0.3)
    c.twitch = max(0.0, c.twitch - dt)

    sw = c.swim
    hf = (not sw) and c.head_first
    amp_t = 1.0 if sw else (0.9 if hf else (0.55 if c.twitch > 0 else 0.06))
    c.amp += (amp_t - c.amp) * (1 - math.exp(-dt * (14 if sw or hf or c.twitch > 0 else 4)))
    # furche divaricate nella fase attiva coda-avanti, tenute unite nella fase passiva
    c.spread += ((1.05 if sw else 0.12) - c.spread) * (1 - math.exp(-dt * 4))
    f = c.freq if sw else (0.75 * c.freq if hf else (12.0 if c.twitch > 0 else 0.5))
    c.dphase = 2 * math.pi * f
    c.phase += c.dphase * dt
    c.pulse += dt * (2.4 if sw else 0.7)
    c.bend += (c.bend_target * (0 if sw else 1) - c.bend) * (1 - math.exp(-dt * 1.5))
    if not sw:
        c.roll_rate += (0.25 * math.sin(t * 0.3 + c.psi) - c.roll_rate) * (1 - math.exp(-dt * 2))
    c.roll += c.roll_rate * dt

    if sw:
        c.psi += 2 * math.pi * 1.4 * dt
        a = unit(np.cross(c.dir, [0, 0, 1.0]))
        b = np.cross(c.dir, a)
        target = unit(c.dir + 0.22 * (math.cos(c.psi) * a + math.sin(c.psi) * b))
        rate = 4.0
    else:
        if hf:
            ca, sa = math.cos(c.arc * dt), math.sin(c.arc * dt)
            d = c.rest_dir
            d = unit(np.array([d[0] * ca - d[1] * sa, d[0] * sa + d[1] * ca, d[2]]))
            if d[1] > -0.4:          # resta diretta verso il basso (coda in alto)
                c.arc = -c.arc
            c.rest_dir = d
        target = c.rest_dir
        rate = 2.0 if hf else 1.3
    c.e = unit(c.e + (target - c.e) * (1 - math.exp(-dt * rate)))

    sp_t = c.vmax * (0.4 + 0.6 * c.amp) if sw else (-0.8 * c.vmax if hf else 0.0)   # negativa: verso la testa
    c.speed += (sp_t - c.speed) * (1 - math.exp(-dt * (5 if sw or hf else 3)))
    v = c.e * c.speed
    if not sw and not hf:
        v = v + np.array([0.0, 115.0 * (1 - c.amp), 0.0])        # affondamento passivo, corpo in avanti
    v = v + c.drift + np.array([8 * math.sin(t * 0.11), 0, 0])
    # limiti morbidi in profondità
    z = c.pos[2]
    if abs(z) > 1500:
        v[2] -= (z - math.copysign(1500, z)) * 2.0
    c.v = v
    c.pos = c.pos + v * dt
    m = 420
    if c.pos[0] < -m:
        c.pos[0] += FW + 2 * m
    elif c.pos[0] > FW + m:
        c.pos[0] -= FW + 2 * m


def snapshot(c):
    return dict(pos=c.pos.copy(), v=c.v.copy(), e=c.e.copy(), roll=c.roll, roll_rate=c.roll_rate,
                phase=c.phase, dphase=c.dphase, amp=c.amp, spread=c.spread, pulse=c.pulse,
                bend=c.bend, size=c.size, static=c.static,
                elong=c.elong, girth=c.girth, taper=c.taper, bbend=c.bbend)


def simulate(n_frames, seed, warm=6.0):
    rng = np.random.default_rng(seed)
    depths = [-1300, -820, -520, -300, -150, -60, 10, 70, 160, 360, 700, 1150]
    rng.shuffle(depths)
    cs = [new_cerc(rng, z) for z in depths[:N_CERC]]
    sub = 8
    dt = 1 / (FPS * sub)
    t = -warm
    for _ in range(int(warm / dt)):
        for c in cs:
            step_cerc(c, dt, t, rng)
        t += dt
    # protagoniste vicino al piano di fuoco, nella zona centrale dell'inquadratura
    heroes = [(0.34 * FW, 0.40 * FH, -15.0), (0.64 * FW, 0.55 * FH, 30.0), (0.50 * FW, 0.22 * FH, 75.0)]
    for c, (x, y, z) in zip(cs, heroes):
        c.pos = np.array([x, y, z])
        start_rest(c, rng)
        c.e = c.rest_dir.copy()
        c.amp = 0.06
        c.speed = 0.0
        c.timer = rng.uniform(1.0, 3.5)
        c.sink_to = y + rng.uniform(80, 200)
    # due individui che si agitano quasi sul posto: uno a fuoco, uno un po' fuori fuoco
    thrashers = ((cs[1], (0.76 * FW, 0.62 * FH, 20.0), 0.4), (cs[3], (0.18 * FW, 0.70 * FH, 190.0), 2.5)) if THRASH else ()
    for c, (x, y, z), t0 in thrashers:
        c.thrasher = True
        c.pos = np.array([x, y, z])
        c.anchor = c.pos.copy()
        c.e = unit(np.array([0.5, -1.0, 0.1]))
        c.rest_dir = c.e.copy()
        c.v = np.zeros(3)
        c.tmode = 'calm'
        c.ttimer = t0
        c.jerk = 0.0
        c.bend_sign = 1
        c.amp = 0.08
    frames = []
    for f in range(n_frames):
        frames.append([snapshot(c) for c in cs])
        for _ in range(sub):
            for c in cs:
                step_cerc(c, dt, t, rng)
            t += dt
    return frames


# ---------------------------------------------------------------- particelle e organismi accessori
def make_particles(seed):
    rng = np.random.default_rng(seed + 1000)
    P = []

    def base(kind, n_ell, absc, dens):
        return dict(kind=kind, p0=np.array([rng.uniform(0, FW), rng.uniform(0, FH), rng.uniform(-1600, 1600)]),
                    v=np.array([rng.normal(0, 6), rng.uniform(2, 14), rng.normal(0, 3)]),
                    jit=rng.uniform(0.3, 2.0), jf=rng.uniform(0.5, 2.5, 2), jp=rng.uniform(0, 6.28, 2),
                    absc=absc, dens=dens, rot0=rng.uniform(0, 6.28), rotv=rng.normal(0, 0.15),
                    tilt=rng.uniform(0, 1.2), ell=[])

    for _ in range(330):           # particolato minerale / organico fine
        org = rng.random() < 0.6
        p = base('fine', 1, ABS_ORGANIC if org else ABS_MINERAL, rng.uniform(0.004, 0.02) if org else rng.uniform(0.03, 0.08))
        r = rng.lognormal(math.log(2.2), 0.5)
        p['ell'] = [(np.zeros(3), np.array([r * rng.uniform(1, 1.6), r, r * rng.uniform(0.7, 1)]))]
        P.append(p)
    for _ in range(40):            # fiocchi organici irregolari
        p = base('floc', 0, ABS_ORGANIC, rng.uniform(0.006, 0.02))
        n = rng.integers(3, 8)
        for k in range(n):
            r = rng.uniform(2.5, 7)
            p['ell'].append((rng.normal(0, 6, 3), np.array([r, r * rng.uniform(0.6, 1), r * rng.uniform(0.6, 1)])))
        p['jit'] = rng.uniform(0.1, 0.5)
        P.append(p)
    for _ in range(14):            # alghe unicellulari (verdi)
        p = base('alga', 1, ABS_ALGA, rng.uniform(0.010, 0.022))
        r = rng.uniform(5, 10)
        p['ell'] = [(np.zeros(3), np.array([r * rng.uniform(1, 1.25), r, r]))]
        p['jit'] = 0.2
        P.append(p)
    for _ in range(9):             # diatomee pennate
        p = base('diatom', 1, ABS_DIATOM, rng.uniform(0.010, 0.018))
        L = rng.uniform(22, 45)
        p['ell'] = [(np.zeros(3), np.array([L, L * rng.uniform(0.16, 0.25), L * 0.15]))]
        p['jit'] = 0.1
        P.append(p)
    for _ in range(3):             # frammenti di alga filamentosa, lontani
        p = base('filament', 0, ABS_ALGA, 0.012)
        p['p0'][2] = rng.choice([-1, 1]) * rng.uniform(800, 1500)
        n = rng.integers(8, 15)
        cl = rng.uniform(14, 20)
        for k in range(n):
            p['ell'].append((np.array([(k - n / 2) * cl * 1.02, 0.0, 0.0]), np.array([cl * 0.55, 7.0, 7.0])))
        p['v'] *= 0.3
        p['jit'] = 0.05
        P.append(p)
    return P


def particle_state(p, t):
    pos = p['p0'] + p['v'] * t
    pos[0] += p['jit'] * math.sin(p['jf'][0] * t + p['jp'][0]) + 0.6 * p['jit'] * math.sin(2.7 * p['jf'][1] * t)
    pos[1] += p['jit'] * math.sin(p['jf'][1] * t + p['jp'][1])
    m = 200
    pos[0] = (pos[0] + m) % (FW + 2 * m) - m
    pos[1] = (pos[1] + m) % (FH + 2 * m) - m
    ang = p['rot0'] + p['rotv'] * t
    ca, sa = math.cos(ang), math.sin(ang)
    ct, st = math.cos(p['tilt']), math.sin(p['tilt'])
    Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
    return pos, Rz @ Rx


# ---------------------------------------------------------------- primitive di rasterizzazione
def ell_extent(R, radii):
    Minv = (R * radii ** 2) @ R.T
    return math.sqrt(Minv[0, 0]), math.sqrt(Minv[1, 1])


def add_ellipsoid(T, ox, oy, c, R, radii, scale=1.0, mode='max'):
    """Spessore (µm) lungo la vista di un ellissoide centrato in c (µm), assi = colonne di R."""
    radii = np.asarray(radii, float)
    M = (R / radii ** 2) @ R.T
    ex, ey = ell_extent(R, radii)
    cx = c[0] * PX - ox
    cy = c[1] * PX - oy
    h, w = T.shape
    x0 = max(0, int(math.floor(cx - ex * PX)) - 1)
    x1 = min(w, int(math.ceil(cx + ex * PX)) + 2)
    y0 = max(0, int(math.floor(cy - ey * PX)) - 1)
    y1 = min(h, int(math.ceil(cy + ey * PX)) + 2)
    if x1 <= x0 or y1 <= y0:
        return
    dx = ((np.arange(x0, x1, dtype=np.float32) - cx) / PX)[None, :]
    dy = ((np.arange(y0, y1, dtype=np.float32) - cy) / PX)[:, None]
    qa = M[2, 2]
    qb = M[0, 2] * dx + M[1, 2] * dy
    qc = M[0, 0] * dx * dx + 2 * M[0, 1] * dx * dy + M[1, 1] * dy * dy - 1.0
    disc = qb * qb - qa * qc
    th = (2.0 / qa) * np.sqrt(np.maximum(disc, 0.0)) * scale
    reg = T[y0:y1, x0:x1]
    if mode == 'max':
        np.maximum(reg, th, out=reg)
    else:
        reg += th.astype(np.float32)


def add_tube(T, pts, radii):
    """pts: (N,2) in px del patch; radii: raggi in px. Spessore in µm, unione per massimo."""
    h, w = T.shape
    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        ra, rb = radii[i], radii[i + 1]
        rm = max(ra, rb)
        x0 = max(0, int(math.floor(min(ax, bx) - rm)) - 1)
        x1 = min(w, int(math.ceil(max(ax, bx) + rm)) + 2)
        y0 = max(0, int(math.floor(min(ay, by) - rm)) - 1)
        y1 = min(h, int(math.ceil(max(ay, by) + rm)) + 2)
        if x1 <= x0 or y1 <= y0:
            continue
        X = np.arange(x0, x1, dtype=np.float32)[None, :]
        Y = np.arange(y0, y1, dtype=np.float32)[:, None]
        abx, aby = bx - ax, by - ay
        L2 = abx * abx + aby * aby + 1e-9
        t = np.clip(((X - ax) * abx + (Y - ay) * aby) / L2, 0.0, 1.0)
        ddx = X - (ax + t * abx)
        ddy = Y - (ay + t * aby)
        rr = ra + (rb - ra) * t
        th = (2.0 / PX) * np.sqrt(np.maximum(rr * rr - ddx * ddx - ddy * ddy, 0.0))
        reg = T[y0:y1, x0:x1]
        np.maximum(reg, th, out=reg)


def splat(img, xs, ys, wts):
    h, w = img.shape
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    fx = (xs - x0).astype(np.float32)
    fy = (ys - y0).astype(np.float32)
    wts = np.broadcast_to(np.asarray(wts, np.float32), xs.shape)
    for ddx, ddy, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi = x0 + ddx
        yi = y0 + ddy
        m = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        np.add.at(img, (yi[m], xi[m]), (wts * ww)[m])


def blur(img, sigma):
    if sigma <= 3.5:
        return cv2.GaussianBlur(img, (0, 0), sigma), 1
    f = max(1, int(sigma / 1.8))
    h, w = img.shape
    small = cv2.resize(img, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
    s2 = math.sqrt(max(sigma * sigma / (f * f) - 0.3, 0.4))
    return cv2.GaussianBlur(small, (0, 0), s2), f


def upsample(img, f, shape):
    if f == 1:
        return img
    return cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def sigma_for(dz):
    return math.sqrt(SIGMA0 ** 2 + (KDOF * abs(dz)) ** 2)


def optics(T, A, dz, absc):
    """Perturbazione del logaritmo dell'intensità (h,w,3) prodotta da uno strato a profondità dz."""
    sigma = sigma_for(dz)
    Tb, f = blur(T, sigma)
    lap = cv2.Laplacian(Tb, cv2.CV_32F, ksize=1) / (f * f)
    Ab, _ = blur(A, sigma)
    shape = T.shape
    Tb = upsample(Tb, f, shape)
    lap = upsample(lap, f, shape)
    Ab = upsample(Ab, f, shape)
    out = np.empty(shape + (3,), np.float32)
    base = -SCAT * Tb
    for ch in range(3):
        k = KAPPA * (dz + DZ0 + CHROMA[ch]) / 100.0
        out[..., ch] = base - Ab * absc[ch] + k * lap
    return out


def composite(pert, ox, oy, patch):
    h, w = patch.shape[:2]
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(W, ox + w), min(H, oy + h)
    if x1 <= x0 or y1 <= y0:
        return
    pert[y0:y1, x0:x1] += patch[y0 - oy:y1 - oy, x0 - ox:x1 - ox]


# ---------------------------------------------------------------- geometria della cercaria
def frame_axes(e, roll):
    ref = np.array([0, 0, 1.0]) if abs(e[2]) < 0.9 else np.array([1.0, 0, 0])
    a = unit(np.cross(e, ref))
    b = np.cross(e, a)
    n1 = math.cos(roll) * a + math.sin(roll) * b
    n2 = np.cross(e, n1)
    return n1, n2


def tail_geom(s, tau):
    sz = s['size']
    e = s['e']
    J = s['pos'] + s['v'] * tau
    phase = s['phase'] + s['dphase'] * tau
    roll = s['roll'] + s['roll_rate'] * tau
    amp = s['amp']
    n1, n2 = frame_axes(e, roll)
    N = 18
    sv = np.linspace(0, 1, N)
    A = 62 * amp * sz
    env = 0.05 + 0.95 * sv ** 1.6
    y = A * env * np.sin(2 * math.pi * 1.1 * sv + phase) + s['bend'] * TAIL_L * sv ** 2
    y2 = 0.18 * A * env * np.cos(2 * math.pi * 1.1 * sv + phase)
    L = TAIL_L * sz * (1 - 0.12 * amp)
    P = J[None, :] + np.outer(sv * L, e) + np.outer(y, n1) + np.outer(y2, n2)
    radii = (TAIL_R0 + (TAIL_R1 - TAIL_R0) * sv) * sz
    # furche
    tip = P[-1]
    tdir = unit(P[-1] - P[-3])
    nf = unit(math.cos(0.5) * n1 + math.sin(0.5) * n2)
    nf = unit(nf - np.dot(nf, tdir) * tdir)
    furcae = []
    for side in (-1, 1):
        d = unit(tdir * math.cos(s['spread']) + side * nf * math.sin(s['spread']))
        q = np.linspace(0, 1, 7)
        bend = 0.22 * amp * math.sin(phase + 1.3 + 0.6 * side) - 0.08
        Q = tip[None, :] + np.outer(q * FURCA_L * sz, d) + np.outer(q ** 2 * FURCA_L * sz * bend, tdir)
        fr = (FURCA_R0 + (FURCA_R1 - FURCA_R0) * q) * sz
        furcae.append((Q, fr))
    return P, radii, furcae, (n1, n2), sv


def body_geom(s):
    sz = s['size']
    e = s['e']
    n1, n2 = frame_axes(e, s['roll'])
    rec = 0.10 * s['amp'] * math.sin(s['phase'] + math.pi)
    eb = unit(e * math.cos(rec) + n1 * math.sin(rec))
    n1b = unit(n1 - np.dot(n1, eb) * eb)
    n2b = np.cross(eb, n1b)
    p = math.sin(s['pulse'])
    a = BODY_A * sz * (1 + 0.08 * p) * s['elong']
    r1 = BODY_R1 * sz * (1 - 0.05 * p) * s['girth']
    r2 = BODY_R2 * sz * (1 - 0.05 * p) * s['girth']
    J = s['pos']
    C = J - eb * (a - 6)                   # centro del corpo (l'asse va dal corpo verso la coda)
    R = np.column_stack([eb, n1b, n2b])
    return C, R, a, r1, r2


BEND = [0.0]   # curvatura del corpo dell'individuo in corso di rendering


def bend_off(u, a):
    # curvatura a falce: nulla alla giunzione con la coda (u=+1), massima all'apice (u=-1)
    return 0.25 * BEND[0] * a * (1 - u) ** 2


def body_point(C, R, a, vec):
    return C + R @ vec + R[:, 1] * bend_off(vec[0] / a, a)


def local_to_world(C, R, a, r1, r2, uvw):
    return C[None, :] + (uvw * np.array([a, r1, r2])) @ R.T + np.outer(bend_off(uvw[:, 0], a), R[:, 1])


def render_cerc(pert, s, t):
    C, R, a, r1, r2 = body_geom(s)
    BEND[0] = s['bbend']
    taus = np.linspace(0, EXPOSURE, TAIL_SAMPLES) if s['amp'] > 0.25 else np.array([EXPOSURE / 2])
    tails = [tail_geom(s, tau) for tau in taus]

    # profondità degli strati
    z_body = C[2]
    z_tail = float(np.mean(tails[0][0][:, 2]))
    sig = max(sigma_for(z_body), sigma_for(z_tail))
    margin = int(3 * sig + 8)
    ext = a * (1 + abs(s["bbend"])) + r1
    allp = [C[:2] + d for d in (np.array([ext, ext]), -np.array([ext, ext]))]
    for P, rad, fur, _, _ in tails:
        allp.append(P[:, :2].min(0) - 20)
        allp.append(P[:, :2].max(0) + 20)
        for Q, _ in fur:
            allp.append(Q[:, :2].min(0) - 10)
            allp.append(Q[:, :2].max(0) + 10)
    allp = np.array(allp) * PX
    ox = int(math.floor(allp[:, 0].min())) - margin
    oy = int(math.floor(allp[:, 1].min())) - margin
    w = int(math.ceil(allp[:, 0].max())) + margin - ox
    h = int(math.ceil(allp[:, 1].max())) + margin - oy
    if ox > W or oy > H or ox + w < 0 or oy + h < 0:
        return
    # ritaglio prudente del patch alla parte che può influire sul fotogramma
    cx0, cy0 = max(ox, -margin), max(oy, -margin)
    cx1, cy1 = min(ox + w, W + margin), min(oy + h, H + margin)
    ox, oy, w, h = cx0, cy0, cx1 - cx0, cy1 - cy0
    if w <= 2 or h <= 2:
        return

    # ---- corpo: profilo continuo a uovo (anteriore più affusolato, raccordo posteriore con la coda)
    Tprof = np.zeros((h, w), np.float32)
    Tb = np.zeros((h, w), np.float32)      # strutture interne (si sommano al profilo)
    Ab = np.zeros((h, w), np.float32)
    uu = np.linspace(-1, 1, 33)
    prof = np.sqrt(np.clip(1 - uu ** 2, 0, 1)) ** 0.85 * (1 + 0.10 * uu) * (1 - 0.30 * np.clip(uu - 0.55, 0, 1) / 0.45)
    prof = prof * (1 - 0.85 * s['taper'] * np.clip(-uu, 0, 1) ** 1.3)   # testa allungata e appuntita
    prof = np.maximum(prof, 0.06)
    rmean = 0.5 * (r1 + r2)
    cen = C[None, :] + np.outer(uu * a * 0.97, R[:, 0]) + np.outer(bend_off(uu, a), R[:, 1])
    add_tube(Tprof, cen[:, :2] * PX - np.array([ox, oy]), prof * rmean * PX)
    # organo anteriore (testa): zona più densa
    c = body_point(C, R, a, np.array([-0.70 * a, 0, 0]))
    tmp = np.zeros_like(Tb)
    add_ellipsoid(tmp, ox, oy, c, R, (0.28 * a, 0.62 * r1, 0.62 * r2))
    tmp = cv2.GaussianBlur(tmp, (0, 0), 2.0)
    Tb += 0.10 * tmp
    Ab += 0.0022 * tmp
    # acetabolo (ventosa ventrale): anello
    c = body_point(C, R, a, np.array([0.30 * a, 0.0, 0.62 * r2]))
    tmp[:] = 0
    add_ellipsoid(tmp, ox, oy, c, R, (12, 12, 8), mode='add')
    tmp2 = np.zeros_like(Tb)
    add_ellipsoid(tmp2, ox, oy, c, R, (6.5, 6.5, 5), mode='add')
    ring = cv2.GaussianBlur(np.maximum(tmp - 1.3 * tmp2, 0), (0, 0), 0.8)
    Tb += 0.32 * ring
    Ab += 0.0025 * ring
    # ghiandole di penetrazione con granuli
    st = s['static']
    gx, gy, gw = [], [], []
    for g in st['glands']:
        cg = body_point(C, R, a, np.array([g['u'] * a, g['v'] * r1, g['w'] * r2]))
        tmp[:] = 0
        add_ellipsoid(tmp, ox, oy, cg, R, (g['r'] * 1.2, g['r'], g['r'] * 0.9), mode='add')
        tmp = cv2.GaussianBlur(tmp, (0, 0), 1.2)
        Tb += 0.12 * tmp
        Ab += (0.0016 if g['coarse'] else 0.0010) * tmp
        pts = cg[None, :] + (g['pts'] * g['r']) @ R.T
        gx.append(pts[:, 0] * PX - ox)
        gy.append(pts[:, 1] * PX - oy)
        gw.append(np.full(len(pts), g['gs']))
    gx, gy, gw = np.concatenate(gx), np.concatenate(gy), np.concatenate(gw)
    gran = np.zeros_like(Tb)
    splat(gran, gx, gy, gw)
    # granulazione diffusa
    pp = local_to_world(C, R, a, r1, r2, st['paren'])
    splat(gran, pp[:, 0] * PX - ox, pp[:, 1] * PX - oy, 0.45)
    # dotti ghiandolari verso l'apice
    q = np.linspace(0, 1, 46)[:, None]
    for side, vg, ug, wg in st['ducts']:
        p0 = np.array([ug, side * vg, wg])
        p1 = np.array([-0.35, side * vg * 0.45, wg * 0.5])
        p2 = np.array([-0.97, side * 0.06, 0.0])
        bez = (1 - q) ** 2 * p0 + 2 * (1 - q) * q * p1 + q ** 2 * p2
        pw = local_to_world(C, R, a, r1, r2, bez)
        splat(gran, pw[:, 0] * PX - ox, pw[:, 1] * PX - oy, 0.35)
    gran = cv2.GaussianBlur(gran, (0, 0), 0.45)
    Tb += 8.0 * gran
    Ab += 0.030 * gran

    # ---- coda (media sui campioni temporali dell'esposizione)
    Tt = np.zeros((h, w), np.float32)
    canal = np.zeros((h, w), np.float32)
    inv = 1.0 / len(tails)
    tmp = np.zeros_like(Tt)
    for P, rad, fur, (n1, n2), sv in tails:
        tmp[:] = 0
        pts = P[:, :2] * PX - np.array([ox, oy])
        add_tube(tmp, pts, rad * PX)
        for Q, fr in fur:
            add_tube(tmp, Q[:, :2] * PX - np.array([ox, oy]), fr * PX)
        Tt += tmp * inv
        splat(canal, pts[:, 0], pts[:, 1], 0.9 * inv)
        nuc = st['tailnuc']
        idx = nuc[:, 0] * (len(sv) - 1)
        i0 = np.clip(idx.astype(int), 0, len(sv) - 2)
        fr_ = (idx - i0)[:, None]
        base = P[i0] * (1 - fr_) + P[i0 + 1] * fr_
        rr = (TAIL_R0 + (TAIL_R1 - TAIL_R0) * nuc[:, 0])[:, None]
        npos = base + rr * (nuc[:, 1:2] * n1[None, :] + nuc[:, 2:3] * n2[None, :])
        splat(canal, npos[:, 0] * PX - ox, npos[:, 1] * PX - oy, 0.7 * inv)
    canal = cv2.GaussianBlur(canal, (0, 0), 0.8)
    # unico strato ottico: unione senza giunture tra corpo e coda
    Tbase = np.maximum(Tprof, Tt)
    inside = np.clip(Tbase / 6.0, 0, 1)
    T = Tbase + (Tb + 0.8 * canal) * inside
    A = Tprof * 0.0034 + np.maximum(Tt - Tprof, 0) * 0.0021 + (Ab + 0.012 * canal) * inside
    composite(pert, ox, oy, optics(T, A, 0.6 * z_body + 0.4 * z_tail, ABS_CERC))


# ---------------------------------------------------------------- particelle
def render_particle(pert, p, t):
    pos, Rr = particle_state(p, t)
    sigma = sigma_for(pos[2])
    ext = 0.0
    for off, rad in p['ell']:
        ext = max(ext, np.linalg.norm(off) + rad.max())
    size_px = ext * PX
    # contrasto trascurabile se piccolo e molto sfocato
    if size_px < 1.2 and sigma > 6:
        return
    if (size_px / sigma) ** 2 < 0.004:
        return
    margin = int(3 * sigma + 4)
    cx, cy = pos[0] * PX, pos[1] * PX
    ox = int(math.floor(cx - size_px)) - margin
    oy = int(math.floor(cy - size_px)) - margin
    w = int(2 * size_px) + 2 * margin + 2
    h = w
    if ox > W or oy > H or ox + w < 0 or oy + h < 0:
        return
    T = np.zeros((h, w), np.float32)
    for off, rad in p['ell']:
        add_ellipsoid(T, ox, oy, pos + Rr @ off, Rr, rad)
    A = T * p['dens']
    if p['kind'] == 'alga':
        A *= 1.6
    composite(pert, ox, oy, optics(T, A, pos[2], p['absc']))


# ---------------------------------------------------------------- sfondo e camera
class Static:
    pass


def build_static(seed):
    rng = np.random.default_rng(seed + 77)
    S = Static()
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r2 = ((xx - W * 0.50) / (W * 0.60)) ** 2 + ((yy - H * 0.44) / (W * 0.60)) ** 2
    prof = np.clip(1.0 - 0.50 * r2 - 0.14 * r2 ** 2, 0.25, 1.0)
    noise = rng.normal(0, 1, (H // 60 + 3, W // 60 + 3)).astype(np.float32)
    noise = cv2.resize(cv2.GaussianBlur(noise, (0, 0), 1.2), (W, H), interpolation=cv2.INTER_CUBIC)
    prof = prof * (1 + 0.035 * noise)
    lamp = np.array([1.0, 0.970, 0.90], np.float32)      # lampada calda
    water = np.array([0.96, 0.990, 0.92], np.float32)    # acqua dolce leggermente giallo-verde
    S.I0 = (0.97 * prof[..., None] * lamp * water).astype(np.float32)
    S.noise2 = cv2.resize(cv2.GaussianBlur(rng.normal(0, 1, (H // 90 + 3, W // 90 + 3)).astype(np.float32), (0, 0), 1),
                          (W, H), interpolation=cv2.INTER_CUBIC)
    # polvere sull'ottica: anelli sfocati statici + qualche granello sul sensore
    pert = np.zeros((H, W, 3), np.float32)
    for _ in range(7):
        d = dict(kind='dust', p0=np.array([rng.uniform(0, FW), rng.uniform(0, FH), rng.choice([-1, 1]) * rng.uniform(2200, 3600)]),
                 v=np.zeros(3), jit=0, jf=np.zeros(2), jp=np.zeros(2), absc=ABS_MINERAL, dens=0.05,
                 rot0=0, rotv=0, tilt=0, ell=[(np.zeros(3), np.full(3, rng.uniform(15, 30)))])
        old_kdof = None
        render_particle(pert, d, 0.0)
    for _ in range(5):
        d = dict(kind='dust', p0=np.array([rng.uniform(0, FW), rng.uniform(0, FH), rng.uniform(250, 500)]),
                 v=np.zeros(3), jit=0, jf=np.zeros(2), jp=np.zeros(2), absc=ABS_MINERAL, dens=0.06,
                 rot0=0, rotv=0, tilt=0, ell=[(np.zeros(3), np.full(3, rng.uniform(1.5, 3)))])
        render_particle(pert, d, 0.0)
    S.dust = pert
    return S


def finish(S, pert, t, fi):
    rng = np.random.default_rng(10_000 + fi)
    flick = 1.0 + 0.004 * math.sin(t * 2 * math.pi * 0.7) + 0.002 * rng.normal()
    I0 = S.I0 * (1 + 0.012 * math.sin(t * 0.17) * S.noise2[..., None])
    I = I0 * np.exp(pert + S.dust) * flick
    # rumore del sensore (dipendente dal segnale) + rumore cromatico tenue
    lum = rng.normal(0, 1, (H, W, 1)).astype(np.float32)
    chrom = rng.normal(0, 1, (H, W, 3)).astype(np.float32)
    I = I + (0.007 + 0.010 * np.sqrt(np.clip(I, 0, 1))) * (0.75 * lum + 0.45 * chrom)
    I = np.clip(I, 0, 1)
    # leggera curva di risposta
    I = I ** 0.92
    return (I * 255 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------- driver
G = {}


def render_frame(fi):
    t = fi / FPS
    pert = np.zeros((H, W, 3), np.float32)
    items = [('p', p, p['p0'][2] + p['v'][2] * t) for p in G['particles']]
    items += [('c', s, s['pos'][2]) for s in G['frames'][fi]]
    items.sort(key=lambda it: it[2])
    for kind, obj, _ in items:
        if kind == 'p':
            render_particle(pert, obj, t)
        else:
            render_cerc(pert, obj, t)
    return finish(G['static'], pert, t, fi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float, default=20)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--preview', type=str, default='')
    ap.add_argument('--outdir', type=str, default='anteprime')
    ap.add_argument('--video', type=str, default='')
    ap.add_argument('--test-morph', action='store_true')
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 2)
    args = ap.parse_args()

    n = int(round(args.seconds * FPS))
    t0 = time.time()
    G['frames'] = simulate(n, args.seed)
    G['particles'] = make_particles(args.seed)
    G['static'] = build_static(args.seed)
    print(f'simulazione {time.time() - t0:.1f}s', flush=True)

    if args.test_morph:
        os.makedirs(args.outdir, exist_ok=True)
        rng = np.random.default_rng(3)
        tiles = []
        for roll, amp, ph, z in ((0.0, 0.06, 0.0, 0.0), (1.2, 0.06, 0.0, 0.0), (0.4, 1.0, 1.0, 0.0), (0.0, 0.06, 0.0, 120.0)):
            c = new_cerc(rng, z)
            c.pos = np.array([FW / 2 + 60, FH / 2 + 120, z])
            c.e = np.array([0.0, -1.0, 0.0]); c.roll = roll; c.amp = amp; c.phase = ph
            c.spread = 1.05 if amp < 0.5 else 0.72; c.bend = 0.03
            c.dphase = 2 * math.pi * 20 if amp > 0.5 else 0.0
            c.roll_rate = 0.0; c.v = np.array([0, -600.0 if amp > 0.5 else 0, 0])
            pert = np.zeros((H, W, 3), np.float32)
            render_cerc(pert, snapshot(c), 0.0)
            img = finish(G['static'], pert, 0.0, 0)
            tiles.append(img[H // 2 - 330:H // 2 + 330, W // 2 - 170:W // 2 + 290])
        cv2.imwrite(os.path.join(args.outdir, 'morph.png'), np.concatenate(tiles, axis=1)[..., ::-1])
        print('morph ok')
        return

    if args.preview:
        os.makedirs(args.outdir, exist_ok=True)
        for fi in [int(x) for x in args.preview.split(',')]:
            t1 = time.time()
            img = render_frame(fi)
            cv2.imwrite(os.path.join(args.outdir, f'frame_{fi:04d}.png'), img[..., ::-1])
            print(f'frame {fi}: {time.time() - t1:.2f}s', flush=True)
        return

    if args.video:
        cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
               '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', '17', '-tune', 'grain',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', args.video]
        ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        ctx = mp.get_context('fork')
        with ctx.Pool(args.workers) as pool:
            for i, img in enumerate(pool.imap(render_frame, range(n), chunksize=2)):
                ff.stdin.write(img.tobytes())
                if i % 30 == 0:
                    print(f'frame {i}/{n}  {time.time() - t0:.0f}s', flush=True)
        ff.stdin.close()
        ff.wait()
        print('fatto', args.video, f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
