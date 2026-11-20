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
EXPOSURE = 1 / 120            # s: coda leggibile come C con una scia di mosso
TAIL_SAMPLES = 18             # campioni temporali della coda durante l'esposizione

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
THRASH = True    # due cercarie che si agitano quasi sul posto (brief §5)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


# ---------------------------------------------------------------- simulazione
class Cerc:
    pass


BEAT_HZ = 15.6        # Krishnamurthy et al. 2017 (dal brief)
DT_SUB = 16           # passi di simulazione per fotogramma (~31 per battito)
ZETA_T, ZETA_N = 1.0, 2.0
THRASH_SPEED = 250.0  # µm/s, spostamento medio delle agitate durante le raffiche (stima)
THRASH_TURN = 16.0    # rad/s di picco della rotazione prescritta per le agitate (tarata a occhio)
RIGHTING = 3.0       # coppia di raddrizzamento in unità del peso (tarata a occhio)   # coefficienti di resistenza tangenziale/normale (RFT, rapporto ~2)


def beat(ph):
    # la coda passa rapidamente da una C all'altra e resta piegata per gran parte del ciclo
    sn = math.sin(ph)
    return math.copysign(abs(sn) ** 0.45, sn)


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def rotmat(axis, ang):
    K = skew(axis)
    return np.eye(3) * math.cos(ang) + math.sin(ang) * K + (1 - math.cos(ang)) * np.outer(axis, axis)


def rotate(v, axis, ang):
    return rotmat(axis, ang) @ v


def rand_axis(e, rng):
    return unit(np.cross(e, rng.normal(0, 1, 3)))


def orthonorm(R):
    e = unit(R[:, 0])
    n1 = unit(R[:, 1] - np.dot(R[:, 1], e) * e)
    return np.column_stack([e, n1, np.cross(e, n1)])


def frame_from_e(e, spin=0.0):
    ref = np.array([0, 0, 1.0]) if abs(e[2]) < 0.9 else np.array([1.0, 0, 0])
    a = unit(np.cross(e, ref))
    b = np.cross(e, a)
    n1 = math.cos(spin) * a + math.sin(spin) * b
    return np.column_stack([e, n1, np.cross(e, n1)])


# ------------------------------------------------ forma prescritta, nel riferimento proprio
# origine = giunzione corpo-coda; x = direzione della base della coda. Il moto rigido di questo
# riferimento NON è prescritto: lo calcola la RFT.
def cerc_shape(p, sz):
    x = np.array([1.0, 0.0, 0.0])
    psi = p['psi']
    b = np.array([0.0, math.cos(psi), math.sin(psi)])      # piano di battito
    bz = np.cross(x, b)
    amp, ph, lag = p['amp'], p['phase'], p['lag']
    # fusto della coda: curvatura a C che si inverte
    kL = p['kmax'] * amp * beat(ph)
    N = 18
    sv = np.linspace(0, 1, N)
    L = TAIL_L * sz
    th = -0.30 * kL + kL * sv + 2.0 * p['arch'] * sv + p['tws'] * np.sin(2 * math.pi * sv)   # + coda a S
    thm = 0.5 * (th[1:] + th[:-1])
    ds = L / (N - 1)
    steps = np.outer(np.cos(thm) * ds, x) + np.outer(np.sin(thm) * ds, b)
    P = np.vstack([np.zeros(3), np.cumsum(steps, axis=0)])
    radii = (TAIL_R0 + (TAIL_R1 - TAIL_R0) * sv) * sz
    # furca: cerniera morbida con ritardo
    flap = p['famp'] * amp * beat(ph - 0.75 * math.pi * lag)
    td = math.cos(th[-1] + flap) * x + math.sin(th[-1] + flap) * b
    nf = unit(math.cos(0.5) * b + math.sin(0.5) * bz)
    nf = unit(nf - np.dot(nf, td) * td)
    furcae = []
    q = np.linspace(0, 1, 7)
    for side in (-1, 1):
        d = unit(td * math.cos(p['spread']) + side * nf * math.sin(p['spread']))
        bendf = 0.45 * amp * beat(ph - 0.9 * math.pi * lag + 0.3 * side) - 0.06
        Q = P[-1][None, :] + np.outer(q * FURCA_L * sz, d) + np.outer(q ** 2 * FURCA_L * sz * bendf, td)
        furcae.append((Q, (FURCA_R0 + (FURCA_R1 - FURCA_R0) * q) * sz))
    # corpo: cerniera con ritardo (in controfase rispetto alla coda)
    hb = -p['rec_amp'] * amp * beat(ph - 0.5 * math.pi * lag)
    eb = math.cos(hb) * x + math.sin(hb) * b
    n1b = unit(b - np.dot(b, eb) * eb)
    n2b = np.cross(eb, n1b)
    pul = math.sin(p['pulse'])
    a = BODY_A * sz * (1 + 0.03 * pul) * p['elong']
    r1 = BODY_R1 * sz * (1 - 0.02 * pul) * p['girth']
    r2 = BODY_R2 * sz * (1 - 0.02 * pul) * p['girth']
    C = -eb * (a - 6)
    uu = np.linspace(-1, 1, 12)
    boff = 0.25 * p['bbend'] * a * (1 - uu) ** 2 + 0.30 * p['sbend'] * a * np.sin(math.pi * (1 - uu))   # C + S
    B = C[None, :] + np.outer(uu * a * 0.97, eb) + np.outer(boff, n1b)
    # punti per la RFT: posizione, tangente, peso (lunghezza del tratto x fattore di spessore)
    X, T, Wt = [], [], []
    for pts, wf in ((B, 2.0 * p['girth']), (P, 1.0), (furcae[0][0], 0.6), (furcae[1][0], 0.6)):
        tg = np.gradient(pts, axis=0)
        tg = tg / np.linalg.norm(tg, axis=1, keepdims=True)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1).mean()
        X.append(pts)
        T.append(tg)
        Wt.append(np.full(len(pts), seg * wf))
    return dict(C=C, Rb=np.column_stack([eb, n1b, n2b]), a=a, r1=r1, r2=r2, P=P, radii=radii, sv=sv,
                furcae=furcae, lat=(b, bz), X=np.vstack(X), T=np.vstack(T), Wt=np.concatenate(Wt))


def rft_solve(X, T, w, rhs_list, anchor=None):
    """Resistive force theory: f_i = -K_i (U + Om x X_i + u_def_i). Somma forze = -F_ext,
    somma momenti = -M_ext. Ogni rhs è (u_def (n,3), F_ext (3), punto di applicazione (3))."""
    tt = T[:, :, None] * T[:, None, :]
    K = (ZETA_T * tt + ZETA_N * (np.eye(3)[None] - tt)) * w[:, None, None]
    S = np.zeros((len(X), 3, 3))
    S[:, 0, 1], S[:, 0, 2] = -X[:, 2], X[:, 1]
    S[:, 1, 0], S[:, 1, 2] = X[:, 2], -X[:, 0]
    S[:, 2, 0], S[:, 2, 1] = -X[:, 1], X[:, 0]
    KS = K @ S
    A = np.zeros((6, 6))
    A[:3, :3] = -K.sum(0)
    A[:3, 3:] = KS.sum(0)
    A[3:, :3] = -(S @ K).sum(0)
    A[3:, 3:] = (S @ KS).sum(0)
    if anchor is not None:
        # punto vincolato (attaccato a un detrito): velocità nulla, la reazione Fa entra nei bilanci
        Xa = anchor
        Sa = skew(Xa)
        A9 = np.zeros((9, 9))
        A9[:6, :6] = A
        A9[:3, 6:] = np.eye(3)
        A9[3:6, 6:] = Sa
        A9[6:, :3] = np.eye(3)
        A9[6:, 3:6] = -Sa
        A = A9
    out = []
    for item in rhs_list:
        udef, F, Xg = item[:3]
        Mx = item[3] if len(item) > 3 else np.zeros(3)
        Ku = np.einsum('nij,nj->ni', K, udef)
        rhs = np.concatenate([Ku.sum(0) - F, np.einsum('nij,nj->ni', S, Ku).sum(0) - np.cross(Xg, F) - Mx])
        if anchor is not None:
            ua = item[4] if len(item) > 4 else np.zeros(3)
            rhs = np.concatenate([rhs, -ua])
        sol = np.linalg.solve(A, rhs)
        out.append((sol[:3], sol[3:6]))
    return out


def default_params(rng):
    return dict(amp=0.0, phase=rng.uniform(0, 2 * math.pi), lag=1.0, kmax=2.0, famp=0.55, rec_amp=0.70,
                spread=0.55, arch=0.0, bbend=0.0, sbend=0.0, tws=0.0, elong=1.0, girth=1.0, taper=0.0, pulse=rng.uniform(0, 6.28),
                psi=rng.uniform(-0.4, 0.4))


def new_cerc(rng, z):
    c = Cerc()
    c.J = np.array([rng.uniform(-100, FW + 100), rng.uniform(0.08, 0.95) * FH, z], float)
    c.Rf = frame_from_e(unit(np.array([rng.normal(0, 0.3), -1.0, rng.normal(0, 0.2)])), rng.uniform(0, 6.28))
    c.size = rng.normal(1.0, 0.05)
    c.p = default_params(rng)
    c.swim = rng.random() < 0.3
    c.timer = rng.uniform(0.2, 2.5 if c.swim else 12.0)
    c.sink_to = rng.uniform(0.3, 0.9) * FH
    c.freq = BEAT_HZ * rng.uniform(0.85, 1.15)
    c.jt = rng.uniform(0, 0.4)
    c.phase = c.p['phase']
    c.dphase = 0.0
    c.twitch = 0.0
    c.bend_target = rng.normal(0, 0.05)
    c.drift = np.array([rng.normal(0, 8), 0.0, rng.normal(0, 4)])
    c.vmax = rng.uniform(660, 740)          # µm/s: ~0,7 mm/s (brief)
    c.sink_v = rng.uniform(70, 130)         # µm/s: ~0,1 mm/s, diversa per individuo
    c.fwd = rng.random() < 0.5              # fase passiva: nuoto a testa avanti (True) o solo affondamento
    c.head_first = False
    c.arc_bias = 0.0
    c.thrasher = False
    c.def_f = rng.uniform(0.35, 0.9, 2)
    c.def_p = rng.uniform(0, 6.28, 2)
    c.prevX = None
    c.U = np.zeros(3)
    c.Om = np.zeros(3)
    c.vlp = np.zeros(3)
    c.vlp1 = np.zeros(3)
    c.olp = np.zeros(3)
    c.olp1 = np.zeros(3)
    c.right = RIGHTING
    c.anchored = False
    c.A = None
    c.mass_off = rng.normal(0, 0.5)
    c.static = make_static(rng)
    calibrate_weight(c)
    return c


def calibrate_weight(c, factor=1.0):
    """Peso applicato al centro del corpo, tarato perché l'affondamento passivo valga sink_v."""
    p = dict(c.p, amp=0.0)
    sh = cerc_shape(p, c.size)
    R = frame_from_e(np.array([0.0, -1.0, 0.0]))
    X, T = sh['X'] @ R.T, sh['T'] @ R.T
    (U, _), = rft_solve(X, T, sh['Wt'], [(np.zeros_like(X), np.array([0.0, 1.0, 0.0]), R @ sh['C'])])
    c.Wg = factor * c.sink_v / max(U[1], 1e-9)


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



def jitter_beat(c, dt, rng):
    """Frequenza 15,6 Hz con variazioni ±15% e reset di fase casuali ogni 0,2-0,5 s (brief §5.2)."""
    c.jt -= dt
    if c.jt <= 0:
        c.jt = rng.uniform(0.2, 0.5)
        c.freq = BEAT_HZ * rng.uniform(0.85, 1.15)
        if rng.random() < 0.5:
            c.phase += rng.uniform(0, 2 * math.pi)


def start_swim(c, rng):
    c.swim = True
    c.timer = rng.uniform(1.0, 3.0)                       # brief: episodi di 1-3 s


def start_rest(c, rng):
    c.swim = False
    c.timer = rng.uniform(10.0, 45.0)                     # brief: affondamento lungo, durata casuale
    c.sink_to = rng.uniform(0.2, 0.95) * FH
    c.drift = np.array([rng.normal(0, 8), 0.0, rng.normal(0, 4)])
    c.bend_target = rng.normal(0, 0.15)     # forma un po' curva: in affondamento ciascuna si inclina a modo suo
    c.head_first = c.fwd and rng.random() < 0.6
    if c.head_first:
        # nuoto corpo-avanti (Panzner et al. 2025, fig. 3); la curvatura dell'arco nasce
        # da una lieve asimmetria del battito
        c.timer = rng.uniform(2.6, 3.8)
        c.arc_bias = rng.choice([-1, 1]) * rng.uniform(0.04, 0.10)


def behave_normal(c, dt, t, rng):
    c.timer -= dt
    y = c.J[1]
    if c.swim:
        if c.timer <= 0 or y < -40:
            start_rest(c, rng)
    else:
        if (c.timer <= 0 and y >= c.sink_to) or y > FH + 80:
            start_swim(c, rng)
    jitter_beat(c, dt, rng)
    if not c.swim and not c.head_first and c.twitch <= 0 and rng.random() < dt * 0.1:
        c.twitch = rng.uniform(0.15, 0.4)
    c.twitch = max(0.0, c.twitch - dt)
    sw, hf = c.swim, (not c.swim) and c.head_first
    tw = (not sw) and (not hf) and c.twitch > 0
    p = c.p
    amp_t = 1.0 if sw else (0.9 if hf else (0.7 if tw else 0.0))
    p['amp'] += (amp_t - p['amp']) * (1 - math.exp(-dt * 30))     # parte e si ferma di colpo
    p['lag'] = -1.0 if hf else 1.0                                  # il verso del ritardo decide coda/testa avanti
    p['spread'] += ((1.45 if sw else (0.10 if hf else 0.55)) - p['spread']) * (1 - math.exp(-dt * 5))
    arch_t = 0.0 if sw else (c.arc_bias if hf else c.bend_target)
    p['arch'] += (arch_t - p['arch']) * (1 - math.exp(-dt * 2))
    p['psi'] += rng.normal(0, 0.15) * math.sqrt(dt)
    dfm = 0.13 * math.sin(2 * math.pi * c.def_f[0] * t + c.def_p[0]) + 0.07 * math.sin(2 * math.pi * c.def_f[1] * t + c.def_p[1])
    p['elong'] += (((1.10 if sw else 1.0) + dfm) - p['elong']) * (1 - math.exp(-dt * 6))
    p['girth'] = 1.0 / math.sqrt(p['elong']) ** 1.2
    p['taper'] = 0.5 * max(0.0, p['elong'] - 1.05)
    p['bbend'] *= math.exp(-dt * 3)
    p['pulse'] += dt * (2.4 if sw else 0.5)
    c.dphase = 2 * math.pi * c.freq if (sw or hf or tw) else 0.0
    # velocità media di riferimento (vedi rft_step): risalita e testa avanti tarate sui dati
    c.right = RIGHTING * (2.0 if (sw or hf) else 0.35)
    return (c.vmax + c.sink_v) * 1.15 if sw else (0.7 * c.vmax if hf else None)


def behave_thrash(c, dt, t, rng):
    """Agitazione a episodi: fasi agitate di 2,5-6 s (raffiche e brevi pause) alternate a fasi
    tranquille di 1,5-5 s in cui il battito si ferma e l'animale scende piano. Durante l'agitazione
    corpo e coda si contorcono a C e a S, il corpo si allunga e si accorcia."""
    c.qtimer -= dt
    if c.qtimer <= 0:
        c.quiet = not c.quiet
        c.qtimer = rng.uniform(1.5, 5.0) if c.quiet else rng.uniform(2.5, 6.0)
        c.burst = not c.quiet
        c.ttimer = rng.uniform(1.0, 3.0)
    c.ttimer -= dt
    if c.ttimer <= 0 and not c.quiet:
        c.burst = not c.burst
        c.ttimer = rng.uniform(1.0, 3.0) if c.burst else rng.uniform(0.15, 0.4)
    if c.ttimer <= 0 or c.newb != c.burst:
        c.newb = c.burst
        c.cf = rng.uniform(3.5, 5.5)
        c.psi_w = rng.choice([-1, 1]) * rng.uniform(1.0, 3.0)
        c.p['lag'] = rng.choice([-1.0, 1.0])
        c.cside = rng.choice([-1.0, 1.0])
    jitter_beat(c, dt, rng)
    p = c.p
    c.act += ((1.0 if c.burst else 0.0) - c.act) * (1 - math.exp(-dt * (12 if c.burst else (2.5 if c.quiet else 6))))
    act = c.act
    p['amp'] += ((1.0 if c.burst else (0.0 if c.quiet else 0.15)) - p['amp']) * (1 - math.exp(-dt * 25))
    c.cph += dt * 2 * math.pi * c.cf * (0.3 + act)
    ph = c.cph
    p['bbend'] = c.cside * 1.6 * act * (0.5 + 0.5 * math.sin(ph))
    p['sbend'] = 0.9 * act * math.sin(1.7 * ph + 1.0)
    p['arch'] = c.cside * 1.4 * act * (0.5 + 0.5 * math.sin(ph + 0.5 * math.pi))   # sfasato: ciclo non reciproco
    p['tws'] = 1.0 * act * math.sin(2.3 * ph + 0.4)
    cr = act * abs(math.sin(ph))
    quiet_shape = 0.10 * math.sin(2 * math.pi * 0.5 * t) * (1 - act)
    p['elong'] += ((1.22 - 0.62 * cr + quiet_shape) - p['elong']) * (1 - math.exp(-dt * 25))
    p['girth'] = 1.0 / math.sqrt(max(p['elong'], 0.5)) ** 1.3
    p['taper'] = 0.5 * max(0.0, p['elong'] - 1.0)
    p['spread'] += ((1.2 - 0.6 * cr) - p['spread']) * (1 - math.exp(-dt * 15))
    p['psi'] += c.psi_w * act * dt
    p['rec_amp'] = 0.4
    c.dphase = 2 * math.pi * c.freq if not c.quiet or p['amp'] > 0.02 else 0.0
    return THRASH_SPEED if c.burst else None


def rft_step(c, dt, t, rng):
    target = behave_thrash(c, dt, t, rng) if c.thrasher else behave_normal(c, dt, t, rng)
    c.phase += c.dphase * dt
    c.p['phase'] = c.phase
    sh = cerc_shape(c.p, c.size)
    Xb = sh['X']
    udef_b = np.zeros_like(Xb) if c.prevX is None or len(c.prevX) != len(Xb) else (Xb - c.prevX) / dt
    c.prevX = Xb
    R = c.Rf
    X, T, ud = Xb @ R.T, sh['T'] @ R.T, udef_b @ R.T
    Fg = np.array([0.0, c.Wg, 0.0])
    # coppia di raddrizzamento: corpo più denso, coda più leggera (forza netta nulla)
    Xg = R @ (sh['C'] + c.mass_off * sh['r1'] * sh['Rb'][:, 1])
    Fc = np.array([0.0, c.right * c.Wg, 0.0])
    Mc = np.cross(Xg, Fc) - np.cross(R @ sh['P'][len(sh['P']) // 2], Fc)
    if c.anchored:
        # la punta anteriore (primo punto del corpo) resta ferma sul detrito
        Xa = X[0]
        (Ud, Od), (Ug, Og) = rft_solve(X, T, sh['Wt'], [(ud, np.zeros(3), np.zeros(3), np.zeros(3), ud[0]),
                                                       (np.zeros_like(X), Fg, Xg, Mc, np.zeros(3))], anchor=Xa)
    else:
        (Ud, Od), (Ug, Og) = rft_solve(X, T, sh['Wt'], [(ud, np.zeros(3), np.zeros(3)), (np.zeros_like(X), Fg, Xg, Mc)])
    # La RFT con questa forma sottostima la velocità media (~0,2 contro 0,7 mm/s misurati):
    # si scala SOLO la componente lenta (media) della propulsione; oscillazioni, rinculo e
    # rotazioni restano quelle calcolate.
    g = 1 - math.exp(-dt / 0.15)                      # filtro a due stadi: residuo a 15 Hz < 1%
    c.vlp1 += (Ud - c.vlp1) * g
    c.vlp += (c.vlp1 - c.vlp) * g
    corr = np.zeros(3)
    nv = np.linalg.norm(c.vlp)
    if target is not None and nv > 1e-6:
        corr = (target / nv - 1.0) * c.vlp * min(1.0, c.p['amp'])
    U = Ud + corr + Ug + (np.zeros(3) if c.anchored else c.drift + np.array([4 * math.sin(t * 0.11), 0, 0]))
    z = c.J[2]
    if abs(z) > 1500:
        U[2] -= (z - math.copysign(1500, z)) * 2.0
    Om = Od + Og
    if c.thrasher:
        # Libera in acqua, la RFT dà a un'agitata pochissima rotazione netta: le girate rapide sono
        # prescritte come velocità angolare continua e liscia, sommata alle oscillazioni calcolate.
        c.tsw -= dt
        if c.tsw <= 0:                    # nuova girata: asse casuale, verso e velocità
            c.tsw = rng.uniform(0.25, 0.7)
            c.tax_cur = rand_axis(c.Rf[:, 0], rng)
            c.tspd = rng.uniform(0.6, 1.0) * THRASH_TURN
        w_t = c.tspd * c.act * (c.tax_cur - np.dot(c.tax_cur, c.Rf[:, 0]) * c.Rf[:, 0])
        c.wcur += (w_t - c.wcur) * (1 - math.exp(-dt * 12))          # accelera e frena in modo liscio
        Om = Om + c.wcur
    c.U, c.Om = U, Om
    c.J = c.J + U * dt
    wn = np.linalg.norm(Om)
    if wn > 1e-12:
        c.Rf = orthonorm(rotmat(Om / wn, wn * dt) @ c.Rf)
    if c.anchored:
        if c.A is None:
            c.A = c.J + c.Rf @ Xb[0]
        c.J = c.A - c.Rf @ Xb[0]          # il punto d'attacco resta esattamente sul detrito
    m = 420
    if c.J[0] < -m:
        c.J[0] += FW + 2 * m
    elif c.J[0] > FW + m:
        c.J[0] -= FW + 2 * m


def snapshot(c):
    p = dict(c.p)
    return dict(pos=c.J.copy(), R=c.Rf.copy(), U=c.U.copy(), Om=c.Om.copy(), p=p, dphase=c.dphase,
                amp=p['amp'], bbend=p['bbend'], size=c.size, static=c.static, thrasher=c.thrasher,
                anchor=c.A)


def simulate(n_frames, seed, warm=6.0):
    rng = np.random.default_rng(seed)
    depths = [-1300, -820, -520, -300, -150, -60, 10, 70, 160, 360, 700, 1150]
    rng.shuffle(depths)
    cs = [new_cerc(rng, z) for z in depths[:N_CERC]]
    dt = 1 / (FPS * DT_SUB)
    t = -warm
    for _ in range(int(warm / dt)):
        for c in cs:
            rft_step(c, dt, t, rng)
        t += dt
    # protagoniste vicino al piano di fuoco, nella zona centrale dell'inquadratura
    heroes = [(0.34 * FW, 0.40 * FH, -15.0), (0.64 * FW, 0.55 * FH, 30.0), (0.50 * FW, 0.22 * FH, 75.0)]
    for c, (x, y, z) in zip(cs, heroes):
        c.J = np.array([x, y, z])
        start_rest(c, rng)
        c.head_first = False
        c.timer = rng.uniform(1.0, 3.5)
        c.sink_to = y + rng.uniform(80, 200)
        c.prevX = None
    # due individui che si agitano quasi sul posto: uno a fuoco, uno un po' fuori fuoco
    thrashers = ((cs[1], (0.70 * FW, 0.28 * FH, 20.0), 0.4), (cs[3], (0.26 * FW, 0.25 * FH, 190.0), 2.5)) if THRASH else ()
    for c, (x, y, z), t0 in thrashers:
        c.thrasher = True
        c.J = np.array([x, y, z])
        c.Rf = frame_from_e(unit(np.array([0.5, -1.0, 0.1])))
        c.burst, c.ttimer, c.act, c.cph = False, t0, 0.0, rng.uniform(0, 6.28)
        c.cf, c.psi_w, c.cside = 2.0, 4.0, 1.0
        c.tsw, c.tax_cur, c.tspd, c.wcur = 0.0, np.array([0, 0, 1.0]), 0.0, np.zeros(3)
        c.quiet, c.qtimer, c.newb = False, rng.uniform(2.5, 6.0), False
        c.tax = [unit(v) for v in rng.normal(0, 1, (3, 3))]
        c.tf = rng.uniform(0.6, 1.6, 3)
        c.tp = rng.uniform(0, 6.28, 3)
        c.drift = np.zeros(3)
        c.prevX = None
        calibrate_weight(c, factor=0.25)
        c.right = 0.0
        c.anchored = False                    # libere di andare in giro
    frames = []
    for f in range(n_frames):
        frames.append([snapshot(c) for c in cs])
        for _ in range(DT_SUB):
            for c in cs:
                rft_step(c, dt, t, rng)
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
def frame_at(s, tau):
    Om = s['Om']
    wn = np.linalg.norm(Om)
    R = rotmat(Om / wn, wn * tau) @ s['R'] if wn > 1e-9 else s['R']
    return s['pos'] + s['U'] * tau, R


def tail_geom(s, tau):
    J, R = frame_at(s, tau)
    p = dict(s['p'])
    p['phase'] += s['dphase'] * tau
    sh = cerc_shape(p, s['size'])
    P = J[None, :] + sh['P'] @ R.T
    furcae = [(J[None, :] + Q @ R.T, fr) for Q, fr in sh['furcae']]
    b, bz = sh['lat']
    return P, sh['radii'], furcae, (R @ b, R @ bz), sh['sv']


def body_geom(s):
    sh = cerc_shape(s['p'], s['size'])
    R = s['R']
    return s['pos'] + R @ sh['C'], R @ sh['Rb'], sh['a'], sh['r1'], sh['r2']


BEND = [0.0, 0.0]   # curvature (C, S) del corpo dell'individuo in corso di rendering


def bend_off(u, a):
    # curvatura a falce: nulla alla giunzione con la coda (u=+1), massima all'apice (u=-1)
    return 0.25 * BEND[0] * a * (1 - u) ** 2 + 0.30 * BEND[1] * a * np.sin(math.pi * (1 - u))


def body_point(C, R, a, vec):
    return C + R @ vec + R[:, 1] * bend_off(vec[0] / a, a)


def local_to_world(C, R, a, r1, r2, uvw):
    return C[None, :] + (uvw * np.array([a, r1, r2])) @ R.T + np.outer(bend_off(uvw[:, 0], a), R[:, 1])


def render_cerc(pert, s, t):
    C, R, a, r1, r2 = body_geom(s)
    BEND[0] = s['bbend']
    BEND[1] = s['p']['sbend']
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
    prof = prof * (1 - 0.85 * s['p']['taper'] * np.clip(-uu, 0, 1) ** 1.3)   # testa allungata e appuntita
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
    # detriti a cui sono attaccate le cercarie che si agitano
    rngd = np.random.default_rng(args.seed + 5)
    for sn in G['frames'][min(1, len(G['frames']) - 1)]:
        if sn['anchor'] is None:
            continue
        d = dict(kind='floc', p0=sn['anchor'].copy(), v=np.zeros(3), jit=0.0, jf=np.zeros(2), jp=np.zeros(2),
                 absc=ABS_ORGANIC, dens=0.03, rot0=0.0, rotv=0.0, tilt=0.0, ell=[])
        for k in range(9):
            r_ = rngd.uniform(4, 10)
            off = rngd.normal(0, 9, 3) - 0.6 * r_ * unit(sn['pos'] - sn['anchor'])
            d['ell'].append((off, np.array([r_, r_ * rngd.uniform(0.6, 1), r_ * rngd.uniform(0.6, 1)])))
        G['particles'].append(d)
    G['static'] = build_static(args.seed)
    print(f'simulazione {time.time() - t0:.1f}s', flush=True)

    if args.test_morph:
        os.makedirs(args.outdir, exist_ok=True)
        rng = np.random.default_rng(3)
        tiles = []
        for roll, amp, ph, z in ((0.0, 0.0, 0.0, 0.0), (1.2, 0.0, 0.0, 0.0), (0.4, 1.0, 1.0, 0.0), (0.0, 0.0, 0.0, 120.0)):
            c = new_cerc(rng, z)
            c.J = np.array([FW / 2 + 60, FH / 2 + 120, z])
            c.Rf = frame_from_e(np.array([0.0, -1.0, 0.0]), roll)
            c.p.update(amp=amp, phase=ph, spread=1.45 if amp > 0.5 else 0.55, arch=0.03)
            c.dphase = 2 * math.pi * BEAT_HZ if amp > 0.5 else 0.0
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
