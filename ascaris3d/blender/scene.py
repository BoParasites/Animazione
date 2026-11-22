# Scena Blender/Cycles del ciclo di Ascaris lumbricoides.
# Uso: python scene.py --state state.json --glb anat_plain.glb --out frames_dir --frames 0:1980:1 [--samples 48] [--res 1920x1080]
import bpy, json, math, sys, os, argparse, random
from mathutils import Vector

ap = argparse.ArgumentParser()
ap.add_argument('--state'); ap.add_argument('--glb'); ap.add_argument('--out')
ap.add_argument('--frames', default='0:1980:1'); ap.add_argument('--samples', type=int, default=48)
ap.add_argument('--res', default='1920x1080'); ap.add_argument('--threads', type=int, default=0)
ap.add_argument('--blend', default='')
ap.add_argument('--layer', default='anat', choices=['anat', 'over', 'all'])
ap.add_argument('--liver-only', action='store_true')
A = ap.parse_args()
S = json.load(open(A.state))
FR, ST, ROUTE, SOIL = S['frames'], S['stages'], S['route'], S['soil']
BOUNDS = S['bounds']

def V3(p):  # three.js (Y in alto) -> Blender (Z in alto)
    return Vector((p[0], -p[2], p[1]))

def smooth(x): return 0.0 if x <= 0 else 1.0 if x >= 1 else x * x * (3 - 2 * x)

# ---------------------------------------------------------------- scena vuota
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
bpy.ops.import_scene.gltf(filepath=A.glb)
for o in list(bpy.data.objects):
    if o.type != 'MESH': continue
    o.data.materials.clear()
    for poly in o.data.polygons: poly.use_smooth = True

# ---------------------------------------------------------------- materiali
def node_mat(name):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    return m, nt, out

def tissue(name, col, alpha=0.5, edge=0.5, rough=0.38, sss=0.25, coat=0.35, emit=0.0, sss_r=(0.006, 0.003, 0.002)):
    """Tessuto semitrasparente: più opaco di taglio (fresnel), con diffusione sottocutanea e velo umido."""
    m, nt, out = node_mat(name)
    b = nt.nodes.new('ShaderNodeBsdfPrincipled')
    b.inputs['Base Color'].default_value = (*col, 1)
    b.inputs['Roughness'].default_value = rough
    b.inputs['Subsurface Weight'].default_value = 0.0  # SSS disattivato: differenza minima, -20% di tempo
    b.inputs['Subsurface Radius'].default_value = sss_r
    b.inputs['Subsurface Scale'].default_value = 1.0
    b.inputs['Coat Weight'].default_value = coat
    b.inputs['Coat Roughness'].default_value = 0.12
    if emit > 0:
        b.inputs['Emission Color'].default_value = (*col, 1); b.inputs['Emission Strength'].default_value = emit
    lw = nt.nodes.new('ShaderNodeLayerWeight'); lw.inputs['Blend'].default_value = 0.35
    pw = nt.nodes.new('ShaderNodeMath'); pw.operation = 'POWER'; pw.inputs[1].default_value = 2.0
    mr = nt.nodes.new('ShaderNodeMapRange')
    mr.inputs['To Min'].default_value = alpha; mr.inputs['To Max'].default_value = min(1.0, alpha + edge)
    nt.links.new(lw.outputs['Facing'], pw.inputs[0]); nt.links.new(pw.outputs[0], mr.inputs['Value'])
    nt.links.new(mr.outputs['Result'], b.inputs['Alpha'])
    nt.links.new(b.outputs['BSDF'], out.inputs['Surface'])
    m.blend_method = 'BLEND' if hasattr(m, 'blend_method') else None
    return m

def emissive(name, col, strength, alpha=1.0):
    m, nt, out = node_mat(name)
    e = nt.nodes.new('ShaderNodeEmission'); e.inputs['Color'].default_value = (*col, 1); e.inputs['Strength'].default_value = strength
    if alpha < 1:
        t = nt.nodes.new('ShaderNodeBsdfTransparent'); mx = nt.nodes.new('ShaderNodeMixShader')
        mx.inputs['Fac'].default_value = alpha
        nt.links.new(t.outputs[0], mx.inputs[1]); nt.links.new(e.outputs[0], mx.inputs[2]); nt.links.new(mx.outputs[0], out.inputs['Surface'])
    else:
        nt.links.new(e.outputs[0], out.inputs['Surface'])
    return m

def srgb(h):
    h = h.lstrip('#'); c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple(((x + 0.055) / 1.055) ** 2.4 if x > 0.04045 else x / 12.92 for x in c)

M = {
    'skin':    tissue('skin', srgb('#9fb9da'), alpha=0.012, edge=0.16, rough=0.35, sss=0.0, coat=0.25),
    'bone':    tissue('bone', srgb('#d9cfb6'), alpha=0.04, edge=0.18, rough=0.5, sss=0.0, coat=0.0),
    'lung':    tissue('lung', srgb('#d98a94'), alpha=0.30, edge=0.45, rough=0.55, sss=0.4, coat=0.15),
    'heart':   tissue('heart', srgb('#9e2230'), alpha=0.55, edge=0.35, rough=0.32, sss=0.5, coat=0.5),
    'liver':   tissue('liver', srgb('#6e2418'), alpha=0.55, edge=0.35, rough=0.28, sss=0.45, coat=0.6),
    'gall':    tissue('gall', srgb('#4d7a35'), alpha=0.6, edge=0.3, rough=0.2, sss=0.4, coat=0.7),
    'stomach': tissue('stomach', srgb('#d99a7c'), alpha=0.40, edge=0.40, rough=0.3, sss=0.5, coat=0.6),
    'spleen':  tissue('spleen', srgb('#6a2238'), alpha=0.5, edge=0.35, rough=0.3, sss=0.4, coat=0.5),
    'pancreas':tissue('pancreas', srgb('#e0b57c'), alpha=0.45, edge=0.35, rough=0.45, sss=0.4, coat=0.3),
    'si':      tissue('si', srgb('#e7a2a0'), alpha=0.34, edge=0.42, rough=0.3, sss=0.5, coat=0.6),
    'colon':   tissue('colon', srgb('#c9a273'), alpha=0.32, edge=0.42, rough=0.32, sss=0.45, coat=0.55),
    'eso':     tissue('eso', srgb('#d8877a'), alpha=0.45, edge=0.4, rough=0.3, sss=0.4, coat=0.5),
    'air':     tissue('air', srgb('#cfe4ea'), alpha=0.40, edge=0.4, rough=0.35, sss=0.3, coat=0.4),
    'soft':    tissue('soft', srgb('#d9a99a'), alpha=0.10, edge=0.25, rough=0.4, sss=0.3, coat=0.2),
    'kidney':  tissue('kidney', srgb('#8e3a32'), alpha=0.45, edge=0.35, rough=0.3, sss=0.4, coat=0.4),
    'bladder': tissue('bladder', srgb('#e2cf86'), alpha=0.3, edge=0.35, rough=0.3, sss=0.3, coat=0.5),
    'artery':  tissue('artery', srgb('#b3202a'), alpha=0.5, edge=0.35, rough=0.3, sss=0.3, coat=0.5),
    'vein':    tissue('vein', srgb('#2c4fa8'), alpha=0.55, edge=0.35, rough=0.3, sss=0.3, coat=0.5),
    'pulm':    tissue('pulm', srgb('#6b58b8'), alpha=0.35, edge=0.3, rough=0.3, sss=0.3, coat=0.4),
    'brain':   tissue('brain', srgb('#d9b8b0'), alpha=0.08, edge=0.2, rough=0.5, sss=0.3, coat=0.2),
    'faint':   tissue('faint', srgb('#c08080'), alpha=0.02, edge=0.06, rough=0.5, sss=0.0, coat=0.0),
}
ASSIGN = {
    'skin': 'skin', 'brain': 'brain', 'eye_l': 'brain', 'eye_r': 'brain',
    'lung_l': 'lung', 'lung_r': 'lung', 'heart': 'heart', 'liver': 'liver', 'gallbladder': 'gall',
    'stomach': 'stomach', 'spleen': 'spleen', 'pancreas': 'pancreas', 'small_intestine': 'si',
    'colon': 'colon', 'appendix': 'colon', 'esophagus': 'eso', 'tongue': 'soft', 'pharynx': 'soft',
    'trachea': 'air', 'bronchi': 'air', 'kidney_l': 'kidney', 'kidney_r': 'kidney', 'adrenal_l': 'pancreas', 'adrenal_r': 'pancreas',
    'ureter_l': 'bladder', 'ureter_r': 'bladder', 'bladder': 'bladder', 'aorta': 'artery',
    'vena_cava_inferior': 'vein', 'vena_cava_superior': 'vein', 'veins_abdomen': 'vein', 'pulmonary_vessels': 'pulm', 'diaphragm': 'faint',
}
for o in bpy.data.objects:
    if o.type == 'MESH':
        key = o.name.split('.')[0]
        o.data.materials.append(M[ASSIGN.get(key, 'bone')])
        o.visible_shadow = key not in ('skin', 'diaphragm')

# ---------------------------------------------------------------- curve
def make_curve(name, pts, radius, mat, cyclic=False, res=2):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    cu.bevel_depth = radius; cu.bevel_resolution = res; cu.use_fill_caps = True
    sp = cu.splines.new('POLY'); sp.points.add(len(pts) - 1)
    for i, p in enumerate(pts): sp.points[i].co = (*p, 1)
    sp.use_cyclic_u = cyclic
    ob = bpy.data.objects.new(name, cu); sc.collection.objects.link(ob)
    ob.data.materials.append(mat)
    return ob, sp

RP = [V3(p) for p in ROUTE]
def trail_mat(name, col, peak):
    """Scia: emissione che cresce dalla coda (Ut) alla testa (Uh) lungo la coordinata U della curva;
    con Rv > 0 l'intero tratto si riaccende (riepilogo finale)."""
    m, nt, out = node_mat(name)
    uv = nt.nodes.new('ShaderNodeUVMap'); sx = nt.nodes.new('ShaderNodeSeparateXYZ')
    nt.links.new(uv.outputs['UV'], sx.inputs[0])
    vals = {}
    for k, v in (('Uh', 0.0), ('Ut', 0.0), ('Rv', 0.0)):
        n = nt.nodes.new('ShaderNodeValue'); n.name = k; n.outputs[0].default_value = v; vals[k] = n
    sub = nt.nodes.new('ShaderNodeMath'); sub.operation = 'SUBTRACT'
    nt.links.new(sx.outputs['X'], sub.inputs[0]); nt.links.new(vals['Ut'].outputs[0], sub.inputs[1])
    span = nt.nodes.new('ShaderNodeMath'); span.operation = 'SUBTRACT'
    nt.links.new(vals['Uh'].outputs[0], span.inputs[0]); nt.links.new(vals['Ut'].outputs[0], span.inputs[1])
    span2 = nt.nodes.new('ShaderNodeMath'); span2.operation = 'MAXIMUM'; span2.inputs[1].default_value = 1e-6
    nt.links.new(span.outputs[0], span2.inputs[0])
    div = nt.nodes.new('ShaderNodeMath'); div.operation = 'DIVIDE'; div.use_clamp = True
    nt.links.new(sub.outputs[0], div.inputs[0]); nt.links.new(span2.outputs[0], div.inputs[1])
    pw = nt.nodes.new('ShaderNodeMath'); pw.operation = 'POWER'; pw.inputs[1].default_value = 1.8
    nt.links.new(div.outputs[0], pw.inputs[0])
    rv = nt.nodes.new('ShaderNodeMath'); rv.operation = 'MULTIPLY'; rv.inputs[1].default_value = 0.4
    nt.links.new(vals['Rv'].outputs[0], rv.inputs[0])
    mx = nt.nodes.new('ShaderNodeMath'); mx.operation = 'MAXIMUM'
    nt.links.new(pw.outputs[0], mx.inputs[0]); nt.links.new(rv.outputs[0], mx.inputs[1])
    st = nt.nodes.new('ShaderNodeMath'); st.operation = 'MULTIPLY'; st.inputs[1].default_value = peak
    nt.links.new(mx.outputs[0], st.inputs[0])
    e = nt.nodes.new('ShaderNodeEmission'); e.inputs['Color'].default_value = (*col, 1)
    nt.links.new(st.outputs[0], e.inputs['Strength'])
    # la coda sfuma anche in opacità
    t = nt.nodes.new('ShaderNodeBsdfTransparent'); mix = nt.nodes.new('ShaderNodeMixShader')
    a = nt.nodes.new('ShaderNodeMath'); a.operation = 'SQRT'; a.use_clamp = True
    nt.links.new(mx.outputs[0], a.inputs[0]); nt.links.new(a.outputs[0], mix.inputs['Fac'])
    nt.links.new(t.outputs[0], mix.inputs[1]); nt.links.new(e.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs['Surface'])
    return m, vals

ROUTE_MAT, ROUTE_VAL = trail_mat('route', (1.0, 0.62, 0.2), 30.0)
route_ob, _ = make_curve('route', RP, 0.0012, ROUTE_MAT)
route_ob.data.bevel_factor_mapping_start = 'SPLINE'
route_ob.data.bevel_factor_mapping_end = 'SPLINE'
route_ob.visible_shadow = False
# traccia tenue del percorso già fatto (sotto la soglia del bagliore)
guide_ob, _ = make_curve('guide', RP, 0.0007, emissive('guide', (1.0, 0.66, 0.30), 3.5, alpha=0.75))
guide_ob.data.bevel_factor_mapping_end = 'SPLINE'
guide_ob.visible_shadow = False

# lunghezze cumulative del percorso per interpolare la testa
CUM = [0.0]
for i in range(1, len(RP)): CUM.append(CUM[-1] + (RP[i] - RP[i - 1]).length)
LEN = CUM[-1]
import bisect
def route_at(frac):
    s = max(0.0, min(1.0, frac)) * LEN
    i = max(0, min(len(RP) - 2, bisect.bisect_right(CUM, s) - 1))
    seg = CUM[i + 1] - CUM[i]; f = (s - CUM[i]) / seg if seg > 0 else 0
    p = RP[i].lerp(RP[i + 1], f); t = (RP[i + 1] - RP[i]).normalized()
    return p, t

NPT = len(RP) - 1
def uidx(frac):
    """frazione di lunghezza -> frazione di indice dei punti (bevel 'SPLINE' e UV della curva)"""
    s = max(0.0, min(1.0, frac)) * LEN
    i = max(0, min(NPT - 1, bisect.bisect_right(CUM, s) - 1))
    seg = CUM[i + 1] - CUM[i]; f = (s - CUM[i]) / seg if seg > 0 else 0
    return (i + f) / NPT

# fegato attenuato quando la testa è nello stomaco
from mathutils.bvhtree import BVHTree
_dg = bpy.context.evaluated_depsgraph_get()
_st = [o for o in bpy.data.objects if o.type == 'MESH' and o.name.split('.')[0] == 'stomach'][0]
_me = _st.evaluated_get(_dg).to_mesh()
STOM = BVHTree.FromPolygons([_st.matrix_world @ v.co for v in _me.vertices], [tuple(p.vertices) for p in _me.polygons])
_raw = []
for _fr in FR:
    _p, _ = route_at(_fr['frac'])
    _c = STOM.find_nearest(_p)
    _raw.append(1.0 if (_c[0] is not None and _c[3] < 0.012) else 0.0)
LIVER_FADE = []
for _i in range(len(_raw)):
    _w = _raw[max(0, _i - 20):_i + 21]
    LIVER_FADE.append(smooth(sum(_w) / len(_w) * 1.6))
print('liver fade frames:', sum(1 for v in LIVER_FADE if v > 0.01), flush=True)

# testa luminosa: piccola sfera emissiva + luce che illumina i tessuti dall'interno
head = bpy.data.objects.new('headlight', bpy.data.lights.new('headlight', 'POINT'))
head.data.energy = 0.35; head.data.color = (1.0, 0.72, 0.38); head.data.shadow_soft_size = 0.004
sc.collection.objects.link(head)
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.0024, segments=16, ring_count=8)
head_orb = bpy.context.active_object; head_orb.data.materials.append(emissive('orb', (1.0, 0.8, 0.55), 30.0))
head_orb.visible_shadow = False

# ---------------------------------------------------------------- uova e larve (ingrandite)
EGG_SC = 1.8  # uova ingrandite (reali: 45–75 µm)
def egg_object(name, fertile=True):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1, segments=48, ring_count=24)
    shell = bpy.context.active_object; shell.name = name
    if fertile:
        shell.scale = Vector((0.0078, 0.0064, 0.0064)) * EGG_SC
        tex = bpy.data.textures.new(name + '_tx', 'VORONOI'); tex.noise_scale = 0.18
        d = shell.modifiers.new('mamm', 'DISPLACE'); d.texture = tex; d.strength = 0.07; d.mid_level = 0.6
        shell.modifiers.new('sub', 'SUBSURF').levels = 1
        shell.data.materials.append(tissue(name + '_m', srgb('#a8742e'), alpha=0.55, edge=0.4, rough=0.45, sss=0.5, coat=0.2, emit=0.6, sss_r=(0.002, 0.0015, 0.001)))
    else:
        shell.scale = Vector((0.0095, 0.0052, 0.0052)) * EGG_SC
        shell.data.materials.append(tissue(name + '_m', srgb('#8f6a33'), alpha=0.5, edge=0.4, rough=0.5, sss=0.4, coat=0.1, emit=0.5))
    shell.visible_shadow = False
    # contenuto: morula (cellule) e larva arrotolata
    cells = []
    for k in range(8):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1, segments=16, ring_count=8)
        c = bpy.context.active_object; c.parent = shell; c.data.materials.append(M_CELL); cells.append(c)
    coil_pts = []
    for k in range(60):
        a = k / 59 * math.pi * 2.6; r = 0.55 - 0.25 * k / 59
        coil_pts.append(Vector((math.cos(a) * r, math.sin(a) * r * 0.85, (k / 59 - 0.5) * 0.35)))
    lv, _ = make_curve(name + '_larva', coil_pts, 0.07, M_LARVA)
    lv.parent = shell
    return shell, cells, lv

M_CELL = tissue('cell', srgb('#f0d9a8'), alpha=0.8, edge=0.2, rough=0.4, sss=0.6, coat=0.1, emit=0.4)
M_LARVA = tissue('larva', srgb('#f4e6d2'), alpha=0.95, edge=0.05, rough=0.35, sss=0.3, coat=0.4, emit=3.0)

def set_embryo(egg, emb, visible=True):
    shell, cells, lv = egg
    shell.hide_render = not visible
    q = emb / 0.55
    n = 0 if emb >= 0.55 else (1 if q < 0.25 else 2 if q < 0.5 else 4 if q < 0.75 else 8)
    lay = {1: [(0, 0, 0, 0.42)], 2: [(-0.2, 0, 0, 0.3), (0.2, 0, 0, 0.3)],
           4: [(-0.19, -0.15, 0, 0.22), (0.19, -0.15, 0, 0.22), (-0.19, 0.15, 0, 0.22), (0.19, 0.15, 0, 0.22)],
           8: [(math.cos(i / 8 * 6.283) * 0.32, math.sin(i / 8 * 6.283) * 0.24, (i % 2 - 0.5) * 0.15, 0.15) for i in range(8)]}.get(n, [])
    for i, c in enumerate(cells):
        if i < len(lay):
            x, y, z, r = lay[i]; c.location = (x, y, z); c.scale = (r, r, r); c.hide_render = not visible
        else:
            c.hide_render = True
    lv.hide_render = not (visible and emb >= 0.55)

def place(obj, p, t=None):
    obj.location = p
    if t is not None:
        obj.rotation_mode = 'QUATERNION'; obj.rotation_quaternion = t.to_track_quat('X', 'Z')

MAIN_EGG = egg_object('egg')
TRAIL = [egg_object(f'trail{i}') for i in range(5)]
SOIL_EGGS = [(egg_object(f'soil{i}', d['fertile']), d) for i, d in enumerate(SOIL)]

larva_ob, larva_sp = make_curve('larva', [Vector((0, 0, 0))] * 24, 0.0015, M_LARVA)
def set_larva(p, t, ph, length=0.034, visible=True):
    larva_ob.hide_render = not visible
    if not visible: return
    up = Vector((0, 0, 1)); n = t.cross(up)
    if n.length < 1e-6: n = Vector((1, 0, 0))
    n.normalize(); b = t.cross(n).normalized()
    for i in range(24):
        f = i / 23; off = 0.0028 * math.sin(f * 6.283 * 1.2 - ph) * (0.35 + 0.65 * (1 - f))
        off2 = 0.0012 * math.cos(f * 6.283 - ph * 0.7)
        q = p - t * (length * f) + n * off + b * off2
        larva_sp.points[i].co = (*q, 1)
        larva_sp.points[i].radius = 1.0 if 0.08 < f < 0.85 else 0.55

# ---------------------------------------------------------------- vermi adulti (scala reale)
ADULT = [V3(p) for p in S.get('adult', [])] if 'adult' in S else None
def catmull(pts, n):
    out = []
    P = [pts[0]] + pts + [pts[-1]]
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for j in range(n):
            t = j / n; t2 = t * t; t3 = t2 * t
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(pts[-1]); return out
COIL = catmull([V3(p) for p in S['adult_ctrl']], 12)
CC = [0.0]
for i in range(1, len(COIL)): CC.append(CC[-1] + (COIL[i] - COIL[i - 1]).length)
def coil_at(s):
    s = max(0, min(CC[-1], s)); i = max(0, min(len(COIL) - 2, bisect.bisect_right(CC, s) - 1))
    seg = CC[i + 1] - CC[i]; f = (s - CC[i]) / seg if seg > 0 else 0
    return COIL[i].lerp(COIL[i + 1], f), (COIL[i + 1] - COIL[i]).normalized()
M_WORM = tissue('worm', srgb('#d9938a'), alpha=0.95, edge=0.05, rough=0.3, sss=0.6, coat=0.8, sss_r=(0.004, 0.003, 0.002))
taper = bpy.data.curves.new('taper', 'CURVE'); tsp = taper.splines.new('BEZIER'); tsp.bezier_points.add(2)
for i, (x, y) in enumerate([(0, 0.35), (0.5, 1.0), (1, 0.25)]):
    bp = tsp.bezier_points[i]; bp.co = (x, y, 0); bp.handle_left_type = bp.handle_right_type = 'AUTO'
taper_ob = bpy.data.objects.new('taper', taper); sc.collection.objects.link(taper_ob); taper_ob.hide_render = True
WORMS = []
for name, ln, r, male, ph in [('female', 0.27, 0.0026, False, 0.0), ('male', 0.21, 0.0021, True, 1.7)]:
    ob, sp = make_curve(name, [Vector((0, 0, 0))] * 90, r, M_WORM, res=4)
    ob.data.taper_object = taper_ob
    WORMS.append((ob, sp, ln, male, ph))
def set_worms(start_f, start_m, clock, alpha):
    for (ob, sp, ln, male, ph), st in zip(WORMS, (start_f, start_m)):
        ob.hide_render = alpha < 0.05
        if ob.hide_render: continue
        side = -1 if male else 1
        for i in range(90):
            f = i / 89; p, t = coil_at(st + f * ln)
            n = t.cross(Vector((0, -1, 0)))
            if n.length < 1e-6: n = Vector((1, 0, 0))
            n.normalize(); b = t.cross(n).normalized()
            off = side * 0.0085 + 0.0035 * math.sin(f * ln * 70 - clock * 1.8 + ph)
            offb = 0.003 * math.cos(f * ln * 45 - clock * 1.3 + ph)
            if male and f > 0.88: off += ((f - 0.88) / 0.12) ** 2 * 0.012
            q = p + n * off + b * offb
            sp.points[i].co = (*q, 1)

# ---------------------------------------------------------------- suolo
bpy.ops.mesh.primitive_plane_add(size=2.6, location=V3([0.15, 0.0, 0.12]))
ground = bpy.context.active_object
ground.modifiers.new('sub', 'SUBSURF').levels = 3
gm, nt, out = node_mat('soil')
b = nt.nodes.new('ShaderNodeBsdfPrincipled'); b.inputs['Roughness'].default_value = 0.95
nz = nt.nodes.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = 120; nz.inputs['Detail'].default_value = 8
vr = nt.nodes.new('ShaderNodeTexVoronoi'); vr.inputs['Scale'].default_value = 260
cr = nt.nodes.new('ShaderNodeValToRGB')
cr.color_ramp.elements[0].color = (*srgb('#0d0805'), 1); cr.color_ramp.elements[1].color = (*srgb('#3b2b1c'), 1)
b.inputs['Specular IOR Level'].default_value = 0.15
mx = nt.nodes.new('ShaderNodeMath'); mx.operation = 'MULTIPLY'
nt.links.new(nz.outputs['Fac'], mx.inputs[0]); nt.links.new(vr.outputs['Distance'], mx.inputs[1])
nt.links.new(nz.outputs['Fac'], cr.inputs['Fac']); nt.links.new(cr.outputs['Color'], b.inputs['Base Color'])
bump = nt.nodes.new('ShaderNodeBump'); bump.inputs['Strength'].default_value = 0.8
nt.links.new(mx.outputs[0], bump.inputs['Height']); nt.links.new(bump.outputs['Normal'], b.inputs['Normal'])
# bordo sfumato: alfa radiale
tc = nt.nodes.new('ShaderNodeTexCoord'); gr = nt.nodes.new('ShaderNodeTexGradient'); gr.gradient_type = 'SPHERICAL'
mp = nt.nodes.new('ShaderNodeMapping'); mp.inputs['Scale'].default_value = (1.6, 1.6, 1.6)
nt.links.new(tc.outputs['Object'], mp.inputs['Vector']); nt.links.new(mp.outputs['Vector'], gr.inputs['Vector'])
ar = nt.nodes.new('ShaderNodeMapRange'); ar.inputs['From Min'].default_value = 0.0; ar.inputs['From Max'].default_value = 0.6
nt.links.new(gr.outputs['Fac'], ar.inputs['Value']); nt.links.new(ar.outputs['Result'], b.inputs['Alpha'])
nt.links.new(b.outputs['BSDF'], out.inputs['Surface'])
ground.data.materials.append(gm)

# ---------------------------------------------------------------- luci e mondo
w = bpy.data.worlds.new('w'); sc.world = w; w.use_nodes = True
w.node_tree.nodes['Background'].inputs['Color'].default_value = (0.004, 0.007, 0.016, 1)
w.node_tree.nodes['Background'].inputs['Strength'].default_value = 1.0
def area(name, loc, rot, size, energy, color):
    l = bpy.data.lights.new(name, 'AREA'); l.size = size; l.energy = energy; l.color = color
    o = bpy.data.objects.new(name, l); o.location = loc; o.rotation_euler = rot; sc.collection.objects.link(o); return o
area('key', (-1.4, 2.4, 2.6), (math.radians(-55), math.radians(-25), 0), 1.6, 120, (1.0, 0.96, 0.92))
area('fill', (1.8, 2.0, 1.2), (math.radians(-70), math.radians(35), 0), 2.0, 30, (0.85, 0.9, 1.0))
area('rimL', (-1.6, -2.2, 1.6), (math.radians(65), math.radians(-30), 0), 1.4, 420, (0.45, 0.65, 1.0))
area('rimR', (1.6, -2.2, 1.6), (math.radians(65), math.radians(30), 0), 1.4, 420, (0.45, 0.65, 1.0))

# ---------------------------------------------------------------- camera e render
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); sc.collection.objects.link(cam); sc.camera = cam
cam.data.sensor_fit = 'VERTICAL'; cam.data.angle = math.radians(28); cam.data.clip_start = 0.02; cam.data.clip_end = 30
cam.data.dof.use_dof = False; cam.data.dof.aperture_fstop = 4.0
W, H = map(int, A.res.split('x'))
r = sc.render; r.engine = 'CYCLES'; r.resolution_x = W; r.resolution_y = H; r.resolution_percentage = 100
r.image_settings.file_format = 'PNG'; r.film_transparent = False
cy = sc.cycles; cy.device = 'CPU'; cy.samples = A.samples; cy.use_adaptive_sampling = True; cy.adaptive_threshold = 0.05
cy.use_denoising = (A.layer != 'over')  # la linea emissiva non ha rumore; cy.denoiser = 'OPENIMAGEDENOISE'
cy.max_bounces = 4; cy.transparent_max_bounces = 16; cy.transmission_bounces = 0; cy.diffuse_bounces = 1; cy.glossy_bounces = 1; cy.volume_bounces = 0
cy.caustics_reflective = False; cy.caustics_refractive = False; cy.blur_glossy = 1.0
if A.threads: r.threads_mode = 'FIXED'; r.threads = A.threads
r.use_persistent_data = True
try:
    sc.view_settings.view_transform = 'AgX'; sc.view_settings.look = 'AgX - Medium High Contrast'
except Exception:
    pass

# ---------------------------------------------------------------- stato per fotogramma
def set_frame(i):
    fr = FR[i]; k = fr['stage']; u = fr['u']; frac = fr['frac']; key = ST[k]['k']; clock = i / 30.0
    cp, at = V3(fr['cam']), V3(fr['at'])
    cam.location = cp; cam.rotation_mode = 'QUATERNION'; cam.rotation_quaternion = (at - cp).to_track_quat('-Z', 'Y')
    cam.data.dof.focus_distance = (at - cp).length
    uh = uidx(frac)
    ut = uidx(FR[max(0, i - 75)]['frac']) if i >= 75 else 0.0  # scia: strada percorsa negli ultimi 2,5 s
    if ut > uh: ut = 0.0
    rv = smooth((i - 1890) / 36.0)  # riepilogo: l'intero percorso si riaccende alla fine
    route_ob.data.bevel_factor_start = ut * (1 - rv)
    route_ob.data.bevel_factor_end = max(0.0005, uh)
    guide_ob.data.bevel_factor_end = max(0.0005, uh)
    ROUTE_VAL['Uh'].outputs[0].default_value = uh
    ROUTE_VAL['Ut'].outputs[0].default_value = ut
    ROUTE_VAL['Rv'].outputs[0].default_value = rv
    lf = LIVER_FADE[i]
    for mn, a0, a1 in (('liver', 0.55, 0.90), ('gall', 0.6, 0.9)):
        mr = [n for n in M[mn].node_tree.nodes if n.type == 'MAP_RANGE'][0]
        mr.inputs['To Min'].default_value = a0 * (1 - 0.8 * lf)
        mr.inputs['To Max'].default_value = a1 * (1 - 0.55 * lf)
    p, t = route_at(frac)
    head.location = p; head_orb.location = p
    # glifi
    set_embryo(MAIN_EGG, 1, False); set_larva(p, t, 0, visible=False)
    for e in TRAIL: set_embryo(e, 0, False)
    if key in ('ingest', 'contagion'):
        set_embryo(MAIN_EGG, 1); place(MAIN_EGG[0], p, t)
    elif key == 'hatch':
        if u < 0.45: set_embryo(MAIN_EGG, 1); place(MAIN_EGG[0], p, t)
        else: set_larva(p, t, clock * 9)
    elif key in ('liver', 'heart', 'lungs', 'ascent', 'swallow'):
        set_larva(p, t, clock * 9)
    elif key == 'adult':
        if u < 0.7: set_larva(p, t, clock * 9, 0.026 + 0.03 * u)
    elif key == 'eggs':
        s0 = BOUNDS[k][0]
        for j, e in enumerate([MAIN_EGG] + TRAIL[:4]):
            fj = frac - j * 0.014 / LEN * LEN / LEN  # passo in frazione
            fj = frac - j * (0.014 / LEN)
            if fj < s0: continue
            q, tq = route_at(fj); set_embryo(e, 0); place(e[0], q, tq); e[0].scale = Vector((0.0078, 0.0064, 0.0064)) * EGG_SC * 0.8
    elif key == 'soil':
        set_embryo(MAIN_EGG, smooth((u - 0.15) / 0.75)); place(MAIN_EGG[0], p, t)
    if key not in ('eggs',): MAIN_EGG[0].scale = Vector((0.0078, 0.0064, 0.0064)) * EGG_SC
    # uova nel suolo
    for (egg, d) in SOIL_EGGS:
        if key == 'soil': vis = frac >= d['s']; emb = smooth((u - 0.15) / 0.75)
        elif key == 'contagion': vis = True; emb = 1
        else: vis = False; emb = 0
        if d['fertile']: set_embryo(egg, emb, vis)
        else: set_embryo(egg, 0, vis)
        place(egg[0], V3(d['p']) + Vector((0, 0, -0.004)), Vector((1, 0.3 * (hash(str(d['p'])) % 3 - 1), 0)).normalized())
    # vermi adulti
    wa = 1.0 if k > 7 else (smooth((u - 0.2) / 0.5) if k == 7 else 0.0)
    slide = math.sin(clock * 0.3) * 0.02
    set_worms(CC[-1] * 0.08 + slide, CC[-1] * 0.55 - slide, clock, wa)

# ---------------------------------------------------------------- livelli
# anat: anatomia, suolo, uova nel suolo, vermi, luce della testa
# over: linea luminosa, testa, uova/larve in viaggio (sempre visibili, sommate in post-produzione)
OVER = {route_ob, guide_ob, head_orb, larva_ob, MAIN_EGG[0], *MAIN_EGG[1], MAIN_EGG[2]}
for e in TRAIL: OVER |= {e[0], *e[1], e[2]}
OVER |= {w[0] for w in WORMS}  # i vermi adulti devono leggersi attraverso l'intestino
def apply_layer(obj_hidden_by_glyph):
    for o in sc.objects:
        if o.type in ('LIGHT', 'CAMERA'): continue
        if A.layer == 'anat' and o in OVER: o.hide_render = True
        if A.layer == 'over' and o not in OVER: o.hide_render = True
if A.layer == 'over':
    r.film_transparent = True; head.hide_render = True
_orig_set_frame = set_frame
def set_frame(i):
    _orig_set_frame(i); apply_layer(None)

if A.blend:
    set_frame(int(A.frames.split(':')[0])); bpy.ops.wm.save_as_mainfile(filepath=A.blend)

a, b_, st = map(int, A.frames.split(':'))
os.makedirs(A.out, exist_ok=True)
import time
for i in range(a, b_, st):
    path = os.path.join(A.out, f'b_{i:05d}.png')
    if os.path.exists(path): continue
    if A.liver_only and LIVER_FADE[i] <= 0.005: continue
    t0 = time.time(); set_frame(i)
    r.filepath = path; bpy.ops.render.render(write_still=True)
    print(f'frame {i} {time.time() - t0:.1f}s', flush=True)
