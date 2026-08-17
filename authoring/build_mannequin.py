"""Rebuild mannequin armor shells from the SMPL-X template surface.

Run: blender --background --python build_mannequin.py -- <repo_dir> <out_dir>

Keeps the skeleton and all ball/bearing "joint" parts from the existing NPZ
assets; regenerates every "armor" part by carving the SMPL-X neutral template
by LBS-weight ownership, eroding panel gaps at articulations, capping the
openings, and subdividing. Right-side parts are exact mirrors of left ones.
"""

import heapq
import sys
from collections import defaultdict

import bmesh
import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1 :]
REPO, OUT = argv[0], argv[1]
ORIG = argv[2] if len(argv) > 2 else f"{REPO}/src/mannequin/assets"
SMPLX_PATH = "/CT/aboscolo/static00/smplx/SMPLX_NEUTRAL.npz"

ARMOR_BUDGET = {0: 24292, 1: 9712, 2: 2740}
HEAD_SHARE_CAP = 0.30
ALLOC_POWER = 0.8
HEAD_SMOOTH_ITERS = 320  # Taubin passes that turn the SMPL-X face into a soft mannequin face
EGG_BLEND = 0.55  # blend of the smoothed skull toward its bounding ellipsoid (0 = full face)

# torso blocks, thighs, shins, forearms, and feet keep the original
# hand-authored design; geometry is copied verbatim from the pre-polish
# assets instead of being carved from SMPL-X (feet stay toe-less)
RESTORE_OLD = {
    "pelvis_shell",
    "abdomen_shell",
    "chest_shell",
    "thigh_L",
    "thigh_R",
    "shin_L",
    "shin_R",
    "forearm_L",
    "forearm_R",
    "rear_foot_L",
    "rear_foot_R",
    "forefoot_L",
    "forefoot_R",
}

# ---------------------------------------------------------------- SMPL-X data
sm = np.load(SMPLX_PATH, allow_pickle=True)
V = sm["v_template"].astype(np.float64)
F = sm["f"].astype(np.int64)
W = sm["weights"].astype(np.float64)
J = sm["J_regressor"].astype(np.float64) @ V
j2n = sm["joint2num"].item()
S = {name: int(i) for name, i in j2n.items()}


# ------------------------------------------------------------------- regions
# score columns: one per competing region; parts we keep reference a region id
def col(*joints, extra=None):
    c = W[:, list(joints)].sum(axis=1)
    if extra is not None:
        c = c + extra
    return c


# Plane cutoffs / reach targets that close the gaps restored parts leave:
# the ankle-owned lower calf joins the shin (feet are the toe-less blocks),
# the upper-arm cone is stretched to reach the shoulder ball, and the head
# keeps only a short neck stub that meets the original neck_pedestal joint
# (top at y=0.172) instead of the full trapezius flare.
ARM_REACH = 0.168  # just outboard of the shoulder ball center (x=0.138)
Y_SHIN_CUT = -1.200  # mid ankle ball, matches the original shin bottom
Y_NECK_CUT = 0.158  # rim overlaps the pedestal top by ~14mm

# limb ends morph into circular sockets centered on their joint balls: the
# SMPL-X ankle/axilla surface is off the ball axis, and the mannequin look
# wants clean sphere-aligned part mouths more than anatomical accuracy
SHOULDER_BALL_YZ = np.array([0.0730, -0.0127])  # ball center, r=0.045
ANKLE_BALL_XZ = np.array([0.0865, -0.0235])  # ball center, r=0.031
ARM_MOUTH_R = 0.040
# mostly-inward taper: wide enough to keep the calf silhouette, narrow enough
# that the naturally slim front of the ankle is never pushed outward (outward
# pushes flare and leave a dimple band above the hem)
SHIN_MOUTH_R = 0.027
SHIN_TOP_TARGET = -0.838  # tuck the shin top against the knee ball (orig -0.842)


def round_end(verts, ax, end, blend, cidx, center, radius):
    """Blend cross-sections near coord ax==end into a circle around center."""
    t = np.clip(1.0 - (verts[:, ax] - end) / blend, 0.0, 1.0)
    m = t > 0.0
    p = verts[np.ix_(m, cidx)] - center
    r = np.linalg.norm(p, axis=1, keepdims=True)
    np.maximum(r, 1e-9, out=r)
    target = center + p / r * radius
    w = t[m][:, None]
    verts[np.ix_(m, cidx)] = verts[np.ix_(m, cidx)] * (1.0 - w) + target * w


# the shin mask reaches 2cm past the plane cut; bisect_plane trims it exactly
ext_shin = V[:, 1] > Y_SHIN_CUT - 0.02
ext_neck = V[:, 1] > Y_NECK_CUT

REGION_SCORES = {
    "pelvis": col(S["Pelvis"]),
    "abdomen": col(S["Spine1"], S["Spine2"]),
    "chest": col(S["Spine3"], S["L_Collar"], S["R_Collar"]),
    "head": col(S["Head"], S["Jaw"], S["L_Eye"], S["R_Eye"]) + col(S["Neck"]) * ext_neck,
}
for side in ("L", "R"):
    REGION_SCORES[f"thigh_{side}"] = col(S[f"{side}_Hip"])
    REGION_SCORES[f"shin_{side}"] = col(S[f"{side}_Knee"]) + col(S[f"{side}_Ankle"]) * ext_shin
    REGION_SCORES[f"rear_foot_{side}"] = col(S[f"{side}_Ankle"])
    REGION_SCORES[f"forefoot_{side}"] = col(S[f"{side}_Foot"])
    REGION_SCORES[f"upper_arm_{side}"] = col(S[f"{side}_Shoulder"])
    REGION_SCORES[f"forearm_{side}"] = col(S[f"{side}_Elbow"])
    REGION_SCORES[f"palm_{side}"] = col(S[f"{side}_Wrist"])
    for finger in ("Index", "Middle", "Pinky", "Ring", "Thumb"):
        for seg in (1, 2, 3):
            REGION_SCORES[f"{finger.lower()}{seg}_{side}"] = col(S[f"{side}_{finger}{seg}"])

region_names = list(REGION_SCORES)
scores = np.stack([REGION_SCORES[n] for n in region_names], axis=1)
assign = scores.argmax(axis=1)
face_assign = scores[F].mean(axis=1).argmax(axis=1)


def face_mode(name):
    """Hands and feet: tile every face so parts butt-joint with no eroded gap."""
    return any(k in name for k in ("palm", "foot", "index", "middle", "pinky", "ring", "thumb"))


# gap radii (one-sided, meters) per region
def gap_for(name):
    if name in ("pelvis", "abdomen", "chest"):
        return 0.002
    if name == "head":
        return 0.002
    if "upper_arm" in name or "shin" in name:
        return 0.0005  # keep the deltoid at the shoulder ball and the calf at the ankle
    if any(k in name for k in ("thigh", "forearm")):
        return 0.0012
    if "foot" in name or "palm" in name:
        return 0.0008
    return 0.0  # fingers: the mixed-weight face ring already forms the gap


# mesh adjacency
edges = defaultdict(set)
for a, b, c in F:
    edges[a].update((b, c))
    edges[b].update((a, c))
    edges[c].update((a, b))


def erode_region(rid, gap, seed_keep=None):
    """Erode ≥ one face ring at region boundaries (gap grows it geodesically).

    seed_keep: bool vertex mask; boundary verts outside it do not erode —
    used to protect rims that a plane cut will trim exactly instead.
    """
    fmask = (assign[F] == rid).all(axis=1)
    faces = F[fmask]
    if not len(faces) or gap <= 0.0:
        return faces
    inpart = np.zeros(len(V), bool)
    inpart[np.unique(faces)] = True
    # boundary verts: part verts that touch a face not fully in the part
    outside_faces = F[~fmask]
    touch = np.zeros(len(V), bool)
    touch[np.unique(outside_faces)] = True
    if seed_keep is not None:
        touch &= seed_keep
    seeds = np.where(inpart & touch)[0]
    if not len(seeds):
        return faces
    dist = {int(s): 0.0 for s in seeds}
    heap = [(0.0, int(s)) for s in seeds]
    heapq.heapify(heap)
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, np.inf) or d > gap:
            continue
        for v in edges[u]:
            if not inpart[v]:
                continue
            nd = d + np.linalg.norm(V[u] - V[v])
            if nd < dist.get(v, np.inf):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    near = np.zeros(len(V), bool)
    for u, d in dist.items():
        if d < gap:
            near[u] = True
    return faces[~near[faces].any(axis=1)]


def components(faces):
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for f in faces:
        for v in f:
            parent.setdefault(int(v), int(v))
        a = find(int(f[0]))
        for v in f[1:]:
            b = find(int(v))
            parent[b] = a
    groups = defaultdict(list)
    for i, f in enumerate(faces):
        groups[find(int(f[0]))].append(i)
    return [faces[np.array(ix)] for ix in groups.values()]


# parts that end on a plane cut: (axis, cut) with the part on the coord>cut
# side.  The carved region overshoots the plane and gets bisected exactly on
# it, so the rim is a clean straight edge instead of a ragged triangle zigzag.
PLANE_CUTS = {
    "shin_L": (1, Y_SHIN_CUT),
    "upper_arm_L": (0, ARM_REACH),  # the stretch overshoots; bisect trims flat
}

# ------------------------------------------------------- build region meshes
FINGERS = ("index", "middle", "pinky", "ring", "thumb")

part_faces = {}
for name in region_names:
    if name == "neck" or name.endswith("_R"):
        continue
    if any(name.startswith(finger) and name[len(finger)].isdigit() for finger in FINGERS):
        continue  # finger segments become capsules, not carved shells
    rid = region_names.index(name)
    if face_mode(name):
        faces = F[face_assign == rid]
    else:
        gap = gap_for(name)
        seed_keep = None
        if name in PLANE_CUTS:
            ax, cut = PLANE_CUTS[name]
            seed_keep = V[:, ax] > cut + 0.005  # the plane rim is cut, not eroded
        faces = erode_region(rid, gap, seed_keep)
        while not len(faces) and gap > 0.0:
            gap = 0.0 if gap < 2e-4 else gap * 0.5
            faces = erode_region(rid, gap, seed_keep)
    comps = components(faces)
    if not comps:
        raise RuntimeError(f"region {name} vanished")
    # head keeps only the skull skin: eyeballs are dropped and the eyelid
    # openings get capped, so heavy smoothing yields soft closed eyes
    faces = max(comps, key=len)
    if len(faces) < 8:
        raise RuntimeError(f"region {name} degenerate: {len(faces)} faces")
    part_faces[name] = faces

# rename to the NPZ part vocabulary
RENAME = {"pelvis": "pelvis_shell", "abdomen": "abdomen_shell", "chest": "chest_shell", "head": "mannequin_head"}
part_faces = {RENAME.get(k, k): v for k, v in part_faces.items() if RENAME.get(k, k) not in RESTORE_OLD}

# ------------------------------------------------- finger capsules (L side)
# Each phalanx is a capsule with end caps centered on its joints. A knuckle
# ball sits on every finger joint (elbow-style), slightly proud of the
# segments, and boolean-carves a clearance cup into both abutting capsules —
# and into the palm at the base knuckles — so the parts nest with a visible
# groove and stay closed when bent. The distal capsule runs joint3 ->
# fingertip, absorbing the old fingertip part.
CAPSULE_RADIUS_SCALE = 0.84
KNUCKLE_BALL_SCALE = 1.28  # ball radius vs the parent-side segment radius: proud like the elbow balls
KNUCKLE_MAX_FRACTION = 0.30  # cap ball radius vs the shorter adjacent segment, so short segments survive
KNUCKLE_CLEARANCE = 0.0018  # visible air gap between ball and the carved segment cups
CAPSULE_TIP_KEEP = 0.55  # rounded-end fraction kept outside the carve: only the tip gets dished
MIN_SEGMENT_LENGTH = 0.004  # never shorten a capsule below this cylinder length


def capsule_mesh(p0, p1, radius, segments=14, rings=5):
    """Closed capsule with hemispherical end caps centered on p0 and p1."""
    axis = p1 - p0
    length = np.linalg.norm(axis)
    axis = axis / length
    ortho = np.array([1.0, 0.0, 0.0]) if abs(axis[1]) > 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(ortho, axis)
    x /= np.linalg.norm(x)
    y = np.cross(axis, x)

    phi_bottom = np.linspace(-np.pi / 2, 0.0, rings + 1)[1:]
    phi_top = np.linspace(0.0, np.pi / 2, rings + 1)[:-1]
    ring_specs = [(np.sin(p) * radius, np.cos(p) * radius) for p in phi_bottom]
    ring_specs += [(length + np.sin(p) * radius, np.cos(p) * radius) for p in phi_top]

    angles = np.linspace(0.0, 2 * np.pi, segments, endpoint=False)
    circle = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    verts = [p0 - axis * radius]
    for offset, ring_radius in ring_specs:
        center = p0 + axis * offset
        verts.extend(center + ring_radius * (cx * x + cy * y) for cx, cy in circle)
    verts.append(p1 + axis * radius)

    faces = [[0, 1 + (i + 1) % segments, 1 + i] for i in range(segments)]
    for ring in range(len(ring_specs) - 1):
        a = 1 + ring * segments
        b = a + segments
        for i in range(segments):
            j = (i + 1) % segments
            faces.append([a + i, a + j, b + j, b + i])
    tip = len(verts) - 1
    base = 1 + (len(ring_specs) - 1) * segments
    faces.extend([tip, base + i, base + (i + 1) % segments] for i in range(segments))
    return np.array(verts), faces


def sphere_mesh(center, radius, segments=16, rings=8):
    phis = np.linspace(-np.pi / 2, np.pi / 2, rings + 1)[1:-1]
    angles = np.linspace(0.0, 2 * np.pi, segments, endpoint=False)
    verts = [center - np.array([0.0, radius, 0.0])]
    for phi in phis:
        ring_radius = np.cos(phi) * radius
        for a in angles:
            verts.append(center + np.array([np.cos(a) * ring_radius, np.sin(phi) * radius, np.sin(a) * ring_radius]))
    verts.append(center + np.array([0.0, radius, 0.0]))

    faces = [[0, 1 + (i + 1) % segments, 1 + i] for i in range(segments)]
    for ring in range(len(phis) - 1):
        a = 1 + ring * segments
        b = a + segments
        for i in range(segments):
            j = (i + 1) % segments
            faces.append([a + i, a + j, b + j, b + i])
    top = len(verts) - 1
    base = 1 + (len(phis) - 1) * segments
    faces.extend([top, base + i, base + (i + 1) % segments] for i in range(segments))
    return np.array(verts), faces


def build_finger_parts():
    """Capsule segments, knuckle balls, boolean cutter spheres, and carve map."""
    capsules, knuckles, cutters = {}, {}, {}
    carves = defaultdict(list)
    for finger in FINGERS:
        joint = finger.capitalize()
        j = [J[S[f"L_{joint}{seg}"]] for seg in (1, 2, 3)]
        seg_vids = [np.unique(F[face_assign == region_names.index(f"{finger}{seg}_L")]) for seg in (1, 2, 3)]
        tip = V[seg_vids[2][np.linalg.norm(V[seg_vids[2]] - j[2], axis=1).argmax()]]
        ends = [j[0], j[1], j[2], tip]

        radii = []
        for seg in range(3):
            axis = (ends[seg + 1] - ends[seg]) / np.linalg.norm(ends[seg + 1] - ends[seg])
            rel = V[seg_vids[seg]] - ends[seg]
            radial = rel - np.outer(rel @ axis, axis)
            radii.append(np.median(np.linalg.norm(radial, axis=1)) * CAPSULE_RADIUS_SCALE)

        lengths = [np.linalg.norm(ends[seg + 1] - ends[seg]) for seg in range(3)]
        ball_radii = {}
        for seg in (1, 2, 3):
            adjacent = lengths[seg - 1] if seg == 1 else min(lengths[seg - 2], lengths[seg - 1])
            ball_radii[seg] = min(KNUCKLE_BALL_SCALE * radii[seg - 1], KNUCKLE_MAX_FRACTION * adjacent)
            knuckles[f"{finger}_knuckle{seg}_L"] = (j[seg - 1], ball_radii[seg])
            cutters[f"{finger}{seg}"] = (j[seg - 1], ball_radii[seg] + KNUCKLE_CLEARANCE)

        for seg, part in enumerate(("proximal", "middle", "distal")):
            p0, p1 = ends[seg], ends[seg + 1]
            axis = (p1 - p0) / np.linalg.norm(p1 - p0)
            if part == "distal":
                p1 = p1 - axis * radii[seg] * 1.5  # stop short of the skin tip
            # pull the rounded ends back from the knuckle balls so the balls sit
            # in open air and the carve only dishes the very tip of the capsule
            pull_start = ball_radii[seg + 1] + KNUCKLE_CLEARANCE + CAPSULE_TIP_KEEP * radii[seg]
            pull_end = (
                0.0 if part == "distal" else ball_radii[seg + 2] + KNUCKLE_CLEARANCE + CAPSULE_TIP_KEEP * radii[seg]
            )
            raw = np.linalg.norm(p1 - p0)
            scale = min(1.0, (raw - MIN_SEGMENT_LENGTH) / (pull_start + pull_end))
            p0 = p0 + axis * pull_start * scale
            p1 = p1 - axis * pull_end * scale
            capsules[f"{finger}_{part}_L"] = capsule_mesh(p0, p1, radii[seg])
        carves[f"{finger}_proximal_L"] = [f"{finger}1", f"{finger}2"]
        carves[f"{finger}_middle_L"] = [f"{finger}2", f"{finger}3"]
        carves[f"{finger}_distal_L"] = [f"{finger}3"]
        carves["palm_L"].append(f"{finger}1")
    return capsules, knuckles, cutters, carves


FINGER_CAPSULES, FINGER_KNUCKLES, FINGER_CUTTERS, FINGER_CARVES = build_finger_parts()

# carve the palm around the existing wrist ball with the same air gap the
# hand-authored forearm keeps on its side of the ball
_orig0 = np.load(f"{ORIG}/lod0.npz", allow_pickle=False)
_o_names = _orig0["link_names"].tolist()
_o_offsets = _orig0["local_offsets"].astype(np.float64)
_o_joints = np.zeros_like(_o_offsets)
_o_joints[0] = _o_offsets[0]
for _j in range(1, len(_orig0["parents"])):
    _o_joints[_j] = _o_joints[_orig0["parents"][_j]] + _o_offsets[_j]


def _orig_link_world(idx):
    s, c = int(_orig0["link_vertex_starts"][idx]), int(_orig0["link_vertex_counts"][idx])
    return _orig0["vertices"][s : s + c].astype(np.float64) + _o_joints[int(_orig0["link_joint_indices"][idx])]


_ball = _orig_link_world(next(i for i, n in enumerate(_o_names) if "__wrist_ball_L__" in n))
_wrist_center = (_ball.min(0) + _ball.max(0)) / 2
_wrist_radius = np.linalg.norm(_ball - _wrist_center, axis=1).mean()
_forearm = _orig_link_world(next(i for i, n in enumerate(_o_names) if "__forearm_L__" in n))
_wrist_gap = np.linalg.norm(_forearm - _wrist_center, axis=1).min() - _wrist_radius
FINGER_CUTTERS["wrist"] = (_wrist_center, _wrist_radius + _wrist_gap)
FINGER_CARVES["palm_L"].append("wrist")

# ---------------------------------------------------- Blender part processing
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
depsgraph_objects = {}

part_geoms = {}
for name, faces in part_faces.items():
    vids = np.unique(faces)
    remap = -np.ones(len(V), np.int64)
    remap[vids] = np.arange(len(vids))
    part_geoms[name] = (V[vids].copy(), remap[faces].tolist())
part_geoms.update(FINGER_CAPSULES)


def make_cutter(name, center, radius):
    cverts, cfaces = sphere_mesh(center, radius, segments=24, rings=12)
    cmesh = bpy.data.meshes.new(name)
    cmesh.from_pydata(cverts.tolist(), [], cfaces)
    cmesh.validate()
    # outward normals are required: the boolean silently no-ops on an
    # inside-out cutter volume
    cbm = bmesh.new()
    cbm.from_mesh(cmesh)
    bmesh.ops.recalc_face_normals(cbm, faces=cbm.faces)
    cbm.to_mesh(cmesh)
    cbm.free()
    cutter = bpy.data.objects.new(name, cmesh)
    scene.collection.objects.link(cutter)
    return cutter


cutter_objects = {}
recut_objects = {}
for key, (center, radius) in FINGER_CUTTERS.items():
    cutter_objects[key] = make_cutter(f"cut_{key}", center, radius)
    # recut cutters are slightly inflated: cutting twice with the same sphere
    # hits coincident surfaces and the exact solver destroys the mesh, and a
    # too-small inflation leaves serrated slivers along the old rim
    recut_objects[key] = make_cutter(f"recut_{key}", center, radius + 0.0005)

for name, (verts, fl) in part_geoms.items():
    if name == "upper_arm_L":
        # stretch the proximal cone (erosion shortened it by a face ring) so
        # its rounded cap reaches the shoulder ball; the distal half is fixed
        x = verts[:, 0]
        xmid = 0.28
        m = x < xmid
        verts[m, 0] = xmid + (x[m] - xmid) * (xmid - (ARM_REACH - 0.005)) / (xmid - x.min())
        round_end(verts, 0, ARM_REACH, 0.105, [1, 2], SHOULDER_BALL_YZ, ARM_MOUTH_R)
    if name == "shin_L":
        # lift the knee-end rim to the ball and center the ankle mouth on it
        y = verts[:, 1]
        ymid = -1.02
        m = y > ymid
        verts[m, 1] = ymid + (y[m] - ymid) * (SHIN_TOP_TARGET - ymid) / (y.max() - ymid)
        round_end(verts, 1, Y_SHIN_CUT, 0.14, [0, 2], ANKLE_BALL_XZ, SHIN_MOUTH_R)

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts.tolist(), [], fl)
    mesh.validate()
    bm = bmesh.new()
    bm.from_mesh(mesh)
    if name in PLANE_CUTS:
        # trim the overshoot exactly on the plane: a straight mechanical rim
        ax, cut = PLANE_CUTS[name]
        no = Vector((0.0, 0.0, 0.0))
        no[ax] = 1.0
        co = Vector((0.0, 0.0, 0.0))
        co[ax] = cut
        bmesh.ops.bisect_plane(
            bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], plane_co=co, plane_no=no, clear_inner=True
        )
        if name == "upper_arm_L":
            # the jagged eroded rim outlives the bisect where it sits outboard
            # of the plane; snap it flush so the cap is one flat disk
            for v in bm.verts:
                if v.co[0] < 0.185 and any(e.is_boundary for e in v.link_edges):
                    v.co[0] = cut
    # smooth the jagged cut rims along the surface before capping;
    # face-mode parts get only a light pass so shared boundaries barely move
    rim = [v for v in bm.verts if any(e.is_boundary for e in v.link_edges)]
    if name in PLANE_CUTS:
        ax, cut = PLANE_CUTS[name]
        planar = [v for v in rim if v.co[ax] <= cut + 1e-5]
        rim = [v for v in rim if v.co[ax] > cut + 1e-5]  # only off-plane rims move freely
        # round the coarse polygonal cut rim without leaving the plane
        axes = {"use_axis_x": ax != 0, "use_axis_y": ax != 1, "use_axis_z": ax != 2}
        for _ in range(3):
            bmesh.ops.smooth_vert(bm, verts=planar, factor=0.5, **axes)
    if face_mode(name):
        iters, factor = 2, 0.4
    else:
        iters, factor = (
            (5, 0.5) if name in ("pelvis_shell", "abdomen_shell", "chest_shell", "mannequin_head") else (3, 0.5)
        )
    for _ in range(iters):
        bmesh.ops.smooth_vert(bm, verts=rim, factor=factor, use_axis_x=True, use_axis_y=True, use_axis_z=True)
    # cap every boundary loop with a deterministic centroid fan
    boundary = [e for e in bm.edges if e.is_boundary]
    if boundary:
        adj = defaultdict(list)
        for e in boundary:
            adj[e.verts[0]].append(e)
            adj[e.verts[1]].append(e)
        unvisited = set(boundary)
        centroids = []
        loops_info = []
        while unvisited:
            e0 = next(iter(unvisited))
            loop = [e0.verts[0], e0.verts[1]]
            unvisited.discard(e0)
            while True:
                tail = loop[-1]
                nxt = next((e for e in adj[tail] if e in unvisited), None)
                if nxt is None:
                    break
                unvisited.discard(nxt)
                loop.append(nxt.other_vert(tail))
            if loop[0] is loop[-1]:
                loop.pop()
            if len(loop) < 3:
                continue
            centroid = Vector((0.0, 0.0, 0.0))
            for v in loop:
                centroid += v.co
            centroid /= len(loop)
            cv = bm.verts.new(centroid)
            centroids.append(cv)
            length = sum((a.co - b.co).length for a, b in zip(loop, loop[1:] + loop[:1]))
            loops_info.append((loop, cv, length))
            for a, b in zip(loop, loop[1:] + loop[:1]):
                try:
                    bm.faces.new((a, b, cv))
                except ValueError:
                    pass
        if [e for e in bm.edges if e.is_boundary]:
            print("OPEN BOUNDARY REMAINS:", name)
        for _ in range(2):
            bmesh.ops.smooth_vert(
                bm,
                verts=[v for v in centroids if v.is_valid],
                factor=0.5,
                use_axis_x=True,
                use_axis_y=True,
                use_axis_z=True,
            )
        if name in PLANE_CUTS:
            # crease the planar rim and cap so subsurf keeps the straight cut
            # instead of retracting it over the coarse boundary triangles
            ax, cut = PLANE_CUTS[name]
            crease = bm.edges.layers.float.new("crease_edge")
            # a partial crease on the shin lets subsurf chamfer the bottom
            # edge into the ball (soft hem); the arm socket stays sharp
            cval = 0.6 if name == "shin_L" else 1.0
            for e in bm.edges:
                if all(abs(v.co[ax] - cut) < 1e-4 for v in e.verts):
                    e[crease] = cval
        if name == "mannequin_head" and loops_info:
            # Taubin smoothing (shrink-free) softens the face into a mannequin
            # look; the neck rim (longest loop) stays pinned in place
            neck_loop, neck_cv, _ = max(loops_info, key=lambda t: t[2])
            pinned = {v.index for v in neck_loop if v.is_valid} | {neck_cv.index}
            bm.verts.index_update()
            bm.verts.ensure_lookup_table()
            coords = np.array([v.co[:] for v in bm.verts])
            nbrs = [[] for _ in range(len(bm.verts))]
            for e in bm.edges:
                a, b = e.verts[0].index, e.verts[1].index
                nbrs[a].append(b)
                nbrs[b].append(a)
            pin = np.zeros(len(coords), bool)
            pin[list(pinned)] = True
            for _ in range(HEAD_SMOOTH_ITERS):
                for factor in (0.5, -0.53):
                    lap = np.array([coords[ns].mean(axis=0) if ns else coords[i] for i, ns in enumerate(nbrs)])
                    moved = coords + factor * (lap - coords)
                    moved[pin] = coords[pin]
                    coords = moved
            if EGG_BLEND > 0.0:
                # blend the skull toward its bounding ellipsoid; the ramp keeps
                # the neck stub and jaw base untouched
                y = coords[:, 1]
                t = EGG_BLEND * np.clip((y - 0.20) / 0.06, 0.0, 1.0)
                t[pin] = 0.0
                dome = coords[y > 0.20]
                center = (dome.min(0) + dome.max(0)) / 2
                radii = (dome.max(0) - dome.min(0)) / 2
                q = (coords - center) / radii
                norm = np.linalg.norm(q, axis=1, keepdims=True)
                proj = center + radii * q / np.maximum(norm, 1e-9)
                coords = coords * (1 - t[:, None]) + proj * t[:, None]
            for i, v in enumerate(bm.verts):
                v.co = coords[i]
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    if name == "mannequin_head":
        # voxel remesh seals the eye/mouth slits into one closed surface,
        # completing the featureless-but-suggested mannequin face
        rem = obj.modifiers.new("rem", "REMESH")
        rem.mode = "VOXEL"
        rem.voxel_size = 0.004
        smo = obj.modifiers.new("smo", "SMOOTH")
        smo.factor = 0.55
        smo.iterations = 56
    else:
        sub = obj.modifiers.new("sub", "SUBSURF")
        sub.levels = sub.render_levels = 1
    for key in FINGER_CARVES.get(name, ()):
        cut = obj.modifiers.new(f"cut_{key}", "BOOLEAN")
        cut.operation = "DIFFERENCE"
        cut.object = cutter_objects[key]
    if name == "palm_L":
        # soften the edgy carve rims, then carve again so the smoothing cannot
        # push material back into the clearance gaps
        smo = obj.modifiers.new("smo", "SMOOTH")
        smo.factor = 0.5
        smo.iterations = 10
        for key in FINGER_CARVES[name]:
            cut = obj.modifiers.new(f"recut_{key}", "BOOLEAN")
            cut.operation = "DIFFERENCE"
            cut.object = recut_objects[key]
    dec = obj.modifiers.new("dec", "DECIMATE")
    dec.ratio = 1.0
    dec.use_collapse_triangulate = True
    obj.modifiers.new("tri", "TRIANGULATE")
    depsgraph_objects[name] = obj


def eval_arrays(obj):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    me.calc_loop_triangles()
    n = len(me.vertices)
    verts = np.empty(n * 3)
    me.vertices.foreach_get("co", verts)
    tris = np.empty(len(me.loop_triangles) * 3, np.int64)
    me.loop_triangles.foreach_get("vertices", tris)
    out = verts.reshape(-1, 3).copy(), tris.reshape(-1, 3).copy()
    ev.to_mesh_clear()
    return out


# baseline subdivided counts
base_counts = {}
for name, obj in depsgraph_objects.items():
    v, _ = eval_arrays(obj)
    base_counts[name] = len(v)
print("BASE_TOTAL(one-sided):", sum(base_counts.values()))

# multiplicity: left parts count twice (mirrored to right)
mult = {n: (2 if n.endswith("_L") else 1) for n in base_counts}


def allocate(budget):
    keys = list(base_counts)
    t = {}
    head_target = min(budget * HEAD_SHARE_CAP, base_counts["mannequin_head"])
    rem_budget = budget - head_target
    wother = np.array([base_counts[k] ** ALLOC_POWER * mult[k] for k in keys if k != "mannequin_head"])
    othersum = wother.sum()
    for k in keys:
        if k == "mannequin_head":
            t[k] = int(head_target)
        else:
            share = base_counts[k] ** ALLOC_POWER * mult[k] / othersum
            t[k] = max(12, round(rem_budget * share / mult[k]))
    return t


restored_counts = {}
for lod in ARMOR_BUDGET:
    d = np.load(f"{ORIG}/lod{lod}.npz", allow_pickle=False)
    restored_counts[lod] = sum(
        int(c) for n, c in zip(d["link_names"].tolist(), d["link_vertex_counts"]) if n.split("__")[1] in RESTORE_OLD
    )

lod_arrays = {}
for lod, budget in ARMOR_BUDGET.items():
    targets = allocate(budget - restored_counts[lod])
    arrays = {}
    for name, obj in depsgraph_objects.items():
        ratio = min(1.0, targets[name] / base_counts[name])
        obj.modifiers["dec"].ratio = ratio
        v, f = eval_arrays(obj)
        arrays[name] = (v, f)
    lod_arrays[lod] = arrays
    total = sum(len(v) * mult[n] for n, (v, f) in arrays.items())
    print(f"LOD{lod}: armor total {total} (budget {budget})")

# knuckle balls bypass the subsurf/decimate pipeline: they stay exact analytic
# spheres, generated per LOD at a matching resolution
KNUCKLE_LOD_RES = {0: (16, 8), 1: (12, 6), 2: (8, 4)}


def triangulated(faces):
    tris = []
    for face in faces:
        tris.append(list(face[:3]))
        if len(face) == 4:
            tris.append([face[0], face[2], face[3]])
    return np.array(tris, np.int64)


for lod, (segments, rings) in KNUCKLE_LOD_RES.items():
    for name, (center, radius) in FINGER_KNUCKLES.items():
        verts, faces = sphere_mesh(center, radius, segments=segments, rings=rings)
        lod_arrays[lod][name] = (verts, triangulated(faces))


# ------------------------------------------------------------- NPZ assembly
def mirror(varr, farr):
    v = varr * np.array([-1.0, 1.0, 1.0])
    f = farr[:, ::-1].copy()
    return v, f


for lod in (0, 1, 2):
    old = np.load(f"{ORIG}/lod{lod}.npz", allow_pickle=False)
    parents = old["parents"]
    offsets = old["local_offsets"].astype(np.float64)
    joints = np.zeros_like(offsets)
    joints[0] = offsets[0]
    for j in range(1, len(parents)):
        joints[j] = joints[parents[j]] + offsets[j]

    names = old["link_names"].tolist()
    owners = old["link_joint_indices"]
    overts, ofaces = old["vertices"], old["faces"]
    ostarts, ocounts = old["link_vertex_starts"], old["link_vertex_counts"]
    ofstarts, ofcounts = old["link_face_starts"], old["link_face_counts"]

    new_names, new_owners, vparts, fparts = [], [], [], []
    vstart = 0
    starts, counts, fstarts, fcounts = [], [], [], []
    for link, lname in enumerate(names):
        if "_fingertip_" in lname:
            continue  # fingertips are absorbed into the distal capsules
        part = lname.split("__")[1]
        if "_bearing_" in part and part.split("_")[0] in FINGERS:
            continue  # the old finger bearings are replaced by the knuckle balls
        owner = int(owners[link])
        joint_name = lname.split("__")[0]
        middle = lname.split("__")[1]
        if "__joint_" in lname or lname.split("__")[1] in RESTORE_OLD:
            vs, vc = int(ostarts[link]), int(ocounts[link])
            fs, fc = int(ofstarts[link]), int(ofcounts[link])
            v = overts[vs : vs + vc].astype(np.float64)
            f = ofaces[fs : fs + fc] - vs
            new_names.append(lname)
        else:
            key = "mannequin_head" if middle == "featureless_mannequin_head" else middle
            suffix = lname.split("__armor_")[1]
            if key.endswith("_R"):
                lv, lf = lod_arrays[lod][key[:-2] + "_L"]
                vw, f = mirror(lv, lf)
            else:
                vw, f = lod_arrays[lod][key]
            v = vw - joints[owner]  # world -> joint-local
            new_names.append(f"{joint_name}__{key}__armor_{suffix}")
        new_owners.append(owner)
        starts.append(vstart)
        counts.append(len(v))
        fstarts.append(sum(fcounts))
        fcounts.append(len(f))
        vparts.append(v)
        fparts.append(f + vstart)
        vstart += len(v)

    # knuckle balls: new joint links owned by the parent-side segment's joint
    jnames = old["joint_names"].tolist()
    knuckle_id = 200
    for finger in FINGERS:
        joint = finger.capitalize()
        for seg in (1, 2, 3):
            lv, lf = lod_arrays[lod][f"{finger}_knuckle{seg}_L"]
            for side in ("L", "R"):
                vw, f = (lv, lf) if side == "L" else mirror(lv, lf)
                owner_name = f"{side}_Hand" if seg == 1 else f"{side}_{joint}{seg - 1}"
                owner = jnames.index(owner_name)
                new_names.append(
                    f"{owner_name}_J{owner}__{finger}_knuckle{seg}_{side}__joint_{knuckle_id}_{knuckle_id}"
                )
                new_owners.append(owner)
                v = vw.astype(np.float64) - joints[owner]
                starts.append(vstart)
                counts.append(len(v))
                fstarts.append(sum(fcounts))
                fcounts.append(len(f))
                vparts.append(v)
                fparts.append(f + vstart)
                vstart += len(v)
                knuckle_id += 1

    L = len(new_names)
    np.savez_compressed(
        f"{OUT}/lod{lod}.npz",
        joint_names=old["joint_names"],
        parents=old["parents"],
        local_offsets=old["local_offsets"],
        rest_local_rotations=old["rest_local_rotations"],
        vertices=np.concatenate(vparts).astype(np.float32),
        faces=np.concatenate(fparts).astype(np.int64),
        link_joint_indices=np.array(new_owners, dtype=old["link_joint_indices"].dtype),
        link_vertex_starts=np.array(starts, np.int64),
        link_vertex_counts=np.array(counts, np.int64),
        link_face_starts=np.array(fstarts, np.int64),
        link_face_counts=np.array(fcounts, np.int64),
        link_geom_positions=np.zeros((L, 3), np.float32),
        link_geom_rotations=np.tile(np.eye(3, dtype=np.float32), (L, 1, 1)),
        link_names=np.array(new_names),
        actuated_joint_indices=old["actuated_joint_indices"],
        actuated_joint_limits=old["actuated_joint_limits"],
        actuated_joint_names=old["actuated_joint_names"],
        actuated_joint_types=old["actuated_joint_types"],
    )
    total_v = sum(counts)
    total_f = sum(fcounts)
    print(f"WROTE lod{lod}.npz verts={total_v} faces={total_f}")
