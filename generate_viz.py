"""
generate_viz.py — Build a self-contained PuzzleFusion++ assembly viewer HTML.

Takes a single inference run directory and a sample data_id, loads the
inference outputs + canonical val point clouds, and writes a single HTML
file with Three.js and OrbitControls inlined (no internet required).

Usage
-----
    python generate_viz.py \\
        --inference_dir output/denoiser/everyday_epoch2000_bs64/inference/test_run \\
        --data_id 97 \\
        --val_dir data/pc_data/everyday/val/ \\
        --output assembly_viz.html

    # Pick the sample with highest part accuracy automatically:
    python generate_viz.py \\
        --inference_dir output/denoiser/everyday_epoch2000_bs64/inference/test_run \\
        --val_dir data/pc_data/everyday/val/ \\
        --output assembly_viz.html \\
        --pick best

    # List all available data_ids in an inference dir:
    python generate_viz.py \\
        --inference_dir output/denoiser/everyday_epoch2000_bs64/inference/test_run \\
        --list
"""

import argparse
import glob
import json
import os
import sys
from typing import List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
PROJDIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VAL_DIR = os.path.join(PROJDIR, "data/pc_data/everyday/val/")
DEFAULT_THREEJS_DIR = os.path.join(PROJDIR, "renderer/threejs")
THREEJS_CDN = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"
ORBIT_CDN = "https://unpkg.com/three@0.134.0/examples/js/controls/OrbitControls.js"

PTS_PER_FRAG = 60  # downsampled points per fragment for web rendering

COLORS = [
    "#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA",
    "#00ACC1", "#E040FB", "#F4511E", "#7CB342", "#3949AB",
    "#00897B", "#FFB300", "#D81B60", "#6D4C41", "#546E7A",
    "#C0CA33", "#039BE5", "#F06292", "#80CBC4", "#FFF176",
]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def fps_downsample(pts: np.ndarray, k: int) -> np.ndarray:
    """Farthest-point sampling for roughly uniform coverage."""
    if pts.shape[0] <= k:
        return pts
    selected = [0]
    dists = np.full(pts.shape[0], np.inf)
    for _ in range(k - 1):
        d = np.linalg.norm(pts - pts[selected[-1]], axis=1)
        dists = np.minimum(dists, d)
        selected.append(int(np.argmax(dists)))
    return pts[selected]


def quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    """Angular error in degrees between two scalar-first [w,x,y,z] quaternions."""
    r1 = R.from_quat(q1[[1, 2, 3, 0]])  # convert to scipy [x,y,z,w]
    r2 = R.from_quat(q2[[1, 2, 3, 0]])
    deg = float((r1 * r2.inv()).magnitude() * 180.0 / np.pi)
    return min(deg, 360.0 - deg)


def make_tf(pos: np.ndarray, quat_wxyz: np.ndarray) -> List[float]:
    """Flatten [tx,ty,tz,qw,qx,qy,qz] to a plain Python list (7 floats)."""
    return [round(float(v), 6) for v in [*pos, *quat_wxyz]]


Q_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Inference data loading
# ---------------------------------------------------------------------------

def list_samples(inference_dir: str) -> List[Tuple[int, float]]:
    """Return [(data_id, part_acc), ...] sorted by data_id for all saved samples."""
    samples = []
    for entry in os.scandir(inference_dir):
        if not entry.is_dir():
            continue
        try:
            data_id = int(entry.name)
        except ValueError:
            continue
        pfiles = glob.glob(os.path.join(entry.path, "predict_*.npy"))
        if not pfiles:
            continue
        acc = float(os.path.basename(pfiles[0]).replace("predict_", "").replace(".npy", ""))
        samples.append((data_id, acc))
    return sorted(samples, key=lambda x: x[0])


def load_inference(inference_dir: str, data_id: int):
    """
    Returns:
        gt_saved  : (n_parts, 7)  [tx,ty,tz,qw,qx,qy,qz]
        pred_all  : (T, n_parts, 7)
        mesh_path : str  e.g. "everyday/Bottle/.../fractured_12"
        part_acc  : float
    """
    sample_dir = os.path.join(inference_dir, str(data_id))
    if not os.path.isdir(sample_dir):
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")

    gt_saved = np.load(os.path.join(sample_dir, "gt.npy"))          # (n, 7)

    pfiles = glob.glob(os.path.join(sample_dir, "predict_*.npy"))
    if not pfiles:
        raise FileNotFoundError(f"No predict_*.npy in {sample_dir}")
    pfile = pfiles[0]
    part_acc = float(os.path.basename(pfile).replace("predict_", "").replace(".npy", ""))
    pred_all = np.load(pfile)                                         # (T, n, 7)

    mesh_path_file = os.path.join(sample_dir, "mesh_file_path.txt")
    mesh_path = ""
    if os.path.isfile(mesh_path_file):
        with open(mesh_path_file) as f:
            mesh_path = f.read().strip()

    return gt_saved, pred_all, mesh_path, part_acc


def load_val_pcs(val_dir: str, data_id: int, n_parts: int):
    """
    Returns:
        pcs_gt   : (n_parts, 1000, 3)  canonical assembled point clouds
        ref_part : (n_parts,)  boolean, True for anchor fragment(s)
    """
    for fname in sorted(os.listdir(val_dir)):
        if not fname.endswith(".npz"):
            continue
        d = np.load(os.path.join(val_dir, fname), allow_pickle=True)
        if int(d["data_id"].item()) == data_id:
            pcs_gt = d["part_pcs_gt"][:n_parts].astype(np.float64)   # (n, 1000, 3)
            ref_part = d["ref_part"][:n_parts].astype(bool)
            print(f"  Loaded val pcs from {fname}")
            return pcs_gt, ref_part
    raise LookupError(f"No val sample with data_id={data_id} found in {val_dir}")


# ---------------------------------------------------------------------------
# Transform computation
# ---------------------------------------------------------------------------

def compute_pred_tf(
    gt_saved: np.ndarray,
    pred_poses: np.ndarray,          # (n_parts, 7)
    centroids_gt: np.ndarray,
    anchor_idx: int,
    anchor_centroid: np.ndarray,
) -> List[List[float]]:
    """
    Compute visual transform for each fragment in the predicted assembly.

    The predicted poses live in the denoiser's coordinate frame, not in the
    canonical assembled frame.  We express each fragment as:
      - rotation: the relative rotation error  R_pred * R_gt^{-1}
      - position: canonical centroid + a scaled displacement of the translation error

    This makes the 'Predicted' mode a visual diff vs Ground Truth: perfectly
    predicted fragments appear exactly where they should; errors are visible as
    offsets and rotations away from their GT positions.
    """
    n_parts = gt_saved.shape[0]
    result = []
    for i in range(n_parts):
        if i == anchor_idx:
            result.append(make_tf(centroids_gt[i], Q_IDENTITY))
            continue

        q_gt   = gt_saved[i][3:]       # [w,x,y,z]
        q_pred = pred_poses[i][3:]
        t_gt   = gt_saved[i][:3]
        t_pred = pred_poses[i][:3]

        R_gt   = R.from_quat(q_gt[[1, 2, 3, 0]])
        R_pred = R.from_quat(q_pred[[1, 2, 3, 0]])
        R_rel  = R_pred * R_gt.inv()
        q_rel  = R_rel.as_quat()[[3, 0, 1, 2]]  # → scalar-first

        t_disp      = t_pred - t_gt
        t_norm      = np.linalg.norm(t_gt) or 1.0
        canon_norm  = np.linalg.norm(centroids_gt[i] - anchor_centroid) or 1e-6
        scale       = canon_norm / (t_norm + 1e-6) * 0.3
        pos         = centroids_gt[i] + t_disp * scale

        result.append(make_tf(pos, q_rel))
    return result


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------

def build_payload(
    inference_dir: str,
    data_id: int,
    val_dir: str,
    pts_per_frag: int = PTS_PER_FRAG,
) -> dict:
    print(f"\n── Loading inference for data_id={data_id} ─────────────────────")
    gt_saved, pred_all, mesh_path, part_acc = load_inference(inference_dir, data_id)
    T, n_parts = pred_all.shape[0], pred_all.shape[1]
    print(f"  {n_parts} fragments, {T} denoising steps, part_acc={part_acc:.4f}")
    print(f"  mesh: {mesh_path}")

    print(f"\n── Loading canonical val point clouds ──────────────────────────")
    pcs_gt, ref_part = load_val_pcs(val_dir, data_id, n_parts)

    # Canonical centroids and centered geometry
    centroids_gt  = np.array([pcs_gt[i].mean(axis=0) for i in range(n_parts)])
    pcs_centered  = [pcs_gt[i] - centroids_gt[i] for i in range(n_parts)]
    base_points   = [fps_downsample(p, pts_per_frag) for p in pcs_centered]

    anchor_idx      = int(np.where(ref_part)[0][0])
    anchor_centroid = centroids_gt[anchor_idx]

    # Scattered grid layout
    scene_scale   = float(np.max(np.abs(pcs_gt)))
    grid_spacing  = scene_scale * 2.0
    cols          = int(np.ceil(np.sqrt(n_parts)))
    scattered_tf  = []
    for i in range(n_parts):
        row, col = divmod(i, cols)
        offset = np.array([col * grid_spacing, row * grid_spacing, 0.0])
        scattered_tf.append(make_tf(offset, Q_IDENTITY))

    # Ground-truth transforms: centered geometry at canonical centroid, identity rot
    gt_tf = [make_tf(centroids_gt[i], Q_IDENTITY) for i in range(n_parts)]

    # Predicted transforms (final denoising step)
    pred_tf = compute_pred_tf(gt_saved, pred_all[-1], centroids_gt, anchor_idx, anchor_centroid)

    # Per-fragment errors
    rot_errors   = [quat_angle_deg(pred_all[-1][i][3:], gt_saved[i][3:]) for i in range(n_parts)]
    trans_errors = [float(np.linalg.norm(pred_all[-1][i][:3] - gt_saved[i][:3])) for i in range(n_parts)]

    # Mean rotation error across denoising trajectory
    traj_mean_rot_err = [
        float(np.mean([quat_angle_deg(pred_all[t][i][3:], gt_saved[i][3:]) for i in range(n_parts)]))
        for t in range(T)
    ]

    # Trajectory keyframe transforms (for optional animation)
    stride     = max(1, T // 12)
    traj_steps = list(range(0, T, stride))
    if T - 1 not in traj_steps:
        traj_steps.append(T - 1)
    traj_tf = [
        compute_pred_tf(gt_saved, pred_all[t], centroids_gt, anchor_idx, anchor_centroid)
        for t in traj_steps
    ]

    # Determine a human-readable label from mesh_path, e.g. "everyday/Bottle/.../fractured_5"
    parts_of_path = mesh_path.replace("\\", "/").split("/")
    category  = parts_of_path[1] if len(parts_of_path) > 1 else "Unknown"
    label     = f"{category} · {n_parts} fragments"

    print(f"\n── Metrics ─────────────────────────────────────────────────────")
    print(f"  rot_error  min/mean/max: "
          f"{min(rot_errors):.1f}° / {np.mean(rot_errors):.1f}° / {max(rot_errors):.1f}°")
    print(f"  trans_err  min/mean/max: "
          f"{min(trans_errors):.4f} / {np.mean(trans_errors):.4f} / {max(trans_errors):.4f}")
    print(f"  traj mean rot: {traj_mean_rot_err[0]:.1f}° → {traj_mean_rot_err[-1]:.1f}°")

    return {
        "n_parts":          n_parts,
        "anchor_idx":       anchor_idx,
        "data_id":          data_id,
        "mesh_path":        mesh_path,
        "label":            label,
        "part_acc":         float(part_acc),
        "colors":           COLORS[:n_parts],
        "rot_errors":       rot_errors,
        "trans_errors":     trans_errors,
        "traj_mean_rot_err": traj_mean_rot_err,
        "traj_steps":       traj_steps,
        "T":                int(T),
        "base_points":      [p.tolist() for p in base_points],
        "scattered_tf":     scattered_tf,
        "gt_tf":            gt_tf,
        "pred_tf":          pred_tf,
        "traj_tf":          traj_tf,
    }


# ---------------------------------------------------------------------------
# Three.js asset loading (local first, CDN fallback)
# ---------------------------------------------------------------------------

def load_asset(local_path: str, cdn_url: str, name: str) -> str:
    if os.path.isfile(local_path):
        with open(local_path) as f:
            return f.read()
    print(f"  {name} not found locally at {local_path}, downloading from CDN...")
    try:
        import urllib.request
        with urllib.request.urlopen(cdn_url, timeout=30) as resp:
            content = resp.read().decode("utf-8")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "w") as f:
            f.write(content)
        print(f"  Saved to {local_path}")
        return content
    except Exception as e:
        sys.exit(f"ERROR: Could not load {name}: {e}")


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def generate_html(payload: dict, threejs_dir: str) -> str:
    three_js = load_asset(
        os.path.join(threejs_dir, "three.min.js"), THREEJS_CDN, "Three.js"
    )
    orbit_js = load_asset(
        os.path.join(threejs_dir, "OrbitControls.js"), ORBIT_CDN, "OrbitControls"
    )

    data_json = json.dumps(payload, separators=(",", ":"))

    n_ok     = sum(1 for e in payload["rot_errors"] if e < 30 or payload["rot_errors"].index(e) == payload["anchor_idx"])
    label    = payload["label"]
    part_acc = payload["part_acc"] * 100

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>PuzzleFusion++ — {label}</title>
  <style>
    *{{margin:0;padding:0;box-sizing:border-box}}
    body{{background:#050712;color:#e0e0e0;font-family:system-ui,sans-serif;overflow:hidden}}
    #c{{position:absolute;inset:0}}
    #ui{{
      position:absolute;top:16px;right:16px;width:275px;padding:14px;
      border-radius:8px;background:rgba(5,7,18,0.93);border:1px solid #1f2937;
      box-shadow:0 18px 45px rgba(0,0,0,.6);display:flex;flex-direction:column;gap:10px;z-index:10
    }}
    h1{{font-size:13px;letter-spacing:2px;text-transform:uppercase;color:#93c5fd;
        border-bottom:1px solid #111827;padding-bottom:6px}}
    .metric{{font-size:11px;color:#9ca3af}}
    .metric span{{color:#e5e7eb;font-weight:600}}
    #mode-buttons{{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}}
    .mode-btn{{
      padding:6px 4px;border-radius:4px;border:1px solid #1f2937;
      background:#020617;color:#9ca3af;font-size:10px;text-transform:uppercase;
      letter-spacing:1px;cursor:pointer;transition:all .15s ease-out
    }}
    .mode-btn:hover{{border-color:#2563eb;color:#bfdbfe}}
    .mode-btn.active{{border-color:#3b82f6;background:#0b1120;color:#dbeafe}}
    #frag-table{{font-size:10px;color:#9ca3af;border-collapse:collapse;width:100%}}
    #frag-table td{{padding:2px 4px}}
    .ok{{color:#4ade80}}.bad{{color:#f87171}}.anchor{{color:#fbbf24}}
    #traj-section{{display:flex;flex-direction:column;gap:4px}}
    #traj-bar-bg{{height:4px;background:#1e293b;border-radius:2px;overflow:hidden}}
    #traj-bar{{height:100%;width:0%;background:#3b82f6;border-radius:2px;transition:width .1s}}
    #traj-controls{{display:flex;gap:6px;align-items:center}}
    #traj-play{{
      padding:4px 10px;border-radius:4px;border:1px solid #1f2937;background:#020617;
      color:#9ca3af;font-size:10px;cursor:pointer;flex-shrink:0
    }}
    #traj-play:hover{{border-color:#2563eb;color:#bfdbfe}}
    #traj-label{{font-size:10px;color:#6b7280;flex:1}}
    #mode-label{{
      position:absolute;top:16px;left:16px;padding:4px 8px;border-radius:999px;
      border:1px solid #1f2937;background:rgba(15,23,42,.95);font-size:10px;
      letter-spacing:2px;text-transform:uppercase;color:#bfdbfe;pointer-events:none;z-index:10
    }}
    #hint{{font-size:10px;color:#6b7280}}
  </style>
</head>
<body>
  <div id="c"></div>
  <div id="mode-label">Scattered</div>
  <div id="ui">
    <h1>PuzzleFusion++</h1>
    <div class="metric">Sample: <span>{label}</span></div>
    <div class="metric">Part accuracy: <span>{part_acc:.1f}%</span></div>
    <div id="mode-buttons">
      <button class="mode-btn active" data-mode="scattered">Scattered</button>
      <button class="mode-btn"        data-mode="pred">Predicted</button>
      <button class="mode-btn"        data-mode="gt">Ground truth</button>
    </div>
    <table id="frag-table"><tbody id="frag-rows"></tbody></table>
    <div id="traj-section">
      <div class="metric" style="margin-bottom:2px">Denoising trajectory</div>
      <div id="traj-bar-bg"><div id="traj-bar"></div></div>
      <div id="traj-controls">
        <button id="traj-play">▶ Play</button>
        <span id="traj-label">Step 0 / 0</span>
      </div>
    </div>
    <div id="hint">Drag to orbit · scroll to zoom</div>
  </div>

  <script>{three_js}</script>
  <script>{orbit_js}</script>
  <script>
    const DATA = {data_json};

    const MODE_SCAT='scattered', MODE_PRED='pred', MODE_GT='gt', MODE_TRAJ='traj';
    let scene, camera, renderer, controls;
    let mode = MODE_SCAT;
    let trajFrame = 0, trajPlaying = false, trajTimer = null;
    const fragmentPoints = [];

    // ── Build fragment status table ──────────────────────────────────────
    (function(){{
      const tbody = document.getElementById('frag-rows');
      DATA.rot_errors.forEach((err, i) => {{
        const isAnchor = i === DATA.anchor_idx;
        const isOk     = err < 30;
        const cls    = isAnchor ? 'anchor' : (isOk ? 'ok' : 'bad');
        const status = isAnchor ? '⚓' : (isOk ? '✓' : '✗');
        const tr = document.createElement('tr');
        tr.innerHTML =
          '<td><span style="display:inline-block;width:10px;height:10px;border-radius:2px;'
          + 'vertical-align:middle;background:' + DATA.colors[i] + ';margin-right:4px"></span>F' + i + '</td>'
          + '<td class="' + cls + '">' + status + '</td>'
          + '<td>' + err.toFixed(1) + '°</td>';
        tbody.appendChild(tr);
      }});
    }})();

    // ── Transform helpers ────────────────────────────────────────────────
    function tfForMode() {{
      if (mode === MODE_SCAT) return DATA.scattered_tf;
      if (mode === MODE_PRED) return DATA.pred_tf;
      if (mode === MODE_GT)   return DATA.gt_tf;
      return DATA.traj_tf[trajFrame];
    }}

    function applyTransforms(tfs) {{
      for (let i = 0; i < fragmentPoints.length; i++) {{
        const tf = tfs[i];
        const [tx,ty,tz,qw,qx,qy,qz] = tf;
        fragmentPoints[i].position.set(tx, ty, tz);
        fragmentPoints[i].setRotationFromQuaternion(new THREE.Quaternion(qx,qy,qz,qw));
      }}
    }}

    function updateModeLabel() {{
      const labels = {{scattered:'Scattered',pred:'Predicted',gt:'Ground truth',traj:'Trajectory'}};
      document.getElementById('mode-label').textContent = labels[mode] || mode;
    }}

    // ── Scene setup ──────────────────────────────────────────────────────
    function buildScene() {{
      const container = document.getElementById('c');
      const w = window.innerWidth, h = window.innerHeight;

      renderer = new THREE.WebGLRenderer({{antialias:true}});
      renderer.setPixelRatio(window.devicePixelRatio || 1);
      renderer.setSize(w, h);
      renderer.setClearColor(0x020617, 1);
      container.appendChild(renderer.domElement);

      scene  = new THREE.Scene();
      camera = new THREE.PerspectiveCamera(45, w/h, 0.001, 20);
      camera.position.set(0.8, 0.6, 1.5);

      controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.1;

      scene.add(new THREE.HemisphereLight(0xffffff, 0x111111, 1.0));
      const grid = new THREE.GridHelper(4, 16, 0x1e293b, 0x111827);
      grid.position.y = -0.6;
      scene.add(grid);
      scene.add(Object.assign(new THREE.AxesHelper(0.3), {{position: new THREE.Vector3(0,-0.6,0)}}));

      for (let i = 0; i < DATA.n_parts; i++) {{
        const pts  = DATA.base_points[i];
        const geom = new THREE.BufferGeometry();
        const pos  = new Float32Array(pts.length * 3);
        for (let j = 0; j < pts.length; j++) {{
          pos[3*j]=pts[j][0]; pos[3*j+1]=pts[j][1]; pos[3*j+2]=pts[j][2];
        }}
        geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        const mat   = new THREE.PointsMaterial({{size:0.012, color:new THREE.Color(DATA.colors[i])}});
        const cloud = new THREE.Points(geom, mat);
        scene.add(cloud);
        fragmentPoints.push(cloud);
      }}

      applyTransforms(tfForMode());
      updateModeLabel();

      window.addEventListener('resize', () => {{
        renderer.setSize(window.innerWidth, window.innerHeight);
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
      }});

      animate();
    }}

    function animate() {{
      requestAnimationFrame(animate);
      if (controls) controls.update();
      if (renderer && scene && camera) renderer.render(scene, camera);
    }}

    // ── Mode switching ───────────────────────────────────────────────────
    function setMode(next) {{
      stopTraj();
      mode = next;
      document.querySelectorAll('.mode-btn').forEach(b =>
        b.classList.toggle('active', b.dataset.mode === mode));
      applyTransforms(tfForMode());
      updateModeLabel();
    }}

    document.querySelectorAll('.mode-btn').forEach(b =>
      b.addEventListener('click', () => setMode(b.dataset.mode)));

    // ── Trajectory playback ──────────────────────────────────────────────
    function setTrajFrame(f) {{
      trajFrame = Math.max(0, Math.min(f, DATA.traj_tf.length - 1));
      const pct  = DATA.traj_steps[trajFrame] / (DATA.T - 1) * 100;
      const err  = DATA.traj_mean_rot_err[DATA.traj_steps[trajFrame]];
      document.getElementById('traj-bar').style.width = pct.toFixed(1) + '%';
      document.getElementById('traj-label').textContent =
        'Step ' + DATA.traj_steps[trajFrame] + ' / ' + (DATA.T-1)
        + ' · ' + err.toFixed(1) + '°';
      if (mode === MODE_TRAJ) applyTransforms(DATA.traj_tf[trajFrame]);
    }}

    function startTraj() {{
      trajPlaying = true;
      document.getElementById('traj-play').textContent = '⏹ Stop';
      mode = MODE_TRAJ;
      updateModeLabel();
      document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
      setTrajFrame(0);
      trajTimer = setInterval(() => {{
        if (trajFrame >= DATA.traj_tf.length - 1) {{
          stopTraj(); return;
        }}
        setTrajFrame(trajFrame + 1);
      }}, 150);
    }}

    function stopTraj() {{
      if (trajTimer) {{ clearInterval(trajTimer); trajTimer = null; }}
      trajPlaying = false;
      document.getElementById('traj-play').textContent = '▶ Play';
    }}

    document.getElementById('traj-play').addEventListener('click', () => {{
      if (trajPlaying) stopTraj();
      else startTraj();
    }});

    setTrajFrame(0);  // initialise label
    buildScene();
  </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a self-contained PuzzleFusion++ assembly viewer HTML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--inference_dir", "-i", required=False,
        help="Path to the inference output directory (contains numbered sub-dirs).",
    )
    parser.add_argument(
        "--data_id", "-d", type=int, default=None,
        help="Which sample to visualise (integer data_id).",
    )
    parser.add_argument(
        "--pick", choices=["best", "worst", "first", "last"], default=None,
        help="Auto-pick a sample by part accuracy instead of --data_id.",
    )
    parser.add_argument(
        "--val_dir", "-v", default=DEFAULT_VAL_DIR,
        help=f"Path to val point-cloud .npz directory (default: {DEFAULT_VAL_DIR}).",
    )
    parser.add_argument(
        "--output", "-o", default=os.path.join(PROJDIR, "assembly_viz.html"),
        help="Output HTML file path.",
    )
    parser.add_argument(
        "--pts_per_frag", type=int, default=PTS_PER_FRAG,
        help=f"Points per fragment to embed (default {PTS_PER_FRAG}).",
    )
    parser.add_argument(
        "--threejs_dir", default=DEFAULT_THREEJS_DIR,
        help="Directory containing three.min.js and OrbitControls.js.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available data_ids and part accuracies, then exit.",
    )
    args = parser.parse_args()

    if not args.inference_dir:
        parser.error("--inference_dir is required")

    if not os.path.isdir(args.inference_dir):
        sys.exit(f"ERROR: inference_dir not found: {args.inference_dir}")

    samples = list_samples(args.inference_dir)
    if not samples:
        sys.exit(f"ERROR: No valid sample directories found in {args.inference_dir}")

    if args.list:
        print(f"{'data_id':>8}  {'part_acc':>10}")
        print("-" * 22)
        for did, acc in samples:
            print(f"{did:>8}  {acc*100:>9.2f}%")
        print(f"\nTotal: {len(samples)} samples")
        return

    # Resolve which data_id to use
    if args.data_id is not None:
        data_id = args.data_id
    elif args.pick:
        if args.pick == "best":
            data_id = max(samples, key=lambda x: x[1])[0]
        elif args.pick == "worst":
            data_id = min(samples, key=lambda x: x[1])[0]
        elif args.pick == "first":
            data_id = samples[0][0]
        else:  # last
            data_id = samples[-1][0]
        print(f"Auto-picked data_id={data_id} (--pick {args.pick})")
    else:
        # Default: pick sample with best part accuracy
        data_id = max(samples, key=lambda x: x[1])[0]
        print(f"No --data_id specified; picking best: data_id={data_id}")

    payload = build_payload(
        inference_dir=args.inference_dir,
        data_id=data_id,
        val_dir=args.val_dir,
        pts_per_frag=args.pts_per_frag,
    )

    print(f"\n── Generating HTML ─────────────────────────────────────────────")
    html = generate_html(payload, args.threejs_dir)

    out_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"  Written: {out_path}  ({size_kb:.0f} KB)")
    print(f"\nOpen in any modern browser (Chrome/Firefox/Edge/Safari).")
    print(f"No server needed — the file is fully self-contained.\n")


if __name__ == "__main__":
    main()
