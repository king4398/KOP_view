#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_schism_webgl_unstructured_current_lookup.py

Independent GitHub Pages sub-page for SCHISM unstructured-mesh WebGL rendering.

This version keeps the existing WebGL scalar idea, but upgrades Current from
regular-grid JSON particles to SCHISM unstructured-mesh particles:

  - Temperature/Elevation: node-based scalar Float32 BIN + WebGL triangles
  - Current: node-based u/v Float32 BIN + spatial lookup index + barycentric particles
  - Current overlay on Temperature/Elevation: white particles
  - Current standalone: jet-colored particles

Output:
  webgl_unstructured/
    index.html
    webgl_scalar.js
    mesh_meta.json
    mesh_nodes.bin
    mesh_elems.bin
    lookup_meta.json
    lookup_offsets.bin
    lookup_triangles.bin
    temp_bin/frame_0000.bin ...
    ssh_bin/frame_0000.bin ...
    current_u_bin/frame_0000.bin ...
    current_v_bin/frame_0000.bin ...

Run:
  cd /home/nyj/kocean/Validation/schism/avi/KOP_view_upload
  python3 make_schism_webgl_unstructured_current_lookup.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from netCDF4 import Dataset, num2date


# =============================================================================
# User settings
# =============================================================================

REPO_DIR = Path("/home/nyj/kocean/Validation/schism/avi/KOP_view_upload")

SCHISM_DIR = Path(
    "/home/nyj/kocean/Result/schism/hgrid_test/"
    "test8v4_Small_clim_watertype_iwind0_khoadepth"
)

OUT_DIR = REPO_DIR / "webgl_unstructured"

OUT2D_FILE = SCHISM_DIR / "out2d_1.nc"
TEMP_FILE = SCHISM_DIR / "temperature_1.nc"
HVELX_FILE = SCHISM_DIR / "horizontalVelX_1.nc"
HVELY_FILE = SCHISM_DIR / "horizontalVelY_1.nc"

START_IDX = 0
END_IDX = 24
STRIDE = 1

# Surface layer for temperature/current(time,node,layer) or similar.
K_IDX = -1

INVALID_VALUE = np.float32(-9999.0)

TEMP_VMIN = 0.0
TEMP_VMAX = 32.0

SSH_VMIN = -1.0
SSH_VMAX = 1.0

CURRENT_VMIN = 0.0
CURRENT_VMAX = 1.0

# Lookup grid for fast triangle search in browser.
# 600x600 is usually a good balance for ~580k triangles.
LOOKUP_NX = 600
LOOKUP_NY = 600

# If True, lookup generation prints progress every N triangles.
LOOKUP_PROGRESS_EVERY = 50000


# =============================================================================
# NetCDF helpers
# =============================================================================

def _find_var(ds: Dataset, names: Sequence[str], contains: Optional[Sequence[str]] = None):
    for name in names:
        if name in ds.variables:
            return ds.variables[name]

    if contains:
        lowers = [c.lower() for c in contains]
        for name, var in ds.variables.items():
            low = name.lower()
            if all(c in low for c in lowers):
                return var

    raise KeyError(f"Could not find variable. Tried names={names}, contains={contains}")


def find_node_xy(ds: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    xvar = _find_var(
        ds,
        ["SCHISM_hgrid_node_x", "node_x", "lon", "longitude", "x"],
        contains=["node", "x"],
    )
    yvar = _find_var(
        ds,
        ["SCHISM_hgrid_node_y", "node_y", "lat", "latitude", "y"],
        contains=["node", "y"],
    )

    lon = np.asarray(xvar[:], dtype=np.float64)
    lat = np.asarray(yvar[:], dtype=np.float64)

    if lon.ndim != 1 or lat.ndim != 1:
        raise ValueError(f"Expected 1D lon/lat arrays, got {lon.shape}, {lat.shape}")
    if lon.size != lat.size:
        raise ValueError(f"lon/lat length mismatch: {lon.size}, {lat.size}")

    return lon, lat


def find_face_nodes(ds: Dataset) -> np.ndarray:
    candidates = [
        "SCHISM_hgrid_face_nodes",
        "SCHISM_hgrid_face_node",
        "face_nodes",
        "element",
        "ele",
        "nv",
    ]

    for name in candidates:
        if name in ds.variables:
            arr = np.asarray(ds.variables[name][:])
            break
    else:
        arr = None
        for name, var in ds.variables.items():
            low = name.lower()
            if ("face" in low or "elem" in low or low == "nv") and len(var.dimensions) == 2:
                arr = np.asarray(var[:])
                break
        if arr is None:
            raise KeyError("Could not find SCHISM face node connectivity variable")

    if np.ma.isMaskedArray(arr):
        arr = arr.filled(0)

    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D face_nodes, got {arr.shape}")

    # Some files store as (max_nodes, n_faces), transpose to (n_faces, max_nodes).
    if arr.shape[0] in (3, 4) and arr.shape[1] > arr.shape[0]:
        arr = arr.T

    arr = arr.astype(np.int64)

    positive = arr[arr > 0]
    if positive.size == 0:
        raise ValueError("face_nodes has no positive node indices")

    if positive.min() == 1:
        print("Detected face_nodes start_index: 1")
        arr = np.where(arr > 0, arr - 1, -1)
    else:
        print("Detected face_nodes start_index: 0")
        arr = np.where(arr >= 0, arr, -1)

    return arr


def triangulate_faces(face_nodes: np.ndarray, node_count: int) -> np.ndarray:
    triangles: List[Tuple[int, int, int]] = []

    for row in face_nodes:
        nodes = [int(v) for v in row if int(v) >= 0]
        if len(nodes) < 3:
            continue

        if any(v < 0 or v >= node_count for v in nodes):
            continue

        if len(nodes) == 3:
            triangles.append((nodes[0], nodes[1], nodes[2]))
        elif len(nodes) == 4:
            triangles.append((nodes[0], nodes[1], nodes[2]))
            triangles.append((nodes[0], nodes[2], nodes[3]))
        else:
            for k in range(1, len(nodes) - 1):
                triangles.append((nodes[0], nodes[k], nodes[k + 1]))

    tri = np.asarray(triangles, dtype=np.uint32)
    if tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError(f"Invalid triangle array: {tri.shape}")

    return tri


def read_time_labels(ds: Dataset, frame_indices: Sequence[int]) -> List[str]:
    if "time" not in ds.variables:
        return [f"frame {i}" for i in frame_indices]

    tvar = ds.variables["time"]
    vals = np.asarray(tvar[:])

    labels = []
    units = getattr(tvar, "units", None)
    calendar = getattr(tvar, "calendar", "standard")

    for i in frame_indices:
        if i < 0 or i >= len(vals):
            labels.append(f"frame {i}")
            continue

        if units:
            try:
                dt = num2date(vals[i], units=units, calendar=calendar)
                labels.append(dt.strftime("%Y-%m-%d %H:%M:%S"))
                continue
            except Exception:
                pass

        labels.append(str(vals[i]))

    return labels


def infer_time_dim_index(var) -> Optional[int]:
    dims = [d.lower() for d in var.dimensions]
    for k, d in enumerate(dims):
        if "time" in d:
            return k
    return 0 if len(var.shape) >= 1 else None


def read_node_scalar(
    var,
    time_index: int,
    node_count: int,
    layer_index: Optional[int] = None,
    invalid_value: float = float(INVALID_VALUE),
) -> np.ndarray:
    """Read one time slice and return a node_count Float32 vector."""
    nd = len(var.shape)

    if nd == 1:
        arr = np.asarray(var[:])
    else:
        time_dim = infer_time_dim_index(var)
        key = [slice(None)] * nd
        if time_dim is not None:
            key[time_dim] = time_index
        arr = np.asarray(var[tuple(key)])

    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)

    arr = np.asarray(arr)

    if arr.ndim == 1:
        out = arr
    elif arr.ndim == 2:
        matches = [k for k, n in enumerate(arr.shape) if int(n) == int(node_count)]
        if not matches:
            raise ValueError(f"Cannot find node axis after time slicing: shape={arr.shape}")
        node_axis = matches[0]
        layer_axis = 1 - node_axis
        li = layer_index if layer_index is not None else K_IDX
        out = np.take(arr, li, axis=layer_axis)
    else:
        cur = arr
        while cur.ndim > 1:
            matches = [k for k, n in enumerate(cur.shape) if int(n) == int(node_count)]
            if not matches:
                raise ValueError(f"Cannot reduce variable {var.name}; current shape={cur.shape}")
            node_axis = matches[0]
            other_axes = [k for k in range(cur.ndim) if k != node_axis]
            if not other_axes:
                break
            take_axis = other_axes[-1]
            li = layer_index if layer_index is not None else K_IDX
            cur = np.take(cur, li, axis=take_axis)
        out = cur

    out = np.asarray(out, dtype=np.float32).reshape(-1)
    if out.size != node_count:
        raise ValueError(f"Scalar size mismatch for {var.name}: {out.size} != {node_count}")

    bad = ~np.isfinite(out)
    if bad.any():
        out = out.copy()
        out[bad] = invalid_value

    return out.astype(np.float32, copy=False)


def write_float32_bin(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(arr, dtype="<f4").tofile(path)


def write_uint32_bin(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(arr, dtype="<u4").tofile(path)


# =============================================================================
# Lookup index generation
# =============================================================================

def build_triangle_lookup(
    lon: np.ndarray,
    lat: np.ndarray,
    triangles: np.ndarray,
    out_dir: Path,
    nx: int = LOOKUP_NX,
    ny: int = LOOKUP_NY,
) -> dict:
    """Build regular spatial lookup grid mapping cells to candidate triangle IDs."""
    lon_min = float(np.nanmin(lon))
    lon_max = float(np.nanmax(lon))
    lat_min = float(np.nanmin(lat))
    lat_max = float(np.nanmax(lat))

    if not np.isfinite([lon_min, lon_max, lat_min, lat_max]).all():
        raise ValueError("Invalid lon/lat bounds for lookup index")
    if lon_max <= lon_min or lat_max <= lat_min:
        raise ValueError("Degenerate lookup bounds")

    cell_count = nx * ny
    buckets: List[List[int]] = [[] for _ in range(cell_count)]

    dx = (lon_max - lon_min) / nx
    dy = (lat_max - lat_min) / ny

    print(f"Building triangle lookup index: {nx} x {ny} cells")

    for ti, (a, b, c) in enumerate(triangles.astype(np.int64)):
        xs = lon[[a, b, c]]
        ys = lat[[a, b, c]]

        if not (np.isfinite(xs).all() and np.isfinite(ys).all()):
            continue

        x0 = float(xs.min())
        x1 = float(xs.max())
        y0 = float(ys.min())
        y1 = float(ys.max())

        ix0 = int(np.floor((x0 - lon_min) / dx))
        ix1 = int(np.floor((x1 - lon_min) / dx))
        iy0 = int(np.floor((y0 - lat_min) / dy))
        iy1 = int(np.floor((y1 - lat_min) / dy))

        ix0 = max(0, min(nx - 1, ix0))
        ix1 = max(0, min(nx - 1, ix1))
        iy0 = max(0, min(ny - 1, iy0))
        iy1 = max(0, min(ny - 1, iy1))

        for iy in range(iy0, iy1 + 1):
            base = iy * nx
            for ix in range(ix0, ix1 + 1):
                buckets[base + ix].append(int(ti))

        if LOOKUP_PROGRESS_EVERY and ti > 0 and ti % LOOKUP_PROGRESS_EVERY == 0:
            print(f"  lookup triangles processed: {ti:,}/{triangles.shape[0]:,}")

    offsets = np.empty(cell_count + 1, dtype=np.uint32)
    offsets[0] = 0
    total = 0
    max_bucket = 0
    non_empty = 0

    for i, bucket in enumerate(buckets):
        n = len(bucket)
        total += n
        max_bucket = max(max_bucket, n)
        if n:
            non_empty += 1
        offsets[i + 1] = total

    tri_indices = np.empty(total, dtype=np.uint32)
    pos = 0
    for bucket in buckets:
        n = len(bucket)
        if n:
            tri_indices[pos:pos + n] = np.asarray(bucket, dtype=np.uint32)
            pos += n

    write_uint32_bin(out_dir / "lookup_offsets.bin", offsets)
    write_uint32_bin(out_dir / "lookup_triangles.bin", tri_indices)

    avg_non_empty = float(total / non_empty) if non_empty else 0.0

    meta = {
        "nx": int(nx),
        "ny": int(ny),
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "cell_count": int(cell_count),
        "offset_count": int(offsets.size),
        "candidate_count": int(tri_indices.size),
        "non_empty_cells": int(non_empty),
        "max_candidates_per_cell": int(max_bucket),
        "avg_candidates_non_empty_cell": avg_non_empty,
        "offsets_bin": "lookup_offsets.bin",
        "triangles_bin": "lookup_triangles.bin",
    }

    (out_dir / "lookup_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Lookup candidate_count:", f"{tri_indices.size:,}")
    print("Lookup non_empty_cells:", f"{non_empty:,}/{cell_count:,}")
    print("Lookup max_candidates_per_cell:", max_bucket)
    print("Lookup avg_candidates_non_empty_cell:", f"{avg_non_empty:.2f}")

    return meta


# =============================================================================
# Web files
# =============================================================================

INDEX_HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SCHISM WebGL Unstructured Viewer</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

<style>
html, body, #map {
    width: 100%;
    height: 100%;
    margin: 0;
    padding: 0;
}

#map { background: #111; }

#gl-canvas,
#current-canvas {
    position: absolute;
    left: 0;
    top: 0;
    pointer-events: none;
}

#gl-canvas { z-index: 700; }
#current-canvas { z-index: 12000; }

.title-box {
    position: fixed;
    left: 14px;
    bottom: 86px;
    z-index: 50000;
    background: rgba(255,255,255,0.95);
    padding: 8px 12px;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.35);
    font-family: Arial, sans-serif;
    font-weight: bold;
}

.legend-box {
    position: fixed;
    right: 14px;
    bottom: 86px;
    z-index: 50000;
    background: rgba(255,255,255,0.95);
    padding: 10px 12px;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.35);
    font-family: Arial, sans-serif;
    min-width: 230px;
}

.control-box {
    position: fixed;
    left: 50%;
    bottom: 20px;
    transform: translateX(-50%);
    z-index: 60000;
    background: rgba(255,255,255,0.96);
    padding: 10px 14px;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.35);
    font-family: Arial, sans-serif;
    min-width: 720px;
}

.side-panel {
    position: fixed;
    right: 14px;
    top: 72px;
    z-index: 55000;
    width: 255px;
    padding: 12px 14px;
    border-radius: 12px;
    background: rgba(15, 18, 24, 0.84);
    color: white;
    font-family: Arial, sans-serif;
    font-size: 13px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.38);
}

.side-panel-title {
    font-weight: bold;
    font-size: 15px;
    margin-bottom: 9px;
}

.side-panel label {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    margin: 8px 0;
}

.side-panel select {
    width: 135px;
    max-width: 135px;
    padding: 4px 6px;
}

.side-panel input[type="checkbox"] { transform: scale(1.12); }

.status-line {
    font-size: 11px;
    opacity: 0.80;
    margin-top: 8px;
    line-height: 1.35;
}

button, select, input { font-family: Arial, sans-serif; }
</style>
</head>

<body>
<div id="map"></div>

<div class="title-box">SCHISM WebGL Viewer</div>
<div class="legend-box" id="legend-box"></div>

<div class="side-panel">
  <div class="side-panel-title">Layer Options</div>

  <label>
    <span><b>Variable</b></span>
    <select id="var-select">
      <option value="temperature">Temperature</option>
      <option value="ssh">Elevation</option>
      <option value="current">Current</option>
    </select>
  </label>

  <label>
    <span><b>Current overlay</b></span>
    <input id="current-overlay-check" type="checkbox">
  </label>

  <label>
    <span><b>Particles</b></span>
    <select id="particle-density-select">
      <option value="800">Low</option>
      <option value="1600">Mid</option>
      <option value="2800" selected>High</option>
    </select>
  </label>

  <div class="status-line" id="status-line">Loading...</div>
</div>

<div class="control-box">
  <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
    <button id="play-btn" style="padding:5px 12px; cursor:pointer;">Play</button>

    <input id="frame-slider" type="range" min="0" max="0" value="0" step="1" style="width:330px;">

    <select id="speed-select" style="padding:5px 8px;">
      <option value="1">1x</option>
      <option value="2">2x</option>
      <option value="3">3x</option>
    </select>

    <label><b>Opacity</b></label>
    <input id="opacity-slider" type="range" min="0.1" max="1.0" value="0.82" step="0.05" style="width:110px;">

    <span id="time-label" style="font-size:13px; min-width:180px; display:inline-block;"></span>
  </div>
</div>

<script src="webgl_scalar.js?v=20260518_unstructured_current_01"></script>
</body>
</html>
'''


WEBGL_SCALAR_JS = r'''/* SCHISM unstructured WebGL viewer with unstructured current particles. */

"use strict";

const CONFIG = {
    metaUrl: "mesh_meta.json",
    nodesUrl: "mesh_nodes.bin",
    elemsUrl: "mesh_elems.bin",
    lookupMetaUrl: "lookup_meta.json",
    lookupOffsetsUrl: "lookup_offsets.bin",
    lookupTrianglesUrl: "lookup_triangles.bin",
    currentOverlayColor: "rgba(255,255,255,0.96)",
    flowScale: 0.018,
    baryEps: 1.0e-9,
    lastTriangleFirst: true
};

let meta = null;
let lookupMeta = null;
let map = null;
let glCanvas = null;
let gl = null;
let currentCanvas = null;
let currentCtx = null;

let program = null;
let vao = null;
let nodeBuffer = null;
let valueBuffer = null;
let elemBuffer = null;

let uMapSizeLoc = null;
let uVminLoc = null;
let uVmaxLoc = null;
let uOpacityLoc = null;
let uCmapLoc = null;
let uInvalidLoc = null;

let meshNodes = null;       // Float32Array [lon,lat,...]
let meshElems = null;       // Uint32Array [i0,i1,i2,...]
let screenNodes = null;     // Float32Array [x,y,...] in Leaflet container CSS pixels
let lookupOffsets = null;   // Uint32Array cell_count + 1
let lookupTriangles = null; // Uint32Array flat triangle IDs

let currentVar = "temperature";
let currentFrame = 0;
let timer = null;
let speed = 1.0;

let scalarCache = new Map();
let scalarLoading = new Map();

let currentUCache = new Map();
let currentVCache = new Map();
let currentULoading = new Map();
let currentVLoading = new Map();
let currentU = null;
let currentV = null;

let particleCount = 2800;
let particles = [];
let particleAnimId = null;
let particleRunning = false;

const varSelect = document.getElementById("var-select");
const playBtn = document.getElementById("play-btn");
const frameSlider = document.getElementById("frame-slider");
const speedSelect = document.getElementById("speed-select");
const opacitySlider = document.getElementById("opacity-slider");
const particleDensitySelect = document.getElementById("particle-density-select");
const currentOverlayCheck = document.getElementById("current-overlay-check");
const timeLabel = document.getElementById("time-label");
const legendBox = document.getElementById("legend-box");
const statusLine = document.getElementById("status-line");

function setStatus(msg) { statusLine.textContent = msg; }
function pad4(i) { return String(i).padStart(4, "0"); }
function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }

function currentOverlayEnabled() {
    return currentOverlayCheck.checked && (currentVar === "temperature" || currentVar === "ssh");
}

function shouldDrawCurrentParticles() {
    return currentVar === "current" || currentOverlayEnabled();
}

function updateCurrentOverlayAvailability() {
    if (currentVar === "current") {
        currentOverlayCheck.checked = false;
        currentOverlayCheck.disabled = true;
        currentOverlayCheck.title = "Current variable already shows current particles.";
    } else {
        currentOverlayCheck.disabled = false;
        currentOverlayCheck.title = "";
    }
}

function scalarFrameUrl(variable, frameIndex) {
    if (variable === "temperature") return `temp_bin/frame_${pad4(frameIndex)}.bin`;
    if (variable === "ssh") return `ssh_bin/frame_${pad4(frameIndex)}.bin`;
    return null;
}

function currentUFrameUrl(frameIndex) { return `current_u_bin/frame_${pad4(frameIndex)}.bin`; }
function currentVFrameUrl(frameIndex) { return `current_v_bin/frame_${pad4(frameIndex)}.bin`; }

function variableMeta(variable) {
    if (variable === "temperature") return meta.variables.temperature;
    if (variable === "ssh") return meta.variables.ssh;
    return null;
}

function updateLegend() {
    if (!meta) return;

    if (currentVar === "temperature") {
        const v = meta.variables.temperature;
        legendBox.innerHTML = `
            <div style="font-weight:bold; margin-bottom:6px;">Surface Temperature [degC]</div>
            <div style="width:220px; height:16px; background: linear-gradient(to right, #000080, #0000ff, #00ffff, #ffff00, #ff0000, #800000); border:1px solid #666;"></div>
            <div style="display:flex; justify-content:space-between; font-size:12px; margin-top:4px;">
                <span>${v.vmin}</span><span>${((v.vmin + v.vmax) / 2).toFixed(1)}</span><span>${v.vmax}</span>
            </div>`;
    } else if (currentVar === "ssh") {
        const v = meta.variables.ssh;
        legendBox.innerHTML = `
            <div style="font-weight:bold; margin-bottom:6px;">Elevation [m]</div>
            <div style="width:220px; height:16px; background: linear-gradient(to right, #08306b, #6baed6, #f7f7f7, #fb6a4a, #67000d); border:1px solid #666;"></div>
            <div style="display:flex; justify-content:space-between; font-size:12px; margin-top:4px;">
                <span>${v.vmin}</span><span>${((v.vmin + v.vmax) / 2).toFixed(2)}</span><span>${v.vmax}</span>
            </div>`;
    } else {
        const v = meta.variables.current;
        legendBox.innerHTML = `
            <div style="font-weight:bold; margin-bottom:6px;">Current Speed [m/s]</div>
            <div style="width:220px; height:16px; background: linear-gradient(to right, #000080, #0000ff, #00ffff, #ffff00, #ff0000, #800000); border:1px solid #666;"></div>
            <div style="display:flex; justify-content:space-between; font-size:12px; margin-top:4px;">
                <span>${v.vmin}</span><span>${((v.vmin + v.vmax) / 2).toFixed(2)}</span><span>${v.vmax}</span>
            </div>`;
    }
}

async function fetchArrayBuffer(url) {
    const r = await fetch(url, { cache: "force-cache" });
    if (!r.ok) throw new Error(`${url}: ${r.status} ${r.statusText}`);
    return await r.arrayBuffer();
}

async function fetchFloat32(url, expectedLength = null) {
    const buf = await fetchArrayBuffer(url);
    const arr = new Float32Array(buf);
    if (expectedLength !== null && arr.length !== expectedLength) {
        throw new Error(`${url}: Float32 length ${arr.length} != expected ${expectedLength}`);
    }
    return arr;
}

async function fetchUint32(url, expectedLength = null) {
    const buf = await fetchArrayBuffer(url);
    const arr = new Uint32Array(buf);
    if (expectedLength !== null && arr.length !== expectedLength) {
        throw new Error(`${url}: Uint32 length ${arr.length} != expected ${expectedLength}`);
    }
    return arr;
}

function compileShader(type, source) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, source);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        const log = gl.getShaderInfoLog(sh);
        gl.deleteShader(sh);
        throw new Error(`Shader compile failed: ${log}`);
    }
    return sh;
}

function makeProgram(vsSource, fsSource) {
    const vs = compileShader(gl.VERTEX_SHADER, vsSource);
    const fs = compileShader(gl.FRAGMENT_SHADER, fsSource);
    const prg = gl.createProgram();
    gl.attachShader(prg, vs);
    gl.attachShader(prg, fs);
    gl.linkProgram(prg);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(prg, gl.LINK_STATUS)) {
        const log = gl.getProgramInfoLog(prg);
        gl.deleteProgram(prg);
        throw new Error(`Program link failed: ${log}`);
    }
    return prg;
}

const VERTEX_SHADER = `#version 300 es
precision highp float;

layout(location = 0) in vec2 a_screen;
layout(location = 1) in float a_value;

uniform vec2 u_mapSize;

out float v_value;

void main() {
    vec2 clip;
    clip.x = a_screen.x / u_mapSize.x * 2.0 - 1.0;
    clip.y = 1.0 - a_screen.y / u_mapSize.y * 2.0;
    gl_Position = vec4(clip, 0.0, 1.0);
    v_value = a_value;
}
`;

const FRAGMENT_SHADER = `#version 300 es
precision highp float;

in float v_value;

uniform float u_vmin;
uniform float u_vmax;
uniform float u_opacity;
uniform int u_cmap;
uniform float u_invalid;

out vec4 outColor;

vec3 jet(float t) {
    t = clamp(t, 0.0, 1.0);
    float r = clamp(min(4.0 * t - 1.5, -4.0 * t + 4.5), 0.0, 1.0);
    float g = clamp(min(4.0 * t - 0.5, -4.0 * t + 3.5), 0.0, 1.0);
    float b = clamp(min(4.0 * t + 0.5, -4.0 * t + 2.5), 0.0, 1.0);
    return vec3(r, g, b);
}

vec3 mix3(vec3 a, vec3 b, float t) { return a * (1.0 - t) + b * t; }

vec3 rdbu(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.031, 0.188, 0.420);
    vec3 c1 = vec3(0.420, 0.682, 0.839);
    vec3 c2 = vec3(0.969, 0.969, 0.969);
    vec3 c3 = vec3(0.984, 0.416, 0.290);
    vec3 c4 = vec3(0.404, 0.000, 0.051);
    if (t < 0.25) return mix3(c0, c1, t / 0.25);
    if (t < 0.50) return mix3(c1, c2, (t - 0.25) / 0.25);
    if (t < 0.75) return mix3(c2, c3, (t - 0.50) / 0.25);
    return mix3(c3, c4, (t - 0.75) / 0.25);
}

void main() {
    if (v_value <= u_invalid + 1.0) discard;
    float t = clamp((v_value - u_vmin) / (u_vmax - u_vmin), 0.0, 1.0);
    vec3 c = (u_cmap == 1) ? rdbu(t) : jet(t);
    outColor = vec4(c, u_opacity);
}
`;

function initWebGL() {
    glCanvas = document.createElement("canvas");
    glCanvas.id = "gl-canvas";
    map.getContainer().appendChild(glCanvas);

    currentCanvas = document.createElement("canvas");
    currentCanvas.id = "current-canvas";
    map.getContainer().appendChild(currentCanvas);
    currentCtx = currentCanvas.getContext("2d");

    gl = glCanvas.getContext("webgl2", {
        alpha: true,
        antialias: true,
        premultipliedAlpha: false
    });

    if (!gl) throw new Error("WebGL2 is not available in this browser.");

    program = makeProgram(VERTEX_SHADER, FRAGMENT_SHADER);

    uMapSizeLoc = gl.getUniformLocation(program, "u_mapSize");
    uVminLoc = gl.getUniformLocation(program, "u_vmin");
    uVmaxLoc = gl.getUniformLocation(program, "u_vmax");
    uOpacityLoc = gl.getUniformLocation(program, "u_opacity");
    uCmapLoc = gl.getUniformLocation(program, "u_cmap");
    uInvalidLoc = gl.getUniformLocation(program, "u_invalid");

    vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    nodeBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, nodeBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, screenNodes, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    valueBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, valueBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, meta.node_count * 4, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 0, 0);

    elemBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, elemBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, meshElems, gl.STATIC_DRAW);

    gl.bindVertexArray(null);

    resizeCanvases();
    updateScreenCoordinates();

    map.on("move zoom resize", () => {
        resizeCanvases();
        updateScreenCoordinates();
        resetParticles();
        renderScalar();
    });

    window.addEventListener("resize", () => {
        resizeCanvases();
        updateScreenCoordinates();
        resetParticles();
        renderScalar();
    });
}

function resizeCanvases() {
    const size = map.getSize();
    const dpr = window.devicePixelRatio || 1;

    for (const canvas of [glCanvas, currentCanvas]) {
        if (!canvas) continue;
        canvas.width = Math.max(1, Math.round(size.x * dpr));
        canvas.height = Math.max(1, Math.round(size.y * dpr));
        canvas.style.width = size.x + "px";
        canvas.style.height = size.y + "px";
    }

    if (gl) gl.viewport(0, 0, glCanvas.width, glCanvas.height);

    if (currentCtx) {
        currentCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
        clearCurrentCanvas();
    }
}

function updateScreenCoordinates() {
    if (!meshNodes || !screenNodes || !map || !gl) return;

    for (let i = 0; i < meta.node_count; i++) {
        const lon = meshNodes[i * 2];
        const lat = meshNodes[i * 2 + 1];
        const pt = map.latLngToContainerPoint([lat, lon]);
        screenNodes[i * 2] = pt.x;
        screenNodes[i * 2 + 1] = pt.y;
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, nodeBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, screenNodes, gl.DYNAMIC_DRAW);
}

function renderScalar() {
    if (!gl || !program || currentVar === "current") {
        if (gl) {
            gl.clearColor(0, 0, 0, 0);
            gl.clear(gl.COLOR_BUFFER_BIT);
        }
        return;
    }

    const vmeta = variableMeta(currentVar);
    if (!vmeta) return;

    const size = map.getSize();
    const opacity = parseFloat(opacitySlider.value);

    gl.viewport(0, 0, glCanvas.width, glCanvas.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    gl.useProgram(program);
    gl.bindVertexArray(vao);

    gl.uniform2f(uMapSizeLoc, size.x, size.y);
    gl.uniform1f(uVminLoc, vmeta.vmin);
    gl.uniform1f(uVmaxLoc, vmeta.vmax);
    gl.uniform1f(uOpacityLoc, opacity);
    gl.uniform1i(uCmapLoc, vmeta.cmap === "rdbu" ? 1 : 0);
    gl.uniform1f(uInvalidLoc, meta.invalid_value);

    gl.drawElements(gl.TRIANGLES, meta.index_count, gl.UNSIGNED_INT, 0);
    gl.bindVertexArray(null);
}

async function loadScalarFrame(variable, frameIndex) {
    const url = scalarFrameUrl(variable, frameIndex);
    if (!url) return null;

    const key = `${variable}:${frameIndex}`;

    if (scalarCache.has(key)) return scalarCache.get(key);
    if (scalarLoading.has(key)) return await scalarLoading.get(key);

    const promise = fetchFloat32(url, meta.node_count).then(arr => {
        scalarCache.set(key, arr);
        scalarLoading.delete(key);

        for (const k of Array.from(scalarCache.keys())) {
            const [v, f] = k.split(":");
            const fi = parseInt(f);
            if (v !== variable || Math.abs(fi - frameIndex) > 2) scalarCache.delete(k);
        }

        return arr;
    });

    scalarLoading.set(key, promise);
    return await promise;
}

function preloadScalarNeighbors(variable, frameIndex) {
    if (variable === "current") return;
    const frameCount = meta.frames.length;
    loadScalarFrame(variable, Math.max(0, frameIndex - 1)).catch(() => {});
    loadScalarFrame(variable, Math.min(frameCount - 1, frameIndex + 1)).catch(() => {});
}

async function loadCurrentComponent(cache, loading, url) {
    if (cache.has(url)) return cache.get(url);
    if (loading.has(url)) return await loading.get(url);

    const promise = fetchFloat32(url, meta.node_count).then(arr => {
        cache.set(url, arr);
        loading.delete(url);
        return arr;
    });

    loading.set(url, promise);
    return await promise;
}

async function loadCurrentFrame(i) {
    const uUrl = currentUFrameUrl(i);
    const vUrl = currentVFrameUrl(i);

    currentU = await loadCurrentComponent(currentUCache, currentULoading, uUrl);
    currentV = await loadCurrentComponent(currentVCache, currentVLoading, vUrl);

    for (const k of Array.from(currentUCache.keys())) {
        const m = k.match(/frame_(\d+)\.bin/);
        if (m && Math.abs(parseInt(m[1]) - i) > 2) currentUCache.delete(k);
    }
    for (const k of Array.from(currentVCache.keys())) {
        const m = k.match(/frame_(\d+)\.bin/);
        if (m && Math.abs(parseInt(m[1]) - i) > 2) currentVCache.delete(k);
    }

    resetParticles();
}

function preloadCurrentFrame(i) {
    const j = Math.min(meta.frames.length - 1, Math.max(0, i));
    loadCurrentComponent(currentUCache, currentULoading, currentUFrameUrl(j)).catch(() => {});
    loadCurrentComponent(currentVCache, currentVLoading, currentVFrameUrl(j)).catch(() => {});
}

async function setFrame(i) {
    const n = meta.frames.length;
    if (n <= 0) return;

    currentFrame = parseInt(i);
    if (!Number.isFinite(currentFrame)) currentFrame = 0;
    if (currentFrame < 0) currentFrame = 0;
    if (currentFrame >= n) currentFrame = n - 1;

    frameSlider.value = currentFrame;
    timeLabel.textContent = meta.frames[currentFrame].label || `frame ${currentFrame}`;

    updateCurrentOverlayAvailability();
    updateLegend();

    if (currentVar === "current") {
        glCanvas.style.display = "none";
        await loadCurrentFrame(currentFrame);
        clearCurrentCanvas();
        startParticles();
        preloadCurrentFrame(currentFrame + 1);
    } else {
        glCanvas.style.display = "block";

        const arr = await loadScalarFrame(currentVar, currentFrame);
        gl.bindBuffer(gl.ARRAY_BUFFER, valueBuffer);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, arr);

        renderScalar();
        preloadScalarNeighbors(currentVar, currentFrame);

        if (currentOverlayEnabled()) {
            await loadCurrentFrame(currentFrame);
            clearCurrentCanvas();
            startParticles();
            preloadCurrentFrame(currentFrame + 1);
        } else {
            stopParticles();
        }
    }

    setStatus(`${currentVar} frame ${currentFrame + 1}/${n} | current=unstructured`);
}

function startTimer() {
    if (timer !== null) clearInterval(timer);
    const intervalMs = 1000 / speed;
    timer = setInterval(() => {
        currentFrame += 1;
        if (currentFrame >= meta.frames.length) currentFrame = 0;
        setFrame(currentFrame);
    }, intervalMs);
}

function clearCurrentCanvas() {
    if (!currentCtx || !map) return;
    const size = map.getSize();
    currentCtx.clearRect(0, 0, size.x, size.y);
}

function speedToColor(s, vmin, vmax) {
    let t = (s - vmin) / (vmax - vmin);
    if (!isFinite(t)) t = 0;
    t = Math.max(0, Math.min(1, t));

    const four = 4.0 * t;
    const r = Math.round(255 * Math.max(0, Math.min(1, Math.min(four - 1.5, -four + 4.5))));
    const g = Math.round(255 * Math.max(0, Math.min(1, Math.min(four - 0.5, -four + 3.5))));
    const b = Math.round(255 * Math.max(0, Math.min(1, Math.min(four + 0.5, -four + 2.5))));

    return `rgba(${r},${g},${b},0.92)`;
}

function currentParticleColor(speedValue) {
    if (currentVar === "current") {
        const v = meta.variables.current;
        return speedToColor(speedValue, v.vmin, v.vmax);
    }
    return CONFIG.currentOverlayColor;
}

function isValidCurrentValue(v) {
    return Number.isFinite(v) && v > meta.invalid_value + 1.0;
}

function triContainsAndVector(ti, lon, lat) {
    const i0 = meshElems[ti * 3];
    const i1 = meshElems[ti * 3 + 1];
    const i2 = meshElems[ti * 3 + 2];

    const x0 = meshNodes[i0 * 2];
    const y0 = meshNodes[i0 * 2 + 1];
    const x1 = meshNodes[i1 * 2];
    const y1 = meshNodes[i1 * 2 + 1];
    const x2 = meshNodes[i2 * 2];
    const y2 = meshNodes[i2 * 2 + 1];

    const den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
    if (Math.abs(den) < 1e-20) return null;

    const w0 = ((y1 - y2) * (lon - x2) + (x2 - x1) * (lat - y2)) / den;
    const w1 = ((y2 - y0) * (lon - x2) + (x0 - x2) * (lat - y2)) / den;
    const w2 = 1.0 - w0 - w1;

    const eps = CONFIG.baryEps;
    if (w0 < -eps || w1 < -eps || w2 < -eps) return null;

    const u0 = currentU[i0], u1 = currentU[i1], u2 = currentU[i2];
    const v0 = currentV[i0], v1 = currentV[i1], v2 = currentV[i2];

    if (!isValidCurrentValue(u0) || !isValidCurrentValue(u1) || !isValidCurrentValue(u2)) return null;
    if (!isValidCurrentValue(v0) || !isValidCurrentValue(v1) || !isValidCurrentValue(v2)) return null;

    const u = w0 * u0 + w1 * u1 + w2 * u2;
    const v = w0 * v0 + w1 * v1 + w2 * v2;
    const spd = Math.sqrt(u * u + v * v);

    if (!Number.isFinite(spd)) return null;
    return {u, v, speed: spd, tri: ti};
}

function lookupCell(lon, lat) {
    if (!lookupMeta) return -1;
    if (lon < lookupMeta.lon_min || lon > lookupMeta.lon_max || lat < lookupMeta.lat_min || lat > lookupMeta.lat_max) return -1;

    let ix = Math.floor((lon - lookupMeta.lon_min) / (lookupMeta.lon_max - lookupMeta.lon_min) * lookupMeta.nx);
    let iy = Math.floor((lat - lookupMeta.lat_min) / (lookupMeta.lat_max - lookupMeta.lat_min) * lookupMeta.ny);

    ix = clamp(ix, 0, lookupMeta.nx - 1);
    iy = clamp(iy, 0, lookupMeta.ny - 1);

    return iy * lookupMeta.nx + ix;
}

function vectorAt(lon, lat, lastTri = -1) {
    if (!currentU || !currentV) return null;

    if (CONFIG.lastTriangleFirst && lastTri >= 0) {
        const quick = triContainsAndVector(lastTri, lon, lat);
        if (quick) return quick;
    }

    const cell = lookupCell(lon, lat);
    if (cell < 0) return null;

    const start = lookupOffsets[cell];
    const end = lookupOffsets[cell + 1];

    for (let k = start; k < end; k++) {
        const ti = lookupTriangles[k];
        const res = triContainsAndVector(ti, lon, lat);
        if (res) return res;
    }

    return null;
}

function randomValidPoint() {
    if (!currentU || !currentV) return null;

    const b = map.getBounds();
    const west = Math.max(b.getWest(), lookupMeta.lon_min);
    const east = Math.min(b.getEast(), lookupMeta.lon_max);
    const south = Math.max(b.getSouth(), lookupMeta.lat_min);
    const north = Math.min(b.getNorth(), lookupMeta.lat_max);

    if (west >= east || south >= north) return null;

    for (let trial = 0; trial < 200; trial++) {
        const lon = west + Math.random() * (east - west);
        const lat = south + Math.random() * (north - south);
        const vec = vectorAt(lon, lat, -1);
        if (vec) return {lon, lat, tri: vec.tri};
    }

    return null;
}

function resetParticle(p) {
    const ll = randomValidPoint();

    if (!ll) {
        p.lon = meta.bounds[0][1];
        p.lat = meta.bounds[0][0];
        p.tri = -1;
    } else {
        p.lon = ll.lon;
        p.lat = ll.lat;
        p.tri = ll.tri;
    }

    p.age = Math.floor(Math.random() * 100);
    p.maxAge = 80 + Math.floor(Math.random() * 80);
}

function resetParticles() {
    particles = [];
    for (let i = 0; i < particleCount; i++) {
        const p = {};
        resetParticle(p);
        particles.push(p);
    }
}

function startParticles() {
    if (particleAnimId !== null) {
        cancelAnimationFrame(particleAnimId);
        particleAnimId = null;
    }

    particleRunning = true;
    resetParticles();

    function step() {
        if (!particleRunning || !shouldDrawCurrentParticles()) return;

        const size = map.getSize();

        currentCtx.globalCompositeOperation = "destination-in";
        currentCtx.fillStyle = "rgba(0, 0, 0, 0.92)";
        currentCtx.fillRect(0, 0, size.x, size.y);

        currentCtx.globalCompositeOperation = "source-over";
        currentCtx.lineWidth = 1.2;

        for (const p of particles) {
            if (p.age > p.maxAge) {
                resetParticle(p);
                continue;
            }

            const vec = vectorAt(p.lon, p.lat, p.tri);
            if (!vec || !isFinite(vec.u) || !isFinite(vec.v)) {
                resetParticle(p);
                continue;
            }

            p.tri = vec.tri;
            const oldPoint = map.latLngToContainerPoint([p.lat, p.lon]);

            const latRad = p.lat * Math.PI / 180.0;
            let coslat = Math.cos(latRad);
            if (Math.abs(coslat) < 1e-6) coslat = 1e-6;

            const dt = CONFIG.flowScale * speed;
            const newLon = p.lon + (vec.u * dt) / coslat;
            const newLat = p.lat + vec.v * dt;

            const vec2 = vectorAt(newLon, newLat, p.tri);
            if (!vec2) {
                resetParticle(p);
                continue;
            }

            p.lon = newLon;
            p.lat = newLat;
            p.tri = vec2.tri;
            p.age += 1;

            const newPoint = map.latLngToContainerPoint([p.lat, p.lon]);

            if (newPoint.x < -50 || newPoint.x > size.x + 50 || newPoint.y < -50 || newPoint.y > size.y + 50) {
                resetParticle(p);
                continue;
            }

            currentCtx.strokeStyle = currentParticleColor(vec.speed);
            currentCtx.beginPath();
            currentCtx.moveTo(oldPoint.x, oldPoint.y);
            currentCtx.lineTo(newPoint.x, newPoint.y);
            currentCtx.stroke();
        }

        particleAnimId = requestAnimationFrame(step);
    }

    particleAnimId = requestAnimationFrame(step);
}

function stopParticles() {
    particleRunning = false;
    if (particleAnimId !== null) {
        cancelAnimationFrame(particleAnimId);
        particleAnimId = null;
    }
    clearCurrentCanvas();
}

function setupEvents() {
    varSelect.addEventListener("change", e => {
        currentVar = e.target.value;
        updateCurrentOverlayAvailability();
        updateLegend();
        setFrame(currentFrame);
    });

    currentOverlayCheck.addEventListener("change", () => {
        updateCurrentOverlayAvailability();
        setFrame(currentFrame);
    });

    playBtn.addEventListener("click", () => {
        if (timer === null) {
            playBtn.textContent = "Pause";
            startTimer();
        } else {
            playBtn.textContent = "Play";
            clearInterval(timer);
            timer = null;
        }
    });

    frameSlider.addEventListener("input", e => setFrame(e.target.value));

    speedSelect.addEventListener("change", e => {
        speed = parseFloat(e.target.value);
        if (timer !== null) startTimer();
    });

    opacitySlider.addEventListener("input", () => renderScalar());

    particleDensitySelect.addEventListener("change", e => {
        particleCount = parseInt(e.target.value);
        resetParticles();
        clearCurrentCanvas();
    });
}

async function boot() {
    setStatus("Loading metadata...");

    const metaResp = await fetch(CONFIG.metaUrl, { cache: "force-cache" });
    if (!metaResp.ok) throw new Error(`Failed to load ${CONFIG.metaUrl}`);
    meta = await metaResp.json();

    const lookupResp = await fetch(CONFIG.lookupMetaUrl, { cache: "force-cache" });
    if (!lookupResp.ok) throw new Error(`Failed to load ${CONFIG.lookupMetaUrl}`);
    lookupMeta = await lookupResp.json();

    frameSlider.max = meta.frames.length - 1;

    const bounds = meta.bounds;
    map = L.map("map", {
        center: [(bounds[0][0] + bounds[1][0]) / 2, (bounds[0][1] + bounds[1][1]) / 2],
        zoom: 7
    });

    const carto = L.tileLayer(
        "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        { attribution: "&copy; OpenStreetMap contributors &copy; CARTO" }
    );

    const esri = L.tileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        { attribution: "Tiles &copy; Esri" }
    );

    esri.addTo(map);

    L.control.layers({"CartoDB Positron": carto, "Esri Satellite": esri}, null, {collapsed:false}).addTo(map);
    map.fitBounds(bounds);

    setStatus("Loading mesh and lookup index...");

    meshNodes = await fetchFloat32(CONFIG.nodesUrl, meta.node_count * 2);
    meshElems = await fetchUint32(CONFIG.elemsUrl, meta.index_count);
    lookupOffsets = await fetchUint32(lookupMeta.offsets_bin, lookupMeta.offset_count);
    lookupTriangles = await fetchUint32(lookupMeta.triangles_bin, lookupMeta.candidate_count);
    screenNodes = new Float32Array(meta.node_count * 2);

    initWebGL();
    setupEvents();
    updateLegend();
    updateCurrentOverlayAvailability();

    await setFrame(0);

    setStatus(`Ready: ${meta.node_count.toLocaleString()} nodes, ${meta.triangle_count.toLocaleString()} triangles, current=unstructured`);
}

boot().catch(err => {
    console.error(err);
    setStatus("ERROR: " + err.message);
    alert(err.message);
});
'''


def main() -> None:
    print("=== SCHISM WebGL unstructured generator with unstructured current ===")
    print("REPO_DIR:", REPO_DIR)
    print("SCHISM_DIR:", SCHISM_DIR)
    print("OUT_DIR:", OUT_DIR)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "temp_bin").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ssh_bin").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "current_u_bin").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "current_v_bin").mkdir(parents=True, exist_ok=True)

    for path in [OUT2D_FILE, TEMP_FILE, HVELX_FILE, HVELY_FILE]:
        if not path.exists():
            raise FileNotFoundError(path)

    print("Reading mesh from:", OUT2D_FILE)
    with Dataset(OUT2D_FILE) as ds2d:
        lon, lat = find_node_xy(ds2d)
        face_nodes = find_face_nodes(ds2d)
        triangles = triangulate_faces(face_nodes, lon.size)

        node_count = int(lon.size)
        triangle_count = int(triangles.shape[0])
        index_count = int(triangles.size)

        print("node_count:", node_count)
        print("triangle_count:", triangle_count)
        print("index_count:", index_count)

        if "time" in ds2d.dimensions:
            max_time = ds2d.dimensions["time"].size
            frame_indices = [i for i in range(START_IDX, END_IDX, STRIDE) if i < max_time]
        else:
            frame_indices = list(range(START_IDX, END_IDX, STRIDE))

        labels = read_time_labels(ds2d, frame_indices)
        elev_var = _find_var(ds2d, ["elevation", "elev", "zeta"], contains=["elev"])

        print("Writing mesh_nodes.bin and mesh_elems.bin")
        nodes = np.empty(node_count * 2, dtype=np.float32)
        nodes[0::2] = lon.astype(np.float32)
        nodes[1::2] = lat.astype(np.float32)

        write_float32_bin(OUT_DIR / "mesh_nodes.bin", nodes)
        write_uint32_bin(OUT_DIR / "mesh_elems.bin", triangles.reshape(-1))

        lookup_meta = build_triangle_lookup(lon, lat, triangles, OUT_DIR, LOOKUP_NX, LOOKUP_NY)

        print("Writing SSH bin frames")
        for out_i, src_i in enumerate(frame_indices):
            vals = read_node_scalar(elev_var, src_i, node_count, layer_index=None)
            write_float32_bin(OUT_DIR / "ssh_bin" / f"frame_{out_i:04d}.bin", vals)
            print(f"  ssh frame {out_i:04d} <- time {src_i}")

    print("Reading temperature from:", TEMP_FILE)
    with Dataset(TEMP_FILE) as dst:
        temp_var = _find_var(dst, ["temperature", "temp"], contains=["temp"])
        print("Writing temperature bin frames")
        for out_i, src_i in enumerate(frame_indices):
            vals = read_node_scalar(temp_var, src_i, node_count, layer_index=K_IDX)
            write_float32_bin(OUT_DIR / "temp_bin" / f"frame_{out_i:04d}.bin", vals)
            print(f"  temp frame {out_i:04d} <- time {src_i}")

    print("Reading current from:", HVELX_FILE, HVELY_FILE)
    with Dataset(HVELX_FILE) as dsx, Dataset(HVELY_FILE) as dsy:
        ux_var = _find_var(dsx, ["horizontalVelX", "hvel_x", "u", "uvel"], contains=["vel", "x"])
        vy_var = _find_var(dsy, ["horizontalVelY", "hvel_y", "v", "vvel"], contains=["vel", "y"])

        print("Writing unstructured current u/v bin frames")
        for out_i, src_i in enumerate(frame_indices):
            u = read_node_scalar(ux_var, src_i, node_count, layer_index=K_IDX)
            v = read_node_scalar(vy_var, src_i, node_count, layer_index=K_IDX)

            bad = (
                ~np.isfinite(u) | ~np.isfinite(v) |
                (u <= float(INVALID_VALUE) + 1.0) |
                (v <= float(INVALID_VALUE) + 1.0)
            )
            if bad.any():
                u = u.copy()
                v = v.copy()
                u[bad] = INVALID_VALUE
                v[bad] = INVALID_VALUE

            write_float32_bin(OUT_DIR / "current_u_bin" / f"frame_{out_i:04d}.bin", u)
            write_float32_bin(OUT_DIR / "current_v_bin" / f"frame_{out_i:04d}.bin", v)
            print(f"  current frame {out_i:04d} <- time {src_i}")

    meta = {
        "format": "schism-unstructured-webgl-v2-current-lookup",
        "node_count": node_count,
        "triangle_count": triangle_count,
        "index_count": index_count,
        "invalid_value": float(INVALID_VALUE),
        "bounds": [
            [float(np.nanmin(lat)), float(np.nanmin(lon))],
            [float(np.nanmax(lat)), float(np.nanmax(lon))],
        ],
        "frames": [
            {
                "i": int(k),
                "source_time_index": int(src_i),
                "label": str(label),
                "temp_bin": f"temp_bin/frame_{k:04d}.bin",
                "ssh_bin": f"ssh_bin/frame_{k:04d}.bin",
                "current_u_bin": f"current_u_bin/frame_{k:04d}.bin",
                "current_v_bin": f"current_v_bin/frame_{k:04d}.bin",
            }
            for k, (src_i, label) in enumerate(zip(frame_indices, labels))
        ],
        "variables": {
            "temperature": {"vmin": TEMP_VMIN, "vmax": TEMP_VMAX, "cmap": "jet", "units": "degC"},
            "ssh": {"vmin": SSH_VMIN, "vmax": SSH_VMAX, "cmap": "rdbu", "units": "m"},
            "current": {"vmin": CURRENT_VMIN, "vmax": CURRENT_VMAX, "cmap": "jet", "units": "m/s"},
        },
        "lookup": lookup_meta,
    }

    (OUT_DIR / "mesh_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (OUT_DIR / "webgl_scalar.js").write_text(WEBGL_SCALAR_JS, encoding="utf-8")

    print("Wrote:", OUT_DIR / "index.html")
    print("Wrote:", OUT_DIR / "webgl_scalar.js")
    print("Wrote:", OUT_DIR / "mesh_meta.json")
    print("Wrote:", OUT_DIR / "lookup_meta.json")

    total_size = 0
    for root, _, files in os.walk(OUT_DIR):
        for f in files:
            total_size += (Path(root) / f).stat().st_size
    print(f"Output size: {total_size / 1024 / 1024:.2f} MB")

    print("\nNext:")
    print(f"  cd {REPO_DIR}")
    print("  git add webgl_unstructured make_schism_webgl_unstructured_current_lookup.py")
    print('  git commit -m "Add unstructured current WebGL particles"')
    print("  git push origin main")
    print("\nOpen:")
    print("  https://king4398.github.io/KOP_view/webgl_unstructured/")


if __name__ == "__main__":
    main()
