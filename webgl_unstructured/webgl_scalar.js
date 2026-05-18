/* SCHISM unstructured WebGL viewer with unstructured current particles. */

"use strict";

const CONFIG = {
    metaUrl: "mesh_meta.json",
    nodesUrl: "mesh_nodes.bin",
    elemsUrl: "mesh_elems.bin",
    edgesUrl: "mesh_edges.bin",
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
let meshLineBuffer = null;
let meshProgram = null;
let meshMapSizeLoc = null;
let meshColorLoc = null;

let uMapSizeLoc = null;
let uVminLoc = null;
let uVmaxLoc = null;
let uOpacityLoc = null;
let uCmapLoc = null;
let uInvalidLoc = null;

let meshNodes = null;       // Float32Array [lon,lat,...]
let meshElems = null;       // Uint32Array [i0,i1,i2,...]
let meshEdges = null;       // Uint32Array [i0,i1,i0,i2,...] unique edge pairs
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
const meshOverlayCheck = document.getElementById("mesh-overlay-check");
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


const MESH_VERTEX_SHADER = `#version 300 es
precision highp float;
layout(location = 0) in vec2 a_screen;
uniform vec2 u_mapSize;
void main() {
    vec2 clip;
    clip.x = a_screen.x / u_mapSize.x * 2.0 - 1.0;
    clip.y = 1.0 - a_screen.y / u_mapSize.y * 2.0;
    gl_Position = vec4(clip, 0.0, 1.0);
}
`;

const MESH_FRAGMENT_SHADER = `#version 300 es
precision highp float;
uniform vec4 u_color;
out vec4 outColor;
void main() {
    outColor = u_color;
}
`;

function meshOverlayEnabled() {
    return !!(meshOverlayCheck && meshOverlayCheck.checked);
}

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
    meshProgram = makeProgram(MESH_VERTEX_SHADER, MESH_FRAGMENT_SHADER);
    meshMapSizeLoc = gl.getUniformLocation(meshProgram, "u_mapSize");
    meshColorLoc = gl.getUniformLocation(meshProgram, "u_color");

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

    meshLineBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, meshLineBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, meshEdges, gl.STATIC_DRAW);

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
    if (!gl || !program) return;

    const size = map.getSize();

    gl.viewport(0, 0, glCanvas.width, glCanvas.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    if (currentVar !== "current") {
        const vmeta = variableMeta(currentVar);
        if (vmeta) {
            const opacity = parseFloat(opacitySlider.value);

            gl.useProgram(program);
            gl.bindVertexArray(vao);
            gl.uniform2f(uMapSizeLoc, size.x, size.y);
            gl.uniform1f(uVminLoc, vmeta.vmin);
            gl.uniform1f(uVmaxLoc, vmeta.vmax);
            gl.uniform1f(uOpacityLoc, opacity);
            gl.uniform1i(uCmapLoc, vmeta.cmap === "rdbu" ? 1 : 0);
            gl.uniform1f(uInvalidLoc, meta.invalid_value);
            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, elemBuffer);
            gl.drawElements(gl.TRIANGLES, meta.index_count, gl.UNSIGNED_INT, 0);
            gl.bindVertexArray(null);
        }
    }

    if (meshOverlayEnabled()) {
        drawMeshOverlay();
    }
}

function drawMeshOverlay() {
    if (!gl || !meshProgram || !meshLineBuffer || !meshEdges) return;

    const size = map.getSize();
    gl.useProgram(meshProgram);
    gl.uniform2f(meshMapSizeLoc, size.x, size.y);

    // White mesh lines work on satellite and on scalar color fields.
    gl.uniform4f(meshColorLoc, 1.0, 1.0, 1.0, 0.72);

    gl.bindBuffer(gl.ARRAY_BUFFER, nodeBuffer);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, meshLineBuffer);
    gl.drawElements(gl.LINES, meta.edge_index_count, gl.UNSIGNED_INT, 0);
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
        glCanvas.style.display = meshOverlayEnabled() ? "block" : "none";
        if (meshOverlayEnabled()) renderScalar();
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
    if (!currentU || !currentV || !lookupMeta || !lookupOffsets || !lookupTriangles) return null;

    const b = map.getBounds();
    const lonSpan = lookupMeta.lon_max - lookupMeta.lon_min;
    const latSpan = lookupMeta.lat_max - lookupMeta.lat_min;

    const ix0 = clamp(Math.floor((Math.max(b.getWest(), lookupMeta.lon_min) - lookupMeta.lon_min) / lonSpan * lookupMeta.nx), 0, lookupMeta.nx - 1);
    const ix1 = clamp(Math.floor((Math.min(b.getEast(), lookupMeta.lon_max) - lookupMeta.lon_min) / lonSpan * lookupMeta.nx), 0, lookupMeta.nx - 1);
    const iy0 = clamp(Math.floor((Math.max(b.getSouth(), lookupMeta.lat_min) - lookupMeta.lat_min) / latSpan * lookupMeta.ny), 0, lookupMeta.ny - 1);
    const iy1 = clamp(Math.floor((Math.min(b.getNorth(), lookupMeta.lat_max) - lookupMeta.lat_min) / latSpan * lookupMeta.ny), 0, lookupMeta.ny - 1);

    // First sample non-empty lookup cells inside current map view.
    for (let trial = 0; trial < 500; trial++) {
        const ix = Math.floor(ix0 + Math.random() * Math.max(1, ix1 - ix0 + 1));
        const iy = Math.floor(iy0 + Math.random() * Math.max(1, iy1 - iy0 + 1));
        const cell = iy * lookupMeta.nx + ix;
        if (lookupOffsets[cell + 1] <= lookupOffsets[cell]) continue;

        const lon = lookupMeta.lon_min + (ix + Math.random()) / lookupMeta.nx * lonSpan;
        const lat = lookupMeta.lat_min + (iy + Math.random()) / lookupMeta.ny * latSpan;
        const vec = vectorAt(lon, lat, -1);
        if (vec) return {lon, lat, tri: vec.tri};
    }

    // Fallback: old random-in-bounds method.
    const west = Math.max(b.getWest(), lookupMeta.lon_min);
    const east = Math.min(b.getEast(), lookupMeta.lon_max);
    const south = Math.max(b.getSouth(), lookupMeta.lat_min);
    const north = Math.min(b.getNorth(), lookupMeta.lat_max);
    if (west >= east || south >= north) return null;

    for (let trial = 0; trial < 300; trial++) {
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

    if (meshOverlayCheck) {
        meshOverlayCheck.addEventListener("change", () => {
            if (currentVar === "current") {
                glCanvas.style.display = meshOverlayEnabled() ? "block" : "none";
            }
            renderScalar();
        });
    }

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
    meshEdges = await fetchUint32(CONFIG.edgesUrl, meta.edge_index_count);
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


// KOP_HARD_FIX_START
(function () {
    "use strict";

    let kopMeshCanvas = null;
    let kopMeshGl = null;
    let kopMeshProgram = null;
    let kopMeshPosBuffer = null;
    let kopMeshIndexBuffer = null;
    let kopMeshNodes = null;
    let kopMeshEdges = null;
    let kopMeshScreenXY = null;
    let kopMeshReady = false;
    let kopMeshLoading = false;

    function kopGet(id) {
        return document.getElementById(id);
    }

    function kopAddMeshOverlayControl() {
        if (kopGet("mesh-overlay-check")) return;

        const panel = document.querySelector(".side-panel");
        if (!panel) return;

        const row = document.createElement("label");
        row.innerHTML = `
          <span><b>Mesh overlay</b></span>
          <input id="mesh-overlay-check" type="checkbox">
        `;

        panel.appendChild(row);

        const chk = kopGet("mesh-overlay-check");
        chk.addEventListener("change", function () {
            kopUpdateMeshOverlay();
        });
    }

    function kopCompileShader(gl, type, src) {
        const sh = gl.createShader(type);
        gl.shaderSource(sh, src);
        gl.compileShader(sh);

        if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
            throw new Error(gl.getShaderInfoLog(sh));
        }

        return sh;
    }

    function kopCreateMeshProgram(gl) {
        const vs = `#version 300 es
        precision highp float;

        layout(location = 0) in vec2 a_screen;
        uniform vec2 u_mapSize;

        void main() {
            vec2 clip;
            clip.x = a_screen.x / u_mapSize.x * 2.0 - 1.0;
            clip.y = 1.0 - a_screen.y / u_mapSize.y * 2.0;
            gl_Position = vec4(clip, 0.0, 1.0);
        }
        `;

        const fs = `#version 300 es
        precision highp float;

        uniform vec4 u_color;
        out vec4 outColor;

        void main() {
            outColor = u_color;
        }
        `;

        const prog = gl.createProgram();
        gl.attachShader(prog, kopCompileShader(gl, gl.VERTEX_SHADER, vs));
        gl.attachShader(prog, kopCompileShader(gl, gl.FRAGMENT_SHADER, fs));
        gl.linkProgram(prog);

        if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
            throw new Error(gl.getProgramInfoLog(prog));
        }

        return prog;
    }

    function kopEnsureMeshCanvas() {
        if (kopMeshCanvas) return true;
        if (typeof map === "undefined" || !map) return false;

        const container = map.getContainer();
        if (!container) return false;

        if (getComputedStyle(container).position === "static") {
            container.style.position = "relative";
        }

        kopMeshCanvas = document.createElement("canvas");
        kopMeshCanvas.id = "kop-mesh-overlay-canvas";
        kopMeshCanvas.style.position = "absolute";
        kopMeshCanvas.style.left = "0";
        kopMeshCanvas.style.top = "0";
        kopMeshCanvas.style.width = "100%";
        kopMeshCanvas.style.height = "100%";
        kopMeshCanvas.style.pointerEvents = "none";
        kopMeshCanvas.style.zIndex = "50000";
        kopMeshCanvas.style.display = "none";

        container.appendChild(kopMeshCanvas);

        kopMeshGl = kopMeshCanvas.getContext("webgl2", {
            alpha: true,
            antialias: true,
            premultipliedAlpha: false
        });

        if (!kopMeshGl) {
            console.error("[KOP mesh overlay] WebGL2 unavailable");
            return false;
        }

        kopMeshProgram = kopCreateMeshProgram(kopMeshGl);
        kopMeshPosBuffer = kopMeshGl.createBuffer();
        kopMeshIndexBuffer = kopMeshGl.createBuffer();

        map.on("move zoom resize zoomend moveend", function () {
            kopDrawMeshOverlay();
        });

        window.addEventListener("resize", function () {
            kopDrawMeshOverlay();
        });

        return true;
    }

    function kopResizeMeshCanvas() {
        if (!kopMeshCanvas || typeof map === "undefined" || !map) return;

        const size = map.getSize();
        const dpr = window.devicePixelRatio || 1;

        kopMeshCanvas.width = Math.max(1, Math.round(size.x * dpr));
        kopMeshCanvas.height = Math.max(1, Math.round(size.y * dpr));
        kopMeshCanvas.style.width = size.x + "px";
        kopMeshCanvas.style.height = size.y + "px";

        kopMeshGl.viewport(0, 0, kopMeshCanvas.width, kopMeshCanvas.height);
    }

    function kopBuildEdgesFromElems(elems) {
        // fallback only when mesh_edges.bin is absent
        const out = new Uint32Array(elems.length * 2);
        let k = 0;

        for (let i = 0; i < elems.length; i += 3) {
            const a = elems[i];
            const b = elems[i + 1];
            const c = elems[i + 2];

            out[k++] = a; out[k++] = b;
            out[k++] = b; out[k++] = c;
            out[k++] = c; out[k++] = a;
        }

        return out.subarray(0, k);
    }

    async function kopLoadMeshOverlayData() {
        if (kopMeshReady || kopMeshLoading) return;
        kopMeshLoading = true;

        try {
            const nodeResp = await fetch("mesh_nodes.bin", { cache: "force-cache" });
            if (!nodeResp.ok) throw new Error("mesh_nodes.bin " + nodeResp.status);
            kopMeshNodes = new Float32Array(await nodeResp.arrayBuffer());

            let edgeResp = await fetch("mesh_edges.bin", { cache: "force-cache" });

            if (edgeResp.ok) {
                kopMeshEdges = new Uint32Array(await edgeResp.arrayBuffer());
            } else {
                console.warn("[KOP mesh overlay] mesh_edges.bin not found. Fallback to mesh_elems.bin.");
                const elemResp = await fetch("mesh_elems.bin", { cache: "force-cache" });
                if (!elemResp.ok) throw new Error("mesh_elems.bin " + elemResp.status);
                const elems = new Uint32Array(await elemResp.arrayBuffer());
                kopMeshEdges = kopBuildEdgesFromElems(elems);
            }

            kopMeshScreenXY = new Float32Array(kopMeshNodes.length);

            kopMeshReady = true;
            kopMeshLoading = false;

            console.log("[KOP mesh overlay] ready nodes:", kopMeshNodes.length / 2, "edge indices:", kopMeshEdges.length);

            kopDrawMeshOverlay();
        } catch (err) {
            kopMeshLoading = false;
            console.error("[KOP mesh overlay] failed:", err);
        }
    }

    function kopUpdateMeshScreenXY() {
        if (!kopMeshNodes || !kopMeshScreenXY || typeof map === "undefined" || !map) return;

        const n = kopMeshNodes.length / 2;

        for (let i = 0; i < n; i++) {
            const lon = kopMeshNodes[i * 2];
            const lat = kopMeshNodes[i * 2 + 1];
            const pt = map.latLngToContainerPoint([lat, lon]);

            kopMeshScreenXY[i * 2] = pt.x;
            kopMeshScreenXY[i * 2 + 1] = pt.y;
        }
    }

    function kopDrawMeshOverlay() {
        const chk = kopGet("mesh-overlay-check");

        if (!chk || !chk.checked) {
            if (kopMeshCanvas && kopMeshGl) {
                kopMeshCanvas.style.display = "none";
                kopMeshGl.clearColor(0, 0, 0, 0);
                kopMeshGl.clear(kopMeshGl.COLOR_BUFFER_BIT);
            }
            return;
        }

        if (!kopEnsureMeshCanvas()) return;

        kopMeshCanvas.style.display = "block";

        if (!kopMeshReady) {
            kopLoadMeshOverlayData();
            return;
        }

        kopResizeMeshCanvas();
        kopUpdateMeshScreenXY();

        const gl = kopMeshGl;
        const size = map.getSize();

        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.useProgram(kopMeshProgram);

        const uMapSize = gl.getUniformLocation(kopMeshProgram, "u_mapSize");
        const uColor = gl.getUniformLocation(kopMeshProgram, "u_color");

        gl.uniform2f(uMapSize, size.x, size.y);

        // mesh overlay must be visible above scalar/current layers
        gl.uniform4f(uColor, 1.0, 1.0, 1.0, 0.88);

        gl.bindBuffer(gl.ARRAY_BUFFER, kopMeshPosBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, kopMeshScreenXY, gl.DYNAMIC_DRAW);

        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, kopMeshIndexBuffer);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, kopMeshEdges, gl.STATIC_DRAW);

        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

        gl.lineWidth(1.0);
        gl.drawElements(gl.LINES, kopMeshEdges.length, gl.UNSIGNED_INT, 0);
    }

    async function kopForceCurrentStandalone() {
        const sel = kopGet("var-select");
        if (!sel || sel.value !== "current") return;

        try {
            if (typeof currentVar !== "undefined") currentVar = "current";

            if (typeof currentOverlayCheck !== "undefined" && currentOverlayCheck) {
                currentOverlayCheck.checked = false;
                currentOverlayCheck.disabled = true;
            }

            if (typeof glCanvas !== "undefined" && glCanvas) {
                glCanvas.style.display = "none";
            }

            if (typeof updateLegend === "function") updateLegend();
            if (typeof updateCurrentOverlayAvailability === "function") updateCurrentOverlayAvailability();

            const f = typeof currentFrame !== "undefined" ? currentFrame : 0;

            if (typeof loadCurrentFrame === "function") {
                await loadCurrentFrame(f);
            }

            if (typeof clearCurrentCanvas === "function") clearCurrentCanvas();

            if (typeof startParticles === "function") {
                startParticles();
            }

            if (typeof setStatus === "function") {
                setStatus("Current frame " + (f + 1));
            }

            console.log("[KOP current fix] forced current standalone frame", f);
        } catch (err) {
            console.error("[KOP current fix] failed:", err);
        }
    }

    function kopInstallHardFix() {
        kopAddMeshOverlayControl();

        const sel = kopGet("var-select");
        if (sel && !sel.dataset.kopHardCurrentHooked) {
            sel.dataset.kopHardCurrentHooked = "1";
            sel.addEventListener("change", function () {
                setTimeout(kopForceCurrentStandalone, 80);
                setTimeout(kopUpdateMeshOverlay, 120);
            }, true);
        }

        const slider = kopGet("frame-slider");
        if (slider && !slider.dataset.kopHardCurrentHooked) {
            slider.dataset.kopHardCurrentHooked = "1";
            slider.addEventListener("input", function () {
                setTimeout(kopForceCurrentStandalone, 80);
            }, true);
        }

        if (typeof currentCanvas !== "undefined" && currentCanvas) {
            currentCanvas.style.zIndex = "12000";
        }

        if (typeof glCanvas !== "undefined" && glCanvas) {
            glCanvas.style.zIndex = "700";
        }

        setTimeout(kopForceCurrentStandalone, 300);
        setTimeout(kopUpdateMeshOverlay, 500);
    }

    function kopUpdateMeshOverlay() {
        kopDrawMeshOverlay();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", kopInstallHardFix);
    } else {
        kopInstallHardFix();
    }

    window.KOP_FORCE_CURRENT_STANDALONE = kopForceCurrentStandalone;
    window.KOP_DRAW_MESH_OVERLAY = kopDrawMeshOverlay;
})();
// KOP_HARD_FIX_END

