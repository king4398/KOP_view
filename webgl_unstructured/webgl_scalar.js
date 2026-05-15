/* SCHISM unstructured WebGL viewer. */

"use strict";

const CONFIG = {
    metaUrl: "mesh_meta.json",
    nodesUrl: "mesh_nodes.bin",
    elemsUrl: "mesh_elems.bin",
    currentOverlayColor: "rgba(255,255,255,0.96)",
    flowScale: 0.018
};

let meta = null;
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

let uZoomLoc = null;
let uPixelOriginLoc = null;
let uMapSizeLoc = null;
let uVminLoc = null;
let uVmaxLoc = null;
let uOpacityLoc = null;
let uCmapLoc = null;
let uInvalidLoc = null;

let currentVar = "temperature";
let currentFrame = 0;
let timer = null;
let speed = 1.0;

let scalarCache = new Map();
let scalarLoading = new Map();

let currentJsonCache = {};
let currentData = null;
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

function currentFrameUrl(frameIndex) {
    if (!meta.current_json_dir) return null;
    return `${meta.current_json_dir}/frame_${pad4(frameIndex)}.json`;
}

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
        legendBox.innerHTML = `
            <div style="font-weight:bold; margin-bottom:6px;">Current Speed [m/s]</div>
            <div style="width:220px; height:16px; background: linear-gradient(to right, #000080, #0000ff, #00ffff, #ffff00, #ff0000, #800000); border:1px solid #666;"></div>
            <div style="display:flex; justify-content:space-between; font-size:12px; margin-top:4px;">
                <span>0</span><span>0.5</span><span>1.0</span>
            </div>`;
    }
}

async function fetchArrayBuffer(url) {
    const r = await fetch(url);
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

layout(location = 0) in vec2 a_lonlat;
layout(location = 1) in float a_value;

uniform float u_zoom;
uniform vec2 u_pixelOrigin;
uniform vec2 u_mapSize;

out float v_value;

const float PI = 3.141592653589793;

vec2 lonLatToWorldPixel(vec2 lonlat, float zoom) {
    float scale = 256.0 * exp2(zoom);
    float lon = lonlat.x;
    float lat = clamp(lonlat.y, -85.05112878, 85.05112878);
    float x = (lon + 180.0) / 360.0 * scale;
    float latRad = radians(lat);
    float siny = clamp(sin(latRad), -0.9999, 0.9999);
    float y = (0.5 - log((1.0 + siny) / (1.0 - siny)) / (4.0 * PI)) * scale;
    return vec2(x, y);
}

void main() {
    vec2 world = lonLatToWorldPixel(a_lonlat, u_zoom);
    vec2 p = world - u_pixelOrigin;
    vec2 clip;
    clip.x = p.x / u_mapSize.x * 2.0 - 1.0;
    clip.y = 1.0 - p.y / u_mapSize.y * 2.0;
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

vec3 mix3(vec3 a, vec3 b, float t) {
    return a * (1.0 - t) + b * t;
}

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

function initWebGL(nodes, elems) {
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

    uZoomLoc = gl.getUniformLocation(program, "u_zoom");
    uPixelOriginLoc = gl.getUniformLocation(program, "u_pixelOrigin");
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
    gl.bufferData(gl.ARRAY_BUFFER, nodes, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    valueBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, valueBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, meta.node_count * 4, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 0, 0);

    elemBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, elemBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, elems, gl.STATIC_DRAW);

    gl.bindVertexArray(null);

    resizeCanvases();

    map.on("move zoom resize", () => {
        resizeCanvases();
        resetParticles();
        renderScalar();
    });

    window.addEventListener("resize", () => {
        resizeCanvases();
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
    const origin = map.getPixelOrigin();
    const opacity = parseFloat(opacitySlider.value);

    gl.viewport(0, 0, glCanvas.width, glCanvas.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    gl.useProgram(program);
    gl.bindVertexArray(vao);

    gl.uniform1f(uZoomLoc, map.getZoom());
    gl.uniform2f(uPixelOriginLoc, origin.x, origin.y);
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
            if (v !== variable || Math.abs(fi - frameIndex) > 2) {
                scalarCache.delete(k);
            }
        }

        return arr;
    });

    scalarLoading.set(key, promise);
    return await promise;
}

function preloadScalarNeighbors(variable, frameIndex) {
    if (variable === "current") return;

    const frameCount = meta.frames.length;
    const prev = Math.max(0, frameIndex - 1);
    const next = Math.min(frameCount - 1, frameIndex + 1);

    loadScalarFrame(variable, prev).catch(() => {});
    loadScalarFrame(variable, next).catch(() => {});
}

function preloadCurrentJson(frameIndex) {
    const url = currentFrameUrl(Math.min(meta.frames.length - 1, frameIndex + 1));
    if (!url || currentJsonCache[url]) return;

    fetch(url)
        .then(r => r.json())
        .then(j => { currentJsonCache[url] = j; })
        .catch(() => {});
}

async function setFrame(i) {
    const n = meta.frames.length;
    if (n <= 0) return;

    currentFrame = parseInt(i);
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
        preloadCurrentJson(currentFrame);
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
            preloadCurrentJson(currentFrame);
        } else {
            stopParticles();
        }
    }

    setStatus(`${currentVar} frame ${currentFrame + 1}/${n}`);
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
    if (!currentData) return CONFIG.currentOverlayColor;

    if (currentVar === "current") {
        return speedToColor(speedValue, currentData.vmin, currentData.vmax);
    }

    return CONFIG.currentOverlayColor;
}

function gridInfo() {
    if (!currentData) return null;

    return {
        nx: currentData.nx,
        ny: currentData.ny,
        lonMin: currentData.lon_min,
        lonMax: currentData.lon_max,
        latMin: currentData.lat_min,
        latMax: currentData.lat_max,
        invalid: currentData.invalid
    };
}

function idx(ix, iy, nx) { return iy * nx + ix; }

function isValidValue(v) {
    if (!currentData) return false;
    return isFinite(v) && v !== currentData.invalid;
}

function vectorAt(lon, lat) {
    if (!currentData) return null;

    const g = gridInfo();
    if (!g) return null;

    if (lon < g.lonMin || lon > g.lonMax || lat < g.latMin || lat > g.latMax) return null;

    const fx = (lon - g.lonMin) / (g.lonMax - g.lonMin) * (g.nx - 1);
    const fy = (lat - g.latMin) / (g.latMax - g.latMin) * (g.ny - 1);

    const ix0 = Math.floor(fx);
    const iy0 = Math.floor(fy);
    const ix1 = ix0 + 1;
    const iy1 = iy0 + 1;

    if (ix0 < 0 || iy0 < 0 || ix1 >= g.nx || iy1 >= g.ny) return null;

    const tx = fx - ix0;
    const ty = fy - iy0;

    const i00 = idx(ix0, iy0, g.nx);
    const i10 = idx(ix1, iy0, g.nx);
    const i01 = idx(ix0, iy1, g.nx);
    const i11 = idx(ix1, iy1, g.nx);

    const u00 = currentData.u[i00];
    const u10 = currentData.u[i10];
    const u01 = currentData.u[i01];
    const u11 = currentData.u[i11];

    const v00 = currentData.v[i00];
    const v10 = currentData.v[i10];
    const v01 = currentData.v[i01];
    const v11 = currentData.v[i11];

    if (
        !isValidValue(u00) || !isValidValue(u10) ||
        !isValidValue(u01) || !isValidValue(u11) ||
        !isValidValue(v00) || !isValidValue(v10) ||
        !isValidValue(v01) || !isValidValue(v11)
    ) {
        return null;
    }

    const w00 = (1 - tx) * (1 - ty);
    const w10 = tx * (1 - ty);
    const w01 = (1 - tx) * ty;
    const w11 = tx * ty;

    const u = w00 * u00 + w10 * u10 + w01 * u01 + w11 * u11;
    const v = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11;
    const spd = Math.sqrt(u * u + v * v);

    if (!isFinite(spd)) return null;
    return {u, v, speed: spd};
}

function randomValidPoint() {
    if (!currentData) return null;

    const g = gridInfo();
    if (!g) return null;

    const b = map.getBounds();
    const west = Math.max(b.getWest(), g.lonMin);
    const east = Math.min(b.getEast(), g.lonMax);
    const south = Math.max(b.getSouth(), g.latMin);
    const north = Math.min(b.getNorth(), g.latMax);

    if (west >= east || south >= north) return null;

    for (let trial = 0; trial < 200; trial++) {
        const lon = west + Math.random() * (east - west);
        const lat = south + Math.random() * (north - south);

        if (vectorAt(lon, lat)) return {lon, lat};
    }

    return null;
}

function resetParticle(p) {
    const ll = randomValidPoint();

    if (!ll) {
        p.lon = meta.bounds[0][1];
        p.lat = meta.bounds[0][0];
    } else {
        p.lon = ll.lon;
        p.lat = ll.lat;
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

async function loadCurrentFrame(i) {
    const url = currentFrameUrl(i);
    if (!url) {
        currentData = null;
        return;
    }

    if (currentJsonCache[url]) {
        currentData = currentJsonCache[url];
        resetParticles();
        return;
    }

    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`${url}: ${resp.status} ${resp.statusText}`);

    const data = await resp.json();
    currentJsonCache[url] = data;
    currentData = data;
    resetParticles();
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

            const vec = vectorAt(p.lon, p.lat);
            if (!vec || !isFinite(vec.u) || !isFinite(vec.v)) {
                resetParticle(p);
                continue;
            }

            const oldPoint = map.latLngToContainerPoint([p.lat, p.lon]);

            const latRad = p.lat * Math.PI / 180.0;
            let coslat = Math.cos(latRad);
            if (Math.abs(coslat) < 1e-6) coslat = 1e-6;

            const dt = CONFIG.flowScale * speed;

            const newLon = p.lon + (vec.u * dt) / coslat;
            const newLat = p.lat + vec.v * dt;

            const vec2 = vectorAt(newLon, newLat);
            if (!vec2) {
                resetParticle(p);
                continue;
            }

            p.lon = newLon;
            p.lat = newLat;
            p.age += 1;

            const newPoint = map.latLngToContainerPoint([p.lat, p.lon]);

            if (
                newPoint.x < -50 || newPoint.x > size.x + 50 ||
                newPoint.y < -50 || newPoint.y > size.y + 50
            ) {
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

    const metaResp = await fetch(CONFIG.metaUrl);
    if (!metaResp.ok) throw new Error(`Failed to load ${CONFIG.metaUrl}`);
    meta = await metaResp.json();

    frameSlider.max = meta.frames.length - 1;

    const bounds = meta.bounds;
    map = L.map("map", {
        center: [
            (bounds[0][0] + bounds[1][0]) / 2,
            (bounds[0][1] + bounds[1][1]) / 2
        ],
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

    const baseMaps = {
        "CartoDB Positron": carto,
        "Esri Satellite": esri
    };

    L.control.layers(baseMaps, null, {collapsed:false}).addTo(map);
    map.fitBounds(bounds);

    setStatus("Loading mesh geometry...");

    const nodes = await fetchFloat32(CONFIG.nodesUrl, meta.node_count * 2);
    const elems = await fetchUint32(CONFIG.elemsUrl, meta.index_count);

    initWebGL(nodes, elems);
    setupEvents();
    updateLegend();
    updateCurrentOverlayAvailability();

    await setFrame(0);

    setStatus(`Ready: ${meta.node_count.toLocaleString()} nodes, ${meta.triangle_count.toLocaleString()} triangles`);
}

boot().catch(err => {
    console.error(err);
    setStatus("ERROR: " + err.message);
    alert(err.message);
});
