"use strict";

const CONFIG = {
    metaUrl: "mesh_meta.json",
    nodesUrl: "mesh_nodes.bin",
    elemsUrl: "mesh_elems.bin",
    edgesUrl: "mesh_edges.bin",
    lookupMetaUrl: "lookup_meta.json",
    lookupOffsetsUrl: "lookup_offsets.bin",
    lookupTrianglesUrl: "lookup_triangles.bin",
    flowScale: 0.018,
    overlayParticleColor: "rgba(255,255,255,0.96)"
};

let meta = null;
let lookupMeta = null;

let map = null;

let glCanvas = null;
let currentCanvas = null;
let meshCanvas = null;

let gl = null;
let currentCtx = null;
let meshGl = null;

let scalarProgram = null;
let meshProgram = null;

let vao = null;
let nodeBuffer = null;
let valueBuffer = null;
let elemBuffer = null;
let meshEdgeBuffer = null;

let meshVao = null;

let uZoomLoc = null;
let uPixelOriginLoc = null;
let uMapSizeLoc = null;
let uVminLoc = null;
let uVmaxLoc = null;
let uOpacityLoc = null;
let uCmapLoc = null;
let uInvalidLoc = null;

let meshZoomLoc = null;
let meshPixelOriginLoc = null;
let meshMapSizeLoc = null;
let meshColorLoc = null;

let nodes = null;
let elems = null;
let meshEdges = null;
let lookupOffsets = null;
let lookupTriangles = null;

// Leaflet-fixed screen coordinates.
// This is the important part: use Leaflet's own current pixel coordinates,
// not a separate WebMercator calculation inside the shader.
let screenCoords = null;
let meshNodeBuffer = null;

let currentU = null;
let currentV = null;
let currentCache = new Map();
let scalarCache = new Map();
let scalarLoading = new Map();

let currentVar = "temperature";
let currentFrame = 0;
let timer = null;
let speed = 1.0;

let particleCount = 2800;
let particles = [];
let particleAnimId = null;
let particleRunning = false;

// Viewport redraw optimization.
// Reprojecting 300k+ nodes on every Leaflet move/zoom event is expensive.
// We throttle live redraws and force a final accurate redraw on moveend/zoomend.
let coordsDirty = true;
let renderScheduled = false;
let mapInteracting = false;
let lastLiveRedrawMs = 0;
const LIVE_REDRAW_INTERVAL_MS = 80;

// During Leaflet pan/zoom, do not constantly recompute 300k node screen coordinates.
// Instead, transform the already-rendered canvases so they visually follow the map.
// On moveend/zoomend, redraw exactly.
let renderPixelOrigin = null;
let renderZoom = null;
let renderLatLngBounds = null;

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

let meshOverlayCheck = document.getElementById("mesh-overlay-check");

// Leaflet pane-based canvas origin.
// Canvases are positioned in overlay panes using Leaflet layer coordinates.
let canvasTopLeft = null;

// Ghost canvas used during zoom to avoid black flicker.
// It temporarily displays the last rendered view while real WebGL/tile layers update.
let zoomGhostCanvas = null;
let zoomGhostCtx = null;
let zoomGhostTimer = null;

function setStatus(msg) {
    if (statusLine) statusLine.textContent = msg;
}

function pad4(i) {
    return String(i).padStart(4, "0");
}

function frameCount() {
    return meta && meta.frames ? meta.frames.length : 0;
}

function scalarFrameUrl(variable, frameIndex) {
    if (variable === "temperature") return `temp_bin/frame_${pad4(frameIndex)}.bin`;
    if (variable === "ssh") return `ssh_bin/frame_${pad4(frameIndex)}.bin`;
    return null;
}

function currentUUrl(frameIndex) {
    return `current_u_bin/frame_${pad4(frameIndex)}.bin`;
}

function currentVUrl(frameIndex) {
    return `current_v_bin/frame_${pad4(frameIndex)}.bin`;
}

function variableMeta(variable) {
    if (variable === "temperature") return meta.variables.temperature;
    if (variable === "ssh") return meta.variables.ssh;
    return null;
}

function currentOverlayEnabled() {
    return currentOverlayCheck &&
           currentOverlayCheck.checked &&
           (currentVar === "temperature" || currentVar === "ssh");
}

function shouldDrawCurrentParticles() {
    return currentVar === "current" || currentOverlayEnabled();
}

function updateCurrentOverlayAvailability() {
    if (!currentOverlayCheck) return;

    if (currentVar === "current") {
        currentOverlayCheck.checked = false;
        currentOverlayCheck.disabled = true;
        currentOverlayCheck.title = "Current variable already shows current particles.";
    } else {
        currentOverlayCheck.disabled = false;
        currentOverlayCheck.title = "";
        if (!currentOverlayCheck.dataset.userTouched) {
            currentOverlayCheck.checked = true;
        }
    }
}

function ensureMeshOverlayControl() {
    if (document.getElementById("mesh-overlay-check")) {
        meshOverlayCheck = document.getElementById("mesh-overlay-check");
        return;
    }

    const panel = document.querySelector(".side-panel");
    if (!panel) return;

    const row = document.createElement("label");
    row.innerHTML = `
        <span><b>Mesh overlay</b></span>
        <input id="mesh-overlay-check" type="checkbox">
    `;
    panel.appendChild(row);

    meshOverlayCheck = document.getElementById("mesh-overlay-check");
}

function updateLegend() {
    if (!legendBox || !meta) return;

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

async function fetchJson(url) {
    const r = await fetch(url, { cache: "force-cache" });
    if (!r.ok) throw new Error(`${url}: ${r.status} ${r.statusText}`);
    return await r.json();
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

function compileShader(glctx, type, source) {
    const sh = glctx.createShader(type);
    glctx.shaderSource(sh, source);
    glctx.compileShader(sh);

    if (!glctx.getShaderParameter(sh, glctx.COMPILE_STATUS)) {
        const log = glctx.getShaderInfoLog(sh);
        glctx.deleteShader(sh);
        throw new Error(`Shader compile failed: ${log}`);
    }

    return sh;
}

function makeProgram(glctx, vsSource, fsSource) {
    const vs = compileShader(glctx, glctx.VERTEX_SHADER, vsSource);
    const fs = compileShader(glctx, glctx.FRAGMENT_SHADER, fsSource);
    const prg = glctx.createProgram();

    glctx.attachShader(prg, vs);
    glctx.attachShader(prg, fs);
    glctx.linkProgram(prg);

    glctx.deleteShader(vs);
    glctx.deleteShader(fs);

    if (!glctx.getProgramParameter(prg, glctx.LINK_STATUS)) {
        const log = glctx.getProgramInfoLog(prg);
        glctx.deleteProgram(prg);
        throw new Error(`Program link failed: ${log}`);
    }

    return prg;
}

const MAP_VERTEX_SHADER = `#version 300 es
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

const SCALAR_FRAGMENT_SHADER = `#version 300 es
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
    if ((v_value != v_value) || v_value <= u_invalid + 1.0) discard;

    float denom = max(abs(u_vmax - u_vmin), 1e-12);
    float t = clamp((v_value - u_vmin) / denom, 0.0, 1.0);

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



function getLayerCanvases() {
    return [glCanvas, currentCanvas, meshCanvas].filter(Boolean);
}

function rememberExactRenderView() {
    if (!map) return;

    const o = map.getPixelOrigin();
    renderPixelOrigin = { x: o.x, y: o.y };
    renderZoom = map.getZoom();
    renderLatLngBounds = map.getBounds();
}

function resetCanvasTransforms() {
    for (const c of getLayerCanvases()) {
        if (!c) continue;
        c.style.transform = "";
        c.style.transformOrigin = "";
    }
}

function applyCanvasMapTransform() {
    // Disabled.
    // Leaflet pane/mapPane transform handles pan/zoom naturally.
}



function applyCanvasZoomAnimation() {
    // Disabled.
    // Leaflet pane/mapPane transform handles pan/zoom naturally.
}


function renderViewportLayers(includeMesh) {
    resetCanvasTransforms();

    resizeCanvases();
    coordsDirty = true;
    updateScreenCoordinates();
    renderScalar();

    if (includeMesh) {
        drawMeshOverlay();
    }

    rememberExactRenderView();
}

function scheduleViewportRedraw(force=false) {
    const now = performance.now();

    if (mapInteracting && !force) {
        applyCanvasMapTransform();
        return;
    }

    if (!force && now - lastLiveRedrawMs < LIVE_REDRAW_INTERVAL_MS) {
        applyCanvasMapTransform();
        return;
    }

    lastLiveRedrawMs = now;
    coordsDirty = true;

    if (renderScheduled) return;

    renderScheduled = true;

    requestAnimationFrame(() => {
        renderScheduled = false;

        if (mapInteracting) {
            applyCanvasMapTransform();
        } else {
            renderViewportLayers(true);
        }
    });
}

function initMap() {
    const bounds = meta.bounds;

    map = L.map("map", {
        center: [
            (bounds[0][0] + bounds[1][0]) / 2.0,
            (bounds[0][1] + bounds[1][1]) / 2.0
        ],
        zoom: 7,

        // Disable animated zoom to prevent WebGL layer drift/nausea during zoom.
        zoomAnimation: true,
        markerZoomAnimation: true,
        fadeAnimation: true,

        // Softer zoom sensitivity.
        zoomSnap: 0.25,
        zoomDelta: 0.5,
        wheelPxPerZoomLevel: 220,
        wheelDebounceTime: 90,

        // Reduce zoom sensitivity.
        // Larger wheelPxPerZoomLevel = slower wheel zoom.
        zoomSnap: 0.25,
        zoomDelta: 0.5,
        wheelPxPerZoomLevel: 220,
        wheelDebounceTime: 90,
        preferCanvas: true
    });

    const carto = L.tileLayer(
        "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        {
            attribution: "&copy; OpenStreetMap contributors &copy; CARTO",
            keepBuffer: 8,
            updateWhenZooming: false,
            updateWhenIdle: false
        }
    );

    const esri = L.tileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        {
            attribution: "Tiles &copy; Esri",
            keepBuffer: 8,
            updateWhenZooming: false,
            updateWhenIdle: false
        }
    );

    esri.addTo(map);

    L.control.layers({
        "CartoDB Positron": carto,
        "Esri Satellite": esri
    }, null, { collapsed: false }).addTo(map);

    map.fitBounds(bounds);


    map.on("movestart", () => {
        mapInteracting = true;
    });

    map.on("move", () => {
        // Leaflet panes move canvases naturally during pan.
        // Do not redraw or manually transform here.
    });

    map.on("zoomstart", () => {
        mapInteracting = true;

        // Keep the last rendered canvas visible and let Leaflet scale the pane.
        // Pause particles only to avoid drawing into a zooming canvas.
        if (typeof pauseParticlesNoClear === "function") {
            pauseParticlesNoClear();
        }
    });

    map.on("zoomanim", () => {
        // Important:
        // Do nothing. Leaflet's own mapPane transform scales the canvas
        // exactly like the map tiles.
    });

    map.on("resize", () => {
        mapInteracting = false;
        coordsDirty = true;

        renderViewportLayers(true);
        resetParticles();

        if (shouldDrawCurrentParticles()) {
            startParticles();
        }
    });

    map.on("moveend zoomend", () => {
        mapInteracting = false;
        coordsDirty = true;

        // After Leaflet finishes pan/zoom, redraw exactly once using new coordinates.
        renderViewportLayers(true);
        resetParticles();

        if (shouldDrawCurrentParticles()) {
            startParticles();
        }
    });

    window.addEventListener("resize", () => {
        mapInteracting = false;
        coordsDirty = true;

        renderViewportLayers(true);
        resetParticles();

        if (shouldDrawCurrentParticles()) {
            startParticles();
        }
    });
}



function ensureZoomGhostCanvas() {
    if (zoomGhostCanvas) return;

    const container = map.getContainer();

    zoomGhostCanvas = document.createElement("canvas");
    zoomGhostCanvas.id = "zoom-ghost-canvas";
    zoomGhostCanvas.style.position = "absolute";
    zoomGhostCanvas.style.left = "0";
    zoomGhostCanvas.style.top = "0";
    zoomGhostCanvas.style.width = "100%";
    zoomGhostCanvas.style.height = "100%";
    zoomGhostCanvas.style.pointerEvents = "none";
    zoomGhostCanvas.style.zIndex = "30000";
    zoomGhostCanvas.style.opacity = "0";
    zoomGhostCanvas.style.transition = "opacity 220ms ease-out";
    zoomGhostCanvas.style.display = "none";

    container.appendChild(zoomGhostCanvas);
    zoomGhostCtx = zoomGhostCanvas.getContext("2d");
}

function resizeZoomGhostCanvas() {
    if (!zoomGhostCanvas || !map) return;

    const size = map.getSize();
    const dpr = window.devicePixelRatio || 1;

    zoomGhostCanvas.width = Math.max(1, Math.round(size.x * dpr));
    zoomGhostCanvas.height = Math.max(1, Math.round(size.y * dpr));
    zoomGhostCanvas.style.width = size.x + "px";
    zoomGhostCanvas.style.height = size.y + "px";

    zoomGhostCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function showZoomGhost() {
    // Disabled.
    // Leaflet pane/mapPane transform handles pan/zoom naturally.
}

function hideZoomGhostSoon() {
    // Disabled.
    // Leaflet pane/mapPane transform handles pan/zoom naturally.
}

function initCanvases() {
    const panes = map.getPanes();

    const scalarPane = map.createPane ? map.createPane("kop-scalar-pane") : panes.overlayPane;
    const currentPane = map.createPane ? map.createPane("kop-current-pane") : panes.overlayPane;
    const meshPane = map.createPane ? map.createPane("kop-mesh-pane") : panes.overlayPane;

    scalarPane.style.zIndex = "450";
    currentPane.style.zIndex = "650";
    meshPane.style.zIndex = "800";

    scalarPane.style.pointerEvents = "none";
    currentPane.style.pointerEvents = "none";
    meshPane.style.pointerEvents = "none";

    glCanvas = document.createElement("canvas");
    glCanvas.id = "gl-canvas";
    glCanvas.style.position = "absolute";
    glCanvas.style.left = "0";
    glCanvas.style.top = "0";
    glCanvas.style.pointerEvents = "none";
    glCanvas.style.willChange = "transform,width,height";
    scalarPane.appendChild(glCanvas);

    currentCanvas = document.createElement("canvas");
    currentCanvas.id = "current-canvas";
    currentCanvas.style.position = "absolute";
    currentCanvas.style.left = "0";
    currentCanvas.style.top = "0";
    currentCanvas.style.pointerEvents = "none";
    currentCanvas.style.willChange = "transform,width,height";
    currentPane.appendChild(currentCanvas);

    meshCanvas = document.createElement("canvas");
    meshCanvas.id = "mesh-canvas";
    meshCanvas.style.position = "absolute";
    meshCanvas.style.left = "0";
    meshCanvas.style.top = "0";
    meshCanvas.style.pointerEvents = "none";
    meshCanvas.style.willChange = "transform,width,height";
    meshCanvas.style.display = "none";
    meshPane.appendChild(meshCanvas);

    gl = glCanvas.getContext("webgl2", {
        alpha: true,
        antialias: true,
        premultipliedAlpha: false,
        preserveDrawingBuffer: true
    });

    meshGl = meshCanvas.getContext("webgl2", {
        alpha: true,
        antialias: true,
        premultipliedAlpha: false,
        preserveDrawingBuffer: true
    });

    currentCtx = currentCanvas.getContext("2d");

    if (!gl) throw new Error("WebGL2 is not available for scalar canvas.");
    if (!meshGl) throw new Error("WebGL2 is not available for mesh canvas.");

    resizeCanvases();
}



function resizeCanvases() {
    if (!map) return;

    const size = map.getSize();
    const dpr = window.devicePixelRatio || 1;

    canvasTopLeft = map.containerPointToLayerPoint([0, 0]);

    for (const canvas of [glCanvas, currentCanvas, meshCanvas]) {
        if (!canvas) continue;

        canvas.width = Math.max(1, Math.round(size.x * dpr));
        canvas.height = Math.max(1, Math.round(size.y * dpr));
        canvas.style.width = size.x + "px";
        canvas.style.height = size.y + "px";

        if (window.L && L.DomUtil) {
            L.DomUtil.setPosition(canvas, canvasTopLeft);
        } else {
            canvas.style.left = canvasTopLeft.x + "px";
            canvas.style.top = canvasTopLeft.y + "px";
        }
    }

    if (gl && glCanvas) gl.viewport(0, 0, glCanvas.width, glCanvas.height);
    if (meshGl && meshCanvas) meshGl.viewport(0, 0, meshCanvas.width, meshCanvas.height);

    if (currentCtx) {
        currentCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
        clearCurrentCanvas();
    }

    coordsDirty = true;
}




function updateScreenCoordinates() {
    if (!map || !nodes || !screenCoords) return;
    if (!coordsDirty) return;

    if (!canvasTopLeft) {
        canvasTopLeft = map.containerPointToLayerPoint([0, 0]);
    }

    const n = nodes.length / 2;

    for (let i = 0; i < n; i++) {
        const lon = nodes[i * 2];
        const lat = nodes[i * 2 + 1];

        const pt = map.latLngToLayerPoint([lat, lon]);

        screenCoords[i * 2] = pt.x - canvasTopLeft.x;
        screenCoords[i * 2 + 1] = pt.y - canvasTopLeft.y;
    }

    if (gl && nodeBuffer) {
        gl.bindBuffer(gl.ARRAY_BUFFER, nodeBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, screenCoords, gl.DYNAMIC_DRAW);
    }

    if (meshGl && meshNodeBuffer) {
        meshGl.bindBuffer(meshGl.ARRAY_BUFFER, meshNodeBuffer);
        meshGl.bufferData(meshGl.ARRAY_BUFFER, screenCoords, meshGl.DYNAMIC_DRAW);
    }

    coordsDirty = false;
}



function initWebGL() {
    scalarProgram = makeProgram(gl, MAP_VERTEX_SHADER, SCALAR_FRAGMENT_SHADER);

    uZoomLoc = gl.getUniformLocation(scalarProgram, "u_zoom");
    uPixelOriginLoc = gl.getUniformLocation(scalarProgram, "u_pixelOrigin");
    uMapSizeLoc = gl.getUniformLocation(scalarProgram, "u_mapSize");
    uVminLoc = gl.getUniformLocation(scalarProgram, "u_vmin");
    uVmaxLoc = gl.getUniformLocation(scalarProgram, "u_vmax");
    uOpacityLoc = gl.getUniformLocation(scalarProgram, "u_opacity");
    uCmapLoc = gl.getUniformLocation(scalarProgram, "u_cmap");
    uInvalidLoc = gl.getUniformLocation(scalarProgram, "u_invalid");

    vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    screenCoords = new Float32Array(nodes.length);
    updateScreenCoordinates();

    nodeBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, nodeBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, screenCoords, gl.DYNAMIC_DRAW);
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

    meshProgram = makeProgram(meshGl, MESH_VERTEX_SHADER, MESH_FRAGMENT_SHADER);

    meshZoomLoc = meshGl.getUniformLocation(meshProgram, "u_zoom");
    meshPixelOriginLoc = meshGl.getUniformLocation(meshProgram, "u_pixelOrigin");
    meshMapSizeLoc = meshGl.getUniformLocation(meshProgram, "u_mapSize");
    meshColorLoc = meshGl.getUniformLocation(meshProgram, "u_color");

    meshVao = meshGl.createVertexArray();
    meshGl.bindVertexArray(meshVao);

    meshNodeBuffer = meshGl.createBuffer();
    meshGl.bindBuffer(meshGl.ARRAY_BUFFER, meshNodeBuffer);
    meshGl.bufferData(meshGl.ARRAY_BUFFER, screenCoords, meshGl.DYNAMIC_DRAW);
    meshGl.enableVertexAttribArray(0);
    meshGl.vertexAttribPointer(0, 2, meshGl.FLOAT, false, 0, 0);

    meshEdgeBuffer = meshGl.createBuffer();
    meshGl.bindBuffer(meshGl.ELEMENT_ARRAY_BUFFER, meshEdgeBuffer);
    meshGl.bufferData(meshGl.ELEMENT_ARRAY_BUFFER, meshEdges, meshGl.STATIC_DRAW);

    meshGl.bindVertexArray(null);
}

function setMapUniforms(glctx, program, zoomLoc, originLoc, sizeLoc) {
    const size = map.getSize();

    // Screen-coordinate shader only needs map size.
    // zoom/origin uniforms are intentionally unused now.
    if (sizeLoc) {
        glctx.uniform2f(sizeLoc, size.x, size.y);
    }
}

function renderScalar() {
    if (!gl || !scalarProgram) return;

    updateScreenCoordinates();

    if (currentVar === "current") {
        glCanvas.style.display = "none";
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        return;
    }

    const vmeta = variableMeta(currentVar);
    if (!vmeta) return;

    glCanvas.style.display = "block";

    gl.viewport(0, 0, glCanvas.width, glCanvas.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    gl.useProgram(scalarProgram);
    gl.bindVertexArray(vao);

    setMapUniforms(gl, scalarProgram, uZoomLoc, uPixelOriginLoc, uMapSizeLoc);

    gl.uniform1f(uVminLoc, vmeta.vmin);
    gl.uniform1f(uVmaxLoc, vmeta.vmax);
    gl.uniform1f(uOpacityLoc, parseFloat(opacitySlider.value));
    gl.uniform1i(uCmapLoc, vmeta.cmap === "rdbu" ? 1 : 0);
    gl.uniform1f(uInvalidLoc, meta.invalid_value);

    gl.drawElements(gl.TRIANGLES, meta.index_count, gl.UNSIGNED_INT, 0);
    gl.bindVertexArray(null);
}

function drawMeshOverlay() {
    if (!meshOverlayCheck || !meshCanvas || !meshGl || !meshProgram) return;

    updateScreenCoordinates();

    if (!meshOverlayCheck.checked) {
        meshCanvas.style.display = "none";
        meshGl.clearColor(0, 0, 0, 0);
        meshGl.clear(meshGl.COLOR_BUFFER_BIT);
        return;
    }

    meshCanvas.style.display = "block";

    meshGl.viewport(0, 0, meshCanvas.width, meshCanvas.height);
    meshGl.clearColor(0, 0, 0, 0);
    meshGl.clear(meshGl.COLOR_BUFFER_BIT);

    meshGl.enable(meshGl.BLEND);
    meshGl.blendFunc(meshGl.SRC_ALPHA, meshGl.ONE_MINUS_SRC_ALPHA);

    meshGl.useProgram(meshProgram);
    meshGl.bindVertexArray(meshVao);

    setMapUniforms(meshGl, meshProgram, meshZoomLoc, meshPixelOriginLoc, meshMapSizeLoc);

    // High-contrast mesh on top of everything
    meshGl.uniform4f(meshColorLoc, 1.0, 1.0, 1.0, 0.72);

    meshGl.drawElements(meshGl.LINES, meshEdges.length, meshGl.UNSIGNED_INT, 0);
    meshGl.bindVertexArray(null);
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

    const n = frameCount();
    const prev = Math.max(0, frameIndex - 1);
    const next = Math.min(n - 1, frameIndex + 1);

    loadScalarFrame(variable, prev).catch(() => {});
    loadScalarFrame(variable, next).catch(() => {});
}

async function loadCurrentFrame(frameIndex) {
    const key = String(frameIndex);

    if (currentCache.has(key)) {
        const cached = currentCache.get(key);
        currentU = cached.u;
        currentV = cached.v;
        resetParticles();
        return;
    }

    const [u, v] = await Promise.all([
        fetchFloat32(currentUUrl(frameIndex), meta.node_count),
        fetchFloat32(currentVUrl(frameIndex), meta.node_count)
    ]);

    currentCache.set(key, { u, v });

    for (const k of Array.from(currentCache.keys())) {
        const fi = parseInt(k);
        if (Math.abs(fi - frameIndex) > 2) currentCache.delete(k);
    }

    currentU = u;
    currentV = v;

    resetParticles();
}

function preloadCurrentFrame(frameIndex) {
    const n = frameCount();
    const next = Math.min(n - 1, frameIndex + 1);
    const key = String(next);

    if (currentCache.has(key)) return;

    Promise.all([
        fetchFloat32(currentUUrl(next), meta.node_count),
        fetchFloat32(currentVUrl(next), meta.node_count)
    ]).then(([u, v]) => {
        currentCache.set(key, { u, v });
    }).catch(() => {});
}

async function setFrame(i) {
    const n = frameCount();
    if (n <= 0) return;

    currentFrame = parseInt(i);
    if (!Number.isFinite(currentFrame)) currentFrame = 0;
    if (currentFrame < 0) currentFrame = 0;
    if (currentFrame >= n) currentFrame = n - 1;

    frameSlider.value = currentFrame;

    const fmeta = meta.frames[currentFrame] || {};
    timeLabel.textContent = fmeta.label || `frame ${currentFrame}`;

    updateCurrentOverlayAvailability();
    updateLegend();

    if (currentVar === "current") {
        glCanvas.style.display = "none";

        await loadCurrentFrame(currentFrame);
        clearCurrentCanvas();
        startParticles();
        preloadCurrentFrame(currentFrame);

        setStatus(`current frame ${currentFrame + 1}/${n}`);
        drawMeshOverlay();
        return;
    }

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
        preloadCurrentFrame(currentFrame);
    } else {
        stopParticles();
    }

    setStatus(`${currentVar} frame ${currentFrame + 1}/${n}`);
    drawMeshOverlay();
    rememberExactRenderView();
}

function startTimer() {
    if (timer !== null) clearInterval(timer);

    const intervalMs = 1000 / speed;

    timer = setInterval(() => {
        currentFrame += 1;
        if (currentFrame >= frameCount()) currentFrame = 0;
        setFrame(currentFrame);
    }, intervalMs);
}


function latLngToCanvasPoint(lat, lon) {
    if (!canvasTopLeft) {
        canvasTopLeft = map.containerPointToLayerPoint([0, 0]);
    }

    const pt = map.latLngToLayerPoint([lat, lon]);

    return {
        x: pt.x - canvasTopLeft.x,
        y: pt.y - canvasTopLeft.y
    };
}

function clearCurrentCanvas() {
    if (!currentCtx || !map) return;
    const size = map.getSize();
    currentCtx.clearRect(0, 0, size.x, size.y);
}

function speedToColor(s, vmin, vmax) {
    let t = (s - vmin) / (vmax - vmin);
    if (!Number.isFinite(t)) t = 0;
    t = Math.max(0, Math.min(1, t));

    const four = 4.0 * t;
    const r = Math.round(255 * Math.max(0, Math.min(1, Math.min(four - 1.5, -four + 4.5))));
    const g = Math.round(255 * Math.max(0, Math.min(1, Math.min(four - 0.5, -four + 3.5))));
    const b = Math.round(255 * Math.max(0, Math.min(1, Math.min(four + 0.5, -four + 2.5))));

    return `rgba(${r},${g},${b},0.92)`;
}

function currentSpeedRange() {
    return {
        vmin: Number(meta.current_vmin ?? 0.0),
        vmax: Number(meta.current_vmax ?? 1.0)
    };
}

function currentParticleColor(speedValue) {
    if (currentVar === "current") {
        const r = currentSpeedRange();
        return speedToColor(speedValue, r.vmin, r.vmax);
    }

    return CONFIG.overlayParticleColor;
}

function getLookupBounds() {
    const lm = lookupMeta || {};

    return {
        nx: Number(lm.nx ?? lm.lookup_nx ?? lm.LOOKUP_NX),
        ny: Number(lm.ny ?? lm.lookup_ny ?? lm.LOOKUP_NY),
        lonMin: Number(lm.lon_min ?? lm.lonMin ?? lm.west ?? meta.bounds[0][1]),
        lonMax: Number(lm.lon_max ?? lm.lonMax ?? lm.east ?? meta.bounds[1][1]),
        latMin: Number(lm.lat_min ?? lm.latMin ?? lm.south ?? meta.bounds[0][0]),
        latMax: Number(lm.lat_max ?? lm.latMax ?? lm.north ?? meta.bounds[1][0])
    };
}

function lookupCell(lon, lat) {
    const g = getLookupBounds();

    if (!Number.isFinite(g.nx) || !Number.isFinite(g.ny)) return null;
    if (lon < g.lonMin || lon > g.lonMax || lat < g.latMin || lat > g.latMax) return null;

    const ix = Math.floor((lon - g.lonMin) / (g.lonMax - g.lonMin) * g.nx);
    const iy = Math.floor((lat - g.latMin) / (g.latMax - g.latMin) * g.ny);

    if (ix < 0 || iy < 0 || ix >= g.nx || iy >= g.ny) return null;

    return iy * g.nx + ix;
}

function barycentricVector(lon, lat, triIndex) {
    const ia = elems[triIndex * 3];
    const ib = elems[triIndex * 3 + 1];
    const ic = elems[triIndex * 3 + 2];

    const ax = nodes[ia * 2];
    const ay = nodes[ia * 2 + 1];
    const bx = nodes[ib * 2];
    const by = nodes[ib * 2 + 1];
    const cx = nodes[ic * 2];
    const cy = nodes[ic * 2 + 1];

    const v0x = bx - ax;
    const v0y = by - ay;
    const v1x = cx - ax;
    const v1y = cy - ay;
    const v2x = lon - ax;
    const v2y = lat - ay;

    const den = v0x * v1y - v1x * v0y;
    if (Math.abs(den) < 1e-20) return null;

    const w1 = (v2x * v1y - v1x * v2y) / den;
    const w2 = (v0x * v2y - v2x * v0y) / den;
    const w0 = 1.0 - w1 - w2;

    const eps = -1e-7;
    if (w0 < eps || w1 < eps || w2 < eps) return null;

    const ua = currentU[ia];
    const ub = currentU[ib];
    const uc = currentU[ic];

    const va = currentV[ia];
    const vb = currentV[ib];
    const vc = currentV[ic];

    if (
        !Number.isFinite(ua) || !Number.isFinite(ub) || !Number.isFinite(uc) ||
        !Number.isFinite(va) || !Number.isFinite(vb) || !Number.isFinite(vc) ||
        ua <= meta.invalid_value + 1.0 || ub <= meta.invalid_value + 1.0 || uc <= meta.invalid_value + 1.0 ||
        va <= meta.invalid_value + 1.0 || vb <= meta.invalid_value + 1.0 || vc <= meta.invalid_value + 1.0
    ) {
        return null;
    }

    const u = w0 * ua + w1 * ub + w2 * uc;
    const v = w0 * va + w1 * vb + w2 * vc;
    const speedValue = Math.hypot(u, v);

    if (!Number.isFinite(speedValue)) return null;

    return { u, v, speed: speedValue };
}

function vectorAt(lon, lat) {
    if (!currentU || !currentV || !lookupOffsets || !lookupTriangles) return null;

    const cell = lookupCell(lon, lat);
    if (cell === null) return null;

    let start = lookupOffsets[cell];
    let end = lookupOffsets[cell + 1];

    if (end <= start) return null;

    for (let k = start; k < end; k++) {
        const triIndex = lookupTriangles[k];
        const vec = barycentricVector(lon, lat, triIndex);
        if (vec) return vec;
    }

    return null;
}


function randomValidPoint() {
    const g = getLookupBounds();

    let west = g.lonMin;
    let east = g.lonMax;
    let south = g.latMin;
    let north = g.latMax;

    // Important:
    // Seed particles inside the current visible map bounds, not the whole model domain.
    // Otherwise many particles are created off-screen and visible density looks too low.
    if (map) {
        const b = map.getBounds();
        west = Math.max(g.lonMin, b.getWest());
        east = Math.min(g.lonMax, b.getEast());
        south = Math.max(g.latMin, b.getSouth());
        north = Math.min(g.latMax, b.getNorth());
    }

    // Fallback to whole domain if viewport does not intersect valid lookup bounds.
    if (!(west < east && south < north)) {
        west = g.lonMin;
        east = g.lonMax;
        south = g.latMin;
        north = g.latMax;
    }

    for (let trial = 0; trial < 800; trial++) {
        const lon = west + Math.random() * (east - west);
        const lat = south + Math.random() * (north - south);

        if (vectorAt(lon, lat)) return { lon, lat };
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

    if (!currentU || !currentV) return;

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
        if (!particleRunning || !shouldDrawCurrentParticles()) {
            particleAnimId = requestAnimationFrame(step);
            return;
        }

        const size = map.getSize();

        currentCtx.globalCompositeOperation = "destination-in";
        currentCtx.fillStyle = "rgba(0, 0, 0, 0.92)";
        currentCtx.fillRect(0, 0, size.x, size.y);

        currentCtx.globalCompositeOperation = "source-over";
        currentCtx.lineWidth = 1.2;

        if (particles.length < particleCount * 0.60) {
            resetParticles();
        }

        for (const p of particles) {
            if (!p || p.age > p.maxAge) {
                resetParticle(p);
                continue;
            }

            const vec = vectorAt(p.lon, p.lat);

            if (!vec || !Number.isFinite(vec.u) || !Number.isFinite(vec.v)) {
                resetParticle(p);
                continue;
            }

            const oldPoint = latLngToCanvasPoint(p.lat, p.lon);

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

            const newPoint = latLngToCanvasPoint(p.lat, p.lon);

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


function pauseParticlesNoClear() {
    particleRunning = false;

    if (particleAnimId !== null) {
        cancelAnimationFrame(particleAnimId);
        particleAnimId = null;
    }
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
    currentOverlayCheck.checked = true;
    currentOverlayCheck.dataset.userTouched = "";

    currentOverlayCheck.addEventListener("change", () => {
        currentOverlayCheck.dataset.userTouched = "1";
        updateCurrentOverlayAvailability();
        setFrame(currentFrame);
    });

    if (meshOverlayCheck) {
        meshOverlayCheck.addEventListener("change", () => {
            mapInteracting = false;
            coordsDirty = true;
            renderViewportLayers(true);
        });
    }

    varSelect.addEventListener("change", e => {
        currentVar = e.target.value;
        updateCurrentOverlayAvailability();
        updateLegend();
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

    frameSlider.addEventListener("input", e => {
        setFrame(e.target.value);
    });

    speedSelect.addEventListener("change", e => {
        speed = parseFloat(e.target.value);
        if (timer !== null) startTimer();
    });

    opacitySlider.addEventListener("input", () => {
        coordsDirty = true;
        scheduleViewportRedraw(true);
    });

    particleDensitySelect.addEventListener("change", e => {
        particleCount = parseInt(e.target.value);
        resetParticles();
        clearCurrentCanvas();
    });
}

async function boot() {
    setStatus("Loading metadata...");

    meta = await fetchJson(CONFIG.metaUrl);
    lookupMeta = await fetchJson(CONFIG.lookupMetaUrl);

    ensureMeshOverlayControl();

    frameSlider.max = frameCount() - 1;

    initMap();

    setStatus("Loading mesh and lookup...");

    [nodes, elems, meshEdges, lookupOffsets, lookupTriangles] = await Promise.all([
        fetchFloat32(CONFIG.nodesUrl, meta.node_count * 2),
        fetchUint32(CONFIG.elemsUrl, meta.index_count),
        fetchUint32(CONFIG.edgesUrl),
        fetchUint32(CONFIG.lookupOffsetsUrl ?? CONFIG.lookupOffsetsUrl),
        fetchUint32(CONFIG.lookupTrianglesUrl ?? CONFIG.lookupTrianglesUrl)
    ]).catch(async () => {
        const a = await fetchFloat32(CONFIG.nodesUrl, meta.node_count * 2);
        const b = await fetchUint32(CONFIG.elemsUrl, meta.index_count);
        const c = await fetchUint32(CONFIG.edgesUrl);
        const d = await fetchUint32("lookup_offsets.bin");
        const e = await fetchUint32("lookup_triangles.bin");
        return [a, b, c, d, e];
    });

    initCanvases();
    initWebGL();
    setupEvents();

    updateCurrentOverlayAvailability();
    updateLegend();

    await setFrame(0);

    drawMeshOverlay();

    setStatus(`Ready: ${meta.node_count.toLocaleString()} nodes, ${meta.triangle_count.toLocaleString()} triangles`);
}

boot().catch(err => {
    console.error(err);
    setStatus("ERROR: " + err.message);
    alert(err.message);
});


(function disableZoomGhostIfAny() {
    const g = document.getElementById("zoom-ghost-canvas");
    if (g) {
        g.style.display = "none";
        g.style.opacity = "0";
    }
})();
