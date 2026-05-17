(function () {
  "use strict";

  const ROOT = "./";

  const state = {
    map: null,
    gl: null,
    canvas: null,

    meta: null,
    nodesLonLat: null,
    screenXY: null,
    elems: null,

    scalarData: null,
    scalarCache: {},
    currentCache: {},

    currentVar: "temperature",
    currentFrame: 0,
    playing: false,
    timer: null,
    speed: 1,

    opacity: 0.85,

    particleCanvas: null,
    particleCtx: null,
    currentData: null,
    particles: [],
    particleAnim: null,
    particleCount: 2800,
    currentOverlay: false,

    program: null,
    posBuffer: null,
    valueBuffer: null,
    elemBuffer: null,

    attribPos: -1,
    attribVal: -1,
    uniResolution: null,
    uniVmin: null,
    uniVmax: null,
    uniOpacity: null,
    uniVarType: null,

    needsPositionUpdate: true,
    needsRender: true,
  };

  function $(id) {
    return document.getElementById(id);
  }

  function clamp(x, a, b) {
    return Math.max(a, Math.min(b, x));
  }

  function getFrameCount() {
    return Number(state.meta.frame_count || state.meta.nframe || state.meta.frames || 24);
  }

  function getLabels() {
    return state.meta.labels || state.meta.times || [];
  }

  function frameUrl(kind, i) {
    const name = "frame_" + String(i).padStart(4, "0") + ".bin";
    if (kind === "temperature") return ROOT + "temp_bin/" + name;
    if (kind === "ssh") return ROOT + "ssh_bin/" + name;
    throw new Error("Unknown scalar kind: " + kind);
  }

  function currentJsonUrl(i) {
    return "../frames_multi/current_grid_json/frame_" + String(i).padStart(4, "0") + ".json";
  }

  async function fetchJson(url) {
    const r = await fetch(url, { cache: "force-cache" });
    if (!r.ok) throw new Error(url + " " + r.status);
    return await r.json();
  }

  async function fetchArrayBuffer(url) {
    const r = await fetch(url, { cache: "force-cache" });
    if (!r.ok) throw new Error(url + " " + r.status);
    return await r.arrayBuffer();
  }

  async function loadMetaAndMesh() {
    state.meta = await fetchJson(ROOT + "mesh_meta.json");

    const nodesUrl = ROOT + (state.meta.nodes_bin || state.meta.mesh_nodes_bin || "mesh_nodes.bin");
    const elemsUrl = ROOT + (state.meta.elems_bin || state.meta.mesh_elems_bin || "mesh_elems.bin");

    const nodeBuf = await fetchArrayBuffer(nodesUrl);
    state.nodesLonLat = new Float32Array(nodeBuf);

    const elemBuf = await fetchArrayBuffer(elemsUrl);
    state.elems = new Uint32Array(elemBuf);

    const nodeCount = Math.floor(state.nodesLonLat.length / 2);
    state.screenXY = new Float32Array(nodeCount * 2);

    console.log("[webgl] nodeCount:", nodeCount, "elemIndexCount:", state.elems.length);
  }


  function firstFinite(values) {
    for (const v of values) {
      const x = Number(v);
      if (Number.isFinite(x)) return x;
    }
    return NaN;
  }

  function getMetaValue(names) {
    const m = state.meta || {};
    for (const name of names) {
      if (m[name] !== undefined && m[name] !== null) return Number(m[name]);
    }
    return NaN;
  }

  function boundsFromMeta() {
    const m = state.meta || {};

    let lonMin = getMetaValue(["lon_min", "lonMin", "xmin", "x_min", "west"]);
    let lonMax = getMetaValue(["lon_max", "lonMax", "xmax", "x_max", "east"]);
    let latMin = getMetaValue(["lat_min", "latMin", "ymin", "y_min", "south"]);
    let latMax = getMetaValue(["lat_max", "latMax", "ymax", "y_max", "north"]);

    const b = m.bounds || m.extent || m.bbox;

    if (Array.isArray(b)) {
      if (b.length === 4) {
        const v = b.map(Number);
        if (v.every(Number.isFinite)) {
          // [lon_min, lon_max, lat_min, lat_max]
          if (Math.abs(v[1]) > 90) {
            lonMin = v[0];
            lonMax = v[1];
            latMin = v[2];
            latMax = v[3];
          }
          // [lon_min, lat_min, lon_max, lat_max]
          else {
            lonMin = v[0];
            latMin = v[1];
            lonMax = v[2];
            latMax = v[3];
          }
        }
      } else if (b.length === 2 && Array.isArray(b[0]) && Array.isArray(b[1])) {
        // [[lat_min, lon_min], [lat_max, lon_max]]
        latMin = Number(b[0][0]);
        lonMin = Number(b[0][1]);
        latMax = Number(b[1][0]);
        lonMax = Number(b[1][1]);
      }
    } else if (b && typeof b === "object") {
      lonMin = firstFinite([b.lon_min, b.lonMin, b.xmin, b.x_min, b.west, lonMin]);
      lonMax = firstFinite([b.lon_max, b.lonMax, b.xmax, b.x_max, b.east, lonMax]);
      latMin = firstFinite([b.lat_min, b.latMin, b.ymin, b.y_min, b.south, latMin]);
      latMax = firstFinite([b.lat_max, b.latMax, b.ymax, b.y_max, b.north, latMax]);
    }

    if ([lonMin, lonMax, latMin, latMax].every(Number.isFinite)) {
      return {
        lonMin: Math.min(lonMin, lonMax),
        lonMax: Math.max(lonMin, lonMax),
        latMin: Math.min(latMin, latMax),
        latMax: Math.max(latMin, latMax)
      };
    }

    return null;
  }

  function boundsFromNodes() {
    if (!state.nodesLonLat || state.nodesLonLat.length < 4) return null;

    function scan(swapped) {
      let lonMin = Infinity;
      let lonMax = -Infinity;
      let latMin = Infinity;
      let latMax = -Infinity;
      let valid = 0;

      for (let i = 0; i < state.nodesLonLat.length / 2; i++) {
        const a = Number(state.nodesLonLat[i * 2]);
        const b = Number(state.nodesLonLat[i * 2 + 1]);

        const lon = swapped ? b : a;
        const lat = swapped ? a : b;

        if (!Number.isFinite(lon) || !Number.isFinite(lat)) continue;
        if (lon < -180 || lon > 180 || lat < -90 || lat > 90) continue;

        lonMin = Math.min(lonMin, lon);
        lonMax = Math.max(lonMax, lon);
        latMin = Math.min(latMin, lat);
        latMax = Math.max(latMax, lat);
        valid += 1;
      }

      if (valid <= 0) return null;

      return { lonMin, lonMax, latMin, latMax, valid };
    }

    const normal = scan(false);
    const swapped = scan(true);

    if (normal && !swapped) return normal;
    if (!normal && swapped) {
      console.warn("[webgl] mesh_nodes.bin looks like [lat, lon]. Using swapped interpretation for bounds.");
      state.nodesAreLatLon = true;
      return swapped;
    }

    if (normal && swapped) {
      // For Korea/Japan domain, normal lon range should generally be much larger than lat range.
      const normalScore = normal.valid;
      const swappedScore = swapped.valid;

      if (swappedScore > normalScore) {
        console.warn("[webgl] using swapped [lat, lon] node interpretation for bounds.");
        state.nodesAreLatLon = true;
        return swapped;
      }

      state.nodesAreLatLon = false;
      return normal;
    }

    return null;
  }

  function getDomainBounds() {
    let b = boundsFromMeta();

    if (!b) {
      console.warn("[webgl] mesh_meta.json bounds not found. Computing bounds from mesh_nodes.bin.");
      b = boundsFromNodes();
    }

    if (!b) {
      console.error("[webgl] meta:", state.meta);
      console.error("[webgl] first node values:", state.nodesLonLat ? Array.from(state.nodesLonLat.slice(0, 10)) : null);
      throw new Error("Cannot determine valid lon/lat bounds for Leaflet map.");
    }

    if (![b.lonMin, b.lonMax, b.latMin, b.latMax].every(Number.isFinite)) {
      throw new Error("Domain bounds contain NaN.");
    }

    return b;
  }

  
function initMap() {
    const b = getDomainBounds();

    const bounds = [
      [b.latMin, b.lonMin],
      [b.latMax, b.lonMax]
    ];

    const centerLat = (b.latMin + b.latMax) * 0.5;
    const centerLon = (b.lonMin + b.lonMax) * 0.5;

    console.log("[webgl] domain bounds:", b);
    console.log("[webgl] map center:", centerLat, centerLon);

    state.map = L.map("map", {
      center: [centerLat, centerLon],
      zoom: 7,
      preferCanvas: true
    });

    const carto = L.tileLayer(
      "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
      { attribution: "&copy; OpenStreetMap contributors &copy; CARTO" }
    );

    const esri = L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      { attribution: "Tiles &copy; Esri" }
    );

    esri.addTo(state.map);

    L.control.layers({
      "CartoDB Positron": carto,
      "Esri Satellite": esri
    }, null, { collapsed: false }).addTo(state.map);

    state.map.fitBounds(bounds);

    state.map.on("move zoom resize zoomend moveend", function () {
      state.needsPositionUpdate = true;
      state.needsRender = true;
      resizeCanvases();
      resetParticles();
      requestRender();
    });
  }


  function createCanvasLayer() {
    const container = state.map.getContainer();
    if (getComputedStyle(container).position === "static") {
      container.style.position = "relative";
    }

    state.canvas = document.createElement("canvas");
    state.canvas.id = "webgl-scalar-canvas";
    state.canvas.style.position = "absolute";
    state.canvas.style.left = "0";
    state.canvas.style.top = "0";
    state.canvas.style.width = "100%";
    state.canvas.style.height = "100%";
    state.canvas.style.pointerEvents = "none";
    state.canvas.style.zIndex = "600";

    state.particleCanvas = document.createElement("canvas");
    state.particleCanvas.id = "webgl-current-particle-canvas";
    state.particleCanvas.style.position = "absolute";
    state.particleCanvas.style.left = "0";
    state.particleCanvas.style.top = "0";
    state.particleCanvas.style.width = "100%";
    state.particleCanvas.style.height = "100%";
    state.particleCanvas.style.pointerEvents = "none";
    state.particleCanvas.style.zIndex = "900";

    container.appendChild(state.canvas);
    container.appendChild(state.particleCanvas);

    state.gl = state.canvas.getContext("webgl2", {
      alpha: true,
      antialias: true,
      premultipliedAlpha: false,
      preserveDrawingBuffer: false
    });

    if (!state.gl) {
      alert("WebGL2 not supported in this browser.");
      throw new Error("WebGL2 not supported");
    }

    state.particleCtx = state.particleCanvas.getContext("2d");

    resizeCanvases();
  }

  function resizeCanvases() {
    if (!state.map || !state.canvas) return;

    const size = state.map.getSize();
    const dpr = window.devicePixelRatio || 1;

    for (const c of [state.canvas, state.particleCanvas]) {
      c.width = Math.max(1, Math.round(size.x * dpr));
      c.height = Math.max(1, Math.round(size.y * dpr));
      c.style.width = size.x + "px";
      c.style.height = size.y + "px";
    }

    state.gl.viewport(0, 0, state.canvas.width, state.canvas.height);

    state.particleCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function compileShader(type, src) {
    const gl = state.gl;
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);

    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(sh));
    }

    return sh;
  }

  function initWebGL() {
    const gl = state.gl;

    const vs = `#version 300 es
      precision highp float;

      in vec2 a_pos;
      in float a_value;

      uniform vec2 u_resolution;

      out float v_value;

      void main() {
        vec2 zeroToOne = a_pos / u_resolution;
        vec2 clip = zeroToOne * 2.0 - 1.0;

        gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
        v_value = a_value;
      }
    `;

    const fs = `#version 300 es
      precision highp float;

      in float v_value;

      uniform float u_vmin;
      uniform float u_vmax;
      uniform float u_opacity;
      uniform int u_var_type;

      out vec4 outColor;

      vec3 jet(float t) {
        t = clamp(t, 0.0, 1.0);
        float r = clamp(min(4.0 * t - 1.5, -4.0 * t + 4.5), 0.0, 1.0);
        float g = clamp(min(4.0 * t - 0.5, -4.0 * t + 3.5), 0.0, 1.0);
        float b = clamp(min(4.0 * t + 0.5, -4.0 * t + 2.5), 0.0, 1.0);
        return vec3(r, g, b);
      }

      vec3 rdBu(float t) {
        t = clamp(t, 0.0, 1.0);

        vec3 c0 = vec3(0.031, 0.188, 0.419);
        vec3 c1 = vec3(0.420, 0.682, 0.839);
        vec3 c2 = vec3(0.970, 0.970, 0.970);
        vec3 c3 = vec3(0.984, 0.416, 0.290);
        vec3 c4 = vec3(0.404, 0.000, 0.051);

        if (t < 0.25) return mix(c0, c1, t / 0.25);
        if (t < 0.50) return mix(c1, c2, (t - 0.25) / 0.25);
        if (t < 0.75) return mix(c2, c3, (t - 0.50) / 0.25);
        return mix(c3, c4, (t - 0.75) / 0.25);
      }

      void main() {
        if (!isfinite(v_value) || v_value < -9000.0) {
          discard;
        }

        float t = (v_value - u_vmin) / (u_vmax - u_vmin);
        t = clamp(t, 0.0, 1.0);

        vec3 color;
        if (u_var_type == 0) {
          color = jet(t);
        } else {
          color = rdBu(t);
        }

        outColor = vec4(color, u_opacity);
      }
    `;

    const prog = gl.createProgram();
    gl.attachShader(prog, compileShader(gl.VERTEX_SHADER, vs));
    gl.attachShader(prog, compileShader(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(prog);

    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(prog));
    }

    state.program = prog;
    gl.useProgram(prog);

    state.attribPos = gl.getAttribLocation(prog, "a_pos");
    state.attribVal = gl.getAttribLocation(prog, "a_value");
    state.uniResolution = gl.getUniformLocation(prog, "u_resolution");
    state.uniVmin = gl.getUniformLocation(prog, "u_vmin");
    state.uniVmax = gl.getUniformLocation(prog, "u_vmax");
    state.uniOpacity = gl.getUniformLocation(prog, "u_opacity");
    state.uniVarType = gl.getUniformLocation(prog, "u_var_type");

    state.posBuffer = gl.createBuffer();
    state.valueBuffer = gl.createBuffer();
    state.elemBuffer = gl.createBuffer();

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, state.elemBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, state.elems, gl.STATIC_DRAW);

    gl.enableVertexAttribArray(state.attribPos);
    gl.bindBuffer(gl.ARRAY_BUFFER, state.posBuffer);
    gl.vertexAttribPointer(state.attribPos, 2, gl.FLOAT, false, 0, 0);

    gl.enableVertexAttribArray(state.attribVal);
    gl.bindBuffer(gl.ARRAY_BUFFER, state.valueBuffer);
    gl.vertexAttribPointer(state.attribVal, 1, gl.FLOAT, false, 0, 0);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  }

  function updateScreenPositions() {
    const n = state.nodesLonLat.length / 2;

    for (let i = 0; i < n; i++) {
      const a = state.nodesLonLat[i * 2];
      const b = state.nodesLonLat[i * 2 + 1];
      const lon = state.nodesAreLatLon ? b : a;
      const lat = state.nodesAreLatLon ? a : b;
      const pt = state.map.latLngToContainerPoint([lat, lon]);

      state.screenXY[i * 2] = pt.x;
      state.screenXY[i * 2 + 1] = pt.y;
    }

    const gl = state.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, state.posBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, state.screenXY, gl.DYNAMIC_DRAW);

    state.needsPositionUpdate = false;
  }

  function getScalarRange(kind) {
    if (kind === "temperature") {
      return [
        Number(state.meta.temp_vmin ?? state.meta.temperature_vmin ?? 0.0),
        Number(state.meta.temp_vmax ?? state.meta.temperature_vmax ?? 32.0)
      ];
    }

    return [
      Number(state.meta.ssh_vmin ?? state.meta.elevation_vmin ?? -1.0),
      Number(state.meta.ssh_vmax ?? state.meta.elevation_vmax ?? 1.0)
    ];
  }

  async function loadScalarFrame(kind, frame) {
    const key = kind + ":" + frame;

    if (state.scalarCache[key]) {
      state.scalarData = state.scalarCache[key];
      uploadScalarData();
      return;
    }

    const buf = await fetchArrayBuffer(frameUrl(kind, frame));
    const arr = new Float32Array(buf);
    state.scalarCache[key] = arr;
    state.scalarData = arr;

    uploadScalarData();

    preloadScalar(kind, Math.max(0, frame - 1));
    preloadScalar(kind, Math.min(getFrameCount() - 1, frame + 1));
  }

  function preloadScalar(kind, frame) {
    const key = kind + ":" + frame;
    if (state.scalarCache[key]) return;

    fetchArrayBuffer(frameUrl(kind, frame))
      .then(buf => {
        state.scalarCache[key] = new Float32Array(buf);
      })
      .catch(() => {});
  }

  function uploadScalarData() {
    const gl = state.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, state.valueBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, state.scalarData, gl.DYNAMIC_DRAW);
    state.needsRender = true;
  }

  function renderScalar() {
    if (!state.scalarData) return;

    const gl = state.gl;

    if (state.needsPositionUpdate) {
      updateScreenPositions();
    }

    gl.viewport(0, 0, state.canvas.width, state.canvas.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(state.program);

    const dpr = window.devicePixelRatio || 1;
    const size = state.map.getSize();

    gl.uniform2f(state.uniResolution, size.x, size.y);

    const [vmin, vmax] = getScalarRange(state.currentVar);
    gl.uniform1f(state.uniVmin, vmin);
    gl.uniform1f(state.uniVmax, vmax);
    gl.uniform1f(state.uniOpacity, state.opacity);
    gl.uniform1i(state.uniVarType, state.currentVar === "temperature" ? 0 : 1);

    gl.bindBuffer(gl.ARRAY_BUFFER, state.posBuffer);
    gl.vertexAttribPointer(state.attribPos, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, state.valueBuffer);
    gl.vertexAttribPointer(state.attribVal, 1, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, state.elemBuffer);
    gl.drawElements(gl.TRIANGLES, state.elems.length, gl.UNSIGNED_INT, 0);

    state.needsRender = false;
  }

  function updateLegend() {
    const box = $("legend-box") || $("legend");
    if (!box) return;

    if (state.currentVar === "temperature") {
      box.innerHTML = `
        <div style="font-weight:bold; margin-bottom:6px;">Surface Temperature [degC]</div>
        <div style="width:220px; height:16px; background:linear-gradient(to right, blue, cyan, yellow, red); border:1px solid #666;"></div>
        <div style="display:flex; justify-content:space-between; font-size:12px; margin-top:4px;">
          <span>0</span><span>8</span><span>16</span><span>24</span><span>32</span>
        </div>`;
    } else if (state.currentVar === "ssh") {
      box.innerHTML = `
        <div style="font-weight:bold; margin-bottom:6px;">Elevation [m]</div>
        <div style="width:220px; height:16px; background:linear-gradient(to right, #08306b, #6baed6, #f7f7f7, #fb6a4a, #67000d); border:1px solid #666;"></div>
        <div style="display:flex; justify-content:space-between; font-size:12px; margin-top:4px;">
          <span>-1.0</span><span>-0.5</span><span>0</span><span>0.5</span><span>1.0</span>
        </div>`;
    } else {
      box.innerHTML = `
        <div style="font-weight:bold; margin-bottom:6px;">Current Speed [m/s]</div>
        <div style="width:220px; height:16px; background:linear-gradient(to right, #000080, #0000ff, #00ffff, #ffff00, #ff0000, #800000); border:1px solid #666;"></div>
        <div style="display:flex; justify-content:space-between; font-size:12px; margin-top:4px;">
          <span>0</span><span>0.25</span><span>0.5</span><span>0.75</span><span>1.0</span>
        </div>`;
    }
  }

  function updateTimeLabel() {
    const labels = getLabels();
    const el = $("time-label");
    if (!el) return;
    el.innerHTML = labels[state.currentFrame] || String(state.currentFrame);
  }

  async function setFrame(i) {
    const n = getFrameCount();

    state.currentFrame = clamp(parseInt(i), 0, n - 1);

    const slider = $("frame-slider");
    if (slider) slider.value = state.currentFrame;

    updateTimeLabel();

    if (state.currentVar === "current") {
      state.canvas.style.display = "none";
      clearParticleCanvas();
      await loadCurrentFrame(state.currentFrame);
      resetParticles();
      startParticles();
      updateLegend();
      return;
    }

    state.canvas.style.display = "block";

    await loadScalarFrame(state.currentVar, state.currentFrame);
    renderScalar();

    if (state.currentOverlay) {
      await loadCurrentFrame(state.currentFrame);
      resetParticles();
      startParticles();
    } else {
      stopParticles();
    }

    updateLegend();
  }

  function startTimer() {
    stopTimer();

    const interval = 1000 / state.speed;
    state.timer = setInterval(() => {
      const n = getFrameCount();
      setFrame((state.currentFrame + 1) % n);
    }, interval);
  }

  function stopTimer() {
    if (state.timer !== null) {
      clearInterval(state.timer);
      state.timer = null;
    }
  }

  function setupControls() {
    const frameCount = getFrameCount();

    const slider = $("frame-slider");
    if (slider) {
      slider.min = 0;
      slider.max = frameCount - 1;
      slider.value = 0;
      slider.addEventListener("input", e => setFrame(e.target.value));
    }

    const varSel = $("var-select");
    if (varSel) {
      varSel.addEventListener("change", e => {
        state.currentVar = e.target.value;
        const overlay = $("current-overlay-check");

        if (state.currentVar === "current" && overlay) {
          overlay.checked = false;
          overlay.disabled = true;
          state.currentOverlay = false;
        } else if (overlay) {
          overlay.disabled = false;
        }

        setFrame(state.currentFrame);
      });
    }

    const overlay = $("current-overlay-check");
    if (overlay) {
      overlay.addEventListener("change", e => {
        state.currentOverlay = e.target.checked;
        setFrame(state.currentFrame);
      });
    }

    const playBtn = $("play-btn");
    if (playBtn) {
      playBtn.addEventListener("click", () => {
        state.playing = !state.playing;
        playBtn.innerHTML = state.playing ? "Pause" : "Play";

        if (state.playing) startTimer();
        else stopTimer();
      });
    }

    const speedSel = $("speed-select");
    if (speedSel) {
      speedSel.addEventListener("change", e => {
        state.speed = Number(e.target.value || 1);
        if (state.playing) startTimer();
      });
    }

    const opacity = $("opacity-slider");
    if (opacity) {
      opacity.addEventListener("input", e => {
        state.opacity = Number(e.target.value);
        state.needsRender = true;
        requestRender();
      });
    }

    const density = $("particle-density-select");
    if (density) {
      state.particleCount = Number(density.value || 2800);
      density.addEventListener("change", e => {
        state.particleCount = Number(e.target.value || 2800);
        resetParticles();
      });
    }
  }

  function requestRender() {
    requestAnimationFrame(() => {
      if (state.currentVar !== "current") {
        renderScalar();
      }
    });
  }

  function normalizeCurrentData(j) {
    return j;
  }

  async function loadCurrentFrame(i) {
    const url = currentJsonUrl(i);

    if (state.currentCache[url]) {
      state.currentData = state.currentCache[url];
      preloadCurrent(Math.min(getFrameCount() - 1, i + 1));
      return;
    }

    const j = normalizeCurrentData(await fetchJson(url));
    state.currentCache[url] = j;
    state.currentData = j;

    preloadCurrent(Math.min(getFrameCount() - 1, i + 1));
  }

  function preloadCurrent(i) {
    const url = currentJsonUrl(i);
    if (state.currentCache[url]) return;

    fetchJson(url)
      .then(j => {
        state.currentCache[url] = normalizeCurrentData(j);
      })
      .catch(() => {});
  }

  function currentGridInfo() {
    const d = state.currentData;
    if (!d) return null;

    return {
      nx: d.nx,
      ny: d.ny,
      lonMin: d.lon_min,
      lonMax: d.lon_max,
      latMin: d.lat_min,
      latMax: d.lat_max,
      invalid: d.invalid
    };
  }

  function cidx(ix, iy, nx) {
    return iy * nx + ix;
  }

  function isCurrentValid(v) {
    const d = state.currentData;
    return Number.isFinite(v) && v !== d.invalid;
  }

  function vectorAt(lon, lat) {
    const d = state.currentData;
    if (!d) return null;

    const g = currentGridInfo();
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

    const i00 = cidx(ix0, iy0, g.nx);
    const i10 = cidx(ix1, iy0, g.nx);
    const i01 = cidx(ix0, iy1, g.nx);
    const i11 = cidx(ix1, iy1, g.nx);

    const u00 = d.u[i00], u10 = d.u[i10], u01 = d.u[i01], u11 = d.u[i11];
    const v00 = d.v[i00], v10 = d.v[i10], v01 = d.v[i01], v11 = d.v[i11];

    if (
      !isCurrentValid(u00) || !isCurrentValid(u10) ||
      !isCurrentValid(u01) || !isCurrentValid(u11) ||
      !isCurrentValid(v00) || !isCurrentValid(v10) ||
      !isCurrentValid(v01) || !isCurrentValid(v11)
    ) {
      return null;
    }

    const w00 = (1 - tx) * (1 - ty);
    const w10 = tx * (1 - ty);
    const w01 = (1 - tx) * ty;
    const w11 = tx * ty;

    const u = w00 * u00 + w10 * u10 + w01 * u01 + w11 * u11;
    const v = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11;
    const speed = Math.hypot(u, v);

    return { u, v, speed };
  }

  function randomCurrentPoint() {
    const d = state.currentData;
    if (!d) return null;

    const g = currentGridInfo();
    if (!g) return null;

    const b = state.map.getBounds();
    const west = Math.max(b.getWest(), g.lonMin);
    const east = Math.min(b.getEast(), g.lonMax);
    const south = Math.max(b.getSouth(), g.latMin);
    const north = Math.min(b.getNorth(), g.latMax);

    if (west >= east || south >= north) return null;

    for (let k = 0; k < 80; k++) {
      const lon = west + Math.random() * (east - west);
      const lat = south + Math.random() * (north - south);
      if (vectorAt(lon, lat)) {
        return { lon, lat, age: Math.floor(Math.random() * 80), maxAge: 90 + Math.floor(Math.random() * 80) };
      }
    }

    return null;
  }

  function resetParticles() {
    state.particles = [];
    if (!state.currentData) return;

    for (let i = 0; i < state.particleCount; i++) {
      const p = randomCurrentPoint();
      if (p) state.particles.push(p);
    }
  }

  function currentJet(speed) {
    const d = state.currentData;
    let t = (speed - d.vmin) / (d.vmax - d.vmin);
    if (!Number.isFinite(t)) t = 0;
    t = clamp(t, 0, 1);

    const four = 4.0 * t;
    const r = Math.round(255 * clamp(Math.min(four - 1.5, -four + 4.5), 0, 1));
    const g = Math.round(255 * clamp(Math.min(four - 0.5, -four + 3.5), 0, 1));
    const b = Math.round(255 * clamp(Math.min(four + 0.5, -four + 2.5), 0, 1));

    return `rgba(${r},${g},${b},0.92)`;
  }

  function particleColor(speed) {
    if (state.currentVar === "current") {
      return currentJet(speed);
    }
    return "rgba(255,255,255,0.96)";
  }

  function clearParticleCanvas() {
    if (!state.particleCtx) return;
    const size = state.map.getSize();
    state.particleCtx.clearRect(0, 0, size.x, size.y);
  }

  function startParticles() {
    if (state.particleAnim !== null) {
      cancelAnimationFrame(state.particleAnim);
      state.particleAnim = null;
    }

    const step = () => {
      if (!state.currentData || (state.currentVar !== "current" && !state.currentOverlay)) {
        clearParticleCanvas();
        state.particleAnim = requestAnimationFrame(step);
        return;
      }

      drawParticles();
      state.particleAnim = requestAnimationFrame(step);
    };

    state.particleAnim = requestAnimationFrame(step);
  }

  function stopParticles() {
    if (state.particleAnim !== null) {
      cancelAnimationFrame(state.particleAnim);
      state.particleAnim = null;
    }
    clearParticleCanvas();
  }

  function drawParticles() {
    const ctx = state.particleCtx;
    const size = state.map.getSize();

    ctx.globalCompositeOperation = "destination-in";
    ctx.fillStyle = "rgba(0,0,0,0.92)";
    ctx.fillRect(0, 0, size.x, size.y);

    ctx.globalCompositeOperation = "source-over";
    ctx.lineWidth = 1.2;

    if (state.particles.length < state.particleCount * 0.65) {
      resetParticles();
    }

    const flowScale = 0.018 * state.speed;

    for (let i = 0; i < state.particles.length; i++) {
      const p = state.particles[i];

      if (!p || p.age > p.maxAge) {
        state.particles[i] = randomCurrentPoint();
        continue;
      }

      const vec = vectorAt(p.lon, p.lat);
      if (!vec) {
        state.particles[i] = randomCurrentPoint();
        continue;
      }

      const oldPt = state.map.latLngToContainerPoint([p.lat, p.lon]);

      const latRad = p.lat * Math.PI / 180.0;
      let coslat = Math.cos(latRad);
      if (Math.abs(coslat) < 1e-6) coslat = 1e-6;

      const newLon = p.lon + (vec.u * flowScale) / coslat;
      const newLat = p.lat + vec.v * flowScale;

      if (!vectorAt(newLon, newLat)) {
        state.particles[i] = randomCurrentPoint();
        continue;
      }

      p.lon = newLon;
      p.lat = newLat;
      p.age += 1;

      const newPt = state.map.latLngToContainerPoint([p.lat, p.lon]);

      if (
        newPt.x < -50 || newPt.x > size.x + 50 ||
        newPt.y < -50 || newPt.y > size.y + 50
      ) {
        state.particles[i] = randomCurrentPoint();
        continue;
      }

      ctx.strokeStyle = particleColor(vec.speed);
      ctx.beginPath();
      ctx.moveTo(oldPt.x, oldPt.y);
      ctx.lineTo(newPt.x, newPt.y);
      ctx.stroke();
    }
  }

  async function main() {
    await loadMetaAndMesh();

    initMap();
    createCanvasLayer();
    initWebGL();
    setupControls();

    updateScreenPositions();
    await setFrame(0);

    window.addEventListener("resize", () => {
      resizeCanvases();
      state.needsPositionUpdate = true;
      state.needsRender = true;
      requestRender();
    });

    requestAnimationFrame(function loop() {
      if (state.needsRender && state.currentVar !== "current") {
        renderScalar();
      }
      requestAnimationFrame(loop);
    });
  }

  main().catch(err => {
    console.error(err);
    alert("WebGL viewer error: " + err.message);
  });
})();
