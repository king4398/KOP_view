"use strict";

const DATA_ROOT = "../webgl_maplibre/data/";
const CONFIG = {
  metaUrl: DATA_ROOT + "mesh_meta.json",
  nodesUrl: DATA_ROOT + "mesh_nodes.bin",
  elemsUrl: DATA_ROOT + "mesh_elems.bin",
  edgesUrl: DATA_ROOT + "mesh_edges.bin",
  lookupMetaUrl: DATA_ROOT + "lookup_meta.json",
  lookupOffsetsUrl: DATA_ROOT + "lookup_offsets.bin",
  lookupTrianglesUrl: DATA_ROOT + "lookup_triangles.bin",
  flowScale: 0.005,
  overlayParticleColor: "rgba(235,235,235,0.55)"
};

const els = {
  varSelect: document.getElementById("var-select"), playBtn: document.getElementById("play-btn"),
  frameSlider: document.getElementById("frame-slider"), speedSelect: document.getElementById("speed-select"),
  opacitySlider: document.getElementById("opacity-slider"), densitySelect: document.getElementById("particle-density-select"),
  currentOverlay: document.getElementById("current-overlay-check"), meshOverlay: document.getElementById("mesh-overlay-check"),
  timeLabel: document.getElementById("time-label"), legendBox: document.getElementById("legend-box"), statusLine: document.getElementById("status-line")
};
let map, meta, lookupMeta, nodesLonLat, nodesMerc, elems, meshEdges, lookupOffsets, lookupTriangles;
let currentVar = "temperature", currentFrame = 0, speed = 1.0, timer = null;
let scalarCache = new Map(), scalarLoading = new Map(), currentCache = new Map(), currentU = null, currentV = null;
let particleMapInteracting = false;
let particleCanvas, particleCtx, particleAnimId = null, particleRunning = false, particleCount = 2800, particles = [];

const GLState = { gl:null, scalarProgram:null, meshProgram:null, nodeBuffer:null, valueBuffer:null, elemBuffer:null, meshNodeBuffer:null, meshEdgeBuffer:null,
  aPos:null, aVal:null, uMatrix:null, uVmin:null, uVmax:null, uOpacity:null, uCmap:null, uInvalid:null,
  meshAPos:null, meshUMatrix:null, meshUColor:null, ready:false };

function setStatus(msg){ if(els.statusLine) els.statusLine.textContent = msg; }
function pad4(i){ return String(i).padStart(4,"0"); }
function frameCount(){ return meta && meta.frames ? meta.frames.length : 0; }
function scalarFrameUrl(v,i){ if(v==="temperature") return DATA_ROOT+`temp_bin/frame_${pad4(i)}.bin`; if(v==="salinity") return DATA_ROOT+`salinity_bin/frame_${pad4(i)}.bin`; if(v==="ssh") return DATA_ROOT+`ssh_bin/frame_${pad4(i)}.bin`; return null; }
function currentUUrl(i){ return DATA_ROOT+`current_u_bin/frame_${pad4(i)}.bin`; }
function currentVUrl(i){ return DATA_ROOT+`current_v_bin/frame_${pad4(i)}.bin`; }
function variableMeta(v){ if(v==="temperature") return meta.variables.temperature; if(v==="salinity") return meta.variables.salinity; if(v==="ssh") return meta.variables.ssh; return null; }
function currentOverlayEnabled(){ return els.currentOverlay && els.currentOverlay.checked && (currentVar==="temperature" || currentVar==="salinity" || currentVar==="ssh"); }
function shouldDrawCurrentParticles(){ return currentVar==="current" || currentOverlayEnabled(); }
function updateCurrentOverlayAvailability(){ if(!els.currentOverlay) return; if(currentVar==="current"){ els.currentOverlay.checked=false; els.currentOverlay.disabled=true; } else { els.currentOverlay.disabled=false; if(!els.currentOverlay.dataset.userTouched) els.currentOverlay.checked=true; } }
async function fetchJson(url){ const sep = url.includes("?") ? "&" : "?"; const r=await fetch(url + sep + "v=july_meta_nostore_01", {cache:"no-store"}); if(!r.ok) throw new Error(`${url}: ${r.status}`); return await r.json(); }
async function fetchArrayBuffer(url){ const r=await fetch(url,{cache:"force-cache"}); if(!r.ok) throw new Error(`${url}: ${r.status}`); return await r.arrayBuffer(); }
async function fetchFloat32(url,n=null){ const a=new Float32Array(await fetchArrayBuffer(url)); if(n!==null && a.length!==n) throw new Error(`${url}: ${a.length} != ${n}`); return a; }
async function fetchUint32(url,n=null){ const a=new Uint32Array(await fetchArrayBuffer(url)); if(n!==null && a.length!==n) throw new Error(`${url}: ${a.length} != ${n}`); return a; }


function halfToFloat(h) {
    const sign = (h & 0x8000) ? -1 : 1;
    const exp = (h >> 10) & 0x1f;
    const frac = h & 0x03ff;

    if (exp === 0) {
        if (frac === 0) return sign * 0.0;
        return sign * Math.pow(2, -14) * (frac / 1024.0);
    }

    if (exp === 31) {
        return frac ? NaN : sign * Infinity;
    }

    return sign * Math.pow(2, exp - 15) * (1.0 + frac / 1024.0);
}

async function fetchFloat16AsFloat32(url, expectedLen){
  const res = await fetch(url, {cache:"no-store"});
  if(!res.ok) throw new Error(`fetch failed ${url}: ${res.status}`);

  const buf = await res.arrayBuffer();
  const h = new Uint16Array(buf);

  // Float16: 1 value = 2 bytes
  // expectedLen is node_count, so compare with Uint16Array length, not byteLength.
  if(expectedLen != null && h.length !== expectedLen){
    throw new Error(`Float16 value length ${h.length} != expected ${expectedLen} for ${url}; byteLength=${buf.byteLength}`);
  }

  const out = new Float32Array(h.length);
  for(let i=0; i<h.length; i++){
    out[i] = halfToFloat(h[i]);
  }
  return out;
}


function fmtLegendNumber(x, digits=1){
  const n = Number(x);
  if(!Number.isFinite(n)) return String(x);
  if(Math.abs(n - Math.round(n)) < 1e-9) return String(Math.round(n));
  return n.toFixed(digits).replace(/\.?0+$/,"");
}

function cmapCode(name){
  const c = String(name || "").toLowerCase();
  if(c === "rdbu" || c === "bluewhitered" || c === "bwr") return 1;
  if(c === "ylgnbu" || c === "ylgnbbu" || c === "ylognbu") return 2;
  return 0;
}

function updateLegend(){
  if(!els.legendBox || !meta || !meta.variables) return;

  const jetGrad = "linear-gradient(to right,#0000cc,#0066ff,#00ccff,#00cc66,#ffff00,#ff9900,#cc0000)";
  const elevGrad = "linear-gradient(to right,#0000cc,#ffffff,#cc0000)";
  const currentGrad = jetGrad;
  const ylgnbuGrad = "linear-gradient(to right,#ffffcc,#c7e9b4,#7fcdbb,#41b6c4,#2c7fb8,#253494)";

  function box(title, grad, a, b, c){
    els.legendBox.innerHTML =
      `<div class="legend-title">${title}</div>` +
      `<div style="height:14px; width:100%; min-width:190px; margin:6px 0 4px 0; border-radius:3px; background:${grad};"></div>` +
      `<div class="legend-ticks" style="display:flex; justify-content:space-between; gap:12px;">` +
      `<span>${a}</span><span>${b}</span><span>${c}</span>` +
      `</div>`;
  }

  if(currentVar==="temperature"){
    const v = meta.variables.temperature || {vmin:0, vmax:32};
    box("Temperature [degC]", jetGrad,
        fmtLegendNumber(v.vmin),
        fmtLegendNumber((v.vmin+v.vmax)/2),
        fmtLegendNumber(v.vmax));
  } else if(currentVar==="salinity"){
    const v = meta.variables.salinity || {vmin:25, vmax:35};
    box("Salinity [psu]", ylgnbuGrad,
        fmtLegendNumber(v.vmin),
        fmtLegendNumber((v.vmin+v.vmax)/2),
        fmtLegendNumber(v.vmax));
  } else if(currentVar==="ssh"){
    const v = meta.variables.ssh || {vmin:-1, vmax:1};
    box("Elevation [m]", elevGrad,
        fmtLegendNumber(v.vmin,2),
        fmtLegendNumber((v.vmin+v.vmax)/2,2),
        fmtLegendNumber(v.vmax,2));
  } else {
    box("Current Speed [m/s]", currentGrad, "0", "0.5", "1");
  }
}

function compileShader(gl,type,src){ const sh=gl.createShader(type); gl.shaderSource(sh,src); gl.compileShader(sh); if(!gl.getShaderParameter(sh,gl.COMPILE_STATUS)){ const log=gl.getShaderInfoLog(sh); gl.deleteShader(sh); throw new Error(log); } return sh; }
function makeProgram(gl,vs,fs){ const p=gl.createProgram(); const a=compileShader(gl,gl.VERTEX_SHADER,vs); const b=compileShader(gl,gl.FRAGMENT_SHADER,fs); gl.attachShader(p,a); gl.attachShader(p,b); gl.linkProgram(p); gl.deleteShader(a); gl.deleteShader(b); if(!gl.getProgramParameter(p,gl.LINK_STATUS)){ const log=gl.getProgramInfoLog(p); gl.deleteProgram(p); throw new Error(log); } return p; }
const SCALAR_VS=`precision highp float; attribute vec2 a_pos; attribute float a_value; uniform mat4 u_matrix; varying float v_value; void main(){ gl_Position=u_matrix*vec4(a_pos,0.0,1.0); v_value=a_value; }`;
const SCALAR_FS=`precision highp float; varying float v_value; uniform float u_vmin; uniform float u_vmax; uniform float u_opacity; uniform int u_cmap; uniform float u_invalid; vec3 jet(float t){t=clamp(t,0.0,1.0);float r=clamp(min(4.0*t-1.5,-4.0*t+4.5),0.0,1.0);float g=clamp(min(4.0*t-0.5,-4.0*t+3.5),0.0,1.0);float b=clamp(min(4.0*t+0.5,-4.0*t+2.5),0.0,1.0);return vec3(r,g,b);} vec3 mix3(vec3 a,vec3 b,float t){return a*(1.0-t)+b*t;} 
vec3 smoothJet(float t) {
    // Smooth jet-like colormap:
    // blue -> cyan -> green -> yellow -> orange -> red
    t = clamp(t, 0.0, 1.0);

    vec3 c0 = vec3(0.05, 0.18, 0.95); // blue
    vec3 c1 = vec3(0.05, 0.62, 1.00); // sky/cyan
    vec3 c2 = vec3(0.10, 0.78, 0.42); // green
    vec3 c3 = vec3(0.92, 0.86, 0.22); // yellow
    vec3 c4 = vec3(0.95, 0.55, 0.10); // orange
    vec3 c5 = vec3(0.82, 0.12, 0.08); // red

    if (t < 0.20) return mix(c0, c1, t / 0.20);
    if (t < 0.40) return mix(c1, c2, (t - 0.20) / 0.20);
    if (t < 0.60) return mix(c2, c3, (t - 0.40) / 0.20);
    if (t < 0.80) return mix(c3, c4, (t - 0.60) / 0.20);
    return mix(c4, c5, (t - 0.80) / 0.20);
}


vec3 ylgnbu(float t) {
    // ColorBrewer-like YlGnBu:
    // low -> high: yellow -> green -> cyan -> blue
    t = clamp(t, 0.0, 1.0);

    vec3 c0 = vec3(1.000, 1.000, 0.800); // #ffffcc
    vec3 c1 = vec3(0.780, 0.914, 0.706); // #c7e9b4
    vec3 c2 = vec3(0.498, 0.804, 0.733); // #7fcdbb
    vec3 c3 = vec3(0.255, 0.714, 0.769); // #41b6c4
    vec3 c4 = vec3(0.173, 0.498, 0.722); // #2c7fb8
    vec3 c5 = vec3(0.145, 0.204, 0.580); // #253494

    if (t < 0.20) return mix(c0, c1, t / 0.20);
    if (t < 0.40) return mix(c1, c2, (t - 0.20) / 0.20);
    if (t < 0.60) return mix(c2, c3, (t - 0.40) / 0.20);
    if (t < 0.80) return mix(c3, c4, (t - 0.60) / 0.20);
    return mix(c4, c5, (t - 0.80) / 0.20);
}


vec3 simpleBlueWhiteRed(float t) {
    // Simple smooth elevation colormap:
    // jet-blue -> white -> jet-red
    t = clamp(t, 0.0, 1.0);

    vec3 blue  = vec3(0.05, 0.18, 0.95); // #0d2ef2
    vec3 white = vec3(0.98, 0.98, 0.96);
    vec3 red   = vec3(0.82, 0.12, 0.08); // #d11f14

    if (t < 0.5) {
        return mix(blue, white, t / 0.5);
    }

    return mix(white, red, (t - 0.5) / 0.5);
}

vec3 smoothRdBu(float t) {
    // Smooth RdBu-like colormap:
    // deep blue -> blue -> light blue -> soft white -> peach -> red -> deep red
    t = clamp(t, 0.0, 1.0);

    vec3 c0 = vec3(0.050, 0.150, 0.420); // deep blue
    vec3 c1 = vec3(0.110, 0.340, 0.700); // blue
    vec3 c2 = vec3(0.420, 0.680, 0.860); // light blue
    vec3 c3 = vec3(0.965, 0.965, 0.940); // soft white
    vec3 c4 = vec3(0.980, 0.700, 0.560); // peach
    vec3 c5 = vec3(0.850, 0.260, 0.180); // red-orange
    vec3 c6 = vec3(0.500, 0.030, 0.090); // deep red

    if (t < 0.1667) return mix(c0, c1, t / 0.1667);
    if (t < 0.3333) return mix(c1, c2, (t - 0.1667) / 0.1666);
    if (t < 0.5000) return mix(c2, c3, (t - 0.3333) / 0.1667);
    if (t < 0.6667) return mix(c3, c4, (t - 0.5000) / 0.1667);
    if (t < 0.8333) return mix(c4, c5, (t - 0.6667) / 0.1666);
    return mix(c5, c6, (t - 0.8333) / 0.1667);
}

vec3 rdbu(float t){t=clamp(t,0.0,1.0);vec3 c0=vec3(0.031,0.188,0.420);vec3 c1=vec3(0.420,0.682,0.839);vec3 c2=vec3(0.969);vec3 c3=vec3(0.984,0.416,0.290);vec3 c4=vec3(0.404,0.0,0.051);if(t<0.25)return mix3(c0,c1,t/0.25);if(t<0.50)return mix3(c1,c2,(t-0.25)/0.25);if(t<0.75)return mix3(c2,c3,(t-0.50)/0.25);return mix3(c3,c4,(t-0.75)/0.25);} void main(){ if((v_value!=v_value)||v_value<=u_invalid+1.0) discard; float den=max(abs(u_vmax-u_vmin),1e-12); float t=clamp((v_value-u_vmin)/den,0.0,1.0); vec3 c = (u_cmap == 1) ? simpleBlueWhiteRed(t) : ((u_cmap == 2) ? ylgnbu(t) : smoothJet(t)); gl_FragColor=vec4(c,u_opacity); }`;
const MESH_VS=`precision highp float; attribute vec2 a_pos; uniform mat4 u_matrix; void main(){ gl_Position=u_matrix*vec4(a_pos,0.0,1.0); }`;
const MESH_FS=`precision highp float; uniform vec4 u_color; void main(){ gl_FragColor=u_color; }`;

function buildMercatorNodes(){ nodesMerc=new Float32Array(meta.node_count*2); for(let i=0;i<meta.node_count;i++){ const lon=nodesLonLat[i*2], lat=nodesLonLat[i*2+1]; const mc=maplibregl.MercatorCoordinate.fromLngLat({lng:lon,lat:lat}); nodesMerc[i*2]=mc.x; nodesMerc[i*2+1]=mc.y; } }
function makeSchismLayer(){ return { id:"schism-custom-layer", type:"custom", renderingMode:"2d", onAdd:function(m,gl){ GLState.gl=gl; gl.getExtension("OES_element_index_uint"); GLState.scalarProgram=makeProgram(gl,SCALAR_VS,SCALAR_FS); GLState.meshProgram=makeProgram(gl,MESH_VS,MESH_FS); GLState.aPos=gl.getAttribLocation(GLState.scalarProgram,"a_pos"); GLState.aVal=gl.getAttribLocation(GLState.scalarProgram,"a_value"); GLState.uMatrix=gl.getUniformLocation(GLState.scalarProgram,"u_matrix"); GLState.uVmin=gl.getUniformLocation(GLState.scalarProgram,"u_vmin"); GLState.uVmax=gl.getUniformLocation(GLState.scalarProgram,"u_vmax"); GLState.uOpacity=gl.getUniformLocation(GLState.scalarProgram,"u_opacity"); GLState.uCmap=gl.getUniformLocation(GLState.scalarProgram,"u_cmap"); GLState.uInvalid=gl.getUniformLocation(GLState.scalarProgram,"u_invalid"); GLState.meshAPos=gl.getAttribLocation(GLState.meshProgram,"a_pos"); GLState.meshUMatrix=gl.getUniformLocation(GLState.meshProgram,"u_matrix"); GLState.meshUColor=gl.getUniformLocation(GLState.meshProgram,"u_color"); GLState.nodeBuffer=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,GLState.nodeBuffer); gl.bufferData(gl.ARRAY_BUFFER,nodesMerc,gl.STATIC_DRAW); GLState.valueBuffer=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,GLState.valueBuffer); gl.bufferData(gl.ARRAY_BUFFER,meta.node_count*4,gl.DYNAMIC_DRAW); GLState.elemBuffer=gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,GLState.elemBuffer); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,elems,gl.STATIC_DRAW); GLState.meshNodeBuffer=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,GLState.meshNodeBuffer); gl.bufferData(gl.ARRAY_BUFFER,nodesMerc,gl.STATIC_DRAW); GLState.meshEdgeBuffer=gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,GLState.meshEdgeBuffer); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,meshEdges,gl.STATIC_DRAW); GLState.ready=true; }, render:function(gl,matrix){ if(!GLState.ready) return; gl.disable(gl.DEPTH_TEST); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA); if(currentVar!=="current"){ const vm=variableMeta(currentVar); if(vm){ gl.useProgram(GLState.scalarProgram); gl.bindBuffer(gl.ARRAY_BUFFER,GLState.nodeBuffer); gl.enableVertexAttribArray(GLState.aPos); gl.vertexAttribPointer(GLState.aPos,2,gl.FLOAT,false,0,0); gl.bindBuffer(gl.ARRAY_BUFFER,GLState.valueBuffer); gl.enableVertexAttribArray(GLState.aVal); gl.vertexAttribPointer(GLState.aVal,1,gl.FLOAT,false,0,0); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,GLState.elemBuffer); gl.uniformMatrix4fv(GLState.uMatrix,false,matrix); gl.uniform1f(GLState.uVmin,vm.vmin); gl.uniform1f(GLState.uVmax,vm.vmax); gl.uniform1f(GLState.uOpacity,parseFloat(els.opacitySlider.value)); gl.uniform1i(GLState.uCmap,cmapCode(vm.cmap)); gl.uniform1f(GLState.uInvalid,meta.invalid_value); gl.drawElements(gl.TRIANGLES,meta.index_count,gl.UNSIGNED_INT,0); } } if(els.meshOverlay && els.meshOverlay.checked){ gl.useProgram(GLState.meshProgram); gl.bindBuffer(gl.ARRAY_BUFFER,GLState.meshNodeBuffer); gl.enableVertexAttribArray(GLState.meshAPos); gl.vertexAttribPointer(GLState.meshAPos,2,gl.FLOAT,false,0,0); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,GLState.meshEdgeBuffer); gl.uniformMatrix4fv(GLState.meshUMatrix,false,matrix); gl.uniform4f(GLState.meshUColor, 0.78, 0.78, 0.78, 0.35); gl.drawElements(gl.LINES,meshEdges.length,gl.UNSIGNED_INT,0); } } }; }

async function loadScalarFrame(v,i){ const url=scalarFrameUrl(v,i); if(!url) return null; const key=`${v}:${i}`; if(scalarCache.has(key)) return scalarCache.get(key); if(scalarLoading.has(key)) return await scalarLoading.get(key); const promise=fetchFloat16AsFloat32(url, meta.node_count).then(arr=>{ scalarCache.set(key,arr); scalarLoading.delete(key); for(const k of Array.from(scalarCache.keys())){ const [vv,ff]=k.split(":"); if(vv!==v || Math.abs(parseInt(ff)-i)>2) scalarCache.delete(k); } return arr; }); scalarLoading.set(key,promise); return await promise; }
function preloadScalarNeighbors(v,i){ if(v==="current") return; loadScalarFrame(v,Math.max(0,i-1)).catch(()=>{}); loadScalarFrame(v,Math.min(frameCount()-1,i+1)).catch(()=>{}); }
async function loadCurrentFrame(i){ const key=String(i); if(currentCache.has(key)){ const c=currentCache.get(key); currentU=c.u; currentV=c.v; resetParticles(); return; } const [u,v]=await Promise.all([fetchFloat16AsFloat32(currentUUrl(i),meta.node_count),fetchFloat16AsFloat32(currentVUrl(i),meta.node_count)]); currentCache.set(key,{u,v}); for(const k of Array.from(currentCache.keys())) if(Math.abs(parseInt(k)-i)>2) currentCache.delete(k); currentU=u; currentV=v; resetParticles(); }
function preloadCurrentFrame(i){ const next=Math.min(frameCount()-1,i+1), key=String(next); if(currentCache.has(key)) return; Promise.all([fetchFloat16AsFloat32(currentUUrl(next), meta.node_count),fetchFloat16AsFloat32(currentVUrl(next), meta.node_count)]).then(([u,v])=>currentCache.set(key,{u,v})).catch(()=>{}); }
async function setFrame(i){ const n=frameCount(); if(n<=0) return; currentFrame=parseInt(i); if(!Number.isFinite(currentFrame)) currentFrame=0; currentFrame=Math.max(0,Math.min(n-1,currentFrame)); els.frameSlider.value=currentFrame; const fm=meta.frames[currentFrame]||{}; els.timeLabel.textContent=fm.label||`frame ${currentFrame}`; updateCurrentOverlayAvailability(); updateLegend(); if(currentVar!=="current"){ const arr=await loadScalarFrame(currentVar,currentFrame); if(GLState.gl && GLState.valueBuffer){ const gl=GLState.gl; gl.bindBuffer(gl.ARRAY_BUFFER,GLState.valueBuffer); gl.bufferSubData(gl.ARRAY_BUFFER,0,arr); } preloadScalarNeighbors(currentVar,currentFrame); } if(shouldDrawCurrentParticles()){ await loadCurrentFrame(currentFrame); clearCurrentCanvas(); startParticles(); preloadCurrentFrame(currentFrame); } else stopParticles(); setStatus(`${currentVar} frame ${currentFrame+1}/${n}`); map.triggerRepaint(); }
function startTimer(){ if(timer!==null) clearInterval(timer); timer=setInterval(()=>{ currentFrame=(currentFrame+1)%frameCount(); setFrame(currentFrame); },1000/speed); }


function initParticleCanvas() {
    const container = map.getContainer();

    if (particleCanvas && particleCanvas.parentNode) {
        particleCanvas.parentNode.removeChild(particleCanvas);
    }

    particleCanvas = document.createElement("canvas");
    particleCanvas.id = "particle-canvas";

    particleCanvas.style.position = "absolute";
    particleCanvas.style.left = "0";
    particleCanvas.style.top = "0";
    particleCanvas.style.width = "100%";
    particleCanvas.style.height = "100%";
    particleCanvas.style.pointerEvents = "none";
    particleCanvas.style.zIndex = "20";
    particleCanvas.style.transform = "none";
    particleCanvas.style.opacity = "1";
    particleCanvas.style.visibility = "visible";

    container.appendChild(particleCanvas);

    particleCtx = particleCanvas.getContext("2d");

    resizeParticleCanvas();

    window.addEventListener("resize", resizeParticleCanvas);
    map.on("resize", resizeParticleCanvas);
}


function resizeParticleCanvas() {
    if (!map || !particleCanvas || !particleCtx) return;

    const rect = map.getContainer().getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(rect.height));
    const dpr = window.devicePixelRatio || 1;

    particleCanvas.width = Math.max(1, Math.round(w * dpr));
    particleCanvas.height = Math.max(1, Math.round(h * dpr));
    particleCanvas.style.width = w + "px";
    particleCanvas.style.height = h + "px";
    particleCanvas.style.left = "0";
    particleCanvas.style.top = "0";
    particleCanvas.style.transform = "none";
    particleCanvas.style.opacity = "1";
    particleCanvas.style.visibility = "visible";

    particleCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    clearCurrentCanvas();
}


function clearCurrentCanvas() {
    if (!particleCtx || !map) return;

    const rect = map.getContainer().getBoundingClientRect();
    particleCtx.clearRect(0, 0, rect.width, rect.height);
}


function speedToColor(s, vmin, vmax) {
    let t = (s - vmin) / (vmax - vmin);
    if (!Number.isFinite(t)) t = 0;
    t = Math.max(0, Math.min(1, t));

    const stops = [
        [0.05, 0.18, 0.95],
        [0.05, 0.62, 1.00],
        [0.10, 0.78, 0.42],
        [0.92, 0.86, 0.22],
        [0.95, 0.55, 0.10],
        [0.82, 0.12, 0.08]
    ];

    const x = t * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.max(0, Math.floor(x)));
    const f = x - i;

    const a = stops[i];
    const b = stops[i + 1];

    const r = Math.round(255 * (a[0] * (1 - f) + b[0] * f));
    const g = Math.round(255 * (a[1] * (1 - f) + b[1] * f));
    const bb = Math.round(255 * (a[2] * (1 - f) + b[2] * f));

    return `rgba(${r},${g},${bb},0.92)`;
}
function currentSpeedRange(){ return {vmin:Number(meta.current_vmin??0), vmax:Number(meta.current_vmax??1)}; }
function currentParticleColor(spd){ if(currentVar==="current"){ const r=currentSpeedRange(); return speedToColor(spd,r.vmin,r.vmax); } return CONFIG.overlayParticleColor; }
function getLookupBounds(){ const lm=lookupMeta||{}; return {nx:Number(lm.nx??lm.lookup_nx), ny:Number(lm.ny??lm.lookup_ny), lonMin:Number(lm.lon_min??meta.bounds[0][1]), lonMax:Number(lm.lon_max??meta.bounds[1][1]), latMin:Number(lm.lat_min??meta.bounds[0][0]), latMax:Number(lm.lat_max??meta.bounds[1][0])}; }
function lookupCell(lon,lat){ const g=getLookupBounds(); if(lon<g.lonMin||lon>g.lonMax||lat<g.latMin||lat>g.latMax) return null; const ix=Math.floor((lon-g.lonMin)/(g.lonMax-g.lonMin)*g.nx); const iy=Math.floor((lat-g.latMin)/(g.latMax-g.latMin)*g.ny); if(ix<0||iy<0||ix>=g.nx||iy>=g.ny) return null; return iy*g.nx+ix; }
function barycentricVector(lon,lat,tri){ const ia=elems[tri*3], ib=elems[tri*3+1], ic=elems[tri*3+2]; const ax=nodesLonLat[ia*2], ay=nodesLonLat[ia*2+1], bx=nodesLonLat[ib*2], by=nodesLonLat[ib*2+1], cx=nodesLonLat[ic*2], cy=nodesLonLat[ic*2+1]; const v0x=bx-ax, v0y=by-ay, v1x=cx-ax, v1y=cy-ay, v2x=lon-ax, v2y=lat-ay; const den=v0x*v1y-v1x*v0y; if(Math.abs(den)<1e-20) return null; const w1=(v2x*v1y-v1x*v2y)/den, w2=(v0x*v2y-v2x*v0y)/den, w0=1-w1-w2; if(w0<-1e-7||w1<-1e-7||w2<-1e-7) return null; const ua=currentU[ia], ub=currentU[ib], uc=currentU[ic], va=currentV[ia], vb=currentV[ib], vc=currentV[ic]; if(!Number.isFinite(ua+ub+uc+va+vb+vc) || ua<=meta.invalid_value+1 || ub<=meta.invalid_value+1 || uc<=meta.invalid_value+1 || va<=meta.invalid_value+1 || vb<=meta.invalid_value+1 || vc<=meta.invalid_value+1) return null; const u=w0*ua+w1*ub+w2*uc, v=w0*va+w1*vb+w2*vc, speed=Math.hypot(u,v); if(!Number.isFinite(speed)) return null; return {u,v,speed}; }
function vectorAt(lon,lat){ if(!currentU||!currentV||!lookupOffsets||!lookupTriangles) return null; const cell=lookupCell(lon,lat); if(cell===null) return null; const start=lookupOffsets[cell], end=lookupOffsets[cell+1]; for(let k=start;k<end;k++){ const vec=barycentricVector(lon,lat,lookupTriangles[k]); if(vec) return vec; } return null; }
function randomValidPoint(){ const g=getLookupBounds(); let west=g.lonMin,east=g.lonMax,south=g.latMin,north=g.latMax; if(map){ const b=map.getBounds(); west=Math.max(west,b.getWest()); east=Math.min(east,b.getEast()); south=Math.max(south,b.getSouth()); north=Math.min(north,b.getNorth()); } if(!(west<east&&south<north)){ west=g.lonMin; east=g.lonMax; south=g.latMin; north=g.latMax; } for(let i=0;i<800;i++){ const lon=west+Math.random()*(east-west), lat=south+Math.random()*(north-south); if(vectorAt(lon,lat)) return {lon,lat}; } return null; }

function isMobileLayout(){
  return document.body && document.body.classList && document.body.classList.contains("device-mobile");
}

function zoomOutParticleDensityMultiplier(){
  if(!map) return 1.0;

  const z = map.getZoom();

  // Revert density boost.
  // z <= 5 : 0.90x
  // z = 7  : 0.95x
  // z >= 9 : 1.00x
  if(z <= 5.0) return 0.90;
  if(z >= 9.0) return 1.00;

  return 0.90 + (z - 5.0) * (0.10 / 4.0);
}

function effectiveParticleCount(){
  const n = Number(particleCount || 0);
  const base = isMobileLayout() ? Math.max(250, Math.round(n * 0.32)) : n;
  const mul = zoomOutParticleDensityMultiplier();
  return Math.max(250, Math.round(base * mul));
}

function resetParticle(p, ll=null) {
    const pos = ll || randomValidPoint();

    if (!pos) {
        p.lon = meta.bounds[0][1];
        p.lat = meta.bounds[0][0];
    } else {
        p.lon = pos.lon;
        p.lat = pos.lat;
    }

    // 길어진 gradient particle은 자주 죽으면 깜빡임/빈공간이 커져서 수명 조금 길게
    p.age = Math.floor(Math.random() * 80);
    p.maxAge = 170 + Math.floor(Math.random() * 140);

    // fade-in용
    p.fadeAge = Math.floor(Math.random() * 10);

    // Last sampled velocity cache.
    // During map interaction, particles keep flowing with these values
    // and do not resample vectorAt().
    p.lastU = null;
    p.lastV = null;
    p.lastSpeed = null;
    p.lastTspeed = null;
    p.lastDt = null;
// Geographic trail. This makes particles stick to the map during pan/zoom.
    p.trail = [{ lon: p.lon, lat: p.lat }];
}
function resetParticles(){
    particles = [];
    if(!currentU || !currentV) return;

    const n = effectiveParticleCount ? effectiveParticleCount() : particleCount;

    for(let i = 0; i < n; i++){
        const p = {};
        resetParticle(p);
        particles.push(p);
    }
}

function particleFlowScale(){
  const base = CONFIG.flowScale * 1.25;
  if(!map) return base;

  const z = map.getZoom();
  const refZoom = 7.0;

  // More movement in zoomed-out view.
  // z <= 5 : 1.60x of boosted base
  // z = 6  : 1.30x of boosted base
  // z >= 7 : original zoom-in slowdown shape
  if(z < refZoom){
    if(z <= 5.0) return base * 1.60;
    return base * (1.0 + (refZoom - z) * 0.30);
  }

  const factor = Math.pow(0.75, Math.max(0, z - refZoom));
  return Math.max(base * 0.20, base * factor);
}

function particleTrailLength(){
  return 3;
}

function speedBasedParticleDrawLength(spd){
  const r = currentSpeedRange ? currentSpeedRange() : {vmin:0, vmax:1};
  const vmin = Number.isFinite(r.vmin) ? r.vmin : 0.0;
  const vmax = Number.isFinite(r.vmax) && r.vmax > vmin ? r.vmax : 1.0;

  let t = (spd - vmin) / (vmax - vmin);
  if(!Number.isFinite(t)) t = 0.0;
  t = Math.max(0.0, Math.min(1.0, t));

  // keep speed-dependent length
  t = Math.pow(t, 0.75);

  // slightly longer than current test
  let minLen = 4.6;
  let maxLen = 24.0;

  if(map){
    const z = map.getZoom();

    // zoomed-out particles were too short.
    // z <= 5 : 0.95x
    // z = 7  : about 1.02x
    // z >= 9 : 1.10x
    let f = 1.10;
    if(z <= 5.0) f = 0.95;
    else if(z < 9.0) f = 0.95 + (z - 5.0) * (0.15 / 4.0);

    minLen *= f;
    maxLen *= f;
  }

  return minLen + (maxLen - minLen) * t;
}

function colorWithAlpha(color, alpha){
    const m = String(color || "").match(/rgba?\(([^)]+)\)/);
    if(!m) return color;

    const p = m[1].split(",").map(s => s.trim());
    if(p.length < 3) return color;

    return `rgba(${p[0]},${p[1]},${p[2]},${alpha})`;
}

function zoomOutParticleWidthMultiplier(){
  if(!map) return 1.0;

  const z = map.getZoom();

  // zoomed out: thinner lines to avoid strong white/thick look.
  // z <= 5 : 0.34x
  // z = 7  : about 0.62x
  // z >= 9 : 1.00x
  if(z <= 5.0) return 0.34;
  if(z >= 9.0) return 1.00;

  return 0.34 + (z - 5.0) * (0.66 / 4.0);
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
            drawMeshOverlayOnParticleCanvas();

        particleAnimId = requestAnimationFrame(step);
            return;
        }

        const trailLen = particleTrailLength();
        const rect = map.getContainer().getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;

        // Soft fade instead of hard clear:
        // remove only part of the previous frame so particles disappear smoothly.
        // Larger alpha = faster fade. 0.10 leaves a short Windy-like trail.
        particleCtx.globalCompositeOperation = "destination-out";
        particleCtx.fillStyle = "rgba(0,0,0,0.10)";
        particleCtx.fillRect(0, 0, width, height);

        particleCtx.globalCompositeOperation = "source-over";
        particleCtx.lineWidth = 1.2;
        particleCtx.lineCap = "round";

        if (particles.length < effectiveParticleCount() * 0.60) {
            resetParticles();
        }

        for (const p of particles) {
            if (!p || p.age > p.maxAge) {
                resetParticle(p);
                continue;
            }

            let vec = null;
            let tSpeed = Number.isFinite(p.lastTspeed) ? p.lastTspeed : 0.0;
            let dt = Number.isFinite(p.lastDt) ? p.lastDt : particleFlowScale();

            if (!particleMapInteracting) {
                // Normal mode:
                // sample vector field from current lon/lat.
                vec = vectorAt(p.lon, p.lat);

                if (!vec || !Number.isFinite(vec.u) || !Number.isFinite(vec.v)) {
                    resetParticle(p);
                    continue;
                }

                const r = currentSpeedRange ? currentSpeedRange() : {vmin:0, vmax:1};
                tSpeed = (vec.speed - r.vmin) / Math.max(1e-12, r.vmax - r.vmin);
                if(!Number.isFinite(tSpeed)) tSpeed = 0.0;
                tSpeed = Math.max(0.0, Math.min(1.0, tSpeed));

                dt = particleFlowScale();

                // Cache latest sampled values.
                // During pan/zoom, we reuse these and avoid vectorAt().
                p.lastU = vec.u;
                p.lastV = vec.v;
                p.lastSpeed = vec.speed;
                p.lastTspeed = tSpeed;
                p.lastDt = dt;
            } else {
                // Map is being panned/zoomed:
                // DO NOT resample vector field.
                // Keep flowing with the last sampled velocity.
                if(!Number.isFinite(p.lastU) || !Number.isFinite(p.lastV)){
                    continue;
                }

                vec = {
                    u: p.lastU,
                    v: p.lastV,
                    speed: Number.isFinite(p.lastSpeed) ? p.lastSpeed : Math.hypot(p.lastU, p.lastV)
                };
            }

            const latRad = p.lat * Math.PI / 180.0;
            let coslat = Math.cos(latRad);
            if (Math.abs(coslat) < 1e-6) coslat = 1e-6;

            const newLon = p.lon + (vec.u * dt) / coslat;
            const newLat = p.lat + vec.v * dt;

            if (!particleMapInteracting) {
                // Only validate/resample destination in normal mode.
                // During map interaction, do not trigger resets from transient viewport changes.
            // Optimization: removed next-position vectorAt(newLon, newLat) pre-check.
}

            p.lon = newLon;
            p.lat = newLat;
            p.age += 1;

            if (!p.trail) p.trail = [];
            p.trail.push({ lon: p.lon, lat: p.lat });

            if (p.trail.length > trailLen) {
                p.trail.shift();
            }

            const head = map.project([p.lon, p.lat]);

            if (
                head.x < -120 || head.x > width + 120 ||
                head.y < -120 || head.y > height + 120
            ) {
                // During map interaction, don't reset because viewport is moving.
                // After interaction ends, resetParticles() will rebuild from scratch.
                if (!particleMapInteracting) {
                    resetParticle(p);
                }
                continue;
            }

            if (p.trail.length < 2) continue;

            const nTrail = p.trail.length;
            const qHead = p.trail[nTrail - 1];
            const qPrev = p.trail[Math.max(0, nTrail - 2)];

            const ptHead = map.project([qHead.lon, qHead.lat]);
            const ptPrev = map.project([qPrev.lon, qPrev.lat]);

            let dx = ptHead.x - ptPrev.x;
            let dy = ptHead.y - ptPrev.y;
            let len = Math.sqrt(dx * dx + dy * dy);

            if(!Number.isFinite(len) || len < 0.05){
                const latRad2 = p.lat * Math.PI / 180.0;
                let coslat2 = Math.cos(latRad2);
                if(Math.abs(coslat2) < 1e-6) coslat2 = 1e-6;

                dx = vec.u / coslat2;
                dy = vec.v;
                len = Math.sqrt(dx * dx + dy * dy);
            }

            if(!Number.isFinite(len) || len <= 0.0){
                continue;
            }

            dx /= len;
            dy /= len;

            const segLen = Math.max(2.0, speedBasedParticleDrawLength(vec.speed) * 0.65);

            const x1 = ptHead.x;
            const y1 = ptHead.y;
            const x0 = x1 - dx * segLen;
            const y0 = y1 - dy * segLen;

            p.fadeAge = (p.fadeAge || 0) + 1;
            const fadeFactor = Math.min(1.0, p.fadeAge / 18.0);

            const baseColor = currentParticleColor(vec.speed);

            const alphaBase = currentVar === "current"
                ? (0.32 + 0.36 * tSpeed)
                : (0.22 + 0.18 * tSpeed);

            const widthMul = zoomOutParticleWidthMultiplier();

            const lineW = currentVar === "current"
                ? (1.05 + 0.65 * tSpeed) * widthMul
                : (0.95 + 0.45 * tSpeed) * widthMul;

            // Cheap head/tail split:
            // faint tail + brighter head for direction.
            const xm = x0 + (x1 - x0) * 0.62;
            const ym = y0 + (y1 - y0) * 0.62;

            particleCtx.lineWidth = lineW * 0.72;
            particleCtx.strokeStyle = colorWithAlpha(baseColor, alphaBase * fadeFactor * 0.32);
            particleCtx.beginPath();
            particleCtx.moveTo(x0, y0);
            particleCtx.lineTo(xm, ym);
            particleCtx.stroke();

            particleCtx.lineWidth = lineW;
            particleCtx.strokeStyle = colorWithAlpha(baseColor, alphaBase * fadeFactor * 0.92);
            particleCtx.beginPath();
            particleCtx.moveTo(xm, ym);
            particleCtx.lineTo(x1, y1);
            particleCtx.stroke();
        }

        particleAnimId = requestAnimationFrame(step);
    }

    particleAnimId = requestAnimationFrame(step);
}

function drawMeshOverlayOnParticleCanvas(){
    if (!particleCtx || !map) return;
    if (currentVar !== "current") return;
    if (!els.meshOverlay || !els.meshOverlay.checked) return;
    if (!meshEdges || !nodesLonLat) return;

    const rect = map.getContainer().getBoundingClientRect();

    particleCtx.save();
    particleCtx.globalCompositeOperation = "source-over";
    particleCtx.strokeStyle = "rgba(0,0,0,0.78)";
    particleCtx.lineWidth = 0.65;
    particleCtx.beginPath();

    const maxEdges = meshEdges.length;
    for (let k = 0; k < maxEdges; k += 2) {
        const ia = meshEdges[k];
        const ib = meshEdges[k + 1];
        if (ia == null || ib == null) continue;

        const lonA = nodesLonLat[ia * 2];
        const latA = nodesLonLat[ia * 2 + 1];
        const lonB = nodesLonLat[ib * 2];
        const latB = nodesLonLat[ib * 2 + 1];

        if (!Number.isFinite(lonA) || !Number.isFinite(latA) ||
            !Number.isFinite(lonB) || !Number.isFinite(latB)) continue;

        const a = map.project([lonA, latA]);
        const b = map.project([lonB, latB]);

        if ((a.x < -50 && b.x < -50) || (a.x > rect.width + 50 && b.x > rect.width + 50) ||
            (a.y < -50 && b.y < -50) || (a.y > rect.height + 50 && b.y > rect.height + 50)) {
            continue;
        }

        particleCtx.moveTo(a.x, a.y);
        particleCtx.lineTo(b.x, b.y);
    }

    particleCtx.stroke();
    particleCtx.restore();
}


function stopParticles(){ particleRunning=false; if(particleAnimId!==null){ cancelAnimationFrame(particleAnimId); particleAnimId=null; } clearCurrentCanvas(); }
function setupEvents(){
    els.currentOverlay.checked = true;
    els.currentOverlay.dataset.userTouched = "";

    els.currentOverlay.addEventListener("change", () => {
        els.currentOverlay.dataset.userTouched = "1";
        updateCurrentOverlayAvailability();
        setFrame(currentFrame);
    });

    els.meshOverlay.addEventListener("change", () => { clearCurrentCanvas(); drawMeshOverlayOnParticleCanvas(); map.triggerRepaint(); });

    els.varSelect.addEventListener("change", e => {
        currentVar = e.target.value;
        updateCurrentOverlayAvailability();
        updateLegend();
        setFrame(currentFrame);
        map.triggerRepaint();
    });

    els.playBtn.addEventListener("click", () => {
        if(timer === null){
            els.playBtn.textContent = "❚❚";
            startTimer();
        } else {
            els.playBtn.textContent = "▶";
            clearInterval(timer);
            timer = null;
        }
    });

    els.frameSlider.addEventListener("input", e => setFrame(e.target.value));

    els.speedSelect.addEventListener("change", e => {
        speed = parseFloat(e.target.value);
        if(timer !== null) startTimer();
    });

    els.opacitySlider.addEventListener("input", () => map.triggerRepaint());

    els.densitySelect.addEventListener("change", e => {
        particleCount = parseInt(e.target.value);
        resetParticles();
        clearCurrentCanvas();
    });

    let particleResetTimer = null;

    function beginParticleInteraction(){
        particleMapInteracting = true;

        if(particleResetTimer !== null){
            clearTimeout(particleResetTimer);
            particleResetTimer = null;
        }
    }

    function scheduleParticleResetAfterInteraction(){
        if(particleResetTimer !== null){
            clearTimeout(particleResetTimer);
        }

        // Wait until pan/zoom really stops.
        particleResetTimer = setTimeout(() => {
            particleResetTimer = null;
            particleMapInteracting = false;
            clearCurrentCanvas();
            resetParticles();
        }, 300);
    }

    map.on("movestart", beginParticleInteraction);
    map.on("zoomstart", beginParticleInteraction);
    map.on("moveend", scheduleParticleResetAfterInteraction);
    map.on("zoomend", scheduleParticleResetAfterInteraction);
}


function makeMapStyle() {
    return {
        version: 8,
        sources: {
            "carto-positron": {
                type: "raster",
                tiles: [
                    "https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
                    "https://b.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
                    "https://c.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
                    "https://d.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png"
                ],
                tileSize: 256,
                attribution: "© OpenStreetMap contributors © CARTO"
            },
            "esri-satellite": {
                type: "raster",
                tiles: [
                    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                ],
                tileSize: 256,
                attribution: "Tiles © Esri"
            }
        },
        layers: [
            {
                id: "basemap-carto",
                type: "raster",
                source: "carto-positron",
                layout: { visibility: "none" }
            },
            {
                id: "basemap-esri",
                type: "raster",
                source: "esri-satellite",
                layout: { visibility: "visible" }
            }
        ]
    };
}
function initMap(){ const b=meta.bounds; const south=b[0][0], west=b[0][1], north=b[1][0], east=b[1][1]; map=new maplibregl.Map({container:"map",style:makeMapStyle(),center:[(west+east)/2,(south+north)/2],zoom:7,minZoom:3,maxZoom:14,dragRotate:false,pitchWithRotate:false,renderWorldCopies:false,attributionControl:true}); map.addControl(new maplibregl.NavigationControl({visualizePitch:false}),"top-left"); map.fitBounds([[west,south],[east,north]],{padding:20,duration:0}); }
async function boot(){ setStatus("Loading metadata..."); meta=await fetchJson(CONFIG.metaUrl); meta.variables=meta.variables||{}; meta.variables.salinity=meta.variables.salinity||{name:"Salinity",units:"psu",vmin:25,vmax:35}; meta.variables.salinity.vmin=25; meta.variables.salinity.vmax=35; meta.variables=meta.variables||{}; meta.variables.salinity=meta.variables.salinity||{name:"Salinity",units:"psu",vmin:25,vmax:35}; meta.variables.salinity.vmin=25; meta.variables.salinity.vmax=35; lookupMeta=await fetchJson(CONFIG.lookupMetaUrl); els.frameSlider.max=frameCount()-1; setStatus("Loading mesh and lookup..."); [nodesLonLat,elems,meshEdges,lookupOffsets,lookupTriangles]=await Promise.all([fetchFloat32(CONFIG.nodesUrl,meta.node_count*2),fetchUint32(CONFIG.elemsUrl,meta.index_count),fetchUint32(CONFIG.edgesUrl),fetchUint32(CONFIG.lookupOffsetsUrl),fetchUint32(CONFIG.lookupTrianglesUrl)]); initMap(); map.on("load",async()=>{ setStatus("Preparing WebGL layer..."); buildMercatorNodes(); initParticleCanvas(); map.addLayer(makeSchismLayer()); setupEvents(); updateCurrentOverlayAvailability(); updateLegend(); await setFrame(0); setStatus(`Ready: ${meta.node_count.toLocaleString()} nodes, ${meta.triangle_count.toLocaleString()} triangles`); }); }
boot().catch(err=>{ console.error(err); setStatus("ERROR: "+err.message); alert(err.message); });


function setBaseMap(name) {
    if (!map) return;

    const useCarto = name === "carto";
    const useEsri = name === "esri";

    if (map.getLayer("basemap-carto")) {
        map.setLayoutProperty("basemap-carto", "visibility", useCarto ? "visible" : "none");
    }

    if (map.getLayer("basemap-esri")) {
        map.setLayoutProperty("basemap-esri", "visibility", useEsri ? "visible" : "none");
    }

    // Keep SCHISM custom layer above base maps.
    try {
        if (map.getLayer("schism-custom-layer")) {
            map.moveLayer("schism-custom-layer");
        }
    } catch (e) {}

    map.triggerRepaint();
    console.log("[basemap] switched to", name);
}


// KOP_FORCE_BASEMAP_SWITCH_START
(function () {
    "use strict";

    function forceSetBaseMap(name) {
        if (typeof map === "undefined" || !map) {
            console.warn("[basemap] map is not ready");
            return;
        }

        const cartoIds = [
            "basemap-carto",
            "basemap-carto-light",
            "basemap-carto-positron",
            "carto-light-layer",
            "carto-positron-layer"
        ];

        const esriIds = [
            "basemap-esri",
            "basemap-esri-satellite",
            "esri-satellite-layer",
            "basemap-esri-layer"
        ];

        function setVisible(layerId, visible) {
            if (!map.getLayer(layerId)) return;

            map.setLayoutProperty(layerId, "visibility", visible ? "visible" : "none");

            try {
                map.setPaintProperty(layerId, "raster-opacity", visible ? 1.0 : 0.0);
            } catch (e) {}

            console.log("[basemap]", layerId, visible ? "visible" : "none");
        }

        const useCarto = name === "carto";
        const useEsri = name === "esri";

        cartoIds.forEach(id => setVisible(id, useCarto));
        esriIds.forEach(id => setVisible(id, useEsri));

        try {
            if (map.getLayer("schism-custom-layer")) {
                map.moveLayer("schism-custom-layer");
            }
        } catch (e) {}

        map.triggerRepaint();
        console.log("[basemap] switched:", name);
    }

    window.setBaseMap = forceSetBaseMap;

    function installBasemapRadios() {
        document.querySelectorAll('input[name="basemap-radio"]').forEach(radio => {
            if (radio.dataset.kopBasemapHooked) return;
            radio.dataset.kopBasemapHooked = "1";

            radio.addEventListener("change", e => {
                if (e.target.checked) {
                    window.setBaseMap(e.target.value);
                }
            });
        });

        console.log("[basemap] radio switch installed");
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", () => {
            setTimeout(installBasemapRadios, 800);
        });
    } else {
        setTimeout(installBasemapRadios, 800);
    }
})();
// KOP_FORCE_BASEMAP_SWITCH_END

// KOP_SMOOTH_JET_STYLE applied

// KOP_MESH_SOFT_GRAY_ALPHA_010


// KOP_SAFE_TUNE_MESH016_FLOW0063_GRAYPARTICLE_RDBU_LEGEND

// KOP_TUNE_MESH04_FLOW005

// KOP_SMOOTH_RDBU_MESH035

// KOP_ELEVATION_SIMPLE_BWR_PARTICLE238


(function removeParticleSnapshotArtifacts() {
    const badIds = [
        "kop-particle-fixed-snapshot",
        "particle-snapshot-canvas"
    ];

    for (const id of badIds) {
        const el = document.getElementById(id);
        if (el && el.parentNode) {
            el.parentNode.removeChild(el);
        }
    }

    window.__KOP_PARTICLE_LOCKED = false;
})();


// KOP_PARTICLE_GEOTRAIL_MAP_ATTACHED

// KOP_FLOAT16_JULY2023_SPRING_TIDE_15DAY

// KOP_FIX_JULY15_META_FLOAT16_CURRENT_01

// KOP_FORCE_REPLACE_FLOAT16_FUNC_02

// KOP_META_JSON_NOSTORE_JULY_01

// KOP_PARTICLE_SPEED_INDEPENDENT_FROM_PLAYBACK_01

// KOP_JULY7_SALINITY_01


// KOP_SALINITY_META_RANGE_28_35

// KOP_SALINITY_LEGEND_CLEANUP_01

// KOP_FORCE_SALINITY_28_35_META

// KOP_LEGEND_FORCEFIX_02

// KOP_LEGEND_TEMPERATURE_TITLE_NO_SURFACE_01


// KOP_MOBILE_PARTICLE_HALF_01

// KOP_MOBILE_PARTICLE_032_01

// KOP_MOBILE_PARTICLE_032_FROM_SCREENSHOT_01

// KOP_PLAY_ICON_BUTTON_01

// KOP_SALINITY_YLGNBU_25_35_01

// KOP_ZOOM_PARTICLE_SPEED_01

// KOP_TEST_GRADIENT_SPRITE_PARTICLE_01

// KOP_TEST_REDUCE_FLICKER_01

// KOP_TEST_PARTICLE_SOFT_FADE_01

// KOP_TEST_PARTICLE_FADE_IN_01

// KOP_TEST_TUNE_GRAY_SHORTER_STABLE_01

// KOP_TEST_WHITE_ALPHA_KEEP_FLOW_01

// KOP_TEST_MOVE_RESET_ONLY_01

// KOP_TEST_GRID_PARTICLE_DISTRIBUTION_01

// KOP_TEST_PARTICLE_COUNT_ORIGINAL_01

// KOP_TEST_ZOOMOUT_PARTICLE_DENSITY_150_01

// KOP_TEST_LOW_SPEED_PARTICLES_VISIBLE_01

// KOP_TEST_PARTICLE_COUNT_SAME_AS_MAIN_01

// KOP_TEST_ZOOMOUT_SHORTER_LESS_01

// KOP_TEST_SPEED_BASED_PARTICLE_LENGTH_01

// KOP_TEST_ZOOMOUT_HALF_LENGTH_DENSITY_01

// KOP_TEST_PARTICLE_HEAD_ALPHA_01

// KOP_TEST_PARTICLE_SOFT_GRAY_225_055_01

// KOP_TEST_PARTICLE_LENGTH_HALF_01

// KOP_TEST_WINDY_SHORT_SEGMENT_01

// KOP_TEST_ZOOMOUT_PARTICLE_WIDTH_01

// KOP_TEST_PARTICLE_OPTIMIZED_01

// KOP_TEST_PARTICLE_ALPHA_050_01

// KOP_TEST_PARTICLE_COLOR_235_055_01

// KOP_APPLY_TEST_PARTICLES_TO_MAIN_01

// KOP_TEST_RESET_FROM_MAIN_01

// KOP_TEST_SIMPLE_WINDY_TUNE_01

// KOP_TEST_NO_RECALC_DURING_INTERACTION_01

// KOP_TEST_LOGO_PARTICLES_TUNE_01

// KOP_TEST_REMOVE_VEC2_CHECK_01

// KOP_TEST_PARTICLE_DENSITY_BACK_01

// KOP_MESH_OVERLAY_BLACK_01

// KOP_MESH_OVERLAY_BLACK_CORRECT_01
