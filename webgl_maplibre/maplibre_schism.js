"use strict";

const DATA_ROOT = "../webgl_unstructured/";
const CONFIG = {
  metaUrl: DATA_ROOT + "mesh_meta.json",
  nodesUrl: DATA_ROOT + "mesh_nodes.bin",
  elemsUrl: DATA_ROOT + "mesh_elems.bin",
  edgesUrl: DATA_ROOT + "mesh_edges.bin",
  lookupMetaUrl: DATA_ROOT + "lookup_meta.json",
  lookupOffsetsUrl: DATA_ROOT + "lookup_offsets.bin",
  lookupTrianglesUrl: DATA_ROOT + "lookup_triangles.bin",
  flowScale: 0.018,
  overlayParticleColor: "rgba(255,255,255,0.96)"
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
let particleCanvas, particleCtx, particleAnimId = null, particleRunning = false, particleCount = 2800, particles = [];

const GLState = { gl:null, scalarProgram:null, meshProgram:null, nodeBuffer:null, valueBuffer:null, elemBuffer:null, meshNodeBuffer:null, meshEdgeBuffer:null,
  aPos:null, aVal:null, uMatrix:null, uVmin:null, uVmax:null, uOpacity:null, uCmap:null, uInvalid:null,
  meshAPos:null, meshUMatrix:null, meshUColor:null, ready:false };

function setStatus(msg){ if(els.statusLine) els.statusLine.textContent = msg; }
function pad4(i){ return String(i).padStart(4,"0"); }
function frameCount(){ return meta && meta.frames ? meta.frames.length : 0; }
function scalarFrameUrl(v,i){ if(v==="temperature") return DATA_ROOT+`temp_bin/frame_${pad4(i)}.bin`; if(v==="ssh") return DATA_ROOT+`ssh_bin/frame_${pad4(i)}.bin`; return null; }
function currentUUrl(i){ return DATA_ROOT+`current_u_bin/frame_${pad4(i)}.bin`; }
function currentVUrl(i){ return DATA_ROOT+`current_v_bin/frame_${pad4(i)}.bin`; }
function variableMeta(v){ if(v==="temperature") return meta.variables.temperature; if(v==="ssh") return meta.variables.ssh; return null; }
function currentOverlayEnabled(){ return els.currentOverlay && els.currentOverlay.checked && (currentVar==="temperature" || currentVar==="ssh"); }
function shouldDrawCurrentParticles(){ return currentVar==="current" || currentOverlayEnabled(); }
function updateCurrentOverlayAvailability(){ if(!els.currentOverlay) return; if(currentVar==="current"){ els.currentOverlay.checked=false; els.currentOverlay.disabled=true; } else { els.currentOverlay.disabled=false; if(!els.currentOverlay.dataset.userTouched) els.currentOverlay.checked=true; } }
async function fetchJson(url){ const r=await fetch(url,{cache:"force-cache"}); if(!r.ok) throw new Error(`${url}: ${r.status}`); return await r.json(); }
async function fetchArrayBuffer(url){ const r=await fetch(url,{cache:"force-cache"}); if(!r.ok) throw new Error(`${url}: ${r.status}`); return await r.arrayBuffer(); }
async function fetchFloat32(url,n=null){ const a=new Float32Array(await fetchArrayBuffer(url)); if(n!==null && a.length!==n) throw new Error(`${url}: ${a.length} != ${n}`); return a; }
async function fetchUint32(url,n=null){ const a=new Uint32Array(await fetchArrayBuffer(url)); if(n!==null && a.length!==n) throw new Error(`${url}: ${a.length} != ${n}`); return a; }

function updateLegend(){
  if(!els.legendBox || !meta) return;
  if(currentVar==="temperature"){
    const v=meta.variables.temperature;
    els.legendBox.innerHTML=`<div style="font-weight:bold; margin-bottom:6px;">Surface Temperature [degC]</div><div style="width:220px; height:16px; background:linear-gradient(to right,#000080,#0000ff,#00ffff,#ffff00,#ff0000,#800000); border:1px solid #666;"></div><div style="display:flex; justify-content:space-between; font-size:12px; margin-top:4px;"><span>${v.vmin}</span><span>${((v.vmin+v.vmax)/2).toFixed(1)}</span><span>${v.vmax}</span></div>`;
  } else if(currentVar==="ssh"){
    const v=meta.variables.ssh;
    els.legendBox.innerHTML=`<div style="font-weight:bold; margin-bottom:6px;">Elevation [m]</div><div style="width:220px; height:16px; background:linear-gradient(to right,#08306b,#6baed6,#f7f7f7,#fb6a4a,#67000d); border:1px solid #666;"></div><div style="display:flex; justify-content:space-between; font-size:12px; margin-top:4px;"><span>${v.vmin}</span><span>${((v.vmin+v.vmax)/2).toFixed(2)}</span><span>${v.vmax}</span></div>`;
  } else {
    els.legendBox.innerHTML=`<div style="font-weight:bold; margin-bottom:6px;">Current Speed [m/s]</div><div style="width:220px; height:16px; background:linear-gradient(to right,#000080,#0000ff,#00ffff,#ffff00,#ff0000,#800000); border:1px solid #666;"></div><div style="display:flex; justify-content:space-between; font-size:12px; margin-top:4px;"><span>0</span><span>0.5</span><span>1.0</span></div>`;
  }
}

function compileShader(gl,type,src){ const sh=gl.createShader(type); gl.shaderSource(sh,src); gl.compileShader(sh); if(!gl.getShaderParameter(sh,gl.COMPILE_STATUS)){ const log=gl.getShaderInfoLog(sh); gl.deleteShader(sh); throw new Error(log); } return sh; }
function makeProgram(gl,vs,fs){ const p=gl.createProgram(); const a=compileShader(gl,gl.VERTEX_SHADER,vs); const b=compileShader(gl,gl.FRAGMENT_SHADER,fs); gl.attachShader(p,a); gl.attachShader(p,b); gl.linkProgram(p); gl.deleteShader(a); gl.deleteShader(b); if(!gl.getProgramParameter(p,gl.LINK_STATUS)){ const log=gl.getProgramInfoLog(p); gl.deleteProgram(p); throw new Error(log); } return p; }
const SCALAR_VS=`precision highp float; attribute vec2 a_pos; attribute float a_value; uniform mat4 u_matrix; varying float v_value; void main(){ gl_Position=u_matrix*vec4(a_pos,0.0,1.0); v_value=a_value; }`;
const SCALAR_FS=`precision highp float; varying float v_value; uniform float u_vmin; uniform float u_vmax; uniform float u_opacity; uniform int u_cmap; uniform float u_invalid; vec3 jet(float t){t=clamp(t,0.0,1.0);float r=clamp(min(4.0*t-1.5,-4.0*t+4.5),0.0,1.0);float g=clamp(min(4.0*t-0.5,-4.0*t+3.5),0.0,1.0);float b=clamp(min(4.0*t+0.5,-4.0*t+2.5),0.0,1.0);return vec3(r,g,b);} vec3 mix3(vec3 a,vec3 b,float t){return a*(1.0-t)+b*t;} vec3 rdbu(float t){t=clamp(t,0.0,1.0);vec3 c0=vec3(0.031,0.188,0.420);vec3 c1=vec3(0.420,0.682,0.839);vec3 c2=vec3(0.969);vec3 c3=vec3(0.984,0.416,0.290);vec3 c4=vec3(0.404,0.0,0.051);if(t<0.25)return mix3(c0,c1,t/0.25);if(t<0.50)return mix3(c1,c2,(t-0.25)/0.25);if(t<0.75)return mix3(c2,c3,(t-0.50)/0.25);return mix3(c3,c4,(t-0.75)/0.25);} void main(){ if((v_value!=v_value)||v_value<=u_invalid+1.0) discard; float den=max(abs(u_vmax-u_vmin),1e-12); float t=clamp((v_value-u_vmin)/den,0.0,1.0); vec3 c=(u_cmap==1)?rdbu(t):jet(t); gl_FragColor=vec4(c,u_opacity); }`;
const MESH_VS=`precision highp float; attribute vec2 a_pos; uniform mat4 u_matrix; void main(){ gl_Position=u_matrix*vec4(a_pos,0.0,1.0); }`;
const MESH_FS=`precision highp float; uniform vec4 u_color; void main(){ gl_FragColor=u_color; }`;

function buildMercatorNodes(){ nodesMerc=new Float32Array(meta.node_count*2); for(let i=0;i<meta.node_count;i++){ const lon=nodesLonLat[i*2], lat=nodesLonLat[i*2+1]; const mc=maplibregl.MercatorCoordinate.fromLngLat({lng:lon,lat:lat}); nodesMerc[i*2]=mc.x; nodesMerc[i*2+1]=mc.y; } }
function makeSchismLayer(){ return { id:"schism-custom-layer", type:"custom", renderingMode:"2d", onAdd:function(m,gl){ GLState.gl=gl; gl.getExtension("OES_element_index_uint"); GLState.scalarProgram=makeProgram(gl,SCALAR_VS,SCALAR_FS); GLState.meshProgram=makeProgram(gl,MESH_VS,MESH_FS); GLState.aPos=gl.getAttribLocation(GLState.scalarProgram,"a_pos"); GLState.aVal=gl.getAttribLocation(GLState.scalarProgram,"a_value"); GLState.uMatrix=gl.getUniformLocation(GLState.scalarProgram,"u_matrix"); GLState.uVmin=gl.getUniformLocation(GLState.scalarProgram,"u_vmin"); GLState.uVmax=gl.getUniformLocation(GLState.scalarProgram,"u_vmax"); GLState.uOpacity=gl.getUniformLocation(GLState.scalarProgram,"u_opacity"); GLState.uCmap=gl.getUniformLocation(GLState.scalarProgram,"u_cmap"); GLState.uInvalid=gl.getUniformLocation(GLState.scalarProgram,"u_invalid"); GLState.meshAPos=gl.getAttribLocation(GLState.meshProgram,"a_pos"); GLState.meshUMatrix=gl.getUniformLocation(GLState.meshProgram,"u_matrix"); GLState.meshUColor=gl.getUniformLocation(GLState.meshProgram,"u_color"); GLState.nodeBuffer=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,GLState.nodeBuffer); gl.bufferData(gl.ARRAY_BUFFER,nodesMerc,gl.STATIC_DRAW); GLState.valueBuffer=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,GLState.valueBuffer); gl.bufferData(gl.ARRAY_BUFFER,meta.node_count*4,gl.DYNAMIC_DRAW); GLState.elemBuffer=gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,GLState.elemBuffer); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,elems,gl.STATIC_DRAW); GLState.meshNodeBuffer=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,GLState.meshNodeBuffer); gl.bufferData(gl.ARRAY_BUFFER,nodesMerc,gl.STATIC_DRAW); GLState.meshEdgeBuffer=gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,GLState.meshEdgeBuffer); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,meshEdges,gl.STATIC_DRAW); GLState.ready=true; }, render:function(gl,matrix){ if(!GLState.ready) return; gl.disable(gl.DEPTH_TEST); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA); if(currentVar!=="current"){ const vm=variableMeta(currentVar); if(vm){ gl.useProgram(GLState.scalarProgram); gl.bindBuffer(gl.ARRAY_BUFFER,GLState.nodeBuffer); gl.enableVertexAttribArray(GLState.aPos); gl.vertexAttribPointer(GLState.aPos,2,gl.FLOAT,false,0,0); gl.bindBuffer(gl.ARRAY_BUFFER,GLState.valueBuffer); gl.enableVertexAttribArray(GLState.aVal); gl.vertexAttribPointer(GLState.aVal,1,gl.FLOAT,false,0,0); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,GLState.elemBuffer); gl.uniformMatrix4fv(GLState.uMatrix,false,matrix); gl.uniform1f(GLState.uVmin,vm.vmin); gl.uniform1f(GLState.uVmax,vm.vmax); gl.uniform1f(GLState.uOpacity,parseFloat(els.opacitySlider.value)); gl.uniform1i(GLState.uCmap,vm.cmap==="rdbu"?1:0); gl.uniform1f(GLState.uInvalid,meta.invalid_value); gl.drawElements(gl.TRIANGLES,meta.index_count,gl.UNSIGNED_INT,0); } } if(els.meshOverlay && els.meshOverlay.checked){ gl.useProgram(GLState.meshProgram); gl.bindBuffer(gl.ARRAY_BUFFER,GLState.meshNodeBuffer); gl.enableVertexAttribArray(GLState.meshAPos); gl.vertexAttribPointer(GLState.meshAPos,2,gl.FLOAT,false,0,0); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,GLState.meshEdgeBuffer); gl.uniformMatrix4fv(GLState.meshUMatrix,false,matrix); gl.uniform4f(GLState.meshUColor,1,1,1,0.65); gl.drawElements(gl.LINES,meshEdges.length,gl.UNSIGNED_INT,0); } } }; }

async function loadScalarFrame(v,i){ const url=scalarFrameUrl(v,i); if(!url) return null; const key=`${v}:${i}`; if(scalarCache.has(key)) return scalarCache.get(key); if(scalarLoading.has(key)) return await scalarLoading.get(key); const promise=fetchFloat32(url,meta.node_count).then(arr=>{ scalarCache.set(key,arr); scalarLoading.delete(key); for(const k of Array.from(scalarCache.keys())){ const [vv,ff]=k.split(":"); if(vv!==v || Math.abs(parseInt(ff)-i)>2) scalarCache.delete(k); } return arr; }); scalarLoading.set(key,promise); return await promise; }
function preloadScalarNeighbors(v,i){ if(v==="current") return; loadScalarFrame(v,Math.max(0,i-1)).catch(()=>{}); loadScalarFrame(v,Math.min(frameCount()-1,i+1)).catch(()=>{}); }
async function loadCurrentFrame(i){ const key=String(i); if(currentCache.has(key)){ const c=currentCache.get(key); currentU=c.u; currentV=c.v; resetParticles(); return; } const [u,v]=await Promise.all([fetchFloat32(currentUUrl(i),meta.node_count),fetchFloat32(currentVUrl(i),meta.node_count)]); currentCache.set(key,{u,v}); for(const k of Array.from(currentCache.keys())) if(Math.abs(parseInt(k)-i)>2) currentCache.delete(k); currentU=u; currentV=v; resetParticles(); }
function preloadCurrentFrame(i){ const next=Math.min(frameCount()-1,i+1), key=String(next); if(currentCache.has(key)) return; Promise.all([fetchFloat32(currentUUrl(next),meta.node_count),fetchFloat32(currentVUrl(next),meta.node_count)]).then(([u,v])=>currentCache.set(key,{u,v})).catch(()=>{}); }
async function setFrame(i){ const n=frameCount(); if(n<=0) return; currentFrame=parseInt(i); if(!Number.isFinite(currentFrame)) currentFrame=0; currentFrame=Math.max(0,Math.min(n-1,currentFrame)); els.frameSlider.value=currentFrame; const fm=meta.frames[currentFrame]||{}; els.timeLabel.textContent=fm.label||`frame ${currentFrame}`; updateCurrentOverlayAvailability(); updateLegend(); if(currentVar!=="current"){ const arr=await loadScalarFrame(currentVar,currentFrame); if(GLState.gl && GLState.valueBuffer){ const gl=GLState.gl; gl.bindBuffer(gl.ARRAY_BUFFER,GLState.valueBuffer); gl.bufferSubData(gl.ARRAY_BUFFER,0,arr); } preloadScalarNeighbors(currentVar,currentFrame); } if(shouldDrawCurrentParticles()){ await loadCurrentFrame(currentFrame); clearCurrentCanvas(); startParticles(); preloadCurrentFrame(currentFrame); } else stopParticles(); setStatus(`${currentVar} frame ${currentFrame+1}/${n}`); map.triggerRepaint(); }
function startTimer(){ if(timer!==null) clearInterval(timer); timer=setInterval(()=>{ currentFrame=(currentFrame+1)%frameCount(); setFrame(currentFrame); },1000/speed); }
function initParticleCanvas(){ const container=map.getCanvasContainer(); particleCanvas=document.createElement("canvas"); particleCanvas.id="particle-canvas"; container.appendChild(particleCanvas); particleCtx=particleCanvas.getContext("2d"); resizeParticleCanvas(); window.addEventListener("resize",resizeParticleCanvas); map.on("resize",resizeParticleCanvas); }
function resizeParticleCanvas(){ if(!map||!particleCanvas||!particleCtx) return; const c=map.getCanvas(), w=c.clientWidth, h=c.clientHeight, dpr=window.devicePixelRatio||1; particleCanvas.width=Math.max(1,Math.round(w*dpr)); particleCanvas.height=Math.max(1,Math.round(h*dpr)); particleCanvas.style.width=w+"px"; particleCanvas.style.height=h+"px"; particleCtx.setTransform(dpr,0,0,dpr,0,0); clearCurrentCanvas(); }
function clearCurrentCanvas(){ if(!particleCtx||!map) return; const c=map.getCanvas(); particleCtx.clearRect(0,0,c.clientWidth,c.clientHeight); }

function speedToColor(s, vmin, vmax) {
    let t = (s - vmin) / (vmax - vmin);
    if (!Number.isFinite(t)) t = 0;
    t = Math.max(0, Math.min(1, t));

    const stops = [
        [0.18995, 0.07176, 0.23217],
        [0.25107, 0.25237, 0.63374],
        [0.27628, 0.60412, 0.96756],
        [0.20400, 0.77900, 0.42300],
        [0.99300, 0.90600, 0.14400],
        [0.97600, 0.45100, 0.08000],
        [0.70600, 0.01600, 0.15000]
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
function resetParticle(p){ const ll=randomValidPoint(); if(ll){p.lon=ll.lon;p.lat=ll.lat;} else {p.lon=meta.bounds[0][1];p.lat=meta.bounds[0][0];} p.age=Math.floor(Math.random()*100); p.maxAge=80+Math.floor(Math.random()*80); }
function resetParticles(){ particles=[]; if(!currentU||!currentV) return; for(let i=0;i<particleCount;i++){ const p={}; resetParticle(p); particles.push(p); } }
function startParticles(){ if(particleAnimId!==null) cancelAnimationFrame(particleAnimId); particleRunning=true; resetParticles(); function step(){ if(!particleRunning||!shouldDrawCurrentParticles()){ particleAnimId=requestAnimationFrame(step); return; } const canvas=map.getCanvas(), width=canvas.clientWidth, height=canvas.clientHeight; particleCtx.globalCompositeOperation="destination-in"; particleCtx.fillStyle="rgba(0,0,0,0.92)"; particleCtx.fillRect(0,0,width,height); particleCtx.globalCompositeOperation="source-over"; particleCtx.lineWidth=1.2; for(const p of particles){ if(!p||p.age>p.maxAge){resetParticle(p);continue;} const vec=vectorAt(p.lon,p.lat); if(!vec){resetParticle(p);continue;} const oldPoint=map.project([p.lon,p.lat]); const latRad=p.lat*Math.PI/180; let coslat=Math.cos(latRad); if(Math.abs(coslat)<1e-6) coslat=1e-6; const dt=CONFIG.flowScale*speed; const newLon=p.lon+(vec.u*dt)/coslat, newLat=p.lat+vec.v*dt; if(!vectorAt(newLon,newLat)){resetParticle(p);continue;} p.lon=newLon; p.lat=newLat; p.age++; const newPoint=map.project([p.lon,p.lat]); if(newPoint.x<-50||newPoint.x>width+50||newPoint.y<-50||newPoint.y>height+50){resetParticle(p);continue;} particleCtx.strokeStyle=currentParticleColor(vec.speed); particleCtx.beginPath(); particleCtx.moveTo(oldPoint.x,oldPoint.y); particleCtx.lineTo(newPoint.x,newPoint.y); particleCtx.stroke(); } particleAnimId=requestAnimationFrame(step); } particleAnimId=requestAnimationFrame(step); }
function stopParticles(){ particleRunning=false; if(particleAnimId!==null){ cancelAnimationFrame(particleAnimId); particleAnimId=null; } clearCurrentCanvas(); }
function setupEvents(){ els.currentOverlay.checked=true; els.currentOverlay.dataset.userTouched=""; els.currentOverlay.addEventListener("change",()=>{els.currentOverlay.dataset.userTouched="1"; updateCurrentOverlayAvailability(); setFrame(currentFrame);}); els.meshOverlay.addEventListener("change",()=>map.triggerRepaint()); els.varSelect.addEventListener("change",e=>{currentVar=e.target.value; updateCurrentOverlayAvailability(); updateLegend(); setFrame(currentFrame); map.triggerRepaint();}); els.playBtn.addEventListener("click",()=>{ if(timer===null){els.playBtn.textContent="Pause"; startTimer();} else {els.playBtn.textContent="Play"; clearInterval(timer); timer=null;} }); els.frameSlider.addEventListener("input",e=>setFrame(e.target.value)); els.speedSelect.addEventListener("change",e=>{ speed=parseFloat(e.target.value); if(timer!==null) startTimer(); }); els.opacitySlider.addEventListener("input",()=>map.triggerRepaint()); els.densitySelect.addEventListener("change",e=>{ particleCount=parseInt(e.target.value); resetParticles(); clearCurrentCanvas(); }); map.on("moveend",()=>resetParticles()); map.on("zoomend",()=>{ resizeParticleCanvas(); resetParticles(); }); }


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
async function boot(){ setStatus("Loading metadata..."); meta=await fetchJson(CONFIG.metaUrl); lookupMeta=await fetchJson(CONFIG.lookupMetaUrl); els.frameSlider.max=frameCount()-1; setStatus("Loading mesh and lookup..."); [nodesLonLat,elems,meshEdges,lookupOffsets,lookupTriangles]=await Promise.all([fetchFloat32(CONFIG.nodesUrl,meta.node_count*2),fetchUint32(CONFIG.elemsUrl,meta.index_count),fetchUint32(CONFIG.edgesUrl),fetchUint32(CONFIG.lookupOffsetsUrl),fetchUint32(CONFIG.lookupTrianglesUrl)]); initMap(); map.on("load",async()=>{ setStatus("Preparing WebGL layer..."); buildMercatorNodes(); initParticleCanvas(); map.addLayer(makeSchismLayer()); setupEvents(); updateCurrentOverlayAvailability(); updateLegend(); await setFrame(0); setStatus(`Ready: ${meta.node_count.toLocaleString()} nodes, ${meta.triangle_count.toLocaleString()} triangles`); }); }
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
