const canvas = document.getElementById('glcanvas');
const gl = canvas.getContext('webgl2');
if (!gl) {
alert('WebGL2 not supported');
throw new Error('WebGL2 not supported');
}

const vsSource = `#version 300 es
in vec2 aPosition;
void main() {
gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

const fsTemplate = `#version 300 es
precision highp float;

uniform vec2 uResolution;
uniform vec3 uPlaneOrigin;
uniform vec3 uPlaneU;
uniform vec3 uPlaneV;
uniform vec3 uPlaneN;
uniform float uPlaneSize;
uniform float uPlaneShift;
uniform float uScale;
uniform sampler2D uColormap;

out vec4 fragColor;

float angular(float x, float y, float z) {
return __ANGULAR_EXPR__;
}

float radial(float x, float y, float z) {
float r = length(vec3(x,y,z));
return __RADIAL_EXPR__;
}

void main() {
vec2 pxy = (gl_FragCoord.xy - 0.5 * uResolution) / uResolution.y * uPlaneSize * 2.0;
vec3 P = uPlaneOrigin + uPlaneN * uPlaneShift + uPlaneU * pxy.x + uPlaneV * pxy.y;
float val = angular(P.x, P.y, P.z) * radial(P.x, P.y, P.z);
float t = clamp(val * uScale * 0.5 + 0.5, 0.0, 1.0);
vec3 color = texture(uColormap, vec2(t, 0.5)).rgb;
fragColor = vec4(color, 1.0);
}
`;

function createShader(type, source) {
const shader = gl.createShader(type);
gl.shaderSource(shader, source);
gl.compileShader(shader);
if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
const err = gl.getShaderInfoLog(shader);
gl.deleteShader(shader);
throw new Error(err);
}
return shader;
}

let program = null;
let uResolution, uPlaneOrigin, uPlaneU, uPlaneV, uPlaneN, uPlaneSize, uPlaneShift, uScale, uColormap;

function buildProgram(angularExpr, radialExpr) {
const fsSource = fsTemplate.replace('__ANGULAR_EXPR__', angularExpr).replace('__RADIAL_EXPR__', radialExpr);
const vs = createShader(gl.VERTEX_SHADER, vsSource);
const fs = createShader(gl.FRAGMENT_SHADER, fsSource);
const newProg = gl.createProgram();
gl.attachShader(newProg, vs);
gl.attachShader(newProg, fs);
gl.linkProgram(newProg);
if (!gl.getProgramParameter(newProg, gl.LINK_STATUS)) {
const err = gl.getProgramInfoLog(newProg);
gl.deleteProgram(newProg);
throw new Error(err);
}
if (program) gl.deleteProgram(program);
program = newProg;
gl.useProgram(program);

const posLoc = gl.getAttribLocation(program, 'aPosition');
gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
gl.enableVertexAttribArray(posLoc);
gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

uResolution = gl.getUniformLocation(program, 'uResolution');
uPlaneOrigin = gl.getUniformLocation(program, 'uPlaneOrigin');
uPlaneU = gl.getUniformLocation(program, 'uPlaneU');
uPlaneV = gl.getUniformLocation(program, 'uPlaneV');
uPlaneN = gl.getUniformLocation(program, 'uPlaneN');
uPlaneSize  = gl.getUniformLocation(program, 'uPlaneSize');
uPlaneShift = gl.getUniformLocation(program, 'uPlaneShift');
uScale      = gl.getUniformLocation(program, 'uScale');
uColormap   = gl.getUniformLocation(program, 'uColormap');
}

const quadBuf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
-1,-1, 1,-1, -1,1,
-1,1, 1,-1, 1,1
]), gl.STATIC_DRAW);

gl.viewport(0, 0, canvas.width, canvas.height);

const overlay = document.getElementById('overlay');
const octx = overlay.getContext('2d');
const colorbarCanvas = document.getElementById('colorbar');
const cbctx = colorbarCanvas.getContext('2d');

/* ---- Colormap textures ---- */
const COLORMAPS = {
  seismic:  [[0,0,60],[0,0,120],[0,40,180],[0,100,220],[255,255,255],[255,100,100],[220,40,0],[180,0,0],[120,0,0]],
  heat:     [[0,0,0],[40,0,0],[80,0,0],[160,0,0],[255,60,0],[255,140,0],[255,200,0],[255,240,100],[255,255,255]],
  viridis:  [[68,1,84],[72,35,115],[64,70,135],[52,100,140],[44,130,140],[36,160,140],[48,180,125],[100,200,100],[253,231,37]],
  coolwarm: [[5,48,97],[40,80,140],[80,120,190],[150,180,220],[255,255,255],[240,180,150],[220,100,100],[160,60,60],[100,0,0]]
};

const cmapTex = gl.createTexture();
gl.bindTexture(gl.TEXTURE_2D, cmapTex);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

function uploadColormap(name) {
  const colors = COLORMAPS[name];
  const data = new Uint8Array(9 * 3);
  for (let i = 0; i < 9; i++) {
    data[i*3] = colors[i][0]; data[i*3+1] = colors[i][1]; data[i*3+2] = colors[i][2];
  }
  gl.bindTexture(gl.TEXTURE_2D, cmapTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, 9, 1, 0, gl.RGB, gl.UNSIGNED_BYTE, data);
  drawColorbar();
}

function drawColorbar() {
  const W = colorbarCanvas.width, H = colorbarCanvas.height;
  const cmap = COLORMAPS[currentCmap];
  for (let y = 0; y < H; y++) {
    const t = 1 - y/(H-1);
    const idx = Math.min(Math.floor(t*8), 7);
    const f = t*8 - idx;
    const c1 = cmap[idx], c2 = cmap[idx+1];
    cbctx.fillStyle = `rgb(${Math.round(c1[0]+(c2[0]-c1[0])*f)},${Math.round(c1[1]+(c2[1]-c1[1])*f)},${Math.round(c1[2]+(c2[2]-c1[2])*f)})`;
    cbctx.fillRect(0, y, W, 1);
  }
  const vMax = (1.0/colorScale).toFixed(2);
  const vMin = (-1.0/colorScale).toFixed(2);
  document.getElementById('cmax').textContent = '+' + vMax;
  document.getElementById('cmin').textContent = vMin;
}

/* ---- State ---- */
let basisAngle = 0.0;
let colorScale = 1.0;
let planeSize = 2.0;
let planeShift = 0.0;
let planeOrigin = [0, 0, 0];
let planeNormal = [0, 0, 1];
let currentCmap = 'seismic';
let isDragging = false;
let lastMouse = [0, 0];

canvas.addEventListener('mousedown', e => { isDragging = true; lastMouse = [e.clientX, e.clientY]; });
window.addEventListener('mouseup', () => isDragging = false);
window.addEventListener('mousemove', e => {
  if (!isDragging) return;
  basisAngle += (e.clientX - lastMouse[0]) * 0.01 + (e.clientY - lastMouse[1]) * 0.01;
  lastMouse = [e.clientX, e.clientY];
});

canvas.addEventListener('wheel', e => {
  e.preventDefault();
  colorScale *= Math.exp(e.deltaY * -0.001);
  colorScale = Math.max(0.1, Math.min(colorScale, 10.0));
  document.getElementById('colorScale').value = colorScale.toFixed(1);
  document.getElementById('val-colorScale').textContent = colorScale.toFixed(1);
}, { passive: false });

/* ---- Controls ---- */
const presetSelect = document.getElementById('preset');
const exprInput   = document.getElementById('expr');
const elRadial    = document.getElementById('radial');
const errorDiv    = document.getElementById('error');
const elPlaneSize = document.getElementById('planeSize');
const elColorScale= document.getElementById('colorScale');
const elShift     = document.getElementById('shift');
const elCmap      = document.getElementById('colormap');
const elOX = document.getElementById('ox');
const elOY = document.getElementById('oy');
const elOZ = document.getElementById('oz');
const elNX = document.getElementById('nx');
const elNY = document.getElementById('ny');
const elNZ = document.getElementById('nz');

const RADIAL_EXPRS = {
  none: '1.0',
  gaussian: 'exp(-r*r)',
  slater: 'exp(-r)',
  lorentz: '1.0/(1.0+r*r)',
  morse: '(1.0-exp(-r))*(1.0-exp(-r))',
  smoothstep: '1.0 - smoothstep(0.0, 1.0, r)'
};

function readInputs() {
  planeSize = parseFloat(elPlaneSize.value);
  colorScale= parseFloat(elColorScale.value);
  planeShift= parseFloat(elShift.value);
  planeOrigin = [parseFloat(elOX.value), parseFloat(elOY.value), parseFloat(elOZ.value)];
  planeNormal = [parseFloat(elNX.value), parseFloat(elNY.value), parseFloat(elNZ.value)];
}
function updateLabels() {
  document.getElementById('val-planeSize').textContent = planeSize.toFixed(1);
  document.getElementById('val-colorScale').textContent = colorScale.toFixed(1);
  document.getElementById('val-shift').textContent = planeShift.toFixed(2);
}
function compile() {
  const radialExpr = RADIAL_EXPRS[elRadial.value] || '1.0';
  try { buildProgram(exprInput.value, radialExpr); errorDiv.textContent = ''; }
  catch (e) { errorDiv.textContent = e.message.split('\n')[0]; }
}
function onControlChange() { readInputs(); updateLabels(); }

elPlaneSize.addEventListener('input', onControlChange);
elColorScale.addEventListener('input', onControlChange);
elShift.addEventListener('input', onControlChange);
[elOX, elOY, elOZ, elNX, elNY, elNZ].forEach(el => el.addEventListener('input', onControlChange));

elCmap.addEventListener('change', () => { currentCmap = elCmap.value; uploadColormap(currentCmap); });

presetSelect.addEventListener('change', () => { if (presetSelect.value !== 'custom') { exprInput.value = presetSelect.value; compile(); } });
exprInput.addEventListener('input', () => { presetSelect.value = 'custom'; compile(); });
elRadial.addEventListener('change', compile);

document.getElementById('resetNormal').addEventListener('click', () => {
  elNX.value = 0; elNY.value = 0; elNZ.value = 1; basisAngle = 0; readInputs();
});

/* ---- Overlay drawing ---- */
function norm3(v) { const len = Math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); return len>1e-6 ? [v[0]/len,v[1]/len,v[2]/len] : [0,0,1]; }
function dot3(a,b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
function cross3(a,b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }

function computeBasis(n, angle) {
  n = norm3(n);
  const tmp = Math.abs(n[2]) < 0.999 ? [0,0,1] : [0,1,0];
  const u0 = norm3(cross3(tmp, n));
  const v0 = cross3(n, u0);
  const ca = Math.cos(angle), sa = Math.sin(angle);
  return { u: [u0[0]*ca+v0[0]*sa, u0[1]*ca+v0[1]*sa, u0[2]*ca+v0[2]*sa],
           v: [-u0[0]*sa+v0[0]*ca, -u0[1]*sa+v0[1]*ca, -u0[2]*sa+v0[2]*ca] };
}

function drawOverlay() {
  const M = 24; // margin around the 512x512 quad
  const W = overlay.width, H = overlay.height;
  octx.clearRect(0, 0, W, H);
  const dataH = H - 2*M;
  const scale = dataH / (planeSize * 2);
  const cx = W/2, cy = H/2;
  const tickLen = 6;

  octx.strokeStyle = 'rgba(200,200,200,0.7)';
  octx.fillStyle = '#bbb';
  octx.lineWidth = 1;

  // X-axis at bottom edge of quad (outside the colored area)
  const yBot = H - M;
  octx.beginPath(); octx.moveTo(M, yBot); octx.lineTo(W - M, yBot); octx.stroke();
  octx.textAlign = 'center'; octx.textBaseline = 'top';
  octx.font = '11px Consolas,monospace';
  for (let s = -planeSize; s <= planeSize + 0.001; s += 0.5) {
    const x = cx + s*scale;
    if (x >= M && x <= W - M) {
      octx.beginPath(); octx.moveTo(x, yBot); octx.lineTo(x, yBot + tickLen); octx.stroke();
      if (Math.abs(s) > 0.01 || s === 0) octx.fillText(s.toFixed(1), x, yBot + tickLen + 2);
    }
  }
  octx.font = 'bold 12px Consolas,monospace';
  octx.fillText('X', W - M - 6, yBot + tickLen + 4);

  // Y-axis at left edge of quad
  const xLeft = M;
  octx.beginPath(); octx.moveTo(xLeft, M); octx.lineTo(xLeft, H - M); octx.stroke();
  octx.textAlign = 'right'; octx.textBaseline = 'middle';
  octx.font = '11px Consolas,monospace';
  for (let s = -planeSize; s <= planeSize + 0.001; s += 0.5) {
    const y = cy - s*scale;
    if (y >= M && y <= H - M) {
      octx.beginPath(); octx.moveTo(xLeft, y); octx.lineTo(xLeft - tickLen, y); octx.stroke();
      if (Math.abs(s) > 0.01 || s === 0) octx.fillText(s.toFixed(1), xLeft - tickLen - 4, y);
    }
  }
  octx.font = 'bold 12px Consolas,monospace';
  octx.fillText('Y', xLeft - tickLen - 4, M + 4);
}

compile();
uploadColormap(currentCmap);

function getBasisVectors() {
  // Precompute orthonormal basis in JS with double precision
  const n = norm3(planeNormal);
  const tmp = Math.abs(n[2]) < 0.999 ? [0,0,1] : [0,1,0];
  const u0 = norm3(cross3(tmp, n));
  const v0 = cross3(n, u0);
  // Apply rotation
  const ca = Math.cos(basisAngle), sa = Math.sin(basisAngle);
  const u = [u0[0]*ca + v0[0]*sa, u0[1]*ca + v0[1]*sa, u0[2]*ca + v0[2]*sa];
  const v = [-u0[0]*sa + v0[0]*ca, -u0[1]*sa + v0[1]*ca, -u0[2]*sa + v0[2]*ca];
  return { u, v, n };
}

function render() {
  if (!program) return;
  gl.useProgram(program);
  const basis = getBasisVectors();
  gl.uniform2f(uResolution, canvas.width, canvas.height);
  gl.uniform3f(uPlaneOrigin, planeOrigin[0], planeOrigin[1], planeOrigin[2]);
  gl.uniform3f(uPlaneU, basis.u[0], basis.u[1], basis.u[2]);
  gl.uniform3f(uPlaneV, basis.v[0], basis.v[1], basis.v[2]);
  gl.uniform3f(uPlaneN, basis.n[0], basis.n[1], basis.n[2]);
  gl.uniform1f(uPlaneSize, planeSize);
  gl.uniform1f(uPlaneShift, planeShift);
  gl.uniform1f(uScale, colorScale);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, cmapTex);
  gl.uniform1i(uColormap, 0);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
  drawOverlay();
  drawColorbar();
  requestAnimationFrame(render);
}
render();
