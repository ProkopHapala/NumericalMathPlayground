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

const fsSlice = `#version 300 es
precision highp float;

uniform vec2 uResolution;
uniform vec3 uPlaneOrigin;
uniform vec3 uPlaneU;  // basis vector along viewport X (already rotated)
uniform vec3 uPlaneV;  // basis vector along viewport Y (already rotated)
uniform vec3 uPlaneN;  // plane normal (shift direction)
uniform float uPlaneSize;
uniform float uPlaneShift;
uniform float uScale;
uniform sampler2D uColormap;

out vec4 fragColor;

float func(float x, float y, float z) {
return __EXPR__;
}

void main() {
// Pixel position on the plane in world space
vec2 pxy = (gl_FragCoord.xy - 0.5 * uResolution) / uResolution.y * uPlaneSize * 2.0;
vec3 P = uPlaneOrigin + uPlaneN * uPlaneShift + uPlaneU * pxy.x + uPlaneV * pxy.y;

// Evaluate function at actual 3D point P (no normalization to sphere)
float val = func(P.x, P.y, P.z);

float t = clamp(val * uScale * 0.5 + 0.5, 0.0, 1.0);
vec3 color = texture(uColormap, vec2(t, 0.5)).rgb;
fragColor = vec4(color, 1.0);
}
`;

const fsVolumeMinMax = `#version 300 es
precision highp float;

uniform vec2 uResolution;
uniform vec3 uPlaneOrigin;
uniform vec3 uPlaneU;
uniform vec3 uPlaneV;
uniform vec3 uPlaneN;
uniform float uPlaneSize;
uniform float uScale;
uniform int uSteps;

out vec4 fragColor;

float func(float x, float y, float z) {
return __EXPR__;
}

// Ray-sphere intersection. Returns vec2(tNear,tFar), tNear>tFar means no hit.
bool raySphere(vec3 ro, vec3 rd, float R, out float tN, out float tF) {
  vec3 oc = ro; // sphere centered at origin
  float b = dot(oc, rd);
  float c = dot(oc, oc) - R*R;
  float disc = b*b - c;
  if (disc < 0.0) return false;
  float s = sqrt(disc);
  tN = -b - s;
  tF = -b + s;
  return tF > 0.0;
}

void main() {
vec2 pxy = (gl_FragCoord.xy - 0.5 * uResolution) / uResolution.y * uPlaneSize * 2.0;
vec3 ro = uPlaneOrigin + uPlaneU*pxy.x + uPlaneV*pxy.y - uPlaneN*uPlaneSize*2.0;
vec3 rd = uPlaneN;
float sR = uPlaneSize * 0.95;
float tN, tF;
if (!raySphere(ro - uPlaneOrigin, rd, sR, tN, tF)) { fragColor = vec4(0.02,0.02,0.08,1.0); return; }
float t0 = max(tN, 0.0);
float rayLen = tF - t0;
float dt = rayLen / float(uSteps);

float fmin =  1e30;
float fmax = -1e30;

for (int i = 0; i < 300; i++) {
if (i >= uSteps) break;
vec3 p = ro + rd * (t0 + float(i) * dt);
float v = func(p.x, p.y, p.z);
fmin = min(fmin, v);
fmax = max(fmax, v);
}

float r = clamp(fmin * uScale * 0.5 + 0.5, 0.0, 1.0);
float b = clamp(fmax * uScale * 0.5 + 0.5, 0.0, 1.0);
fragColor = vec4(r, 0.0, b, 1.0);
}
`;

const fsIsosurface = `#version 300 es
precision highp float;

uniform vec2 uResolution;
uniform vec3 uPlaneOrigin;
uniform vec3 uPlaneU;
uniform vec3 uPlaneV;
uniform vec3 uPlaneN;
uniform float uPlaneSize;
uniform float uScale;
uniform int uSteps;
uniform float uIsoval;

out vec4 fragColor;

float func(float x, float y, float z) {
return __EXPR__;
}

vec3 gradient(vec3 p) {
float e = 0.001;
float fx = func(p.x+e,p.y,p.z) - func(p.x-e,p.y,p.z);
float fy = func(p.x,p.y+e,p.z) - func(p.x,p.y-e,p.z);
float fz = func(p.x,p.y,p.z+e) - func(p.x,p.y,p.z-e);
vec3 g = vec3(fx, fy, fz) / (2.0*e);
float glen = length(g);
if (glen < 1e-6) return vec3(0.0, 0.0, 1.0);
return g / glen;
}

// Ray-sphere intersection.
bool raySphere(vec3 ro, vec3 rd, float R, out float tN, out float tF) {
  float b = dot(ro, rd);
  float c = dot(ro, ro) - R*R;
  float disc = b*b - c;
  if (disc < 0.0) return false;
  float s = sqrt(disc);
  tN = -b - s; tF = -b + s;
  return tF > 0.0;
}

void main() {
vec2 pxy = (gl_FragCoord.xy - 0.5 * uResolution) / uResolution.y * uPlaneSize * 2.0;
vec3 ro = uPlaneOrigin + uPlaneU * pxy.x + uPlaneV * pxy.y - uPlaneN * uPlaneSize * 2.0;
vec3 rd = uPlaneN;
float sR = uPlaneSize * 0.95;
float tN, tF;
if (!raySphere(ro - uPlaneOrigin, rd, sR, tN, tF)) { fragColor = vec4(0.02,0.02,0.08,1.0); return; }
float tStart = max(tN, 0.0);
float dt = (tF - tStart) / float(uSteps);

float prevVal = func((ro+rd*tStart).x, (ro+rd*tStart).y, (ro+rd*tStart).z) - uIsoval;
float t = 0.0;
bool hit = false;
vec3 hitPos;

for (int i = 0; i < 300; i++) {
if (i >= uSteps) break;
t = tStart + float(i) * dt;
vec3 p = ro + rd * t;
float val = func(p.x, p.y, p.z) - uIsoval;

if (i > 0 && prevVal * val < 0.0) {
float t0 = t - dt;
float t1 = t;
float v0 = prevVal;
float v1 = val;
for (int j = 0; j < 8; j++) {
float tm = (t0 + t1) * 0.5;
vec3 pm = ro + rd * tm;
float vm = func(pm.x, pm.y, pm.z) - uIsoval;
if (v0 * vm < 0.0) {
t1 = tm; v1 = vm;
} else {
t0 = tm; v0 = vm;
}
}
hitPos = ro + rd * ((t0 + t1) * 0.5);
hit = true;
break;
}
prevVal = val;
}

if (!hit) {
// dark blue background so we know shader ran but no surface found
fragColor = vec4(0.02, 0.02, 0.08, 1.0);
return;
}

vec3 N = gradient(hitPos);
vec3 L = normalize(vec3(0.5, 0.8, 1.0));
vec3 V = -rd;
float lambert = max(dot(N, L), 0.0);
vec3 R = reflect(-L, N);
float spec = pow(max(dot(R, V), 0.0), 32.0);
vec3 col = vec3(0.12, 0.10, 0.14) + vec3(0.6, 0.55, 0.5) * lambert + vec3(0.4) * spec;
fragColor = vec4(col, 1.0);
}
`;

// Filament: accumulate max of 1/(|Re|+|Im|+eps) — diverges along zero-crossing filament
const fsFilament = `#version 300 es
precision highp float;

uniform vec2 uResolution;
uniform vec3 uPlaneOrigin;
uniform vec3 uPlaneU;
uniform vec3 uPlaneV;
uniform vec3 uPlaneN;
uniform float uPlaneSize;
uniform float uScale;
uniform int uSteps;

out vec4 fragColor;

float funcR(float x, float y, float z) { return __EXPR_R__; }
float funcI(float x, float y, float z) { return __EXPR_I__; }
float funcW(float x, float y, float z) { return __EXPR_W__; }

// Ray-sphere intersection.
bool raySphere(vec3 ro, vec3 rd, float R, out float tN, out float tF) {
  float b = dot(ro, rd);
  float c = dot(ro, ro) - R*R;
  float disc = b*b - c;
  if (disc < 0.0) return false;
  float s = sqrt(disc);
  tN = -b - s; tF = -b + s;
  return tF > 0.0;
}

void main() {
vec2 pxy = (gl_FragCoord.xy - 0.5*uResolution) / uResolution.y * uPlaneSize * 2.0;
vec3 ro = uPlaneOrigin + uPlaneU*pxy.x + uPlaneV*pxy.y - uPlaneN*uPlaneSize*2.0;
vec3 rd = uPlaneN;
float sR = uPlaneSize * 0.95;
float tN, tF;
if (!raySphere(ro - uPlaneOrigin, rd, sR, tN, tF)) { fragColor = vec4(0.02,0.02,0.08,1.0); return; }
float tStart = max(tN, 0.0);
float dt = (tF - tStart) / float(uSteps);
float eps = 0.02 / uScale;

float maxVal = 0.0;
for (int i = 0; i < 300; i++) {
if (i >= uSteps) break;
vec3 p = ro + rd*(tStart + float(i)*dt);
float r = funcR(p.x,p.y,p.z);
float im = funcI(p.x,p.y,p.z);
float w = funcW(p.x,p.y,p.z);
float v = w / (abs(r) + abs(im) + eps);
maxVal = max(maxVal, v);
}

float t = clamp(maxVal * uScale * 0.05, 0.0, 1.0);
vec3 col = mix(vec3(0.02,0.02,0.08), vec3(1.0,0.9,0.3), t*t);
fragColor = vec4(col, 1.0);
}
`;

// Chiral filament: colors the zero-crossing filament by sign(dot(r, cross(∇R,∇I))) = sign(xyz)
// Red = xyz>0 (tetrahedral), Blue = xyz<0 (anti-tetrahedral)
// Brightness = max along ray of W(x,y,z)/(|R|+|I|+eps)
const fsFilamentChiral = `#version 300 es
precision highp float;

uniform vec2 uResolution;
uniform vec3 uPlaneOrigin;
uniform vec3 uPlaneU;
uniform vec3 uPlaneV;
uniform vec3 uPlaneN;
uniform float uPlaneSize;
uniform float uScale;
uniform int uSteps;

out vec4 fragColor;

float funcR(float x, float y, float z) { return __EXPR_R__; }
float funcI(float x, float y, float z) { return __EXPR_I__; }
float funcW(float x, float y, float z) { return __EXPR_W__; }

bool raySphere(vec3 ro, vec3 rd, float R, out float tN, out float tF) {
  float b = dot(ro, rd);
  float c = dot(ro, ro) - R*R;
  float disc = b*b - c;
  if (disc < 0.0) return false;
  float s = sqrt(disc);
  tN = -b - s; tF = -b + s;
  return tF > 0.0;
}

void main() {
vec2 pxy = (gl_FragCoord.xy - 0.5*uResolution) / uResolution.y * uPlaneSize * 2.0;
vec3 ro = uPlaneOrigin + uPlaneU*pxy.x + uPlaneV*pxy.y - uPlaneN*uPlaneSize*2.0;
vec3 rd = uPlaneN;
float sR = uPlaneSize * 0.95;
float tN, tF;
if (!raySphere(ro - uPlaneOrigin, rd, sR, tN, tF)) { fragColor = vec4(0.02,0.02,0.08,1.0); return; }
float tStart = max(tN, 0.0);
float dt = (tF - tStart) / float(uSteps);
float eps = 0.02 / uScale;

// Accumulate max brightness separately for each chirality
float maxRed  = 0.0;  // xyz > 0
float maxBlue = 0.0;  // xyz < 0

for (int i = 0; i < 300; i++) {
if (i >= uSteps) break;
vec3 p = ro + rd*(tStart + float(i)*dt);
float r  = funcR(p.x,p.y,p.z);
float im = funcI(p.x,p.y,p.z);
float w  = funcW(p.x,p.y,p.z);                 // envelope (1.0 if unused)
float brightness = w / (abs(r) + abs(im) + eps);
// chirality = sign of r·(∇R×∇I) = sign(xyz) (for R=x²-y², I=y²-z²; general: numeric)
// We compute it numerically: ∇R×∇I via finite diff, then dot with p
float e = 0.005;
vec3 gR = vec3(funcR(p.x+e,p.y,p.z)-funcR(p.x-e,p.y,p.z),
               funcR(p.x,p.y+e,p.z)-funcR(p.x,p.y-e,p.z),
               funcR(p.x,p.y,p.z+e)-funcR(p.x,p.y,p.z-e));
vec3 gI = vec3(funcI(p.x+e,p.y,p.z)-funcI(p.x-e,p.y,p.z),
               funcI(p.x,p.y+e,p.z)-funcI(p.x,p.y-e,p.z),
               funcI(p.x,p.y,p.z+e)-funcI(p.x,p.y,p.z-e));
float chirality = dot(p, cross(gR, gI));
if (chirality >= 0.0) maxRed  = max(maxRed,  brightness);
else                  maxBlue = max(maxBlue, brightness);
}

float scl = uScale * 0.05;
float red  = clamp(maxRed  * scl, 0.0, 1.0);
float blue = clamp(maxBlue * scl, 0.0, 1.0);
red  = red*red; blue = blue*blue;  // gamma for contrast

vec3 bg  = vec3(0.02,0.02,0.08);
vec3 col = bg + vec3(red*0.9, red*0.15, 0.0) + vec3(0.0, blue*0.2, blue*0.9);
col = min(col, vec3(1.0));
fragColor = vec4(col, 1.0);
}
`;

// Zero planes: front-to-back alpha blend — Re surface red, Im surface blue, both shaded
const fsZeroPlanes = `#version 300 es
precision highp float;

uniform vec2 uResolution;
uniform vec3 uPlaneOrigin;
uniform vec3 uPlaneU;
uniform vec3 uPlaneV;
uniform vec3 uPlaneN;
uniform float uPlaneSize;
uniform float uScale;
uniform int uSteps;

out vec4 fragColor;

float funcR(float x, float y, float z) { return __EXPR_R__; }
float funcI(float x, float y, float z) { return __EXPR_I__; }
float funcW(float x, float y, float z) { return __EXPR_W__; }

vec3 gradR(vec3 p) {
float e = 0.002;
return vec3(
funcR(p.x+e,p.y,p.z)-funcR(p.x-e,p.y,p.z),
funcR(p.x,p.y+e,p.z)-funcR(p.x,p.y-e,p.z),
funcR(p.x,p.y,p.z+e)-funcR(p.x,p.y,p.z-e)) / (2.0*e);
}

vec3 gradI(vec3 p) {
float e = 0.002;
return vec3(
funcI(p.x+e,p.y,p.z)-funcI(p.x-e,p.y,p.z),
funcI(p.x,p.y+e,p.z)-funcI(p.x,p.y-e,p.z),
funcI(p.x,p.y,p.z+e)-funcI(p.x,p.y,p.z-e)) / (2.0*e);
}

vec3 shade(vec3 N, vec3 rd, vec3 albedo) {
N = normalize(N);
vec3 L = normalize(vec3(0.5,0.8,1.0));
float lam = max(dot(N,L),0.0) + max(dot(-N,L),0.0); // double-sided
vec3 R = reflect(-L,N);
float spec = pow(max(dot(R,-rd),0.0),24.0);
return albedo*(0.15 + 0.7*lam) + vec3(0.4)*spec;
}

// Ray-sphere intersection.
bool raySphere(vec3 ro, vec3 rd, float R, out float tN, out float tF) {
  float b = dot(ro, rd);
  float c = dot(ro, ro) - R*R;
  float disc = b*b - c;
  if (disc < 0.0) return false;
  float s = sqrt(disc);
  tN = -b - s; tF = -b + s;
  return tF > 0.0;
}

void main() {
vec2 pxy = (gl_FragCoord.xy - 0.5*uResolution) / uResolution.y * uPlaneSize * 2.0;
vec3 ro = uPlaneOrigin + uPlaneU*pxy.x + uPlaneV*pxy.y - uPlaneN*uPlaneSize*2.0;
vec3 rd = uPlaneN;
float sR = uPlaneSize * 0.95;
float tN, tF;
if (!raySphere(ro - uPlaneOrigin, rd, sR, tN, tF)) { fragColor = vec4(0.02,0.02,0.08,1.0); return; }
float tStart = max(tN, 0.0);
float dt = (tF - tStart) / float(uSteps);

// front-to-back compositing
vec4 accum = vec4(0.0);

vec3 p0 = ro + rd*tStart;
float prevR = funcR(p0.x,p0.y,p0.z);
float prevI = funcI(p0.x,p0.y,p0.z);

for (int i = 1; i < 300; i++) {
if (i >= uSteps) break;
if (accum.a > 0.98) break;

float t = tStart + float(i)*dt;
vec3 p = ro + rd*t;
float curR = funcR(p.x,p.y,p.z);
float curI = funcI(p.x,p.y,p.z);

// Re crossing
if (prevR*curR < 0.0) {
float frac = prevR/(prevR-curR);
vec3 hit = ro + rd*(t - dt + frac*dt);
float ww = max(0.0, funcW(hit.x,hit.y,hit.z));  // envelope gates alpha
vec3 N = gradR(hit);
vec3 col = shade(N, rd, vec3(0.85,0.15,0.1));  // red
float alpha = 0.55 * ww * (1.0 - accum.a);
accum.rgb += col*alpha;
accum.a   += alpha;
}
// Im crossing
if (prevI*curI < 0.0) {
float frac = prevI/(prevI-curI);
vec3 hit = ro + rd*(t - dt + frac*dt);
float ww = max(0.0, funcW(hit.x,hit.y,hit.z));
vec3 N = gradI(hit);
vec3 col = shade(N, rd, vec3(0.1,0.3,0.9));   // blue
float alpha = 0.55 * ww * (1.0 - accum.a);
accum.rgb += col*alpha;
accum.a   += alpha;
}

prevR = curR;
prevI = curI;
}

vec3 bg = vec3(0.02,0.02,0.08);
vec3 final = accum.rgb + bg*(1.0 - accum.a);
fragColor = vec4(final, 1.0);
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

const programs = {};
let program = null;
let uResolution, uPlaneOrigin, uPlaneU, uPlaneV, uPlaneN, uPlaneSize, uPlaneShift, uScale, uColormap, uSteps, uIsoval;

function buildProgram(expr, mode, exprR, exprI, exprW) {
uSteps = undefined;
uIsoval = undefined;
let tpl;
if (mode === 'slice') tpl = fsSlice;
else if (mode === 'volumeMinMax') tpl = fsVolumeMinMax;
else if (mode === 'isosurface') tpl = fsIsosurface;
else if (mode === 'filament') tpl = fsFilament;
else if (mode === 'filamentChiral') tpl = fsFilamentChiral;
else if (mode === 'zeroplanes') tpl = fsZeroPlanes;
else throw new Error('Unknown mode: ' + mode);

let fsSource = tpl.replace(/__EXPR__/g, expr);
const dualMode = mode === 'filament' || mode === 'filamentChiral' || mode === 'zeroplanes';
if (dualMode) {
fsSource = fsSource
  .replace(/__EXPR_R__/g, exprR)
  .replace(/__EXPR_I__/g, exprI)
  .replace(/__EXPR_W__/g, (exprW && exprW.trim()) ? exprW : '1.0');
}
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
const locSteps = gl.getUniformLocation(program, 'uSteps');
if (locSteps !== null) uSteps = locSteps;
const locIsoval = gl.getUniformLocation(program, 'uIsoval');
if (locIsoval !== null) uIsoval = locIsoval;
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
  seismic:  [[0,0,60],[0,0,120],[0,40,180],[0,100,220],[180,180,255],[255,100,100],[220,40,0],[180,0,0],[120,0,0]],
  heat:     [[0,0,0],[40,0,0],[80,0,0],[160,0,0],[255,60,0],[255,140,0],[255,200,0],[255,240,100],[255,255,255]],
  viridis:  [[68,1,84],[72,35,115],[64,70,135],[52,100,140],[44,130,140],[36,160,140],[48,180,125],[100,200,100],[253,231,37]],
  coolwarm: [[5,48,97],[40,80,140],[80,120,190],[150,180,220],[220,220,240],[240,180,150],[220,100,100],[160,60,60],[100,0,0]]
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
let colorScale = 1.0;
let planeSize = 2.0;
let planeShift = 0.0;
let planeOrigin = [0, 0, 0];
let currentCmap = 'seismic';
let renderMode = 'slice';
let volumeSteps = 100;
let isoValue = 0.5;
let exprR = 'x*x*x - 3.0*x*y*y';
let exprI = '(3.0*x*x*y - y*y*y)*z';
let exprW = '1.0';
let isDragging = false;
let lastMouse = [0, 0];
let needsRender = true;
let showAxes = true;
let autoUpdate = true;

/* ---- Quaternion trackball ---- */
// q = [w, x, y, z]
let camQuat = [1, 0, 0, 0];  // identity

function qMul(a, b) {
  return [
    a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
    a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
    a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
    a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]
  ];
}
function qNorm(q) {
  const l = Math.sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
  return l>1e-10 ? [q[0]/l,q[1]/l,q[2]/l,q[3]/l] : [1,0,0,0];
}
function qRotVec(q, v) {
  // rotate vector v by unit quaternion q: q * [0,v] * q^-1
  const [w,x,y,z] = q;
  const [vx,vy,vz] = v;
  const tx = 2*(y*vz - z*vy);
  const ty = 2*(z*vx - x*vz);
  const tz = 2*(x*vy - y*vx);
  return [vx + w*tx + y*tz - z*ty,
          vy + w*ty + z*tx - x*tz,
          vz + w*tz + x*ty - y*tx];
}
// Map 2D canvas pixel to trackball sphere point (unit vec).
// Inside unit circle: project onto hemisphere. Outside: hyperbolic sheet (smooth roll).
function trackballProject(px, py) {
  const W = canvas.width, H = canvas.height;
  const R = Math.min(W, H) * 0.45;
  const cx = W*0.5, cy = H*0.5;
  const x = (px - cx) / R;
  const y = -(py - cy) / R;  // flip canvas Y → math Y
  const d2 = x*x + y*y;
  const z = d2 <= 0.5 ? Math.sqrt(1.0 - d2)   // hemisphere
                      : 0.5 / Math.sqrt(d2);  // hyperbolic sheet, continuous at d²=0.5
  const l = Math.sqrt(x*x + y*y + z*z);
  return [x/l, y/l, z/l];
}
function trackballDelta(p0, p1) {
  // rotation axis = cross(p0,p1), angle = acos(dot)
  const dot = Math.min(1, p0[0]*p1[0]+p0[1]*p1[1]+p0[2]*p1[2]);
  const angle = Math.acos(dot);
  if (angle < 1e-6) return [1,0,0,0];
  const ax = p0[1]*p1[2]-p0[2]*p1[1];
  const ay = p0[2]*p1[0]-p0[0]*p1[2];
  const az = p0[0]*p1[1]-p0[1]*p1[0];
  const al = Math.sqrt(ax*ax+ay*ay+az*az);
  const s = Math.sin(angle*0.5)/al;
  return [Math.cos(angle*0.5), ax*s, ay*s, az*s];
}

canvas.addEventListener('mousedown', e => {
  const rect = canvas.getBoundingClientRect();
  isDragging = true;
  lastMouse = [e.clientX - rect.left, e.clientY - rect.top];
});
window.addEventListener('mouseup', () => { if (isDragging) { isDragging = false; needsRender = true; } });
window.addEventListener('mousemove', e => {
  if (!isDragging) return;
  const rect = canvas.getBoundingClientRect();
  const cur = [e.clientX - rect.left, e.clientY - rect.top];
  const p0 = trackballProject(lastMouse[0], lastMouse[1]);
  const p1 = trackballProject(cur[0], cur[1]);
  const dq = trackballDelta(p0, p1);
  camQuat = qNorm(qMul(camQuat, dq));  // right-multiply: dq in camera/screen frame
  lastMouse = cur;
  if (autoUpdate) needsRender = true;
  else { drawOverlay(); }  // always update axis gizmo even without re-render
});

canvas.addEventListener('wheel', e => {
  e.preventDefault();
  colorScale *= Math.exp(e.deltaY * -0.001);
  colorScale = Math.max(0.1, Math.min(colorScale, 10.0));
  document.getElementById('colorScale').value = colorScale.toFixed(1);
  document.getElementById('val-colorScale').textContent = colorScale.toFixed(1);
  needsRender = true;
}, { passive: false });

/* ---- Controls ---- */
const presetSelect = document.getElementById('preset');
const exprInput   = document.getElementById('expr');
const exprInputR  = document.getElementById('exprR');
const exprInputI  = document.getElementById('exprI');
const errorDiv    = document.getElementById('error');
const elPlaneSize = document.getElementById('planeSize');
const elColorScale= document.getElementById('colorScale');
const elShift     = document.getElementById('shift');
const elCmap      = document.getElementById('colormap');
const elRenderMode= document.getElementById('renderMode');
const elSteps     = document.getElementById('steps');
const elIsoval    = document.getElementById('isoval');
const elOX = document.getElementById('ox');
const elOY = document.getElementById('oy');
const elOZ = document.getElementById('oz');
const exprInputW = document.getElementById('exprW');

function readInputs() {
  planeSize = parseFloat(elPlaneSize.value);
  colorScale= parseFloat(elColorScale.value);
  planeShift= parseFloat(elShift.value);
  volumeSteps = parseInt(elSteps.value);
  isoValue = parseFloat(elIsoval.value);
  renderMode = elRenderMode.value;
  exprR = exprInputR.value;
  exprI = exprInputI.value;
  exprW = exprInputW.value;
  planeOrigin = [parseFloat(elOX.value), parseFloat(elOY.value), parseFloat(elOZ.value)];
}
function updateLabels() {
  document.getElementById('val-planeSize').textContent = planeSize.toFixed(1);
  document.getElementById('val-colorScale').textContent = colorScale.toFixed(1);
  document.getElementById('val-shift').textContent = planeShift.toFixed(2);
  document.getElementById('val-steps').textContent = volumeSteps;
  document.getElementById('val-isoval').textContent = isoValue.toFixed(2);
}
function compile() {
  for (const k in programs) { gl.deleteProgram(programs[k]); delete programs[k]; }
  try { buildProgram(exprInput.value, renderMode, exprR, exprI, exprW); errorDiv.textContent = ''; }
  catch (e) { errorDiv.textContent = e.message.split('\n')[0]; }
}
function onControlChange() { readInputs(); updateLabels(); needsRender = true; }

elPlaneSize.addEventListener('input', onControlChange);
elColorScale.addEventListener('input', onControlChange);
elShift.addEventListener('input', onControlChange);
[elOX, elOY, elOZ].forEach(el => el.addEventListener('input', onControlChange));

elCmap.addEventListener('change', () => { currentCmap = elCmap.value; uploadColormap(currentCmap); needsRender = true; });

elRenderMode.addEventListener('change', () => { readInputs(); updateLabels(); compile(); needsRender = true; });
elSteps.addEventListener('input', onControlChange);
elIsoval.addEventListener('input', onControlChange);

presetSelect.addEventListener('change', () => {
  if (presetSelect.value !== 'custom') {
    const val = presetSelect.value;
    if (val.includes('|')) {
      const parts = val.split('|');
      exprInputR.value = parts[0]; exprInputI.value = parts[1];
      exprR = parts[0]; exprI = parts[1];
      if (parts[2] !== undefined) { exprInputW.value = parts[2]; exprW = parts[2]; }
    } else { exprInput.value = val; }
    compile(); needsRender = true;
  }
});
exprInput.addEventListener('input', () => { presetSelect.value = 'custom'; compile(); needsRender = true; });
exprInputR.addEventListener('input', () => { exprR = exprInputR.value; presetSelect.value = 'custom'; compile(); needsRender = true; });
exprInputI.addEventListener('input', () => { exprI = exprInputI.value; presetSelect.value = 'custom'; compile(); needsRender = true; });
exprInputW.addEventListener('input', () => { exprW = exprInputW.value; compile(); needsRender = true; });

document.getElementById('resetNormal').addEventListener('click', () => {
  camQuat = [1,0,0,0]; readInputs(); needsRender = true;
});
document.getElementById('showAxes').addEventListener('change', e => { showAxes = e.target.checked; needsRender = true; });
document.getElementById('autoUpdate').addEventListener('change', e => { autoUpdate = e.target.checked; });
document.getElementById('btnUpdate').addEventListener('click', () => { needsRender = true; });

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

// Project a 3D world point onto the overlay canvas using orthographic projection
function worldToScreen(p, basis, W, H) {
  const px = p[0]-planeOrigin[0], py = p[1]-planeOrigin[1], pz = p[2]-planeOrigin[2];
  const su = (px*basis.u[0]+py*basis.u[1]+pz*basis.u[2]) / planeSize;
  const sv = (px*basis.v[0]+py*basis.v[1]+pz*basis.v[2]) / planeSize;
  return [W*0.5 + su*W*0.5, H*0.5 - sv*H*0.5];
}

function drawOverlay() {
  const W = overlay.width, H = overlay.height;
  octx.clearRect(0, 0, W, H);

  if (showAxes) {
    const basis = getBasisVectors();
    const axLen = planeSize * 0.7;
    const origin3 = [0,0,0];
    const axes = [
      { end: [axLen,0,0], color: '#ff4444', label: 'X' },
      { end: [0,axLen,0], color: '#44ff44', label: 'Y' },
      { end: [0,0,axLen], color: '#4488ff', label: 'Z' },
    ];
    const os = worldToScreen(origin3, basis, W, H);
    octx.lineWidth = 2;
    for (const ax of axes) {
      const es = worldToScreen(ax.end, basis, W, H);
      // Axis line
      octx.strokeStyle = ax.color;
      octx.beginPath(); octx.moveTo(os[0], os[1]); octx.lineTo(es[0], es[1]); octx.stroke();
      // Arrowhead
      const dx = es[0]-os[0], dy = es[1]-os[1];
      const len = Math.sqrt(dx*dx+dy*dy);
      if (len > 4) {
        const ux = dx/len, uy = dy/len;
        const ah = 10, aw = 5;
        octx.fillStyle = ax.color;
        octx.beginPath();
        octx.moveTo(es[0], es[1]);
        octx.lineTo(es[0]-ah*ux+aw*uy, es[1]-ah*uy-aw*ux);
        octx.lineTo(es[0]-ah*ux-aw*uy, es[1]-ah*uy+aw*ux);
        octx.closePath(); octx.fill();
      }
      // Label
      octx.fillStyle = ax.color;
      octx.font = 'bold 13px Consolas,monospace';
      octx.textAlign = 'center'; octx.textBaseline = 'middle';
      octx.fillText(ax.label, es[0] + (es[0]-os[0])*0.18, es[1] + (es[1]-os[1])*0.18);
    }
    // Origin dot
    octx.fillStyle = '#aaa';
    octx.beginPath(); octx.arc(os[0], os[1], 3, 0, Math.PI*2); octx.fill();
  }
}

readInputs();
compile();
uploadColormap(currentCmap);

function getBasisVectors() {
  // Apply quaternion trackball to world axes
  const u = qRotVec(camQuat, [1,0,0]);
  const v = qRotVec(camQuat, [0,1,0]);
  const n = qRotVec(camQuat, [0,0,1]);
  return { u, v, n };
}

function render() {
  if (!program || !needsRender) {
    requestAnimationFrame(render);
    return;
  }
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
  if (uSteps !== undefined) gl.uniform1i(uSteps, volumeSteps);
  if (uIsoval !== undefined) gl.uniform1f(uIsoval, isoValue);
  if (renderMode === 'slice') {
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, cmapTex);
    gl.uniform1i(uColormap, 0);
  }
  gl.drawArrays(gl.TRIANGLES, 0, 6);
  drawOverlay();
  drawColorbar();
  needsRender = false;
  requestAnimationFrame(render);
}
render();
