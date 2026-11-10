// Flamme 3D de la console Fotia : WebGL2 brut, sans bibliotheque.
//
// POURQUOI SANS THREE.JS. Une seule scene, un seul objet : un volume de flamme
// rendu par lancer de rayons dans un shader. Three.js n'apporterait rien ici,
// et le charger depuis un CDN casserait la page sans internet (le jour du jury).
//
// CE QUI EST CALCULE, CE QUI EST DESSINE.
//   * Calcule (Python, moteur hybride) : diametre de la goutte a chaque instant
//     (loi en d2), taille de l'enveloppe qui la suit, instant d'extinction.
//   * Dessine : couleur (bleu de la zone de reaction, jaune de la suie),
//     turbulence, forme en goutte sous gravite, souffle du curseur. La page le
//     dit sous l'image.
//
// Ce fichier est concatene AVANT console.js par flame/dashboard/console.py :
// il ne fait que declarer `Flame3D`, sans export.

const FLAME_VERT = `#version 300 es
in vec2 aPos;
void main() { gl_Position = vec4(aPos, 0.0, 1.0); }`;

const FLAME_FRAG = `#version 300 es
precision highp float;
uniform vec2 uRes;
uniform float uTime, uDrop, uFlame, uOn, uSoot, uTanHalf;
// uFlow : l'ecoulement autour de la goutte, en « etirements ». Flottabilite
// (vers le haut, 0 en microgravite) + air du curseur. La flamme s'etire et
// s'amincit dans son sens, et se rapproche de la goutte cote amont.
uniform vec3 uRo, uRight, uUp, uFwd, uFlow;
out vec4 outColor;

float hash(vec3 p) {
  p = fract(p * 0.3183099 + vec3(0.1, 0.2, 0.3));
  p *= 17.0;
  return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
float vnoise(vec3 x) {
  vec3 i = floor(x), f = fract(x);
  f = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(mix(hash(i), hash(i + vec3(1, 0, 0)), f.x),
        mix(hash(i + vec3(0, 1, 0)), hash(i + vec3(1, 1, 0)), f.x), f.y),
    mix(mix(hash(i + vec3(0, 0, 1)), hash(i + vec3(1, 0, 1)), f.x),
        mix(hash(i + vec3(0, 1, 1)), hash(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}
float fbm(vec3 p) {
  float s = 0.0, a = 0.5;
  for (int i = 0; i < 3; i++) { s += a * vnoise(p); p = p * 2.02 + vec3(1.7, 9.2, 3.1); a *= 0.5; }
  return s;
}

float flowSize() { return length(uFlow); }
vec3 flowDir() { float s = flowSize(); return s > 1e-4 ? uFlow / s : vec3(0.0, 1.0, 0.0); }
float flowGain() { return clamp(flowSize() / 1.2, 0.0, 1.0); }
// Centre du volume : pousse en aval de l'ecoulement.
vec3 lift() { return flowDir() * uFlame * flowSize() * 0.55; }

// x : distance au front de flamme, en epaisseurs de front ;
// y : position le long de l'ecoulement (-1 amont, +1 aval).
vec2 shape(vec3 p) {
  float R = max(uFlame, 1e-3);
  float s = flowSize(), g = flowGain();
  vec3 dir = flowDir();
  // Repere de l'ecoulement : composante le long de dir, et le reste.
  float a = dot(p, dir) - R * s * 0.55;
  vec3 perp = p - dot(p, dir) * dir;
  a /= 1.0 + s;                                     // etirement en aval
  float h = clamp(a / R, -1.2, 1.2);
  perp *= 1.0 + 0.55 * g * max(h, 0.0);             // pointe plus fine en aval
  perp *= 1.0 - 0.12 * g * max(-h, 0.0);            // tete plus ronde en amont
  vec3 q = perp + a * dir;
  // Turbulence emportee par l'ecoulement, plus forte en aval.
  float speed = 0.15 + 1.6 * g;
  float n = fbm(q / R * 2.4 - dir * uTime * speed) - 0.5;
  float amp = 0.05 + 0.45 * g * smoothstep(-0.3, 1.0, h);
  float d = length(q) + n * amp * R * 2.0;
  float w = R * (0.11 + 0.12 * g * max(h, 0.0));
  return vec2((d - R) / w, h);
}

vec3 emission(vec3 p) {
  vec2 s = shape(p);
  float x = s.x, h = s.y;
  float blue = exp(-(x - 0.9) * (x - 0.9) * 2.6);   // bord exterieur bleu
  float soot = exp(-(x + 0.55) * (x + 0.55) * 1.6);
  // Tete bleue cote amont, suie jaune en aval.
  float sootAmt = uSoot * (1.0 - 0.75 * flowGain() * smoothstep(0.0, -1.0, h));
  float inner = smoothstep(0.0, -6.0, x) * 0.05;
  return vec3(0.16, 0.34, 1.0) * blue * (0.5 + 0.9 * (1.0 - uSoot))
       + vec3(1.0, 0.52, 0.12) * soot * sootAmt * 1.5
       + vec3(1.0, 0.86, 0.55) * soot * soot * soot * sootAmt * 0.5
       + vec3(0.6, 0.3, 0.1) * inner * uSoot * 0.4;
}

void main() {
  vec2 uv = (gl_FragCoord.xy * 2.0 - uRes) / uRes.y;
  vec3 rd = normalize(uFwd + (uv.x * uRight + uv.y * uUp) * uTanHalf);
  vec3 ro = uRo;
  float vig = 1.0 - 0.35 * dot(uv * 0.6, uv * 0.6);
  vec3 col = vec3(0.012, 0.014, 0.022) * vig;

  // La goutte, une sphere a l'origine.
  float tDrop = 1e9;
  {
    float b = dot(ro, rd), c = dot(ro, ro) - uDrop * uDrop, hh = b * b - c;
    if (hh > 0.0) { float t = -b - sqrt(hh); if (t > 0.0) tDrop = t; }
  }


  // Le volume de flamme, parcouru dans sa sphere englobante.
  vec3 acc = vec3(0.0);
  if (uOn > 0.001 && uFlame > 1e-3) {
    vec3 oc = ro - lift();
    float Rb = uFlame * ((1.0 + flowSize()) * 1.15 + 0.6 + 0.6 * flowGain());
    float b = dot(oc, rd), c = dot(oc, oc) - Rb * Rb, hh = b * b - c;
    if (hh > 0.0) {
      float s = sqrt(hh);
      float t0 = max(-b - s, 0.0), t1 = min(-b + s, tDrop);
      const int N = 56;
      float dt = (t1 - t0) / float(N);
      float t = t0 + dt * hash(vec3(gl_FragCoord.xy, fract(uTime) * 61.0));
      for (int i = 0; i < N; i++) {
        if (t > t1) break;
        acc += emission(ro + rd * t);
        t += dt;
      }
      acc *= dt / uFlame * 0.85 * uOn;
    }
  }
  col += acc;

  if (tDrop < 1e8) {
    vec3 p = ro + rd * tDrop, n = normalize(p);
    float fres = pow(1.0 - max(dot(n, -rd), 0.0), 3.0);
    vec3 glow = mix(vec3(0.35, 0.5, 1.0), vec3(1.0, 0.62, 0.3), uSoot) * uOn;
    float spec = pow(max(dot(reflect(rd, n), normalize(vec3(0.4, 0.7, 0.5))), 0.0), 40.0);
    col = acc + vec3(0.10, 0.12, 0.16) * (0.5 + 0.5 * n.y) + glow * (0.25 + 0.9 * fres)
        + vec3(0.9, 0.95, 1.0) * spec * 0.6;
  }
  col = vec3(1.0) - exp(-col * 1.35);
  outColor = vec4(pow(col, vec3(0.9)), 1.0);
}`;

const FLAME_PREFS = "fotia.flame3d";

class Flame3D {
  // Renvoie null si WebGL2 manque ou si le shader ne compile pas : la
  // console garde alors son dessin 2D.
  static create(canvas, view) {
    try {
      const gl = canvas.getContext("webgl2", { antialias: false, premultipliedAlpha: false });
      if (!gl) return null;
      return new Flame3D(canvas, view, gl);
    } catch (err) {
      console.warn("Fotia: 3D flame unavailable,", err);
      return null;
    }
  }

  constructor(canvas, view, gl) {
    this.canvas = canvas;
    this.view = view;
    this.gl = gl;
    this.program = this.link(FLAME_VERT, FLAME_FRAG);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(this.program, "aPos");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    this.u = {};
    for (const name of ["uRes", "uTime", "uDrop", "uFlame", "uOn", "uSoot",
      "uTanHalf", "uRo", "uRight", "uUp", "uFwd", "uFlow"]) {
      this.u[name] = gl.getUniformLocation(this.program, name);
    }
    this.scale = 0.7;             // resolution interne, ajustee a la volee
    this.frames = [];
    this.yaw = 0.6; this.pitch = 0.18; this.zoom = 1;
    this.wind = [0, 0, 0];        // air du curseur, tel que la flamme le ressent
    this.cursor = null;           // position ecran du curseur sur la vue
    this.gust = 0;                // rafale nee du mouvement du curseur
    this.opts = { orbit: true, air: false };
    try {
      const saved = JSON.parse(localStorage.getItem(FLAME_PREFS) || "null");
      if (saved) this.opts = { ...this.opts, ...saved };
    } catch { /* stockage indisponible : reglages par defaut */ }
    this.scene = null;
    this.bindControls();
    this.resize();
  }

  link(vs, fs) {
    const gl = this.gl;
    const make = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const p = gl.createProgram();
    gl.attachShader(p, make(gl.VERTEX_SHADER, vs));
    gl.attachShader(p, make(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    gl.useProgram(p);
    return p;
  }

  setOption(name, value) {
    this.opts[name] = value;
    if (name === "air" && !value) { this.cursor = null; this.gust = 0; }
    try { localStorage.setItem(FLAME_PREFS, JSON.stringify(this.opts)); } catch { /* rien */ }
    this.view.classList.toggle("orbit", this.opts.orbit);
    this.view.classList.toggle("air", this.opts.air);
  }

  resetView() { this.yaw = 0.6; this.pitch = 0.18; this.zoom = 1; }

  resize() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    const r = Math.min(window.devicePixelRatio || 1, 1.5) * this.scale;
    this.canvas.width = Math.max(1, Math.round(w * r));
    this.canvas.height = Math.max(1, Math.round(h * r));
  }

  // --- Camera ---------------------------------------------------------------
  camera() {
    const sc = this.scene;
    const R0 = sc ? sc.flameR0 : 10;
    const st = sc ? sc.stretch : 1;
    const target = [0, R0 * (st - 1) * 0.45, 0];
    const dist = R0 * (1 + (st - 1) * 1.35) * 3.3 * this.zoom;
    const cp = Math.cos(this.pitch);
    const ro = [target[0] + dist * cp * Math.sin(this.yaw), target[1] + dist * Math.sin(this.pitch),
      target[2] + dist * cp * Math.cos(this.yaw)];
    const fwd = norm(sub(target, ro));
    const right = norm(cross(fwd, [0, 1, 0]));
    const up = cross(right, fwd);
    return { ro, fwd, right, up, target, tanHalf: Math.tan((38 * Math.PI) / 360) };
  }

  // Point du plan median (face a la camera) sous le curseur.
  pointUnder(clientX, clientY) {
    const rect = this.canvas.getBoundingClientRect();
    const u = ((clientX - rect.left) * 2 - rect.width) / rect.height;
    const v = (rect.height - (clientY - rect.top) * 2) / rect.height;
    const c = this.camera();
    const rd = norm(add(c.fwd, scale(add(scale(c.right, u), scale(c.up, v)), c.tanHalf)));
    const t = dot(sub(c.target, c.ro), c.fwd) / dot(rd, c.fwd);
    return { p: add(c.ro, scale(rd, t)), c };
  }

  bindControls() {
    const v = this.view;
    let drag = null;
    v.addEventListener("pointerdown", (e) => {
      if (!this.opts.orbit || e.target.closest("button")) return;
      drag = { x: e.clientX, y: e.clientY };
      v.setPointerCapture(e.pointerId);
      v.classList.add("dragging");
    });
    v.addEventListener("pointermove", (e) => {
      if (drag) {
        this.yaw -= (e.clientX - drag.x) * 0.008;
        this.pitch = Math.max(-1.2, Math.min(1.2, this.pitch + (e.clientY - drag.y) * 0.006));
        drag = { x: e.clientX, y: e.clientY };
      }
      if (this.opts.air) {
        // Un mouvement rapide fait une rafale ; elle retombe d'elle-meme.
        if (this.cursor) {
          const moved = Math.hypot(e.clientX - this.cursor.x, e.clientY - this.cursor.y);
          this.gust = Math.min(1.2, this.gust + moved / 140);
        }
        this.cursor = { x: e.clientX, y: e.clientY };
      }
    });
    const end = (e) => {
      if (drag) { drag = null; v.classList.remove("dragging"); try { v.releasePointerCapture(e.pointerId); } catch { /* rien */ } }
    };
    v.addEventListener("pointerup", end);
    v.addEventListener("pointercancel", end);
    v.addEventListener("pointerleave", () => { this.cursor = null; });
    v.addEventListener("dblclick", () => this.resetView());
    // Zoom avec Ctrl + molette seulement : la molette seule fait defiler la page.
    v.addEventListener("wheel", (e) => {
      if (!this.opts.orbit || !(e.ctrlKey || e.metaKey)) return;
      e.preventDefault();
      this.zoom = Math.max(0.45, Math.min(2.5, this.zoom * Math.exp(e.deltaY * 0.0012)));
    }, { passive: false });
    this.setOption("orbit", this.opts.orbit);
    this.setOption("air", this.opts.air);
  }

  // Le curseur est un petit ventilateur qui souffle VERS la flamme : un
  // courant d'air continu, plus fort de pres, plus une rafale quand il bouge.
  // La flamme ne se creuse pas sous le curseur : elle est emportee en aval,
  // comme une bougie dans un courant d'air. Purement illustratif : la vitesse
  // de combustion affichee ne tient pas compte de cet air.
  windTarget(now) {
    if (!this.opts.air || !this.cursor || !this.scene) return [0, 0, 0];
    const R0 = Math.max(this.scene.flameR0, 1e-3);
    const { p, c } = this.pointUnder(this.cursor.x, this.cursor.y);
    const toward = sub(c.target, p);                       // du curseur vers la flamme
    const near = Math.exp(-((Math.hypot(...toward) / (R0 * 2.6)) ** 2));
    let strength = 0.65 * near + this.gust * (0.35 + 0.5 * near);
    // Un vrai courant d'air n'est jamais regulier.
    const t = now / 1000;
    strength *= 1 + 0.12 * Math.sin(t * 9.1) + 0.08 * Math.sin(t * 15.7 + 1.3);
    return scale(norm(toward), Math.min(1.1, strength));
  }

  // --- Image ------------------------------------------------------------------
  frame(now, scene) {
    this.scene = scene;
    const gl = this.gl;
    // Qualite adaptative : on vise ~30 images/s meme sur un portable modeste.
    const last = this.prev ?? now;
    this.prev = now;
    this.frames.push(now - last);
    if (this.frames.length >= 40) {
      const avg = this.frames.reduce((a, b) => a + b, 0) / this.frames.length;
      this.frames = [];
      const before = this.scale;
      if (avg > 40 && this.scale > 0.4) this.scale = Math.max(0.4, this.scale * 0.85);
      else if (avg < 20 && this.scale < 1) this.scale = Math.min(1, this.scale * 1.1);
      if (before !== this.scale) this.resize();
    }
    if (!this.opts.orbit) this.yaw += 0.0025;     // lente rotation sans souris
    // La flamme suit l'air avec un temps de retard (~0,2 s) ; la rafale
    // retombe en ~0,35 s.
    const dt = Math.min(0.1, Math.max(0, (now - last) / 1000));
    this.gust *= Math.exp(-dt / 0.35);
    const target = this.windTarget(now);
    const k = 1 - Math.exp(-dt / 0.22);
    this.wind = this.wind.map((x, i) => x + (target[i] - x) * k);
    const flow = add([0, scene.stretch - 1, 0], this.wind);  // flottabilite + air

    const c = this.camera();
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    const u = this.u;
    gl.uniform2f(u.uRes, this.canvas.width, this.canvas.height);
    gl.uniform1f(u.uTime, now / 1000);
    gl.uniform1f(u.uDrop, scene.dropR);
    gl.uniform1f(u.uFlame, scene.flameR);
    gl.uniform1f(u.uOn, scene.on);
    gl.uniform1f(u.uSoot, scene.soot);
    gl.uniform1f(u.uTanHalf, c.tanHalf);
    gl.uniform3fv(u.uRo, c.ro);
    gl.uniform3fv(u.uRight, c.right);
    gl.uniform3fv(u.uUp, c.up);
    gl.uniform3fv(u.uFwd, c.fwd);
    gl.uniform3fv(u.uFlow, flow);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
}

function add(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function scale(a, s) { return [a[0] * s, a[1] * s, a[2] * s]; }
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function norm(a) { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; }
