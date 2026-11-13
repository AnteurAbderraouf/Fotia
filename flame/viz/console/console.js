// Console Fotia : affichage et saisie seulement. Aucun calcul physique ici,
// tout vient de Python dans `data.result` (voir flame/dashboard/console.py).
//
// Streamlit rappelle cette fonction a chaque rendu avec de nouvelles donnees.
// Elle doit donc etre idempotente : la premiere fois elle construit et lance
// l'animation, ensuite elle ne fait que mettre a jour. L'etat local est range
// sur l'element parent.

const KINDS = {
  fire: "Real fire",
  experiment: "Controlled fire test",
  result: "Published result",
  simulation: "Simulation",
  reference: "Reference",
};

const BADGE = {
  inside: ["Measured here", "a NASA test was run close to this setting"],
  extrapolation: ["Between tests", "each value was tested, not this mix"],
  outside: ["Never tested", "no NASA test this far out"],
};

const GRAVITY_TICKS = [
  [0, "µg"], [1, "Earth"],  // Lune et Mars masquees pour l'instant (console.py)
];

const SEND_DELAY = 140; // ms entre deux envois pendant un glissement

function esc(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

function link(url, text) {
  if (!url) return esc(text);
  return `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(text)} ↗</a>`;
}

function pct(x, digits = 0) { return `${(x * 100).toFixed(digits)}%`; }

function gLabel(g) {
  if (g < 0.01) return "µg";
  if (Math.abs(g - 1) < 1e-9) return "1 g";
  return `${g.toFixed(3)} g`;
}

function pill(verdict, text) {
  return `<span class="verdict ${esc(verdict)}">${esc(text ?? BADGE[verdict][0])}</span>`;
}

// Icones (lucide, trait 1,5) du bouton de lecture.
const ICON = {
  flame: '<path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"/>',
  play: '<polygon points="6 3 20 12 6 21 6 3"/>',
  pause: '<rect x="14" y="4" width="4" height="16" rx="1"/><rect x="6" y="4" width="4" height="16" rx="1"/>',
  replay: '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>',
};
const svgIcon = (name) => `<svg viewBox="0 0 24 24" aria-hidden="true">${ICON[name]}</svg>`;

// --- Definition des commandes ------------------------------------------------

function controlsSpec(cond) {
  const dil = cond.diluent;
  const specs = [
    { k: "o2_frac", label: "Oxygen", step: 0.01, fmt: (v) => pct(v), ticks: null },
    {
      k: "pressure_atm", label: "Pressure", step: 0.01,
      fmt: (v) => `${v.toFixed(2)} atm<small>${(v * 14.696).toFixed(1)} psia</small>`, ticks: null,
    },
    { k: "gravity_g", label: "Gravity", step: 0.001, fmt: gLabel, ticks: GRAVITY_TICKS },
    { seg: "fuel", label: "Fuel" },
    { seg: "diluent", label: "Diluent" },
  ];
  if (dil !== "none") {
    specs.push({
      k: "diluent_frac", label: dil === "co2" ? "Added CO₂" : "Added helium",
      step: 0.01, fmt: (v) => pct(v), ticks: null, band: dil === "co2" ? "co2_frac" : "he_frac",
    });
  }
  specs.push({ k: "d0_mm", label: "Droplet size", step: 0.05, fmt: (v) => `${v.toFixed(2)} mm`, ticks: null });
  return specs;
}

const SEGMENTS = {
  fuel: [["Methanol", "Methanol"], ["Heptane", "Heptane"], ["n-decane", "n-Decane"]],
  diluent: [["none", "Nitrogen"], ["co2", "+ CO₂"], ["he", "+ Helium"]],
};

// --- Construction ----------------------------------------------------------

function build(root, S) {
  const $ = (id) => root.querySelector(`#${id}`);
  S.el = {
    places: $("places"), placeCite: $("placeCite"), controls: $("controls"),
    view: $("view"), cv: $("cv"), badge: $("badge"), clock: $("clock"),
    beyond: $("beyond"), state: $("state"), err: $("err"), curve: $("curve"),
    hatch: $("hatch"), vtools: $("vtools"), vhint: $("vhint"), curveEnd: $("curveEnd"),
    answers: $("answers"), map: $("map"), unplaced: $("unplaced"), std: $("std"),
    tabs: $("tabs"), whyText: $("whyText"), cards: $("cards"), legend: $("legend"),
    foot: $("foot"),
  };
  S.el.legend.innerHTML = Object.entries(BADGE)
    .map(([v, [name, text]]) => `<li>${pill(v, name)}<span>${esc(text)}</span></li>`)
    .join("");

  // Un seul ecouteur par zone, delegue : les elements sont redessines souvent.
  S.el.places.addEventListener("click", (e) => {
    const b = e.target.closest("[data-place]");
    if (b) choosePlace(S, b.dataset.place);
  });
  S.el.controls.addEventListener("input", (e) => {
    const input = e.target.closest("input[data-k]");
    if (!input) return;
    setValue(S, input.dataset.k, parseFloat(input.value));
  });
  // Tant qu'un curseur est tenu, on ne reconstruit pas les commandes.
  S.el.controls.addEventListener("pointerdown", (e) => {
    if (e.target.closest("input[data-k]")) S.dragging = true;
  });
  window.addEventListener("pointerup", () => {
    if (!S.dragging) return;
    S.dragging = false;
    renderControls(S);
  });
  S.el.controls.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-seg]");
    if (!b) return;
    const k = b.dataset.seg;
    const v = b.dataset.v;
    if (S.cond[k] === v) return;
    S.cond[k] = v;
    if (k === "diluent") {
      // Le diluant prend la place qui reste a cote de l'oxygene, 20 % au plus,
      // pour ne pas creer d'office un melange au-dela de 100 %.
      const room = Math.max(0, 1 - S.cond.o2_frac);
      S.cond.diluent_frac = v === "none" ? 0 : Math.round(Math.min(0.2, room) * 100) / 100;
    }
    if (k !== "fuel") markCustom(S);
    renderControls(S);
    send(S, true);
  });
  S.el.tabs.addEventListener("click", (e) => {
    const b = e.target.closest("[data-tab]");
    if (!b) return;
    S.tab = b.dataset.tab;
    renderWhy(S);
  });
  S.el.cards.addEventListener("click", (e) => {
    const q = e.target.closest("blockquote");
    if (q) q.classList.toggle("open");
  });

  // Flamme 3D si WebGL2 repond, sinon le dessin 2D d'origine. Un canevas ne
  // peut porter qu'un seul contexte : en cas d'echec on le remplace.
  S.f3d = typeof Flame3D === "function" ? Flame3D.create(S.el.cv, S.el.view) : null;
  if (!S.f3d) {
    const fresh = document.createElement("canvas");
    fresh.id = "cv";
    S.el.cv.replaceWith(fresh);
    S.el.cv = fresh;
    S.ctx = fresh.getContext("2d");
    // Sans 3D, seul le bouton de lecture reste.
    for (const b of S.el.vtools.querySelectorAll('[data-tg], [data-act="reset"]')) b.style.display = "none";
    S.el.vhint.style.display = "none";
  } else {
    syncToggles(S);
  }
  S.el.vtools.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    if (b.dataset.act === "play") { togglePlay(S); return; }
    if (!S.f3d) return;
    if (b.dataset.tg) S.f3d.setOption(b.dataset.tg, !S.f3d.opts[b.dataset.tg]);
    if (b.dataset.act === "reset") S.f3d.resetView();
    syncToggles(S);
  });

  // Espace allume / met en pause / rejoue. On le prend partout sauf dans un
  // champ de texte : sur un bouton, Espace le « cliquerait » (Entree le fait
  // toujours). Le composant vit dans un shadow DOM : la vraie cible est en
  // tete de composedPath().
  const isSpace = (e) => e.code === "Space" || e.key === " ";
  const typing = (e) => {
    const el = e.composedPath()[0];
    return !!(el && el.closest && el.closest('textarea, select, [contenteditable="true"], input:not([type="range"])'));
  };
  window.addEventListener("keydown", (e) => {
    if (!isSpace(e) || typing(e) || !root.isConnected) return;
    e.preventDefault();
    if (!e.repeat) togglePlay(S);
  }, true);
  window.addEventListener("keyup", (e) => {
    if (isSpace(e) && !typing(e) && root.isConnected) e.preventDefault();
  }, true);
  resetPlay(S);
  const resize = () => {
    if (S.f3d) { S.f3d.resize(); return; }
    const r = window.devicePixelRatio || 1;
    const w = S.el.cv.clientWidth, h = S.el.cv.clientHeight;
    if (!w || !h) return;
    S.el.cv.width = w * r; S.el.cv.height = h * r;
    S.ctx.setTransform(r, 0, 0, r, 0, 0);
  };
  new ResizeObserver(resize).observe(S.el.view);
  resize();
  const loop = (now) => {
    if (!root.isConnected && !S.el.cv.isConnected) return; // composant retire
    drawFrame(S, now);
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

// --- Saisie ----------------------------------------------------------------

function markCustom(S) {
  if (S.cond.preset !== "custom") S.from = S.cond.preset;
  S.cond.preset = "custom";
}

function setValue(S, k, v) {
  if (Number.isNaN(v)) return;
  S.cond[k] = v;
  if (k !== "d0_mm") markCustom(S);
  renderControls(S, k);
  renderPlaces(S);
  send(S, false);
}

function choosePlace(S, key) {
  if (key === "custom") {
    markCustom(S);
  } else {
    const p = S.data.presets.find((x) => x.key === key);
    if (!p) return;
    if (p.pressure_atm != null) S.cond.pressure_atm = p.pressure_atm;
    if (p.o2_frac != null) S.cond.o2_frac = p.o2_frac;
    if (p.gravity_g != null) S.cond.gravity_g = p.gravity_g;
    S.cond.preset = key;
    S.from = null;
  }
  renderPlaces(S);
  renderControls(S);
  send(S, true);
}

function send(S, now) {
  clearTimeout(S.timer);
  const go = () => {
    S.pending = false;
    S.seq += 1;
    S.setState("conditions", { ...S.cond, seq: S.seq });
  };
  if (now) go();
  else { S.pending = true; S.timer = setTimeout(go, SEND_DELAY); }
}

// --- 01 Lieux ------------------------------------------------------------------

function spec(p) {
  const parts = [
    p.pressure_atm == null ? "? atm" : `${p.pressure_atm.toFixed(2)} atm`,
    p.o2_frac == null ? "? O₂" : `${pct(p.o2_frac)} O₂`,
    p.gravity_g == null ? "your g" : gLabel(p.gravity_g),
  ];
  return parts.join(" · ");
}

function renderPlaces(S) {
  const c = S.cond;
  const chips = S.data.presets.map((p) => `
    <button class="place ${c.preset === p.key ? "on" : ""}" data-place="${esc(p.key)}">
      <span class="nm">${esc(p.name)}</span><span class="sp">${esc(spec(p))}</span>
    </button>`);
  chips.push(`
    <button class="place ${c.preset === "custom" ? "on" : ""}" data-place="custom">
      <span class="nm">Your own mix</span>
      <span class="sp">${esc(`${c.pressure_atm.toFixed(2)} atm · ${pct(c.o2_frac)} O₂ · ${gLabel(c.gravity_g)}`)}</span>
    </button>`);
  S.el.places.innerHTML = chips.join("");

  if (c.preset === "custom") {
    const from = S.data.presets.find((x) => x.key === S.from);
    S.el.placeCite.innerHTML = from
      ? `Adjusted from <b>${esc(from.name)}</b>. No spacecraft source for this exact mix.`
      : "Your own mix. No spacecraft source attached.";
    return;
  }
  const p = S.data.presets.find((x) => x.key === c.preset);
  if (!p) { S.el.placeCite.innerHTML = ""; return; }
  const parts = p.citations.map((ct) =>
    `${esc(ct.what)}: ${link(ct.url, ct.source)} <q>${esc(ct.quote)}</q>`);
  if (p.gravity_note) parts.push(esc(p.gravity_note));
  if (p.missing.length) parts.push(`Not in our sources: ${esc(p.missing.join(", "))}.`);
  if (p.caveat) parts.push(esc(p.caveat));
  const verified = p.citations.every((ct) => ct.verified);
  parts.push(`<span class="nv">${verified ? "Checked against the original text" : "Quotes not yet verified against the original text"}</span>`);
  S.el.placeCite.innerHTML = parts.join('<span class="sep">/</span>');
}

// --- 02 Commandes ------------------------------------------------------------

// Pour un carburant sans essai (n-decane), les bandes sont celles du
// carburant dont l'estimation part (heptane), et la legende le dit.
function testedFuel(S) {
  return S.data.surrogate[S.cond.fuel] || S.cond.fuel;
}

function tested(S, k) {
  const r = S.data.ranges[testedFuel(S)];
  return r ? r[k] || null : null;
}

function trackParts(S, spec, value) {
  const [lo, hi] = S.data.limits[spec.k];
  const at = (v) => (100 * (Math.min(Math.max(v, lo), hi) - lo)) / (hi - lo);
  const band = tested(S, spec.band || spec.k);
  let bandHtml = "", gapHtml = "", hint, inside;
  const who = S.data.surrogate[S.cond.fuel] ? `${testedFuel(S)} tested` : "Tested";
  if (!band) {
    inside = false;
    hint = "Never tested";
  } else {
    inside = value >= band[0] - 1e-9 && value <= band[1] + 1e-9;
    const a = at(band[0]), b = at(band[1]);
    bandHtml = `<div class="band" style="left:${a}%;width:${Math.max(b - a, 1)}%"></div>`;
    if (!inside) {
      const edge = value > band[1] ? b : a;
      const [l, r] = [edge, at(value)].sort((x, y) => x - y);
      gapHtml = `<div class="gap" style="left:${l}%;width:${r - l}%"></div>`;
    }
    const f = spec.k === "gravity_g"
      ? "microgravity only"
      : `${fmtPlain(spec, band[0])} – ${fmtPlain(spec, band[1])}`;
    hint = inside ? `${who}: ${f}` : `Beyond the tests (${who.toLowerCase()}: ${f})`;
  }
  return { bandHtml, gapHtml, hint, inside };
}

function fmtPlain(spec, v) {
  if (spec.k === "o2_frac" || spec.k === "diluent_frac") return pct(v);
  if (spec.k === "pressure_atm") return `${v.toFixed(2)} atm`;
  if (spec.k === "d0_mm") return `${v.toFixed(1)} mm`;
  return v.toFixed(3);
}

function renderControls(S, dragging) {
  // Pendant un glissement on ne reconstruit pas le curseur tenu : seulement
  // sa valeur affichee et ses bandes, sinon le navigateur perd la prise.
  const specs = controlsSpec(S.cond);
  if (dragging) {
    const box = S.el.controls.querySelector(`.ctl[data-k="${dragging}"]`);
    const sp = specs.find((x) => x.k === dragging);
    if (box && sp) {
      const v = S.cond[dragging];
      const t = trackParts(S, sp, v);
      box.querySelector("output").innerHTML = sp.fmt(v);
      box.querySelector(".band-slot").innerHTML = t.bandHtml + t.gapHtml;
      const h = box.querySelector(".hint");
      h.textContent = t.hint; h.classList.toggle("out", !t.inside);
      box.querySelector("input").classList.toggle("out", !t.inside);
      return;
    }
  }
  const html = specs.map((sp) => {
    if (sp.seg) {
      const buttons = SEGMENTS[sp.seg].map(([v, name]) =>
        `<button data-seg="${sp.seg}" data-v="${esc(v)}" class="${S.cond[sp.seg] === v ? "on" : ""}">${esc(name)}</button>`).join("");
      let hint = "";
      if (sp.seg === "fuel" && S.data.surrogate[S.cond.fuel]) {
        hint = `<div class="hint out">No NASA droplet test: estimated from ${esc(S.data.surrogate[S.cond.fuel])} and theory</div>`;
      }
      return `<div class="ctl"><div class="row"><span class="lbl">${esc(sp.label)}</span></div><div class="seg">${buttons}</div>${hint}</div>`;
    }
    const [lo, hi] = S.data.limits[sp.k];
    const v = S.cond[sp.k];
    const t = trackParts(S, sp, v);
    const ticks = sp.ticks
      ? `<div class="ticks">${sp.ticks.map(([x, name]) =>
          `<span style="left:${(100 * (x - lo)) / (hi - lo)}%">${esc(name)}</span>`).join("")}</div>`
      : "";
    return `<div class="ctl" data-k="${sp.k}">
      <div class="row"><span class="lbl">${esc(sp.label)}</span><output>${sp.fmt(v)}</output></div>
      <div class="track"><div class="rail"></div><div class="band-slot">${t.bandHtml}${t.gapHtml}</div>
        <input type="range" data-k="${sp.k}" min="${lo}" max="${hi}" step="${sp.step}" value="${v}"
          class="${t.inside ? "" : "out"}" aria-label="${esc(sp.label)}"></div>
      ${ticks}
      <div class="hint ${t.inside ? "" : "out"}">${esc(t.hint)}</div>
    </div>`;
  });
  html.push(`<div class="key"><span><i class="k-band"></i>NASA tested</span><span><i class="k-gap"></i>beyond the tests</span></div>`);
  S.el.controls.innerHTML = html.join("");
}

// --- 03 Flamme -----------------------------------------------------------------

function flameShape(ctx, cx, cy, radius, stretch, now) {
  // Enveloppe : sphere en microgravite, etiree vers le haut avec la gravite.
  ctx.save();
  ctx.translate(cx, cy - radius * (stretch - 1) * 0.6);
  ctx.scale(1, stretch);
  const flick = 1 + 0.025 * Math.sin(now / 85) + 0.015 * Math.sin(now / 37);
  const r = radius * flick;
  const g = ctx.createRadialGradient(0, 0, r * 0.2, 0, 0, r);
  g.addColorStop(0.0, "rgba(120,160,255,0.30)");
  g.addColorStop(0.35, "rgba(80,130,255,0.32)");
  g.addColorStop(0.55, "rgba(140,150,255,0.38)");
  g.addColorStop(0.68, "rgba(255,180,80,0.62)");
  g.addColorStop(0.82, "rgba(255,110,35,0.38)");
  g.addColorStop(1.0, "rgba(255,60,0,0)");
  ctx.fillStyle = g;
  ctx.beginPath(); ctx.arc(0, 0, r, 0, Math.PI * 2); ctx.fill();
  ctx.restore();
}

function scaleBar(ctx, x, y, pxPerMm) {
  ctx.save();
  ctx.strokeStyle = "rgba(255,255,255,0.45)";
  ctx.fillStyle = "rgba(255,255,255,0.55)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x, y); ctx.lineTo(x + pxPerMm, y);
  ctx.moveTo(x, y - 4); ctx.lineTo(x, y + 4);
  ctx.moveTo(x + pxPerMm, y - 4); ctx.lineTo(x + pxPerMm, y + 4);
  ctx.stroke();
  ctx.font = "10px 'DM Mono', monospace";
  ctx.fillText("1 mm", x, y - 8);
  ctx.restore();
}

// Part de suie, pour la couleur seulement (illustratif) : le methanol brule
// d'une flamme bleue presque sans suie, les alcanes d'une flamme jaune.
const SOOT = { Methanol: 0.12, Heptane: 0.85, "n-decane": 0.92 };

// LECTURE A LA DEMANDE. La combustion ne demarre pas seule : la goutte attend,
// flamme allumee mais figee a t = 0, que l'on appuie sur Espace (ou sur le
// bouton). Etats :
//   ready   goutte intacte, flamme figee
//   playing la combustion avance
//   paused  figee en cours de route
//   done    arretee sur l'etat final ; Espace la rejoue
// Un nouveau reglage ramene a « ready ».
function resetPlay(S) {
  S.play = { mode: "ready", t: 0, last: null, doneAt: null };
  syncPlay(S);
}

function togglePlay(S) {
  const p = S.play;
  if (!S.data || !S.data.result.prediction) return;
  if (p.mode === "playing") p.mode = "paused";
  else if (p.mode === "done") { p.t = 0; p.doneAt = null; p.mode = "playing"; }
  else p.mode = "playing";
  p.last = null;
  syncPlay(S);
}

function syncPlay(S) {
  const b = S.el.vtools.querySelector('[data-act="play"]');
  if (!b) return;
  const m = S.play.mode;
  const [icon, label] = m === "playing" ? ["pause", "Pause"] : m === "done" ? ["replay", "Replay"]
    : m === "paused" ? ["play", "Resume"] : ["flame", "Ignite"];
  b.innerHTML = `${svgIcon(icon)}${label}`;
  b.setAttribute("aria-pressed", String(m === "playing"));
}

// L'horloge de la combustion : ou en est la goutte a l'instant `now`.
function timeline(S, now) {
  const pred = S.data && S.data.result.prediction;
  if (!pred) return null;
  const P = pred.flame;
  const p = S.play;
  const speed = Math.max(1, P.duration / 8); // une combustion ~ 8 s d'ecran
  if (p.mode === "playing") {
    if (p.last != null) p.t += ((now - p.last) / 1000) * speed;
    p.last = now;
    if (p.t >= P.duration) {
      p.t = P.duration; p.mode = "done"; p.doneAt = now;
      syncPlay(S);
    }
  }
  const t = Math.min(p.t, P.duration);
  const d = Math.sqrt(Math.max(P.d0 * P.d0 - P.k * t, P.dfinal * P.dfinal, 0));
  let on = 1;
  // « ready » : flamme allumee mais figee a t = 0, en attente d'Espace.
  // A l'extinction la flamme s'efface en ~0,4 s.
  if (p.mode === "done") on = Math.max(0, 1 - (now - p.doneAt) / 400);
  return { P, t, d, on, mode: p.mode };
}

function readout(S, tl) {
  const { P, t, d, mode } = tl;
  S.el.clock.textContent = `t ${t.toFixed(1)} / ${P.duration.toFixed(1)} s
d ${d.toFixed(2)} mm`;
  const end = P.dfinal > 0 ? `Flame out at ${P.dfinal.toFixed(2)} mm` : "Droplet fully burned";
  const key = "<kbd>Space</kbd>";
  const html = {
    ready: `${key} to start the burn`,
    playing: "Burning",
    paused: `Paused · ${key} to resume`,
    done: `${end} · ${key} to replay`,
  }[mode];
  if (S.stateHtml !== html) { S.el.state.innerHTML = html; S.stateHtml = html; }
  if (S.head) {
    const x = (600 * t) / Math.max(P.duration, 1e-6);
    S.head.setAttribute("x1", x); S.head.setAttribute("x2", x);
  }
}

function syncToggles(S) {
  for (const b of S.el.vtools.querySelectorAll("[data-tg]")) {
    b.setAttribute("aria-pressed", String(!!S.f3d.opts[b.dataset.tg]));
  }
  S.el.vhint.textContent = S.f3d.opts.orbit
    ? "Drag to rotate · Ctrl + wheel to zoom · double-click to reset"
    : "Rotating on its own";
}

function drawFrame(S, now) {
  const tl = timeline(S, now);
  if (S.f3d) {
    if (!tl) {
      S.f3d.frame(now, { dropR: 0, flameR: 0, flameR0: 10, stretch: 1, on: 0, soot: 0 });
      return;
    }
    const { P, d, on } = tl;
    S.f3d.frame(now, {
      dropR: d / 2,
      flameR: P.envelope * d,
      flameR0: P.envelope * P.d0,
      stretch: P.stretch,
      on,
      soot: SOOT[S.cond.fuel] ?? 0.8,
    });
    readout(S, tl);
    return;
  }
  draw2D(S, now, tl);
}

function draw2D(S, now, tl) {
  const ctx = S.ctx, cv = S.el.cv;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (!w || !h) return;
  ctx.clearRect(0, 0, w, h);
  if (!tl) return;
  const { P, d, on } = tl;
  const pxPerMm = (Math.min(w, h) * 0.075) / Math.max(P.d0, 0.5);
  const cx = w / 2, cy = h * 0.56;
  if (on > 0.01) {
    ctx.globalAlpha = on;
    flameShape(ctx, cx, cy, d * pxPerMm * P.envelope, P.stretch, now);
    ctx.globalAlpha = 1;
  }
  ctx.fillStyle = "#cfe0ff";
  ctx.beginPath(); ctx.arc(cx, cy, Math.max((d * pxPerMm) / 2, 0.6), 0, Math.PI * 2); ctx.fill();
  scaleBar(ctx, 16, h - 18, pxPerMm);
  readout(S, tl);
}

function renderCurve(S) {
  const svg = S.el.curve;
  const pred = S.data.result.prediction;
  if (!pred) { svg.innerHTML = ""; S.head = null; S.el.curveEnd.textContent = ""; return; }
  const P = pred.flame;
  const W = 600, top = 4, bottom = 99;
  const y = (d) => bottom - ((bottom - top) * d) / P.d0;
  const pts = [];
  for (let i = 0; i <= 60; i++) {
    const t = (P.duration * i) / 60;
    const d = Math.sqrt(Math.max(P.d0 * P.d0 - P.k * t, P.dfinal * P.dfinal, 0));
    pts.push(`${(W * i) / 60},${y(d).toFixed(2)}`);
  }
  const out = P.dfinal > 0
    ? `<line class="out" x1="${W}" x2="${W}" y1="${y(P.dfinal)}" y2="${bottom}"></line>`
    : "";
  svg.innerHTML = `
    <line class="ax" x1="0" x2="${W}" y1="${bottom}" y2="${bottom}"></line>
    <polyline class="ln" points="${pts.join(" ")}"></polyline>
    ${out}
    <line class="head" id="head" x1="0" x2="0" y1="${top}" y2="${bottom}"></line>`;
  S.el.curveEnd.textContent = `${P.duration.toFixed(1)} s`;
  S.head = svg.querySelector("#head");
}

function renderStage(S) {
  const r = S.data.result;
  const pred = r.prediction;
  S.el.err.style.display = r.error ? "flex" : "none";
  S.el.err.textContent = r.error || "";
  if (!pred) {
    S.el.badge.innerHTML = ""; S.el.beyond.style.display = "none";
    S.el.hatch.style.display = "none";
    S.el.clock.textContent = ""; S.el.state.textContent = ""; S.stateHtml = "";
    renderCurve(S);
    return;
  }
  S.el.badge.innerHTML = pill(pred.verdict, pred.badge);
  S.el.hatch.style.display = pred.extrapolated.length ? "block" : "none";
  if (pred.extrapolated.length) {
    S.el.beyond.style.display = "block";
    S.el.beyond.textContent = `Beyond the NASA tests: ${pred.extrapolated.join(", ")}.`;
  } else {
    S.el.beyond.style.display = "none";
    S.el.beyond.textContent = "";
  }
  resetPlay(S);            // un nouveau reglage attend qu'on rallume
  renderCurve(S);
}

// --- 04 Reponse ------------------------------------------------------------------

function renderAnswers(S) {
  const r = S.data.result;
  const pred = r.prediction;
  if (!pred) {
    S.el.answers.innerHTML = `<div class="msg">${esc(r.error || "No answer for this setting.")}</div>`;
    return;
  }
  const c = S.cond;
  let goes, sub, meter = "";
  if (pred.p_ext == null) {
    goes = "Not modelled";
    sub = "No NASA extinction test for this fuel";
  } else {
    goes = pred.extinguishes && pred.dext != null
      ? `Extinguishes at ${pred.dext.toFixed(2)} mm`
      : "Burns to completion";
    const edge = pred.ext_basis === "edge" ? " (value at the edge of the tests)" : "";
    sub = `${pct(pred.p_ext)} extinction probability${edge}`;
    meter = `<div class="meter"><i style="width:${(100 * pred.p_ext).toFixed(1)}%"></i></div>`;
  }
  let cmp = "";
  if (pred.compare) {
    cmp = `${pred.compare.ratio.toFixed(2)}× the rate in ISS cabin air · `;
  }
  const notes = pred.notes.length
    ? `<ul class="notes">${pred.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>`
    : "";
  S.el.answers.innerHTML = `
    <div class="q">
      <div class="qk">Flame extinction</div>
      <div class="qa">${esc(goes)}</div>
      <div class="qs">${esc(sub)}</div>${meter}
    </div>
    <div class="q">
      <div class="qk">Burning rate constant, K</div>
      <div class="qa num">${pred.k.toFixed(2)}<small>mm²/s</small></div>
      <div class="qs">${esc(cmp)}${esc(pred.k_basis)}</div>
    </div>
    <div class="q">
      <div class="qk">Burn time</div>
      <div class="qa num">${pred.duration.toFixed(1)}<small>s</small></div>
      <div class="qs">${esc(`A ${c.d0_mm.toFixed(2)} mm droplet, ${pred.extinguishes ? "until extinction" : "until fully burned"}`)}</div>
    </div>
    <div class="q sure">
      <div class="qk">Confidence</div>
      ${pill(pred.verdict, pred.badge)}
      <div class="qs">${esc(pred.badge_text)}</div>
      ${notes}
    </div>`;
}


// --- 05 Contexte ---------------------------------------------------------------

function renderMap(S) {
  const svg = S.el.map;
  const W = 380, H = 280, L = 44, R = 12, T = 24, B = 40;
  const PMAX = 1.25;
  const x = (p) => L + ((W - L - R) * Math.min(p, PMAX)) / PMAX;
  const y = (o) => T + (H - T - B) * (1 - o);
  const parts = [];
  // grille
  parts.push('<g class="grid">');
  for (const p of [0.25, 0.5, 0.75, 1.0, 1.25]) parts.push(`<line x1="${x(p)}" x2="${x(p)}" y1="${T}" y2="${H - B}"></line>`);
  for (const o of [0.25, 0.5, 0.75, 1.0]) parts.push(`<line x1="${L}" x2="${W - R}" y1="${y(o)}" y2="${y(o)}"></line>`);
  parts.push("</g>");
  // cadre de la norme
  const box = S.data.result.standard.box;
  if (box) {
    parts.push(`<rect class="box" x="${x(box.p_min)}" y="${y(box.o2_max)}" width="${x(box.p_max) - x(box.p_min)}" height="${y(0) - y(box.o2_max)}"></rect>`);
    parts.push(`<text class="boxlab" x="${x(box.p_min) + 6}" y="${y(box.o2_max) + 14}">NASA-STD-3001</text>`);
  }
  // axes
  parts.push('<g class="axis">');
  for (const p of [0, 0.25, 0.5, 0.75, 1.0, 1.25]) parts.push(`<text x="${x(p)}" y="${H - B + 14}" text-anchor="middle">${p}</text>`);
  for (const o of [0, 0.25, 0.5, 0.75, 1.0]) parts.push(`<text x="${L - 6}" y="${y(o) + 3}" text-anchor="end">${o * 100}%</text>`);
  parts.push(`<text class="ttl" x="${(L + W - R) / 2}" y="${H - 6}" text-anchor="middle">cabin pressure (atm)</text>`);
  parts.push(`<text class="ttl" transform="translate(11 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">oxygen</text>`);
  parts.push("</g>");
  // points, regroupes quand ils tombent au meme endroit
  const groups = new Map();
  const unplaced = [];
  for (const a of S.data.atmospheres) {
    if (a.pressure == null || a.o2 == null) { unplaced.push(a); continue; }
    const k = `${a.pressure.toFixed(3)}|${a.o2.toFixed(3)}`;
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(a);
  }
  // Etiquettes : chaque groupe essaie quelques positions et prend la premiere
  // qui ne chevauche ni une etiquette deja posee ni le bord de la carte.
  const placed = [];
  const fits = (b) => b.x0 >= 2 && b.x1 <= W - 2 && b.y0 >= 0 && b.y1 <= H - B &&
    !placed.some((o) => b.x0 < o.x1 && b.x1 > o.x0 && b.y0 < o.y1 && b.y1 > o.y0);
  const around = (px, py, tw) => [
    { tx: px + 9, ty: py + 3, anchor: "start", x0: px + 9, x1: px + 9 + tw },
    { tx: px - 9, ty: py + 3, anchor: "end", x0: px - 9 - tw, x1: px - 9 },
    { tx: px, ty: py + 18, anchor: "middle", x0: px - tw / 2, x1: px + tw / 2 },
    { tx: px, ty: py - 10, anchor: "middle", x0: px - tw / 2, x1: px + tw / 2 },
    { tx: px - 9, ty: py + 18, anchor: "end", x0: px - 9 - tw, x1: px - 9 },
    { tx: px + 9, ty: py + 18, anchor: "start", x0: px + 9, x1: px + 9 + tw },
    { tx: px, ty: py - 24, anchor: "middle", x0: px - tw / 2, x1: px + tw / 2 },
    { tx: px, ty: py + 32, anchor: "middle", x0: px - tw / 2, x1: px + tw / 2 },
  ].map((o) => ({ ...o, y0: o.ty - 10, y1: o.ty + 3 }));

  // Le reglage en cours d'abord : son etiquette passe avant les autres.
  const c = S.cond;
  const yx = x(c.pressure_atm), yy = y(c.o2_frac);
  const beyond = c.pressure_atm > PMAX ? ` → ${c.pressure_atm.toFixed(2)} atm` : "";
  const youText = `You${beyond}`;
  const youOpts = around(yx, yy, youText.length * 7);
  const youPick = [youOpts[2], youOpts[3], youOpts[1], youOpts[0]].find(fits) || youOpts[2];
  placed.push(youPick);
  // Les points des atmospheres, chacun pres de son point ; on garde aussi la
  // place du point « You » pour qu'aucune etiquette ne le couvre.
  placed.push({ x0: yx - 7, x1: yx + 7, y0: yy - 7, y1: yy + 7 });

  for (const list of groups.values()) {
    const a = list[0];
    const px = x(a.pressure), py = y(a.o2);
    const name = list.map((v) => v.short).join(" · ");
    const tip = list.map((v) => `${v.name}: ${v.pressure.toFixed(2)} atm, ${pct(v.o2)} O2 (${v.source})`).join("\n");
    parts.push(`<a href="${esc(a.url)}" target="_blank" rel="noopener noreferrer"><circle class="pt ${esc(a.status.split(" ")[0])}" cx="${px}" cy="${py}" r="5"><title>${esc(tip)}</title></circle></a>`);
    const options = around(px, py, name.length * 6.1);
    // Pas de place libre : le point reste, avec son infobulle, sans etiquette
    // plutot qu'une etiquette posee sur une autre.
    const pick = options.find(fits);
    placed.push({ x0: px - 6, x1: px + 6, y0: py - 6, y1: py + 6 });
    if (!pick) continue;
    placed.push(pick);
    parts.push(`<text class="lab" x="${pick.tx}" y="${pick.ty}" text-anchor="${pick.anchor}">${esc(name)}</text>`);
  }
  parts.push(`<line class="youline" x1="${L}" x2="${yx}" y1="${yy}" y2="${yy}"></line>`);
  parts.push(`<line class="youline" x1="${yx}" x2="${yx}" y1="${yy}" y2="${H - B}"></line>`);
  parts.push(`<circle class="you" cx="${yx}" cy="${yy}" r="6"></circle>`);
  parts.push(`<text class="youlab" x="${youPick.tx}" y="${youPick.ty}" text-anchor="${youPick.anchor}">${esc(youText)}</text>`);
  svg.innerHTML = parts.join("");

  S.el.unplaced.innerHTML =
    `<div class="mkey"><span><i style="background:var(--text)"></i>Spacecraft</span><span><i style="background:var(--c40)"></i>Ground test</span><span><i style="box-shadow:inset 0 0 0 1.5px var(--text-2)"></i>Proposal</span><span><i style="background:var(--accent)"></i>Your setting</span></div>` +
    (unplaced.length
      ? `Not placed: ${unplaced.map((a) => `${esc(a.name)} (${a.pressure == null ? "pressure" : "oxygen"} not given in the source)`).join(", ")}.`
      : "");

  const s = S.data.result.standard;
  const cls = s.inside == null ? "" : s.inside ? "ok" : "no";
  S.el.std.innerHTML = `
    <div class="v ${cls}">${esc(s.verdict)}</div>
    <div>${esc(s.note)} ${link(s.url, s.source)}</div>
    <div class="nv">${s.verified ? "Checked against the original text" : "Quote not yet verified against the original text"}</div>`;
}

function renderWhy(S) {
  const secs = S.data.result.why;
  if (!secs.length) {
    S.el.tabs.innerHTML = "";
    S.el.whyText.innerHTML = "";
    S.el.cards.innerHTML = `<div class="empty">Fix the gas mix to see the context.</div>`;
    return;
  }
  if (!secs.find((s) => s.key === S.tab)) S.tab = secs[0].key;
  S.el.tabs.innerHTML = secs.map((s) =>
    `<button role="tab" data-tab="${esc(s.key)}" class="${s.key === S.tab ? "on" : ""}" aria-selected="${s.key === S.tab}">${esc(s.name)}<span class="ct">${s.cards.length}</span></button>`).join("");
  const sec = secs.find((s) => s.key === S.tab);
  S.el.whyText.innerHTML = sec.why.map((p) => `<p>${esc(p)}</p>`).join("");
  if (!sec.cards.length) {
    S.el.cards.innerHTML = sec.empty ? `<div class="empty">${esc(sec.empty)}</div>` : "";
    return;
  }
  S.el.cards.innerHTML = sec.cards.map((c) => {
    const tags = c.labels.map((l) => `<span class="tag">${esc(l)}</span>`).join("");
    const quote = c.quote ? `<blockquote title="Click to expand">“${esc(c.quote)}”</blockquote>` : "";
    return `<article class="card ${esc(c.kind)}">
      <div class="meta"><span class="kind">${esc(KINDS[c.kind] || c.kind)}</span>${c.year ? `<span>${esc(c.year)}</span>` : ""}${tags}</div>
      <h4>${esc(c.title)}</h4>
      ${c.summary ? `<p>${esc(c.summary)}</p>` : ""}
      ${quote}
      <div class="src">${link(c.url, c.source)}<span class="nv">${c.verified ? "Checked against the original text" : "Not yet verified against the original text"}</span></div>
    </article>`;
  }).join("");
}

function renderFoot(S) {
  S.el.foot.innerHTML = `
    <p>Extinction: learned from ${S.data.tests} NASA FLEX droplet tests flown on the ISS (PSI-69). Burning rate: the same FLEX tests plus ${S.data.msl1_tests} heptane droplets burned on Spacelab MSL-1 in 1997 (${link("https://ntrs.nasa.gov/citations/20040000941", "NASA/TM-2003-212553")}). Beyond all of them, quasi-steady droplet combustion theory.</p>
    <p>Context: fires, spacecraft atmospheres and papers collected by hand from NASA and published sources. Quotes are shown verbatim and are not yet verified against the originals.</p>
    <p>Colours and body typefaces: NASA Horizon Design System. Headings in Saira, after the Fotia logo.</p>`;
}

// --- Point d'entree -------------------------------------------------------------

export default function (component) {
  const { data, setStateValue, parentElement } = component;
  if (!data) return;
  const root = parentElement.querySelector("#fx") || parentElement;
  let S = parentElement.__fotia;
  if (!S || S.root !== root) {
    S = { root, seq: 0, tab: "oxygen", from: null, timer: null, head: null,
          pending: false, dragging: false };
    parentElement.__fotia = S;
    S.data = data;
    S.cond = { ...data.conditions };
    S.seq = data.conditions.seq || 0;
    build(root, S);
  }
  S.setState = setStateValue;

  // Un resultat plus ancien que le dernier envoi est ignore : sinon un
  // curseur deplace vite reviendrait en arriere le temps d'un aller-retour.
  const incoming = data.conditions.seq || 0;
  if (incoming < S.seq) return;
  S.data = data;
  // Un envoi en attente ou un curseur tenu : les valeurs locales sont plus
  // recentes que celles qui reviennent, on les garde.
  const busy = S.pending || S.dragging;
  if (!busy) S.cond = { ...data.conditions };

  renderPlaces(S);
  if (!busy) renderControls(S);
  renderStage(S);
  renderAnswers(S);
  renderMap(S);
  renderWhy(S);
  renderFoot(S);
}
