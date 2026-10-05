// MapWise deck — minimal editorial template: off-white, letter-spaced serif, soft yellow accent.
// 8 slides: cover, idea, how it works, predictions, results, build timeline, applications, close+QR.
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const FIG = path.join(__dirname, "..", "assets", "figures");
const C = {
  paper: "FBF9F5",      // off-white page
  card: "F4F1EA",       // soft panel
  ink: "232323",        // near-black text
  ink2: "4A4742",
  muted: "9A948A",
  line: "E3DED3",
  accent: "F2D04B",     // soft yellow
  accentSoft: "FAEFC4",
  dark: "232323",
};
const SER = "Cormorant Garamond";            // display serif
const SANS = "Segoe UI";                     // quiet body sans
const SANSL = "Segoe UI Light";

let MET = { mIoU: null, pixacc: null, per_class: {}, epochs: null, minutes: null, n_train: null };
try { MET = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "out", "deck_metrics.json"), "utf8")); } catch (e) {}
const pc = (v) => (v === null || v === undefined ? "—" : Math.round(v * 100) + "%");
const cl = (k) => (MET.per_class || {})[k];

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "MapWise";
pres.author = "Priyanshu Singh";
const W = 13.333, H = 7.5, M = 1.15;

function png(f) { const b = fs.readFileSync(f); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; }
function fit(f, x, y, w, h) { const s = png(f), r = Math.min(w / s.w, h / s.h); return { path: f, x: x + (w - s.w * r) / 2, y: y + (h - s.h * r) / 2, w: s.w * r, h: s.h * r }; }
function coverImg(f, x, y, w, h) { const s = png(f), r = Math.max(w / s.w, h / s.h), iw = s.w * r, ih = s.h * r; return { path: f, x, y, w, h, sizing: { type: "crop", x: (iw - w) / 2, y: (ih - h) / 2, w, h } }; }
const has = (f) => fs.existsSync(path.join(FIG, f));

function slide() {
  const s = pres.addSlide();
  s.background = { color: C.paper };
  return s;
}
// the template's signature: wide-tracked serif wordmark, centred at the top
function wordmark(s, y) {
  s.addText("M A P W I S E", { x: 0, y: y === undefined ? 0.42 : y, w: W, h: 0.34,
    fontFace: SER, fontSize: 15, color: C.ink, charSpacing: 6, align: "center", margin: 0 });
}
function kicker(s, t, x, y) {
  s.addText(t, { x: x === undefined ? M : x, y: y === undefined ? 1.45 : y, w: 7, h: 0.28,
    fontFace: SANS, fontSize: 10, color: C.muted, charSpacing: 3, margin: 0 });
}
function heading(s, parts, y, size) {
  s.addText(parts, { x: M, y: y === undefined ? 1.85 : y, w: W - 2 * M, h: 1.25,
    fontFace: SER, fontSize: size === undefined ? 44 : size, color: C.ink,
    charSpacing: 1.5, margin: 0, lineSpacingMultiple: 1.0 });
}
function body(s, t, x, y, w, size, h) {
  s.addText(t, { x: x === undefined ? M : x, y: y, w: w === undefined ? 6.6 : w, h: h === undefined ? 1.4 : h,
    fontFace: SANSL, fontSize: size === undefined ? 13 : size, color: C.ink2, margin: 0, lineSpacingMultiple: 1.45 });
}
// soft yellow underline, the template's only strong colour
function dash(s, x, y, w) {
  s.addShape(pres.shapes.RECTANGLE, { x: x, y: y, w: w === undefined ? 0.95 : w, h: 0.055,
    fill: { color: C.accent }, line: { type: "none" } });
}
// thin outlined circle holding a statistic — the template's stat rings
// A stat ring whose yellow arc actually spans the value. A decorative dot reads as a
// nearly-empty progress bar, which makes a strong number look like a weak one.
function ring(s, x, y, d, frac, value, label, sub) {
  s.addShape(pres.shapes.OVAL, { x: x, y: y, w: d, h: d, fill: { color: C.paper }, line: { color: C.line, width: 1.25 } });
  if (frac !== null && frac !== undefined && frac > 0) {
    s.addShape(pres.shapes.ARC, {
      x: x, y: y, w: d, h: d,
      angleRange: [270, 270 + Math.min(frac, 1) * 360],
      line: { color: C.accent, width: 4 },
    });
  }
  s.addText(value, { x: x, y: y + d * 0.30, w: d, h: d * 0.34, fontFace: SER, fontSize: 30, color: C.ink, align: "center", valign: "middle", margin: 0 });
  s.addText(label, { x: x - 0.25, y: y + d + 0.14, w: d + 0.5, h: 0.26, fontFace: SANS, fontSize: 9.5, color: C.ink, charSpacing: 2, align: "center", margin: 0 });
  if (sub) s.addText(sub, { x: x - 0.45, y: y + d + 0.42, w: d + 0.9, h: 0.5, fontFace: SANSL, fontSize: 9.5, color: C.muted, align: "center", margin: 0, lineSpacingMultiple: 1.25 });
}
function footer(s, n) {
  s.addText(String(n).padStart(2, "0"), { x: W - M - 0.6, y: H - 0.72, w: 0.6, h: 0.3,
    fontFace: SER, fontSize: 13, color: C.muted, align: "right", margin: 0 });
}

// ============================================== 1. COVER
{ const s = slide();
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: W, h: 0.16, fill: { color: C.accent }, line: { type: "none" } });
  s.addText("M A P W I S E", { x: 0, y: 2.45, w: W, h: 1.0, fontFace: SER, fontSize: 62, color: C.ink, charSpacing: 16, align: "center", margin: 0 });
  s.addShape(pres.shapes.RECTANGLE, { x: W / 2 - 0.6, y: 3.62, w: 1.2, h: 0.05, fill: { color: C.accent }, line: { type: "none" } });
  s.addText("EVERY PIXEL OF THE EARTH, CLASSIFIED", { x: 0, y: 3.9, w: W, h: 0.4,
    fontFace: SANS, fontSize: 12, color: C.ink2, charSpacing: 4, align: "center", margin: 0 });
  s.addText("A U-Net that reads satellite imagery and labels every pixel as urban, agriculture,\nforest, water, rangeland or barren — then reports how much ground each one covers.",
    { x: 2.4, y: 4.5, w: W - 4.8, h: 0.9, fontFace: SANSL, fontSize: 13, color: C.muted, align: "center", margin: 0, lineSpacingMultiple: 1.5 });
  s.addText("PRIYANSHU SINGH   ·   AI / ML ENGINEER", { x: 0, y: 6.5, w: W, h: 0.3,
    fontFace: SANS, fontSize: 10, color: C.muted, charSpacing: 3, align: "center", margin: 0 });
}

// ============================================== 2. THE IDEA
{ const s = slide(); wordmark(s); footer(s, 2);
  kicker(s, "THE IDEA");
  heading(s, [{ text: "Satellites see everything.\n" }, { text: "Nobody can look at it all.", options: { italic: true, color: C.ink2 } }], 1.8, 40);
  dash(s, M, 3.35);
  body(s, "Terabytes of imagery arrive every day. The questions asked of it are simple — how much forest was lost, where did the city expand, which fields are planted — but answering them by eye does not scale.\n\nMapWise answers them per pixel, automatically, and returns the percentages rather than just a picture.",
       M, 3.65, 6.2);
  if (has("hero.png")) {
    s.addShape(pres.shapes.RECTANGLE, { x: 7.75, y: 1.62, w: 4.45, h: 3.6, fill: { color: C.card }, line: { type: "none" } });
    s.addImage(fit(path.join(FIG, "hero.png"), 7.9, 2.35, 4.15, 2.1));
    s.addText("SATELLITE TILE → CLASSIFIED MAP", { x: 7.75, y: 4.62, w: 4.45, h: 0.3,
      fontFace: SANS, fontSize: 9, color: C.muted, charSpacing: 2, align: "center", margin: 0 });
  }
  const rings = [[MET.pixacc, pc(MET.pixacc), "PIXELS CORRECT"], [MET.mIoU, pc(MET.mIoU), "MEAN IoU"]];
  rings.forEach(function (r, i) { ring(s, 8.3 + i * 2.1, 5.45, 1.25, r[0], r[1], r[2], ""); });
}

// ============================================== 3. HOW IT WORKS
{ const s = slide(); wordmark(s); footer(s, 3);
  kicker(s, "ARCHITECTURE");
  heading(s, [{ text: "How it works" }], 1.8, 42);
  dash(s, M, 2.95);
  body(s, "A U-Net: a pretrained encoder reads the image, a decoder written from scratch rebuilds it as labels.", M, 3.22, 10.5, 13, 0.34);
  const steps = [["01", "ENCODE", "A ResNet-34 compresses the tile into a feature pyramid, from fine texture to coarse context."],
                 ["02", "DECODE", "Four upsampling blocks rebuild full resolution, each re-joining the matching encoder detail."],
                 ["03", "CLASSIFY", "A 1x1 convolution turns every pixel's features into seven class scores."],
                 ["04", "MEASURE", "Pixel counts become ground-cover percentages — the number a client actually uses."]];
  const cw = (W - 2 * M - 3 * 0.42) / 4;
  steps.forEach(function (st, i) {
    const x = M + i * (cw + 0.42), y = 4.22;
    s.addShape(pres.shapes.RECTANGLE, { x: x, y: y, w: cw, h: 0.045, fill: { color: C.accent }, line: { type: "none" } });
    s.addText(st[0], { x: x, y: y + 0.22, w: cw, h: 0.4, fontFace: SER, fontSize: 24, color: C.muted, margin: 0 });
    s.addText(st[1], { x: x, y: y + 0.78, w: cw, h: 0.3, fontFace: SANS, fontSize: 11, color: C.ink, charSpacing: 2.5, margin: 0 });
    s.addText(st[2], { x: x, y: y + 1.14, w: cw, h: 1.5, fontFace: SANSL, fontSize: 11, color: C.muted, margin: 0, lineSpacingMultiple: 1.4 });
  });
  s.addText("Skip connections are why the edges stay sharp.", { x: M, y: 6.82, w: 9, h: 0.3,
    fontFace: SER, fontSize: 15, italic: true, color: C.ink2, margin: 0 });
}

// ============================================== 4. PREDICTIONS
{ const s = slide(); wordmark(s); footer(s, 4);
  kicker(s, "RESULTS IN THE FIELD");
  heading(s, [{ text: "Satellite in, map out" }], 1.8, 40);
  dash(s, M, 2.95);
  const imgs = ["compare_0.png", "compare_3.png"].filter(has);
  if (imgs.length) {
    const each = (3.55 - 0.2 * (imgs.length - 1)) / imgs.length;
    imgs.forEach(function (f, i) { s.addImage(fit(path.join(FIG, f), M, 3.25 + i * (each + 0.2), W - 2 * M, each)); });
    s.addText("Left: the satellite tile.  Centre: MapWise.  Right: the reference map a human analyst drew.",
      { x: M, y: 6.95, w: W - 2 * M, h: 0.3, fontFace: SANSL, fontSize: 10, color: C.muted, margin: 0 });
  } else {
    s.addShape(pres.shapes.RECTANGLE, { x: M, y: 3.25, w: W - 2 * M, h: 3.55, fill: { color: C.card }, line: { type: "none" } });
    s.addText("PREDICTIONS PENDING", { x: M, y: 3.25, w: W - 2 * M, h: 3.55, fontFace: SANS, fontSize: 11, color: C.muted, align: "center", valign: "middle", charSpacing: 3, margin: 0 });
  }
}

// ============================================== 5. PER-CLASS RESULTS
{ const s = slide(); wordmark(s); footer(s, 5);
  kicker(s, "ACCURACY, CLASS BY CLASS");
  heading(s, [{ text: "What it gets right — " }, { text: "and what it doesn't", options: { italic: true, color: C.ink2 } }], 1.8, 36);
  dash(s, M, 3.0);
  const rings = [[cl("agriculture_land"), "AGRICULTURE"], [cl("urban_land"), "URBAN"],
                 [cl("water"), "WATER"], [cl("forest_land"), "FOREST"]];
  rings.forEach(function (r, i) { ring(s, M + 0.15 + i * 1.95, 3.45, 1.3, r[0], pc(r[0]), r[1], ""); });
  s.addShape(pres.shapes.RECTANGLE, { x: 8.75, y: 3.3, w: 3.45, h: 2.85, fill: { color: C.card }, line: { type: "none" } });
  s.addText("The rare classes are hard.", { x: 9.0, y: 3.48, w: 3.0, h: 0.34, fontFace: SER, fontSize: 16, color: C.ink, margin: 0 });
  s.addText("DeepGlobe is heavily imbalanced. Weighted Dice loss lifts rangeland (" + pc(cl("rangeland")) +
            ") and barren (" + pc(cl("barren_land")) + ") into usefulness.\n\n'Unknown' is a void label — one pixel in the sampled set — so it sits outside the mean, as PASCAL VOC and Cityscapes do with theirs.",
    { x: 9.0, y: 3.95, w: 3.0, h: 2.05, fontFace: SANSL, fontSize: 10, color: C.muted, margin: 0, lineSpacingMultiple: 1.32 });
  s.addText([{ text: pc(MET.pixacc) + " of all pixels land in the right class", options: { color: C.ink } },
             { text: "   ·   mean IoU " + pc(MET.mIoU) + " across the six labelled classes, a deliberately strict measure   ·   " +
                     (MET.epochs || "—") + " epochs in " + (MET.minutes ? Math.round(MET.minutes) : "—") + " minutes on one T4",
               options: { color: C.muted } }],
    { x: M, y: 6.35, w: 11.2, h: 0.3, fontFace: SANS, fontSize: 10.5, charSpacing: 0.5, margin: 0 });
}

// ============================================== 6. THE BUILD
{ const s = slide(); wordmark(s); footer(s, 6);
  kicker(s, "PROCESS");
  heading(s, [{ text: "How it's built" }], 1.8, 42);
  dash(s, M, 2.95);
  const ph = [["DATA", "2448px tiles, random 512px crops, flips and rotations — aerial imagery has no 'up'."],
              ["ARCHITECTURE", "ResNet-34 encoder, U-Net decoder written from scratch."],
              ["TRAINING", "Dice + weighted cross-entropy, mixed precision, cosine schedule."],
              ["DEPLOY", "ONNX export, int8 quantisation, inference inside the browser."]];
  const y = 4.35, x0 = M + 0.9, x1 = W - M - 0.9, span = x1 - x0, step = span / (ph.length - 1);
  s.addShape(pres.shapes.LINE, { x: x0, y: y, w: span, h: 0, line: { color: C.line, width: 1.25 } });
  ph.forEach(function (p, i) {
    const cx = x0 + i * step;
    s.addShape(pres.shapes.OVAL, { x: cx - 0.17, y: y - 0.17, w: 0.34, h: 0.34, fill: { color: C.accent }, line: { type: "none" } });
    s.addText(String(i + 1).padStart(2, "0"), { x: cx - 0.9, y: y - 0.85, w: 1.8, h: 0.3, fontFace: SER, fontSize: 16, color: C.muted, align: "center", margin: 0 });
    s.addText(p[0], { x: cx - 1.25, y: y + 0.4, w: 2.5, h: 0.3, fontFace: SANS, fontSize: 11, color: C.ink, charSpacing: 2.5, align: "center", margin: 0 });
    s.addText(p[1], { x: cx - 1.3, y: y + 0.78, w: 2.6, h: 1.4, fontFace: SANSL, fontSize: 10.5, color: C.muted, align: "center", margin: 0, lineSpacingMultiple: 1.4 });
  });
}

// ============================================== 7. APPLICATIONS
{ const s = slide(); wordmark(s); footer(s, 7);
  kicker(s, "WHO IT'S FOR");
  heading(s, [{ text: "The mask is the demo.\n" }, { text: "The number is the product.", options: { italic: true, color: C.ink2 } }], 1.8, 38);
  dash(s, M, 3.5);
  const uses = [["AGRICULTURE", "Planted area per field, season over season, without anyone walking the land."],
                ["URBAN PLANNING", "Where the built-up edge moved, measured rather than estimated."],
                ["ENVIRONMENT & RISK", "Deforestation, water extent and wildfire fuel — inputs insurers price on."]];
  const cw = (W - 2 * M - 2 * 0.5) / 3;
  uses.forEach(function (u, i) {
    const x = M + i * (cw + 0.5), y = 3.95;
    s.addShape(pres.shapes.RECTANGLE, { x: x, y: y, w: cw, h: 2.3, fill: { color: C.card }, line: { type: "none" } });
    s.addShape(pres.shapes.RECTANGLE, { x: x, y: y, w: cw, h: 0.05, fill: { color: C.accent }, line: { type: "none" } });
    s.addText(u[0], { x: x + 0.32, y: y + 0.42, w: cw - 0.64, h: 0.3, fontFace: SANS, fontSize: 11, color: C.ink, charSpacing: 2.5, margin: 0 });
    s.addText(u[1], { x: x + 0.32, y: y + 0.88, w: cw - 0.64, h: 1.2, fontFace: SANSL, fontSize: 11, color: C.muted, margin: 0, lineSpacingMultiple: 1.45 });
  });
}

// ============================================== 8. CLOSE + QR
{ const s = slide();
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: W, h: 0.16, fill: { color: C.accent }, line: { type: "none" } });
  s.addText("M A P W I S E", { x: M, y: 2.15, w: 7, h: 0.6, fontFace: SER, fontSize: 38, color: C.ink, charSpacing: 10, margin: 0 });
  dash(s, M + 0.04, 3.0, 1.1);
  s.addText("See the ground as data.", { x: M, y: 3.3, w: 7, h: 0.6, fontFace: SER, fontSize: 30, italic: true, color: C.ink2, margin: 0 });
  s.addText("Try it on your own satellite tile. The model runs inside your browser,\nso the image never leaves your device.",
    { x: M, y: 4.2, w: 6.4, h: 0.9, fontFace: SANSL, fontSize: 12.5, color: C.muted, margin: 0, lineSpacingMultiple: 1.5 });
  s.addText("PRIYANSHU SINGH   ·   AI / ML ENGINEER", { x: M, y: 5.75, w: 7, h: 0.3, fontFace: SANS, fontSize: 10, color: C.ink, charSpacing: 3, margin: 0 });
  s.addText("github.com/Priyanshu-Singh-git/mapwise", { x: M, y: 6.12, w: 7, h: 0.3, fontFace: SANSL, fontSize: 10.5, color: C.muted, margin: 0 });

  const qx = 8.85, qy = 1.95, qs = 3.2;
  if (has("qr_demo.png")) {
    s.addShape(pres.shapes.RECTANGLE, { x: qx, y: qy, w: qs, h: qs + 0.95, fill: { color: C.card }, line: { type: "none" } });
    s.addImage(fit(path.join(FIG, "qr_demo.png"), qx + 0.42, qy + 0.42, qs - 0.84, qs - 0.84));
    s.addText("SCAN TO TRY IT LIVE", { x: qx, y: qy + qs - 0.3, w: qs, h: 0.3, fontFace: SANS, fontSize: 10, color: C.ink, charSpacing: 2.5, align: "center", margin: 0 });
    s.addText("priyanshu-singh-git.github.io/mapwise", { x: qx, y: qy + qs + 0.04, w: qs, h: 0.4, fontFace: SANSL, fontSize: 9.5, color: C.muted, align: "center", margin: 0 });
  }
}

(async () => {
  const out = path.join(__dirname, "mapwise_deck.pptx");
  await pres.writeFile({ fileName: out });
  console.log("wrote", out);
})();
