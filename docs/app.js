/* MapWise — land-cover segmentation in the browser.
   Preprocess to 512x512 -> ONNX U-Net -> argmax -> colour mask + per-class area. */
const $ = (s) => document.querySelector(s);
const SIZE = 512;
let META = null, SESSION = null, SRC = null;

const statusEl = $("#status");
const say = (t) => (statusEl.textContent = t);

async function boot() {
  try {
    META = await (await fetch("models/mapwise_meta.json")).json();
    ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
    // Only the 24 MB int8 model is published; the 94 MB fp32 export stays out of the repo.
    // The fp32 path is kept as a fallback for anyone running this locally after an export.
    for (const f of ["models/mapwise_int8.onnx", "models/mapwise.onnx"]) {
      try { SESSION = await ort.InferenceSession.create(f, { executionProviders: ["wasm"] }); break; }
      catch (e) { /* try next */ }
    }
    if (!SESSION) throw new Error("no model file found");
    drawLegend();
    if (META.mIoU) $("#metrics").textContent = ` Validation mean IoU ${(META.mIoU * 100).toFixed(1)}%.`;
    say("Model ready — choose an image.");
    $("#run").disabled = !SRC;
  } catch (e) {
    say("Could not load the model: " + e.message);
  }
}

function drawLegend(shares) {
  if (!META) return;
  $("#legend").innerHTML = META.classes.map((name, i) => {
    const c = META.colors[i], rgb = `rgb(${c[0]},${c[1]},${c[2]})`;
    const pct = shares ? shares[i] * 100 : null;
    return `<tr>
      <td class="sw"><i style="background:${rgb}"></i></td>
      <td>${name}<div class="bar"><i style="width:${pct ?? 0}%;background:${rgb}"></i></div></td>
      <td class="pc">${pct === null ? "—" : pct.toFixed(1) + "%"}</td></tr>`;
  }).join("");
}

function showImage(img) {
  SRC = img;
  const c = $("#imgC"); c.width = SIZE; c.height = SIZE;
  const ctx = c.getContext("2d");
  // The model is trained on 512px crops at the imagery's NATIVE resolution. Shrinking a whole
  // 2448px tile into 512 would show it ground texture ~5x smaller than it ever saw, and the
  // predictions degrade badly. So take a native-scale centre crop when the image is larger,
  // and only upscale when it is smaller than the window.
  const s = Math.min(img.width, img.height, SIZE);
  ctx.drawImage(img, (img.width - s) / 2, (img.height - s) / 2, s, s, 0, 0, SIZE, SIZE);
  const m = $("#maskC"); m.width = SIZE; m.height = SIZE;
  m.getContext("2d").clearRect(0, 0, SIZE, SIZE);
  drawLegend();
  $("#run").disabled = !SESSION;
  const big = Math.min(img.width, img.height) > SIZE;
  say(big
    ? `Ready — showing a ${SIZE}x${SIZE} centre crop at native scale.`
    : "Ready to classify.");
}

function loadFile(f) {
  const img = new Image();
  img.onload = () => showImage(img);
  img.onerror = () => say("That file could not be read as an image.");
  img.src = URL.createObjectURL(f);
}

async function classify() {
  if (!SESSION || !SRC) return;
  $("#run").disabled = true;
  say("Classifying…");
  await new Promise((r) => setTimeout(r, 10));            // let the UI paint
  const t0 = performance.now();

  const px = $("#imgC").getContext("2d").getImageData(0, 0, SIZE, SIZE).data;
  const data = new Float32Array(3 * SIZE * SIZE);
  const { mean, std } = META;
  for (let i = 0, n = SIZE * SIZE; i < n; i++) {
    data[i] = (px[i * 4] / 255 - mean[0]) / std[0];
    data[n + i] = (px[i * 4 + 1] / 255 - mean[1]) / std[1];
    data[2 * n + i] = (px[i * 4 + 2] / 255 - mean[2]) / std[2];
  }

  let out;
  try {
    out = await SESSION.run({ input: new ort.Tensor("float32", data, [1, 3, SIZE, SIZE]) });
  } catch (e) {
    say("Inference failed: " + e.message); $("#run").disabled = false; return;
  }
  const logits = out[Object.keys(out)[0]].data;
  const K = META.classes.length, n = SIZE * SIZE;

  const counts = new Array(K).fill(0);
  const rgba = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    let best = 0, bv = logits[i];
    for (let k = 1; k < K; k++) { const v = logits[k * n + i]; if (v > bv) { bv = v; best = k; } }
    counts[best]++;
    const c = META.colors[best];
    rgba[i * 4] = c[0]; rgba[i * 4 + 1] = c[1]; rgba[i * 4 + 2] = c[2]; rgba[i * 4 + 3] = 255;
  }
  $("#maskC").getContext("2d").putImageData(new ImageData(rgba, SIZE, SIZE), 0, 0);
  drawLegend(counts.map((c) => c / n));
  say(`Done in ${((performance.now() - t0) / 1000).toFixed(1)}s — ${(n / 1000).toFixed(0)}k pixels classified.`);
  $("#run").disabled = false;
}

// ---- wiring ----
$("#drop").onclick = () => $("#file").click();
$("#file").onchange = (e) => e.target.files[0] && loadFile(e.target.files[0]);
$("#drop").ondragover = (e) => { e.preventDefault(); $("#drop").classList.add("hot"); };
$("#drop").ondragleave = () => $("#drop").classList.remove("hot");
$("#drop").ondrop = (e) => {
  e.preventDefault(); $("#drop").classList.remove("hot");
  const f = e.dataTransfer.files[0]; if (f) loadFile(f);
};
$("#run").onclick = classify;
$("#op").oninput = (e) => {
  $("#maskC").style.opacity = e.target.value / 100;
  $("#opv").textContent = e.target.value + "%";
};

fetch("samples/index.json").then((r) => r.json()).then((list) => {
  $("#samples").innerHTML = list.map((f) => `<img src="samples/${f}" data-f="${f}" alt="">`).join("");
  $("#samples").querySelectorAll("img").forEach((el) => {
    el.onclick = () => {
      $("#samples").querySelectorAll("img").forEach((o) => o.classList.remove("sel"));
      el.classList.add("sel");
      const img = new Image();
      img.onload = () => showImage(img);
      img.src = el.src;
    };
  });
}).catch(() => {});

boot();
