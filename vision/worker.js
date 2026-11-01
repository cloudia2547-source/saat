// s'aat Vision - runs the trash model off the main thread so the camera view stays smooth.
// ONNX Runtime Web 1.18 "core" build: single-threaded, normal (non-shared) memory.
// Newer builds reserve 4 GB of shared memory up front, which iPhone/iPad Safari refuses ("RangeError: Out of memory").
importScripts("ort.wasm-core.min.js");

const S = 416;
let session = null;

function iou(a, b) {
  const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const inter = ix * iy;
  const u = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return u > 0 ? inter / u : 0;
}

// output0: [1, 4 + nClasses + 32, N]  (cx, cy, w, h in 416-px letterbox space, then class scores, then mask coeffs)
function decode(d, dims, nClasses, conf) {
  const N = dims[2], cand = [];
  let topScore = 0, topCls = -1;            // strongest guess in the frame, even below the threshold (for diagnostics)
  for (let i = 0; i < N; i++) {
    let best = 0, bc = -1;
    for (let c = 0; c < nClasses; c++) {
      const s = d[(4 + c) * N + i];
      if (s > best) { best = s; bc = c; }
    }
    if (best > topScore) { topScore = best; topCls = bc; }
    if (best < conf) continue;
    const cx = d[i], cy = d[N + i], w = d[2 * N + i], h = d[3 * N + i];
    cand.push({ x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2, score: best, cls: bc });
  }
  cand.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const b of cand) {
    if (keep.every(k => iou(k, b) < 0.5)) keep.push(b);   // class-agnostic: one label per object
    if (keep.length >= 40) break;
  }
  return { keep, topScore, topCls };
}

self.onmessage = async (e) => {
  const m = e.data;
  if (m.type === "init") {
    ort.env.wasm.wasmPaths = new URL("./", self.location.href).href;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    let firstErr = null;
    try {
      const resp = await fetch(m.model);
      if (!resp.ok) throw new Error("model download failed (HTTP " + resp.status + ")");
      const bytes = new Uint8Array(await resp.arrayBuffer());
      // try the fast SIMD engine first, then the plain one for older devices
      for (const simd of [true, false]) {
        try {
          ort.env.wasm.simd = simd;
          session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
          const warm = new Float32Array(3 * S * S);
          await session.run({ images: new ort.Tensor("float32", warm, [1, 3, S, S]) });
          postMessage({ type: "ready", engine: simd ? "SIMD" : "basic" });
          return;
        } catch (err) {
          firstErr = firstErr || err;
          session = null;
        }
      }
      throw firstErr;
    } catch (err) {
      postMessage({ type: "error", message: String((err && err.message) || err) });
    }
  } else if (m.type === "run") {
    const t0 = performance.now();
    if (!session) { postMessage({ type: "result", id: m.id, dets: [], ms: 0, error: "model not loaded" }); return; }
    try {
      const res = await session.run({ images: new ort.Tensor("float32", m.input, [1, 3, S, S]) });
      const out = res.output0;
      const r = decode(out.data, out.dims, m.nClasses, m.conf);
      postMessage({ type: "result", id: m.id, dets: r.keep, topScore: r.topScore, topCls: r.topCls, ms: performance.now() - t0 });
    } catch (err) {
      postMessage({ type: "result", id: m.id, dets: [], ms: 0, error: String((err && err.message) || err) });
    }
  }
};
