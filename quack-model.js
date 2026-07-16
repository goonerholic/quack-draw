/* 꽥드로우 경량 추론 엔진 — 순수 JS, 의존성 없음.
   학습 파이프라인(train.py)의 래스터화와 반드시 1:1로 동일해야 한다. */
(function (global) {
  "use strict";

  const SIZE = 28;
  const BOX = 22;
  // 반경 1.5 디스크 = 3x3 블록 (파이썬 _OFFS 와 동일)
  const OFFS = [];
  for (let dy = -1; dy <= 1; dy++)
    for (let dx = -1; dx <= 1; dx++) OFFS.push([dx, dy]);

  // strokes: [ [ [x...], [y...] ], ... ]  ->  Float32Array(SIZE*SIZE) in [0,1]
  function rasterize(strokes) {
    const img = new Float32Array(SIZE * SIZE);
    let minx = Infinity, maxx = -Infinity, miny = Infinity, maxy = -Infinity;
    let any = false;
    for (const s of strokes) {
      const X = s[0], Y = s[1];
      for (let i = 0; i < X.length; i++) {
        any = true;
        if (X[i] < minx) minx = X[i];
        if (X[i] > maxx) maxx = X[i];
        if (Y[i] < miny) miny = Y[i];
        if (Y[i] > maxy) maxy = Y[i];
      }
    }
    if (!any) return img;
    const w = maxx - minx, h = maxy - miny;
    const scale = BOX / Math.max(w, h, 1e-6);
    const offx = (SIZE - w * scale) / 2 - minx * scale;
    const offy = (SIZE - h * scale) / 2 - miny * scale;

    function stamp(px, py) {
      const ix = Math.floor(px + 0.5), iy = Math.floor(py + 0.5);
      for (let k = 0; k < OFFS.length; k++) {
        const x = ix + OFFS[k][0], y = iy + OFFS[k][1];
        if (x >= 0 && x < SIZE && y >= 0 && y < SIZE) img[y * SIZE + x] = 1;
      }
    }

    for (const s of strokes) {
      const X = s[0], Y = s[1], n = X.length;
      if (n === 0) continue;
      if (n === 1) { stamp(X[0] * scale + offx, Y[0] * scale + offy); continue; }
      for (let i = 0; i < n - 1; i++) {
        const x0 = X[i] * scale + offx, y0 = Y[i] * scale + offy;
        const x1 = X[i + 1] * scale + offx, y1 = Y[i + 1] * scale + offy;
        const dist = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0));
        const steps = Math.floor(dist * 2) + 1;
        for (let t = 0; t <= steps; t++) {
          const f = t / steps;
          stamp(x0 + (x1 - x0) * f, y0 + (y1 - y0) * f);
        }
      }
    }
    return img;
  }

  // model.json 을 받아 가중치를 Float32 로 역양자화해 둔다.
  function prepare(model) {
    const layers = model.layers.map((L) => {
      const w = new Float32Array(L.w.length);
      const sc = L.wscale;
      for (let i = 0; i < L.w.length; i++) w[i] = L.w[i] * sc;
      return { in: L.in, out: L.out, w, b: Float32Array.from(L.b) };
    });
    return { labels: model.labels, cats: model.cats, layers, size: model.size };
  }

  // 순전파 -> softmax 확률 배열
  function predict(prepared, input) {
    let a = input;
    const Ls = prepared.layers;
    for (let li = 0; li < Ls.length; li++) {
      const L = Ls[li], out = new Float32Array(L.out), w = L.w, b = L.b, nin = L.in;
      for (let o = 0; o < L.out; o++) {
        let s = b[o];
        const base = o * nin;
        for (let i = 0; i < nin; i++) s += a[i] * w[base + i];
        out[o] = s;
      }
      if (li < Ls.length - 1) { // 은닉층 ReLU
        for (let i = 0; i < out.length; i++) if (out[i] < 0) out[i] = 0;
      }
      a = out;
    }
    // softmax
    let mx = -Infinity;
    for (let i = 0; i < a.length; i++) if (a[i] > mx) mx = a[i];
    let sum = 0;
    const p = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) { p[i] = Math.exp(a[i] - mx); sum += p[i]; }
    for (let i = 0; i < p.length; i++) p[i] /= sum;
    return p;
  }

  function topk(prepared, probs, k) {
    const idx = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]);
    return idx.slice(0, k).map((i) => ({
      i, cat: prepared.cats[i], label: prepared.labels[i], p: probs[i],
    }));
  }

  const API = { SIZE, BOX, rasterize, prepare, predict, topk };
  if (typeof module !== "undefined" && module.exports) module.exports = API;
  else global.QuackModel = API;
})(typeof globalThis !== "undefined" ? globalThis : this);
