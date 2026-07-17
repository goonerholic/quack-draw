/* 꽥드로우 경량 추론 엔진 — 순수 JS CNN, 의존성 없음.
   학습 파이프라인(tools/train_cnn.py)의 래스터화·연산과 반드시 1:1로 동일해야 한다. */
(function (global) {
  "use strict";

  const SIZE = 28;
  const BOX = 22;
  const OFFS = [];
  for (let dy = -1; dy <= 1; dy++)
    for (let dx = -1; dx <= 1; dx++) OFFS.push([dx, dy]);

  // strokes: [ [ [x...], [y...] ], ... ]  ->  Float32Array(SIZE*SIZE) in [0,1]
  function rasterize(strokes) {
    const img = new Float32Array(SIZE * SIZE);
    let minx = Infinity, maxx = -Infinity, miny = Infinity, maxy = -Infinity, any = false;
    for (const s of strokes) {
      const X = s[0], Y = s[1];
      for (let i = 0; i < X.length; i++) {
        any = true;
        if (X[i] < minx) minx = X[i]; if (X[i] > maxx) maxx = X[i];
        if (Y[i] < miny) miny = Y[i]; if (Y[i] > maxy) maxy = Y[i];
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

  // base64(int8 바이트) -> 역양자화된 Float32Array
  function deq(b64, scale) {
    const bin = atob(b64), n = bin.length, out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const c = bin.charCodeAt(i);
      out[i] = (c > 127 ? c - 256 : c) * scale;
    }
    return out;
  }

  // model.json -> 실행 준비된 레이어들
  function prepare(model) {
    const W = model.weights;
    const layers = model.arch.map((L) => {
      if (L.t === "conv") return { t: "conv", k: L.k, in: L.in, out: L.out,
        w: deq(W[L.key + "w"], L.ws), b: Float32Array.from(W[L.key + "b"]) };
      if (L.t === "dense") return { t: "dense", in: L.in, out: L.out, relu: !!L.relu,
        w: deq(W[L.key + "w"], L.ws), b: Float32Array.from(W[L.key + "b"]) };
      return { t: L.t }; // pool / flatten
    });
    return { labels: model.labels, cats: model.cats, emojis: model.emojis || [],
             size: model.size || SIZE, layers };
  }

  // same-pad 3x3 conv + ReLU (stride 1)
  function conv3(inp, H, W, Cin, ker, bias, Cout) {
    const out = new Float32Array(H * W * Cout), K = 3, pad = 1;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const oBase = (y * W + x) * Cout;
      for (let co = 0; co < Cout; co++) {
        let s = bias[co];
        for (let ky = 0; ky < K; ky++) {
          const iy = y + ky - pad; if (iy < 0 || iy >= H) continue;
          for (let kx = 0; kx < K; kx++) {
            const ix = x + kx - pad; if (ix < 0 || ix >= W) continue;
            const inBase = (iy * W + ix) * Cin;
            const kBase = ((ky * K + kx) * Cin) * Cout + co;
            for (let ci = 0; ci < Cin; ci++) s += inp[inBase + ci] * ker[kBase + ci * Cout];
          }
        }
        out[oBase + co] = s > 0 ? s : 0;
      }
    }
    return out;
  }

  // 2x2 maxpool (stride 2, valid)
  function pool2(inp, H, W, C) {
    const OH = H >> 1, OW = W >> 1, out = new Float32Array(OH * OW * C);
    for (let y = 0; y < OH; y++) for (let x = 0; x < OW; x++) for (let c = 0; c < C; c++) {
      let m = -Infinity;
      for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
        const v = inp[(((y * 2 + dy) * W) + (x * 2 + dx)) * C + c];
        if (v > m) m = v;
      }
      out[(y * OW + x) * C + c] = m;
    }
    return { out, OH, OW };
  }

  function dense(inp, inN, ker, bias, outN, relu) {
    const out = new Float32Array(outN);
    for (let o = 0; o < outN; o++) {
      let s = bias[o];
      for (let i = 0; i < inN; i++) s += inp[i] * ker[i * outN + o];
      out[o] = relu && s < 0 ? 0 : s;
    }
    return out;
  }

  // input: Float32Array(SIZE*SIZE) -> softmax 확률 배열
  function predict(P, input) {
    let data = input, H = P.size, W = P.size, C = 1;
    for (const L of P.layers) {
      if (L.t === "conv") { data = conv3(data, H, W, C, L.w, L.b, L.out); C = L.out; }
      else if (L.t === "pool") { const r = pool2(data, H, W, C); data = r.out; H = r.OH; W = r.OW; }
      else if (L.t === "flatten") { /* (H,W,C) C-order == 그대로 벡터 */ }
      else if (L.t === "dense") { data = dense(data, L.in, L.w, L.b, L.out, L.relu); }
    }
    // softmax
    let mx = -Infinity;
    for (let i = 0; i < data.length; i++) if (data[i] > mx) mx = data[i];
    let sum = 0; const p = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) { p[i] = Math.exp(data[i] - mx); sum += p[i]; }
    for (let i = 0; i < p.length; i++) p[i] /= sum;
    return p;
  }

  function topk(P, probs, k) {
    const idx = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]);
    return idx.slice(0, k).map((i) => ({
      i, cat: P.cats[i], label: P.labels[i], emoji: (P.emojis && P.emojis[i]) || "", p: probs[i],
    }));
  }

  const API = { SIZE, BOX, rasterize, prepare, predict, topk };
  if (typeof module !== "undefined" && module.exports) module.exports = API;
  else global.QuackModel = API;
})(typeof globalThis !== "undefined" ? globalThis : this);
