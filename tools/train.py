#!/usr/bin/env python3
"""꽥드로우 모델 학습 파이프라인.

- QuickDraw simplified ndjson 에서 카테고리별 표본을 range 요청으로 스트리밍
- 결정론적 래스터화(브라우저 JS 와 동일)로 28x28 비트맵 생성
- 작은 MLP 학습 -> int8 양자화 -> model.json 로 내보내기
- JS 교차검증용 표본(stroke + 정답 + 기대 비트맵) 덤프
"""
import json, urllib.request, urllib.parse, sys, os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# QuickDraw 카테고리명 -> 한국어 라벨
CATS = [
    ("duck", "오리"),
    ("cat", "고양이"),
    ("fish", "물고기"),
    ("flower", "꽃"),
    ("tree", "나무"),
    ("house", "집"),
    ("star", "별"),
    ("car", "자동차"),
    ("umbrella", "우산"),
    ("butterfly", "나비"),
    ("apple", "사과"),
    ("ice cream", "아이스크림"),
]

SIZE = 28          # 출력 비트맵 한 변
BOX = 22           # 그림이 들어갈 내부 정사각형(패딩 3px)
N_PER = 9000       # 카테고리당 표본 수
FETCH_BYTES = 22 * 1024 * 1024  # 카테고리당 최대 다운로드
BASE = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/"

# 리포 루트(= tools/ 의 상위)에 model.json 을 쓴다.
OUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fetch_lines(cat, n):
    """recognized=True 인 drawing 을 최대 n 개 반환."""
    url = BASE + urllib.parse.quote(cat) + ".ndjson"
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{FETCH_BYTES-1}"})
    data = urllib.request.urlopen(req, timeout=120).read().decode("utf-8", "ignore")
    out = []
    for line in data.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue  # 마지막 잘린 줄
        if not obj.get("recognized"):
            continue
        out.append(obj["drawing"])
        if len(out) >= n:
            break
    return out


# ── 결정론적 래스터화 (JS 와 1:1 대응) ──────────────────────────
# 반경 1.5 디스크 = 3x3 블록 오프셋
_OFFS = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]  # d2<=2 모두 포함


def rasterize(strokes):
    """strokes -> (SIZE*SIZE,) float32 in [0,1]."""
    xs = [x for s in strokes for x in s[0]]
    ys = [y for s in strokes for y in s[1]]
    if not xs:
        return np.zeros(SIZE * SIZE, np.float32)
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    w = maxx - minx
    h = maxy - miny
    scale = BOX / max(w, h, 1e-6)
    offx = (SIZE - w * scale) / 2.0 - minx * scale
    offy = (SIZE - h * scale) / 2.0 - miny * scale
    img = np.zeros((SIZE, SIZE), np.float32)

    def stamp(px, py):
        ix = int(np.floor(px + 0.5))
        iy = int(np.floor(py + 0.5))
        for dx, dy in _OFFS:
            x, y = ix + dx, iy + dy
            if 0 <= x < SIZE and 0 <= y < SIZE:
                img[y, x] = 1.0

    for s in strokes:
        pxs = [x * scale + offx for x in s[0]]
        pys = [y * scale + offy for y in s[1]]
        n = len(pxs)
        if n == 1:
            stamp(pxs[0], pys[0])
            continue
        for i in range(n - 1):
            x0, y0, x1, y1 = pxs[i], pys[i], pxs[i + 1], pys[i + 1]
            dist = max(abs(x1 - x0), abs(y1 - y0))
            steps = int(dist * 2) + 1
            for t in range(steps + 1):
                f = t / steps
                stamp(x0 + (x1 - x0) * f, y0 + (y1 - y0) * f)
    return img.reshape(-1)


def main():
    X, y = [], []
    dumps = []  # JS 검증용
    for idx, (cat, ko) in enumerate(CATS):
        print(f"[{idx+1}/{len(CATS)}] {cat} ({ko}) 다운로드…", flush=True)
        draws = fetch_lines(cat, N_PER)
        print(f"    {len(draws)}개 표본, 래스터화…", flush=True)
        for d in draws:
            X.append(rasterize(d))
            y.append(idx)
        # 앞쪽 2개는 JS 교차검증용으로 원본 stroke + 기대 비트맵 저장
        for d in draws[:2]:
            dumps.append({"label": idx, "cat": cat,
                          "strokes": d,
                          "bitmap": rasterize(d).tolist()})
    X = np.asarray(X, np.float32)
    y = np.asarray(y, np.int64)
    print("데이터셋:", X.shape, flush=True)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.12,
                                          random_state=42, stratify=y)
    clf = MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu",
                        alpha=1e-4, batch_size=256, learning_rate_init=1e-3,
                        max_iter=60, early_stopping=True, n_iter_no_change=6,
                        random_state=42, verbose=True)
    clf.fit(Xtr, ytr)
    acc = clf.score(Xte, yte)
    print(f"\n검증 정확도: {acc:.4f}", flush=True)

    # per-class 정확도
    pred = clf.predict(Xte)
    print("\n카테고리별 정확도:")
    for i, (cat, ko) in enumerate(CATS):
        m = yte == i
        if m.sum():
            print(f"  {cat:10s} {ko:8s} {(pred[m]==i).mean():.3f} (n={m.sum()})")

    export(clf, acc)
    with open(os.path.join(OUT_DIR, "jscheck.json"), "w") as f:
        json.dump({"size": SIZE, "box": BOX, "samples": dumps}, f)
    print("완료.")


def quantize(W):
    """행렬을 int8 로 양자화. 반환 (list[int], scale)."""
    m = float(np.abs(W).max()) or 1.0
    scale = m / 127.0
    q = np.clip(np.round(W / scale), -127, 127).astype(np.int8)
    return q, scale


def export(clf, acc):
    layers = []
    for W, b in zip(clf.coefs_, clf.intercepts_):
        qW, sW = quantize(W)
        layers.append({
            "in": W.shape[0], "out": W.shape[1], "wscale": sW,
            "w": qW.T.flatten().astype(int).tolist(),  # 행 우선: out x in
            "b": [float(v) for v in b],
        })
    model = {
        "size": SIZE, "box": BOX, "acc": round(float(acc), 4),
        "labels": [ko for _, ko in CATS],
        "cats": [c for c, _ in CATS],
        "activation": "relu",
        "layers": layers,
    }
    path = os.path.join(OUT_DIR, "model.json")
    with open(path, "w") as f:
        json.dump(model, f)
    print(f"model.json 저장: {os.path.getsize(path)/1024:.0f} KB")


if __name__ == "__main__":
    import urllib.parse
    main()
