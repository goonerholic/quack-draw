#!/usr/bin/env python3
"""꽥드로우 CNN 학습 — 36개 카테고리, 순수 JS 추론용 가중치 내보내기."""
import json, urllib.request, urllib.parse, os, base64
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# (QuickDraw 이름, 한국어, 이모지)
CATS = [
    ("cat","고양이","🐱"),("dog","강아지","🐶"),("fish","물고기","🐟"),("duck","오리","🦆"),
    ("bird","새","🐦"),("rabbit","토끼","🐰"),("butterfly","나비","🦋"),("bee","벌","🐝"),
    ("snail","달팽이","🐌"),("snake","뱀","🐍"),
    ("flower","꽃","🌷"),("tree","나무","🌳"),("sun","해","☀️"),("moon","달","🌙"),
    ("star","별","⭐"),("cloud","구름","☁️"),("rainbow","무지개","🌈"),
    ("house","집","🏠"),("car","자동차","🚗"),("bus","버스","🚌"),("bicycle","자전거","🚲"),
    ("airplane","비행기","✈️"),("sailboat","배","⛵"),
    ("apple","사과","🍎"),("banana","바나나","🍌"),("ice cream","아이스크림","🍦"),
    ("cake","케이크","🍰"),("cookie","쿠키","🍪"),("pizza","피자","🍕"),("carrot","당근","🥕"),
    ("umbrella","우산","☂️"),("hat","모자","🎩"),("key","열쇠","🔑"),("scissors","가위","✂️"),
    ("clock","시계","⏰"),("microphone","마이크","🎤"),
]

SIZE, BOX = 28, 22
N_PER = 7000
FETCH_BYTES = 20 * 1024 * 1024
BASE = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
_OFFS = [(dx, dy) for dy in (-1,0,1) for dx in (-1,0,1)]


def fetch_lines(cat, n):
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
            continue
        if not obj.get("recognized"):
            continue
        out.append(obj["drawing"])
        if len(out) >= n:
            break
    return out


def rasterize(strokes):  # quack-model.js 와 1:1 동일해야 함
    xs = [x for s in strokes for x in s[0]]
    ys = [y for s in strokes for y in s[1]]
    if not xs:
        return np.zeros(SIZE*SIZE, np.float32)
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    w, h = maxx-minx, maxy-miny
    scale = BOX / max(w, h, 1e-6)
    offx = (SIZE - w*scale)/2 - minx*scale
    offy = (SIZE - h*scale)/2 - miny*scale
    img = np.zeros((SIZE, SIZE), np.float32)

    def stamp(px, py):
        ix, iy = int(np.floor(px+0.5)), int(np.floor(py+0.5))
        for dx, dy in _OFFS:
            x, y = ix+dx, iy+dy
            if 0 <= x < SIZE and 0 <= y < SIZE:
                img[y, x] = 1.0
    for s in strokes:
        pxs = [x*scale+offx for x in s[0]]
        pys = [y*scale+offy for y in s[1]]
        n = len(pxs)
        if n == 1:
            stamp(pxs[0], pys[0]); continue
        for i in range(n-1):
            x0,y0,x1,y1 = pxs[i],pys[i],pxs[i+1],pys[i+1]
            steps = int(max(abs(x1-x0),abs(y1-y0))*2)+1
            for t in range(steps+1):
                f = t/steps
                stamp(x0+(x1-x0)*f, y0+(y1-y0)*f)
    return img.reshape(-1)


def main():
    X, y, dumps = [], [], []
    for idx,(cat,ko,em) in enumerate(CATS):
        print(f"[{idx+1}/{len(CATS)}] {cat} ({ko})", flush=True)
        draws = fetch_lines(cat, N_PER)
        for d in draws:
            X.append(rasterize(d)); y.append(idx)
        for d in draws[:2]:
            dumps.append({"label": idx, "cat": cat, "strokes": d, "bitmap": rasterize(d).tolist()})
    X = np.asarray(X, np.float32).reshape(-1, SIZE, SIZE, 1)
    y = np.asarray(y, np.int64)
    print("dataset", X.shape, flush=True)

    n = len(X); idx = np.random.RandomState(42).permutation(n)
    X, y = X[idx], y[idx]
    cut = int(n*0.9)
    Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]

    m = keras.Sequential([
        keras.Input((SIZE, SIZE, 1)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(len(CATS)),
    ])
    m.compile(optimizer="adam",
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
    m.fit(Xtr, ytr, validation_data=(Xte, yte), epochs=16, batch_size=256,
          callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True,
                                                   monitor="val_accuracy")], verbose=2)
    loss, acc = m.evaluate(Xte, yte, verbose=0)
    print(f"\n검증 정확도: {acc:.4f}", flush=True)

    export(m, acc)
    with open(os.path.join(OUT_DIR, "jscheck_cnn.json"), "w") as f:
        json.dump({"size": SIZE, "box": BOX, "samples": dumps}, f)
    print("완료.")


def q_int8(W):
    mx = float(np.abs(W).max()) or 1.0
    s = mx/127.0
    q = np.clip(np.round(W/s), -127, 127).astype(np.int8)
    return base64.b64encode(q.tobytes()).decode(), s


def main_export_layer(layer, arch, weights):
    pass


def export(model, acc):
    arch, weights = [], {}
    li = 0
    for layer in model.layers:
        w = layer.get_weights()
        cls = layer.__class__.__name__
        if cls == "Conv2D":
            K, b = w  # (kh,kw,in,out),(out,)
            kb, ks = q_int8(K.astype(np.float32).flatten())
            key = f"c{li}"
            weights[key+"w"] = kb; weights[key+"b"] = [float(v) for v in b]
            arch.append({"t":"conv","k":K.shape[0],"in":K.shape[2],"out":K.shape[3],
                         "ws":ks,"key":key}); li += 1
        elif cls == "MaxPooling2D":
            arch.append({"t":"pool"})
        elif cls == "Flatten":
            arch.append({"t":"flatten"})
        elif cls == "Dense":
            K, b = w  # (in,out)
            kb, ks = q_int8(K.astype(np.float32).flatten())
            key = f"d{li}"
            weights[key+"w"] = kb; weights[key+"b"] = [float(v) for v in b]
            arch.append({"t":"dense","in":K.shape[0],"out":K.shape[1],
                         "relu": layer.get_config().get("activation")=="relu",
                         "ws":ks,"key":key}); li += 1
    model_json = {
        "size": SIZE, "box": BOX, "acc": round(float(acc),4),
        "cats":[c for c,_,_ in CATS], "labels":[k for _,k,_ in CATS], "emojis":[e for _,_,e in CATS],
        "arch": arch, "weights": weights,
    }
    path = os.path.join(OUT_DIR, "model_cnn.json")
    with open(path, "w") as f:
        json.dump(model_json, f)
    print(f"model_cnn.json 저장: {os.path.getsize(path)/1024:.0f} KB")


if __name__ == "__main__":
    main()
