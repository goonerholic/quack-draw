#!/usr/bin/env python3
"""꽥드로우 CNN 학습 — 대규모 카테고리(~100), 순수 JS 추론용 가중치 내보내기.
   존재하지 않거나 다운로드 실패한 카테고리는 자동으로 건너뛴다."""
import json, urllib.request, urllib.parse, os, base64
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# (QuickDraw 이름, 한국어, 이모지)  — 중복/실패는 자동 정리
RAW = [
 ("cat","고양이","🐱"),("dog","강아지","🐶"),("fish","물고기","🐟"),("duck","오리","🦆"),
 ("bird","새","🐦"),("rabbit","토끼","🐰"),("butterfly","나비","🦋"),("bee","벌","🐝"),
 ("snail","달팽이","🐌"),("snake","뱀","🐍"),("ant","개미","🐜"),("spider","거미","🕷️"),
 ("frog","개구리","🐸"),("penguin","펭귄","🐧"),("owl","부엉이","🦉"),("elephant","코끼리","🐘"),
 ("giraffe","기린","🦒"),("lion","사자","🦁"),("tiger","호랑이","🐯"),("bear","곰","🐻"),
 ("monkey","원숭이","🐵"),("pig","돼지","🐷"),("cow","소","🐮"),("horse","말","🐴"),
 ("sheep","양","🐑"),("mouse","쥐","🐭"),("dolphin","돌고래","🐬"),("whale","고래","🐳"),
 ("shark","상어","🦈"),("octopus","문어","🐙"),("crab","게","🦀"),("sea turtle","거북이","🐢"),
 ("swan","백조","🦢"),("crocodile","악어","🐊"),("zebra","얼룩말","🦓"),("camel","낙타","🐫"),
 ("flamingo","홍학","🦩"),("parrot","앵무새","🦜"),("hedgehog","고슴도치","🦔"),("kangaroo","캥거루","🦘"),
 ("apple","사과","🍎"),("banana","바나나","🍌"),("grapes","포도","🍇"),("strawberry","딸기","🍓"),
 ("watermelon","수박","🍉"),("pear","배","🍐"),("pineapple","파인애플","🍍"),("mushroom","버섯","🍄"),
 ("carrot","당근","🥕"),("broccoli","브로콜리","🥦"),("bread","빵","🍞"),("hamburger","햄버거","🍔"),
 ("hot dog","핫도그","🌭"),("pizza","피자","🍕"),("ice cream","아이스크림","🍦"),("donut","도넛","🍩"),
 ("cake","케이크","🍰"),("cookie","쿠키","🍪"),("lollipop","막대사탕","🍭"),("sandwich","샌드위치","🥪"),
 ("sun","해","☀️"),("moon","달","🌙"),("star","별","⭐"),("cloud","구름","☁️"),
 ("rainbow","무지개","🌈"),("flower","꽃","🌷"),("tree","나무","🌳"),("leaf","나뭇잎","🍃"),
 ("cactus","선인장","🌵"),("mountain","산","⛰️"),("snowflake","눈송이","❄️"),("lightning","번개","⚡"),
 ("palm tree","야자수","🌴"),
 ("car","자동차","🚗"),("bus","버스","🚌"),("bicycle","자전거","🚲"),("airplane","비행기","✈️"),
 ("sailboat","배","⛵"),("train","기차","🚂"),("truck","트럭","🚚"),("helicopter","헬리콥터","🚁"),
 ("bulldozer","불도저","🚜"),
 ("house","집","🏠"),("umbrella","우산","☂️"),("hat","모자","🎩"),("key","열쇠","🔑"),
 ("scissors","가위","✂️"),("clock","시계","⏰"),("microphone","마이크","🎤"),("book","책","📖"),
 ("pencil","연필","✏️"),("candle","초","🕯️"),("guitar","기타","🎸"),("crown","왕관","👑"),
 ("camera","카메라","📷"),("door","문","🚪"),("ladder","사다리","🪜"),("hammer","망치","🔨"),
 ("saw","톱","🪚"),("telephone","전화기","☎️"),("television","텔레비전","📺"),("laptop","노트북","💻"),
 ("light bulb","전구","💡"),("envelope","편지","✉️"),("fork","포크","🍴"),("spoon","숟가락","🥄"),
 ("knife","칼","🔪"),("shoe","신발","👟"),("sock","양말","🧦"),("t-shirt","티셔츠","👕"),
 ("pants","바지","👖"),("eyeglasses","안경","👓"),("wristwatch","손목시계","⌚"),("drums","드럼","🥁"),
 ("trumpet","트럼펫","🎺"),("violin","바이올린","🎻"),("piano","피아노","🎹"),("tent","텐트","⛺"),
 ("castle","성","🏰"),("lighthouse","등대","🗼"),("bridge","다리","🌉"),
 ("kite","연","🪁"),("snowman","눈사람","⛄"),("basketball","농구공","🏀"),
 ("soccer ball","축구공","⚽"),("baseball","야구공","⚾"),("tennis racquet","테니스채","🎾"),
 ("cup","컵","🥤"),("chair","의자","🪑"),("bed","침대","🛏️"),("toothbrush","칫솔","🪥"),
 ("skateboard","스케이트보드","🛹"),("eye","눈","👁️"),("hand","손","✋"),("tooth","이빨","🦷"),
 ("smiley face","웃는얼굴","😃"),
]

# 중복 제거(첫 등장 유지)
CATS = []
_seen = set()
for c, k, e in RAW:
    if c not in _seen:
        _seen.add(c); CATS.append((c, k, e))

SIZE, BOX = 28, 22
N_PER = 5000
FETCH_BYTES = 16 * 1024 * 1024
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


def rasterize(strokes):  # quack-model.js 와 1:1 동일
    xs = [x for s in strokes for x in s[0]]
    ys = [y for s in strokes for y in s[1]]
    if not xs:
        return np.zeros(SIZE*SIZE, np.uint8)
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    w, h = maxx-minx, maxy-miny
    scale = BOX / max(w, h, 1e-6)
    offx = (SIZE - w*scale)/2 - minx*scale
    offy = (SIZE - h*scale)/2 - miny*scale
    img = np.zeros((SIZE, SIZE), np.uint8)

    def stamp(px, py):
        ix, iy = int(np.floor(px+0.5)), int(np.floor(py+0.5))
        for dx, dy in _OFFS:
            x, y = ix+dx, iy+dy
            if 0 <= x < SIZE and 0 <= y < SIZE:
                img[y, x] = 1
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
    kept, X_list, y, dumps = [], [], [], []
    for cat, ko, em in CATS:
        try:
            draws = fetch_lines(cat, N_PER)
        except Exception as ex:
            print(f"  SKIP {cat}: {type(ex).__name__}", flush=True); continue
        if len(draws) < 1000:
            print(f"  SKIP {cat}: only {len(draws)}", flush=True); continue
        lbl = len(kept)
        for d in draws:
            X_list.append(rasterize(d)); y.append(lbl)
        for d in draws[:2]:
            dumps.append({"label": lbl, "cat": cat, "strokes": d,
                          "bitmap": rasterize(d).astype(float).tolist()})
        kept.append((cat, ko, em))
        print(f"[{len(kept)}] {cat} ({ko})  n={len(draws)}", flush=True)

    print(f"\n총 카테고리 {len(kept)}개", flush=True)
    X = np.asarray(X_list, np.uint8).reshape(-1, SIZE, SIZE, 1)
    y = np.asarray(y, np.int64)
    del X_list
    print("dataset", X.shape, flush=True)

    n = len(X); idx = np.random.RandomState(42).permutation(n)
    X, y = X[idx], y[idx]
    cut = int(n*0.92)
    Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]

    def ds(Xa, ya, training):
        d = tf.data.Dataset.from_tensor_slices((Xa, ya))
        if training:
            d = d.shuffle(20000)
        d = d.map(lambda a, b: (tf.cast(a, tf.float32), b), num_parallel_calls=tf.data.AUTOTUNE)
        return d.batch(256).prefetch(tf.data.AUTOTUNE)

    m = keras.Sequential([
        keras.Input((SIZE, SIZE, 1)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(len(kept)),
    ])
    m.compile(optimizer="adam",
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
    m.fit(ds(Xtr, ytr, True), validation_data=ds(Xte, yte, False), epochs=22,
          callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True,
                                                   monitor="val_accuracy")], verbose=2)
    loss, acc = m.evaluate(ds(Xte, yte, False), verbose=0)
    print(f"\n검증 정확도(top-1): {acc:.4f}  클래스 {len(kept)}개", flush=True)

    export(m, acc, kept)
    with open(os.path.join(OUT_DIR, "jscheck_big.json"), "w") as f:
        json.dump({"size": SIZE, "box": BOX, "samples": dumps}, f)
    print("완료.")


def q_int8(W):
    mx = float(np.abs(W).max()) or 1.0
    s = mx/127.0
    q = np.clip(np.round(W/s), -127, 127).astype(np.int8)
    return base64.b64encode(q.tobytes()).decode(), s


def export(model, acc, kept):
    arch, weights = [], {}
    li = 0
    for layer in model.layers:
        cls = layer.__class__.__name__
        w = layer.get_weights()
        if cls == "Conv2D":
            K, b = w
            kb, ks = q_int8(K.astype(np.float32).flatten()); key = f"c{li}"
            weights[key+"w"] = kb; weights[key+"b"] = [float(v) for v in b]
            arch.append({"t":"conv","k":K.shape[0],"in":K.shape[2],"out":K.shape[3],"ws":ks,"key":key}); li += 1
        elif cls == "MaxPooling2D":
            arch.append({"t":"pool"})
        elif cls == "Flatten":
            arch.append({"t":"flatten"})
        elif cls == "Dropout":
            pass  # 추론 시 무시
        elif cls == "Dense":
            K, b = w
            kb, ks = q_int8(K.astype(np.float32).flatten()); key = f"d{li}"
            weights[key+"w"] = kb; weights[key+"b"] = [float(v) for v in b]
            arch.append({"t":"dense","in":K.shape[0],"out":K.shape[1],
                         "relu": layer.get_config().get("activation")=="relu","ws":ks,"key":key}); li += 1
    mj = {"size": SIZE, "box": BOX, "acc": round(float(acc),4),
          "cats":[c for c,_,_ in kept], "labels":[k for _,k,_ in kept], "emojis":[e for _,_,e in kept],
          "arch": arch, "weights": weights}
    path = os.path.join(OUT_DIR, "model_big.json")
    with open(path, "w") as f:
        json.dump(mj, f)
    print(f"model_big.json 저장: {os.path.getsize(path)/1024:.0f} KB, 카테고리 {len(kept)}", flush=True)


if __name__ == "__main__":
    main()
