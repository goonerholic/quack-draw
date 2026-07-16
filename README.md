# 🦆 꽥드로우 (Quack Draw)

구글 [Quick, Draw!](https://quickdraw.withgoogle.com/) 를 딸래미 전용으로 만든 그림 맞히기 게임.
그릴 대상이 화면에 뜨면 아이가 그림을 그리고, **오리가 갸우뚱하다가 "아! ○○지?" 하고 소리내어 알아맞힙니다**.

- 그냥 `index.html` 을 열면 바로 플레이 (GitHub Pages 로도 배포 가능)
- 카테고리 12개: 오리·고양이·물고기·꽃·나무·집·별·자동차·우산·나비·사과·아이스크림

## 인프라 / 모델 (초경량 · 완전 자립형)

- **백엔드 0.** 정적 파일만으로 동작. 추론은 전부 브라우저 안에서.
- **외부 의존성 0.** TensorFlow.js 같은 런타임/CDN 없이 순수 JavaScript 로 추론.
- **모델**: 구글 QuickDraw 획(stroke) 데이터로 학습한 작은 MLP(256→128).
  가중치를 int8 로 양자화해 `model.json`(약 0.9MB, gzip 시 훨씬 작음) 하나로 저장.
- **정확도**: 검증셋 약 90% (카테고리별 85~97%).
- **음성**: Web Speech API(TTS)로 오리가 한국어로 추측을 말합니다.
  (기기에 한국어 TTS 목소리가 있어야 소리가 납니다.)

## 구성 파일

| 파일 | 역할 |
|------|------|
| `index.html` | 게임 화면 (캔버스·오리·표정·타이머·점수·색종이·음성) |
| `quack-model.js` | 순수 JS 추론 엔진 (획→28×28 래스터화 + MLP 순전파 + softmax) |
| `model.json` | int8 양자화된 모델 가중치 + 라벨 |
| `tools/train.py` | 재현용 학습 파이프라인 |

## 핵심 포인트

학습용 렌더링(`tools/train.py`)과 브라우저 렌더링(`quack-model.js`)이 **비트 단위로 동일한**
결정론적 래스터화(획→28×28)를 쓰기 때문에, 학습 분포와 실사용 입력이 어긋나지 않습니다.
(Node 교차검증에서 두 구현의 비트맵 차이 0, 표본 예측 24/24 일치 확인.)

## GitHub Pages 배포

리포 Settings → Pages → Source 를 `main` 브랜치 루트로 지정하면
`https://<사용자>.github.io/quack-draw/` 에서 바로 플레이할 수 있습니다.

## 카테고리 바꾸기 / 재학습

`tools/train.py` 의 `CATS` 목록을 수정한 뒤 아래를 실행하면 `model.json` 이 새로 만들어집니다.
(QuickDraw 카테고리명은 https://storage.googleapis.com/quickdraw_dataset/full/simplified/ 참고)

```bash
pip install numpy scikit-learn
python3 tools/train.py     # QuickDraw 데이터를 받아 학습 후 model.json 생성
```
