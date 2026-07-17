# 🦆 꽥드로우 (Quack Draw)

구글 [Quick, Draw!](https://quickdraw.withgoogle.com/) 를 딸래미 전용으로 만든 그림 맞히기 게임.
그릴 대상이 화면에 뜨면 아이가 그림을 그리고, **오리가 갸우뚱하다가 "아! ○○지?" 하고 소리내어 알아맞힙니다**.

- 그냥 `index.html` 을 열면 바로 플레이 (GitHub Pages 로도 배포 가능)
- 카테고리 12개: 오리·고양이·물고기·꽃·나무·집·별·자동차·우산·나비·사과·아이스크림

## 인프라 / 모델 (초경량 · 완전 자립형)

- **백엔드 0.** 정적 파일만으로 동작. 추론은 전부 브라우저 안에서.
- **외부 의존성 0.** TensorFlow.js 같은 런타임/CDN 없이 순수 JavaScript 로 추론.
- **모델**: 구글 QuickDraw 획(stroke) 데이터로 학습한 작은 **CNN**(conv16→conv32→conv64→dense256).
  가중치를 int8 로 양자화해 `model.json`(약 1.1MB) 하나로 저장. **135개 카테고리**(동물·음식·탈것·사물 등).
- **정확도**: 검증셋 top-1 약 71% (135개 클래스 기준, top-3는 훨씬 높음). 게임에선 목표가
  "명확한 1위" 또는 "근소한 2위"면 정답 인정(테스트상 정답 그림 81% 인정 / 오탐 0). 확신이
  낮거나 애매하면 억지로 답하지 않고 "잘 모르겠어"라고 갸우뚱합니다.
- **오리 캐릭터**: 부위별로 움직이는 SVG 오리 — 눈 깜빡, 눈동자 굴리기, 부리 뻐끔,
  머리 갸웃, 날개 파닥, 정답 시 춤, 어지럼 소용돌이 눈까지 표정이 살아있음.
- **소통**: 오리는 말을 하지 않고 **표정 + 말풍선**으로 반응합니다("이거 ○○ 맞지?", "잘 모르겠어",
  "○○ 맞았지? 정답!"). 소리는 귀여운 **"꽥" 효과음**(내장 합성 WAV)만, 말풍선 길이만큼 **부리 뻐끔**.
  (음성 합성 TTS는 사용하지 않음.)

## 구성 파일

| 파일 | 역할 |
|------|------|
| `index.html` | 게임 화면 (캔버스·오리·표정·타이머·점수·색종이·음성) |
| `quack-model.js` | 순수 JS 추론 엔진 (획→28×28 래스터화 + CNN conv/pool/dense + softmax) |
| `model.json` | int8 양자화된 CNN 가중치 + 라벨/이모지 |
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
