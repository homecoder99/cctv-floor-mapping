# 카메라 캘리브레이션

OpenCV를 사용한 단일 카메라 내부 파라미터 계산 프로그램입니다.

## 개요

체커보드 패턴을 이용하여 카메라의 내부 파라미터(intrinsic parameters)를 계산합니다:
- 카메라 내부 행렬 (Camera Matrix)
- 왜곡 계수 (Distortion Coefficients)

## 설치

필요한 패키지를 설치합니다:

```bash
pip install -r requirements.txt
```

## 사용 방법

### 1. 체커보드 준비

#### 방법 A: 체커보드 패턴 생성 (권장)

```bash
python generate_checkerboard.py
```

이 스크립트는 체커보드 이미지(`checkerboard_pattern.png`)를 생성합니다:
- 기본 설정: 9x6 내부 코너 (10x7 정사각형)
- 생성된 이미지를 전체화면으로 모니터에 띄웁니다
- 체커보드 정사각형 한 변의 길이를 자로 측정합니다 (모니터에 실제로 표시된 크기, mm 단위)

#### 방법 B: 체커보드 출력

- 체커보드 패턴을 인쇄하여 평평한 판에 붙입니다
- 체커보드 정사각형 한 변의 길이를 자로 측정합니다 (mm 단위)

### 2. 설정 파일 수정

**중요**: 캘리브레이션을 수행하기 전에 다음 파라미터를 확인하고 수정하세요:

#### `capture_images.py`와 `calibrate.py` 모두에서 설정:

```python
# 체커보드 내부 코너 개수 (가로, 세로)
CHECKERBOARD_SIZE = (9, 6)  # 사용하는 체커보드에 맞게 수정
```

#### `calibrate.py`에서만 설정:

```python
# 체커보드 정사각형 한 변의 길이 (mm)
SQUARE_SIZE = 25.0  # 실제 측정값으로 수정!
```

### 3. 이미지 캡처

#### RTSP 카메라 설정 (Tapo C200 등)

RTSP 스트림을 사용하려면 `capture_images.py` 파일에서 다음 부분을 수정하세요:

```python
# 기본값 (웹캠)
CAMERA_SOURCE = 0

# RTSP 스트림 사용 시 (주석 해제 후 실제 값으로 수정)
CAMERA_SOURCE = "rtsp://username:password@192.168.x.x:554/stream1"
```

**Tapo C200 RTSP URL 형식**:
```
rtsp://username:password@ip_address:554/stream1
```
- `username`: Tapo 앱에서 설정한 사용자명
- `password`: Tapo 앱에서 설정한 비밀번호
- `ip_address`: 카메라의 IP 주소
- `stream1`: 고화질 (1080p), `stream2`: 저화질 (360p)

#### 캡처 실행

```bash
python capture_images.py
```

**사용법**:
- 체커보드를 여러 각도와 위치에서 촬영합니다
- 체커보드가 화면에 검출되면 녹색 코너가 표시됩니다
- **스페이스바**: 현재 프레임을 저장 (체커보드가 검출되었을 때만 가능)
- **ESC**: 프로그램 종료

**촬영 팁**:
- 최소 15-20장의 이미지를 촬영하세요
- 다양한 각도 (정면, 좌우 기울임, 상하 기울임)
- 다양한 위치 (화면 중앙, 모서리)
- 다양한 거리
- 체커보드가 화면에 완전히 보이도록 촬영

### 4. 이미지 품질 검사 (권장)

캘리브레이션 전에 촬영한 이미지의 품질을 검사하세요:

```bash
python check_images.py
```

이 스크립트는 다음을 확인합니다:
- 체커보드 검출 성공/실패
- 이미지 밝기, 선명도
- 체커보드 크기 적절성
- 흔들림, 왜곡 등

문제가 있는 이미지를 찾아내고 해결 방법을 제시합니다.

### 5. 캘리브레이션 수행

```bash
python calibrate.py
```

캘리브레이션이 완료되면 다음 파일이 생성됩니다:
- `output/calibration_data.pkl`: 캘리브레이션 데이터 (Python에서 로드 가능)
- `output/calibration_results.txt`: 결과 요약 (사람이 읽기 쉬운 형태)
- `output/corners_*.jpg`: 검출된 코너가 표시된 이미지들

### 6. 결과 검증 (필수!)

**중요**: 캘리브레이션 후 반드시 결과를 검증하세요!

```bash
python validate_calibration.py
```

이 스크립트는 다음을 확인합니다:
- **fx/fy 비율**: 정사각형 픽셀 검증 (가장 중요!)
- **재투영 오차**: 캘리브레이션 정확도
- **왜곡 계수**: 렌즈 왜곡 합리성

검증 결과가 "부적합"이면 재촬영이 필요합니다.

## 결과 해석

### 카메라 내부 파라미터 행렬 (Camera Matrix)

```
[[fx  0  cx]
 [ 0 fy  cy]
 [ 0  0   1]]
```

- `fx`, `fy`: 초점 거리 (pixels)
- `cx`, `cy`: 주점 (principal point) 위치 (pixels)

### 왜곡 계수 (Distortion Coefficients)

```
[k1, k2, p1, p2, k3]
```

- `k1`, `k2`, `k3`: 방사 왜곡 (radial distortion)
- `p1`, `p2`: 접선 왜곡 (tangential distortion)

### 재투영 오차

- 낮을수록 좋습니다 (일반적으로 0.5 pixels 이하면 좋음)
- 오차가 크다면 더 많은 이미지를 촬영하거나 다양한 각도에서 촬영하세요

## 캘리브레이션 데이터 사용 예제

```python
import pickle
import numpy as np

# 캘리브레이션 데이터 로드
with open('output/calibration_data.pkl', 'rb') as f:
    calib_data = pickle.load(f)

camera_matrix = calib_data['camera_matrix']
dist_coeffs = calib_data['dist_coeffs']

print("Camera Matrix:")
print(camera_matrix)
print("\nDistortion Coefficients:")
print(dist_coeffs)
```

## 문제 해결

### 체커보드가 검출되지 않는 경우

1. 조명을 확인하세요 (너무 어둡거나 밝지 않게)
2. 체커보드가 선명하게 보이는지 확인
3. `CHECKERBOARD_SIZE` 설정이 올바른지 확인
4. 카메라 초점이 맞았는지 확인

### RTSP 연결 문제

1. **"No route to host" 또는 "Connection failed" 에러**:

   `capture_images.py` 파일 상단에서 `RTSP_TRANSPORT` 설정 변경:
   ```python
   # 기본값 (TCP)
   RTSP_TRANSPORT = 'tcp'

   # UDP로 변경 시도
   RTSP_TRANSPORT = 'udp'
   ```

   일반적으로 UDP가 더 빠르고 TCP가 더 안정적입니다. 둘 다 시도해보세요.

1-1. **프레임 지연이 심한 경우 (10초 이상)**:

   `capture_images.py` 파일 상단에서 `RTSP_FRAME_SKIP` 값을 조정:
   ```python
   # 기본값
   RTSP_FRAME_SKIP = 5

   # 지연이 더 크면 높이기 (최대 10)
   RTSP_FRAME_SKIP = 10

   # 영상이 끊기면 낮추기
   RTSP_FRAME_SKIP = 2
   ```

   - 높은 값: 더 최신 프레임 (지연 감소, 프레임 드롭 가능)
   - 낮은 값: 더 부드러운 영상 (지연 증가 가능)

2. **카메라를 열 수 없습니다** 에러:
   - RTSP URL 형식 확인: `rtsp://username:password@ip:554/stream1`
   - 카메라의 RTSP 기능이 활성화되어 있는지 확인
   - 방화벽이 554 포트를 차단하지 않는지 확인
   - `stream1` 대신 `stream2` 시도

3. **첫 프레임을 읽을 수 없습니다** 에러:
   - 사용자명과 비밀번호 재확인
   - 카메라가 네트워크에 연결되어 있는지 확인
   - 다른 프로그램에서 RTSP 스트림을 사용 중인지 확인

4. **연결이 느리거나 끊김**:
   - `stream2` (저화질)로 변경 시도
   - WiFi 신호 강도 확인
   - 유선 랜 연결 고려

### 캘리브레이션 검증 실패 (fx/fy 비율 문제)

**증상**: `validate_calibration.py` 실행 시 "fx/fy 비율이 비정상"이라는 치명적 오류

**원인 및 해결**:

1. **체커보드 크기 설정 오류** (가장 흔한 원인):

   `capture_images.py`와 `calibrate.py`의 `CHECKERBOARD_SIZE` 확인:
   ```python
   CHECKERBOARD_SIZE = (9, 6)  # 내부 코너 개수!
   ```

   - **중요**: 정사각형 개수가 아니라 **내부 교차점(코너) 개수**입니다
   - 10x7 정사각형 체커보드 = 9x6 내부 코너
   - 실제 체커보드를 세어보고 정확히 설정하세요

2. **체커보드 왜곡** (모니터 사용 시):

   - 모니터 화면 비율 설정을 확인하세요
   - 생성된 체커보드 이미지가 늘어나지 않았는지 확인
   - 자로 화면의 정사각형을 측정하여 실제로 정사각형인지 확인
   - **권장**: 모니터 대신 출력된 체커보드 사용

3. **촬영 각도의 다양성 부족**:

   - 정면에서만 찍으면 fx/fy를 정확히 분리할 수 없습니다
   - **필수**: 체커보드를 **45도 이상 기울여서** 촬영
   - X축 회전 (위아래 기울임): 5장 이상
   - Y축 회전 (좌우 기울임): 5장 이상
   - Z축 회전 (평면 회전): 몇 장

4. **정사각형 크기 측정 오류**:

   `calibrate.py`의 `SQUARE_SIZE` 재확인:
   - 자로 실제 체커보드 정사각형 한 변을 mm 단위로 정확히 측정
   - 모니터를 사용한 경우, 화면에 표시된 크기를 측정

### 캘리브레이션 오차가 큰 경우

1. **먼저 이미지 품질을 검사**:
   ```bash
   python check_images.py
   ```

2. 더 많은 이미지를 촬영하세요 (20-30장 권장)
3. **다양한 각도**에서 촬영하세요 (특히 기울임!)
4. 흔들리지 않게 촬영하세요
5. `SQUARE_SIZE`가 정확한지 재확인하세요
6. 문제 이미지를 `images/` 폴더에서 삭제하고 재촬영

## 폴더 구조

```
camera_calibration/
├── README.md                      # 이 파일
├── requirements.txt               # 필요한 패키지 목록
├── config_example.py              # 설정 예시 파일 (RTSP URL 등)
│
├── generate_checkerboard.py       # 체커보드 패턴 생성
├── capture_images.py             # 이미지 캡처 (RTSP 지원)
├── check_images.py               # 🆕 이미지 품질 검사
├── calibrate.py                  # 캘리브레이션 수행
├── validate_calibration.py       # 🆕 결과 검증
│
├── checkerboard_pattern.png      # 생성된 체커보드 이미지
├── images/                       # 캡처된 이미지 저장 폴더
│   └── checkerboard_*.jpg
└── output/                       # 캘리브레이션 결과
    ├── calibration_data.pkl
    ├── calibration_results.txt
    ├── corners_*.jpg
    └── image_quality/            # 이미지 품질 검사 결과
        └── quality_report.txt
```

## 권장 워크플로우

1. **체커보드 생성**: `python generate_checkerboard.py`
2. **이미지 촬영**: `python capture_images.py` (20장 이상, 다양한 각도)
3. **품질 검사**: `python check_images.py` ⭐
4. **캘리브레이션**: `python calibrate.py`
5. **결과 검증**: `python validate_calibration.py` ⭐⭐⭐
6. **검증 실패 시**: 문제 원인 파악 후 재촬영

## 참고 자료

- [OpenCV Camera Calibration Tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [Zhang's Method](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf)
