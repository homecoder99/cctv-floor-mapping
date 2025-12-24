## 실시간 객체 검출 및 추적 시스템

CCTV 영상에서 객체를 검출하고 추적하여 도면 좌표로 변환하는 실시간 시스템입니다.

## 개요

이 시스템은 다음 기능을 제공합니다:

1. **왜곡 보정**: 카메라 캘리브레이션 데이터를 사용한 렌즈 왜곡 보정
2. **객체 검출**: YOLO 모델을 사용한 실시간 객체 검출
3. **객체 추적**: ByteTrack 알고리즘으로 프레임 간 객체 추적
4. **속도 추정**: 객체의 이동 속도 계산
5. **정적/동적 분리**: 움직이는 객체와 정지한 객체 구분
6. **좌표 변환**: 호모그래피를 사용한 영상→도면 좌표 변환
7. **실시간 시각화**: 검출/추적 결과 실시간 표시
8. **도면 위 시각화**: DXF 도면 위에 객체 위치 및 이동 경로 표시

### 지원하는 입력 소스

- **RTSP 스트림**: `rtsp://username:password@ip:port/stream`
- **비디오 파일**: `video.mp4`, `sample.avi`, `movie.mov` 등
- **웹캠**: `0`, `1` (장치 번호)

## 전체 파이프라인

```
RTSP 스트림 / 비디오 파일
    ↓
카메라 왜곡 보정 (cv2.undistort)
    ↓
YOLO 객체 검출
    ↓
ByteTrack 추적 (track_id 부여)
    ↓
속도 계산 (픽셀/프레임)
    ↓
정적/동적 분류
    ↓
호모그래피 변환 (영상→도면 좌표)
    ↓
결과 출력 및 시각화
    ├─ 영상 창: bbox + 궤적
    └─ 도면 창: DXF 위 객체 위치 및 이동 경로 (옵션)
```

## 설치

### 1. 필요한 패키지 설치

```bash
cd realtime_tracking
pip install -r requirements.txt
```

### 2. YOLO 모델 다운로드

처음 실행 시 자동으로 다운로드됩니다. 또는 수동으로:

```bash
# YOLOv8 Nano (빠름, 정확도 낮음)
# 자동 다운로드됨: yolov8n.pt

# 또는 더 정확한 모델 사용
# YOLOv8 Small
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt

# YOLOv8 Medium
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
```

## 사용 방법

### 1. 사전 준비

실시간 추적을 실행하기 전에 다음을 먼저 완료해야 합니다:

1. **카메라 캘리브레이션** (`camera_calibration/`)
   - 체커보드 패턴으로 캘리브레이션 수행
   - `calibration_data.pkl` 파일 생성

2. **호모그래피 계산** (`homography/`)
   - `point_collector.py`로 대응점 수집
   - 호모그래피 행렬 계산 및 저장
   - `.json` 파일 생성

### 2. 기본 실행

#### RTSP 스트림 사용

```bash
python tracker.py \
    --calibration ../camera_calibration/output/calibration_data.pkl \
    --homography ../homography/data/homography_20241224_103000.json \
    --input rtsp://username:password@192.168.1.100:554/stream1
```

#### 비디오 파일 사용

```bash
python tracker.py \
    --calibration ../camera_calibration/output/calibration_data.pkl \
    --homography ../homography/data/homography_20241224_103000.json \
    --input video.mp4
```

#### DXF 도면과 함께 사용

**자동 감지 (권장)**:

homography 파일과 같은 디렉토리에 `.dxf` 파일이 있으면 자동으로 감지하여 사용합니다:

```bash
# homography/data/ 디렉토리에 house.dxf가 있는 경우
python tracker.py \
    --calibration ../camera_calibration/output/calibration_data.pkl \
    --homography ../homography/data/homography_20241224_103000.json \
    --input video.mp4 \
    --save output/tracked.mp4

# DXF 파일 자동 감지: ../homography/data/house.dxf
```

**명시적 지정**:

```bash
python tracker.py \
    --calibration ../camera_calibration/output/calibration_data.pkl \
    --homography ../homography/data/homography_20241224_103000.json \
    --input video.mp4 \
    --drawing ../data/house.dxf \
    --save output/tracked.mp4
```

이 명령은 두 개의 창을 표시하고, 두 개의 영상 파일을 저장합니다:
- **Realtime Tracker**: 영상에서 검출된 객체들 → `tracked.mp4`
- **Floor Plan**: DXF 도면 위에 객체들의 실제 위치와 이동 경로 → `tracked_floorplan.mp4`

#### 웹캠 사용

```bash
python tracker.py \
    --calibration ../camera_calibration/output/calibration_data.pkl \
    --homography ../homography/data/homography_20241224_103000.json \
    --input 0
```

### 3. 고급 옵션

```bash
# RTSP 스트림 + 고급 옵션
python tracker.py \
    --calibration ../camera_calibration/output/calibration_data.pkl \
    --homography ../homography/data/homography_20241224_103000.json \
    --input rtsp://username:password@192.168.1.100:554/stream1 \
    --model yolov8m.pt \
    --conf 0.5 \
    --classes 0 2 \
    --speed-threshold 2.0 \
    --save output_tracking.mp4 \
    --skip-frames 3

# 비디오 파일 + 결과 저장
python tracker.py \
    --calibration ../camera_calibration/output/calibration_data.pkl \
    --homography ../homography/data/homography_20241224_103000.json \
    --input input_video.mp4 \
    --model yolov8s.pt \
    --conf 0.4 \
    --classes 0 \
    --save output_tracked.mp4
```

### 4. 주요 파라미터

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `--calibration` | 카메라 캘리브레이션 파일 경로 | `../camera_calibration/output/calibration_data.pkl` |
| `--homography` | 호모그래피 데이터 파일 경로 | 필수 입력 |
| `--input` | 입력 소스 (RTSP URL, 파일 경로, 웹캠 번호) | 필수 입력 |
| `--model` | YOLO 모델 경로 | `yolov8n.pt` |
| `--conf` | 검출 신뢰도 임계값 | `0.3` |
| `--classes` | 검출할 클래스 ID 리스트 | 전체 (None) |
| `--speed-threshold` | 동적 판단 속도 임계값 (pixels/frame) | `1.0` |
| `--no-display` | 화면 표시 안 함 | False |
| `--save` | 출력 영상 저장 경로 | None |
| `--skip-frames` | 프레임 스킵 수 (RTSP용, 파일은 자동 0) | `5` |
| `--zones` | 구역 정의 JSON 파일 경로 (선택) | None |
| `--drawing` | DXF 도면 파일 경로 (선택) | None |
| `--drawing-size` | 도면 렌더링 크기 (width height) | `1500 1500` |

### 5. YOLO 클래스 ID

일반적인 COCO 데이터셋 클래스:

- `0`: person (사람)
- `1`: bicycle
- `2`: car
- `3`: motorcycle
- `5`: bus
- `7`: truck
- ...

**예시**: 사람과 차량만 검출

```bash
--classes 0 2 3 5 7
```

## 작동 원리

### 1. 왜곡 보정

카메라 내부 파라미터와 왜곡 계수를 사용해 렌즈 왜곡을 보정합니다:

```python
undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
```

### 2. 객체 검출 및 추적

YOLO + ByteTrack을 통합하여 한 번에 검출과 추적을 수행합니다:

```python
results = model.track(
    undistorted,
    conf=0.3,
    persist=True,
    tracker="bytetrack.yaml"
)
```

각 검출된 객체에 고유한 `track_id`가 부여됩니다.

### 3. 속도 계산

최근 5프레임의 이동 거리를 평균하여 속도를 계산합니다:

```python
speed = average_distance_per_frame  # pixels/frame
```

### 4. 정적/동적 분류

- **동적 객체**: `speed >= speed_threshold`
- **정적 객체**: 30프레임 이상 `speed < speed_threshold`

정적 객체는 빨간색, 동적 객체는 초록색으로 표시됩니다.

### 5. 도면 좌표 변환 (바닥 접점 기반)

**중요**: 호모그래피는 **바닥 평면의 점**에만 의미가 있습니다.

따라서 bbox 중심이 아닌 **바닥 접점**을 사용합니다:

- **사람 (class 0)**: bbox 하단 중앙 (발 위치)
- **차량 (car, bus, truck 등)**: bbox 하단 중앙 (바퀴/접지점)
- **기타**: bbox 하단 중앙

```python
# 바닥 접점 계산
foot_pos = get_foot_position(bbox, class_id)  # (u, v)_foot

# 호모그래피 변환
drawing_pos = homography.transform_point(foot_pos)  # (X, Y)
```

**시각화**:
- 작은 원: bbox 중심점 (참고용)
- 큰 노란색 원: 바닥 접점 (호모그래피 변환 기준점)

### 6. DXF 도면 위 시각화 (자동 활성화)

**자동 감지**: homography 파일과 같은 디렉토리에 `.dxf` 파일이 있으면 자동으로 로드하여 시각화합니다.

**수동 지정**: `--drawing` 파라미터로 명시적으로 DXF 파일 경로를 지정할 수 있습니다.

**처리 과정**:

1. **DXF 로딩**: ezdxf로 도면 파일 읽기
2. **좌표 범위 계산**: 도면의 실제 좌표 범위 (min_x, min_y, max_x, max_y) 계산
3. **도면 렌더링**: DXF를 이미지로 렌더링 (기본 1500x1500px)
4. **좌표 변환**: 도면 좌표 → 도면 이미지 픽셀 좌표
5. **객체 시각화**:
   - 이동 궤적: 선으로 표시 (초록=동적, 회색=정적)
   - 현재 위치: 원으로 표시
   - 라벨: track_id + 클래스명

```python
# 도면 좌표를 이미지 픽셀로 변환
pixel_pos = drawing_coord_to_pixel(drawing_point)

# 도면 이미지에 객체 그리기
cv2.circle(floor_plan, pixel_pos, 8, color, -1)
```

**도면 창 (Floor Plan)**:
- 배경: DXF 도면 (흰색 배경)
- 초록색 선/원: 움직이는 객체
- 빨간색 원: 정지한 객체
- 회색 선: 정지한 객체의 과거 궤적

## 출력 정보

### 화면 표시

#### Realtime Tracker 창 (영상)

- **bbox**: 객체 경계 상자 (초록=동적, 빨강=정적)
- **라벨**: `ID:123 person 0.85 MOVING 3.2px/f`
  - track_id
  - 클래스명
  - 신뢰도
  - 상태 (MOVING/STATIC)
  - 속도
- **중심점**: 작은 원으로 표시
- **바닥 접점**: 큰 노란색 원 (호모그래피 변환 기준점)
- **궤적**: 최근 10프레임 경로

#### Floor Plan 창 (도면) - 옵션

`--drawing` 파라미터 사용 시 표시됩니다:

- **DXF 도면**: 배경으로 렌더링된 CAD 도면
- **객체 위치**: 호모그래피로 변환된 실제 위치
  - 초록색 원: 움직이는 객체
  - 빨간색 원: 정지한 객체
- **이동 궤적**: 객체의 과거 이동 경로 (최대 30프레임)
  - 초록색 선: 동적 객체의 궤적
  - 회색 선: 정적 객체의 궤적
- **라벨**: `ID:123 person`

### 콘솔 출력

```
[Frame 150] Moving objects: 3
  - ID:1 person Foot:(640,1020) Drawing:(1245.3,2341.7) Speed:4.52px/f
  - ID:3 car Foot:(850,980) Drawing:(3421.1,1523.9) Speed:12.34px/f
  - ID:5 person Foot:(420,1050) Drawing:(987.4,2109.2) Speed:2.18px/f
```

**출력 설명**:
- `Foot:(x,y)`: 영상에서의 바닥 접점 픽셀 좌표
- `Drawing:(X,Y)`: 도면 좌표 (호모그래피 변환 결과)
- `Speed`: 바닥 접점 기준 이동 속도 (pixels/frame)

### 통계 (화면 좌측 상단)

- FPS: 처리 속도
- Frame: 총 프레임 수
- Detections: 현재 프레임 검출 수
- Active Tracks: 활성 추적 수
- Moving: 동적 객체 수
- Stationary: 정적 객체 수

## 키보드 단축키

- **`q`**: 프로그램 종료
- **`s`**: 현재 프레임 스크린샷 저장
  - 영상: `screenshot_YYYYMMDD_HHMMSS.jpg`
  - 도면 (--drawing 사용 시): `screenshot_floorplan_YYYYMMDD_HHMMSS.jpg`

## 성능 최적화

### 1. 모델 선택

| 모델 | 속도 | 정확도 | 권장 용도 |
|------|------|--------|-----------|
| `yolov8n.pt` | 매우 빠름 | 낮음 | 실시간 (>20 FPS) |
| `yolov8s.pt` | 빠름 | 중간 | 실시간 (10-20 FPS) |
| `yolov8m.pt` | 중간 | 높음 | 정확도 우선 (5-10 FPS) |

### 2. 프레임 스킵

RTSP 지연을 줄이기 위해 프레임 스킵 조정:

```bash
--skip-frames 3  # 지연 감소
```

### 3. 신뢰도 임계값

잘못된 검출 줄이기:

```bash
--conf 0.5  # 높은 신뢰도만 검출
```

### 4. 클래스 필터링

필요한 클래스만 검출:

```bash
--classes 0  # 사람만 검출
```

## 영상 저장

`--save` 파라미터를 사용하면 처리된 영상을 저장할 수 있습니다.

### 저장되는 파일

1. **영상 파일** (항상 저장)
   - 파일명: 지정한 경로 (예: `output.mp4`)
   - 내용: 검출/추적 결과가 표시된 영상

2. **도면 영상 파일** (DXF 파일이 있는 경우 자동 저장)
   - 파일명: 원본 파일명 + `_floorplan` (예: `output_floorplan.mp4`)
   - 내용: DXF 도면 위에 객체 위치와 이동 경로가 표시된 영상

### 예시

**파일명 지정:**

```bash
python tracker.py \
    --calibration ../camera_calibration/output/calibration_data.pkl \
    --homography ../homography/data/homography.json \
    --input video.mp4 \
    --save output/result.mp4

# 저장되는 파일:
# - output/result.mp4 (영상)
# - output/result_floorplan.mp4 (도면, DXF 파일이 있는 경우)
```

**디렉토리만 지정:**

```bash
python tracker.py \
    --calibration ../camera_calibration/output/calibration_data.pkl \
    --homography ../homography/data/homography.json \
    --input video.mp4 \
    --save output/

# 자동으로 타임스탬프가 포함된 파일명 생성:
# - output/tracked_20241224_153045.mp4
# - output/tracked_20241224_153045_floorplan.mp4
```

### 완료 메시지

저장이 완료되면 다음과 같은 메시지가 표시됩니다:

```
영상 저장 완료: output/result.mp4
  파일 크기: 125.34 MB

도면 영상 저장 완료: output/result_floorplan.mp4
  파일 크기: 45.67 MB
```

## 문제 해결

### 1. RTSP 연결 실패

**증상**: "RTSP 연결 실패"

**해결**:
- RTSP URL 확인
- 네트워크 연결 확인
- 카메라 설정에서 RTSP 활성화 확인
- 방화벽 설정 확인

### 2. 비디오 파일 열기 실패

**증상**: "비디오 소스 열기 실패"

**해결**:
- 파일 경로 확인 (절대 경로 또는 상대 경로)
- 파일 존재 여부 확인
- 지원하는 포맷 확인 (MP4, AVI, MOV 등)
- OpenCV가 해당 코덱을 지원하는지 확인

### 3. 프레임 지연 (RTSP)

**증상**: 영상이 10초 이상 지연됨

**해결**:
```bash
--skip-frames 10  # 더 많은 프레임 스킵 (RTSP만 해당)
```

**참고**: 비디오 파일의 경우 프레임 스킵이 자동으로 0으로 설정됩니다.

### 4. FPS가 너무 낮음

**증상**: FPS < 5

**해결**:
- 더 작은 모델 사용 (`yolov8n.pt`)
- 클래스 필터링으로 검출 범위 축소
- GPU 사용 확인

### 5. 너무 많은 정적 객체

**증상**: 모든 객체가 STATIC으로 분류됨

**해결**:
```bash
--speed-threshold 0.5  # 임계값 낮춤
```

### 6. 추적 ID가 계속 바뀜

**증상**: track_id가 프레임마다 변경됨

**해결**:
- 신뢰도 임계값 높이기 (`--conf 0.5`)
- 더 정확한 모델 사용 (`yolov8m.pt`)

## 프로젝트 구조

```
realtime_tracking/
├── README.md                 # 이 파일
├── requirements.txt          # 필요한 패키지
├── tracker.py                # 메인 추적 스크립트
└── output/                   # 출력 영상/스크린샷 (자동 생성)
```

## 전체 워크플로우

```
1. camera_calibration/
   ├── capture_images.py      → RTSP에서 체커보드 이미지 캡처
   ├── calibrate.py           → calibration_data.pkl 생성
   └── validate_calibration.py → 캘리브레이션 검증

2. homography/
   ├── point_collector.py     → 대응점 수집 GUI
   └── data/*.json            → 호모그래피 행렬 저장

3. realtime_tracking/
   └── tracker.py             → 실시간 검출/추적 실행
```

## 고급 사용법

### Python API로 직접 사용

```python
from tracker import RealtimeTracker

# 트래커 초기화 (DXF 도면 포함)
tracker = RealtimeTracker(
    calibration_path='../camera_calibration/output/calibration_data.pkl',
    homography_path='../homography/data/homography.json',
    model_path='yolov8n.pt',
    conf_threshold=0.3,
    target_classes=[0, 2],  # 사람, 자동차
    speed_threshold=1.5,
    stationary_frames=30,
    drawing_path='../data/house.dxf',  # DXF 도면 (선택)
    drawing_size=(1500, 1500)  # 도면 렌더링 크기
)

# RTSP 스트림 실행
tracker.run_video(
    video_source='rtsp://username:password@192.168.1.100:554/stream1',
    display=True,
    save_output=True,
    output_path='tracking_result.mp4',
    skip_frames=5
)

# 비디오 파일 실행
tracker.run_video(
    video_source='input_video.mp4',
    display=True,
    save_output=True,
    output_path='tracking_result.mp4',
    skip_frames=0  # 파일은 프레임 스킵 0
)

# 웹캠 실행
tracker.run_video(
    video_source=0,  # 또는 '0'
    display=True,
    save_output=False,
    skip_frames=0
)
```

### 동적 객체만 필터링

```python
# 프레임 처리
undistorted, annotated, detections = tracker.process_frame(frame)

# 동적 객체만 가져오기
moving_objects = tracker.get_moving_objects(detections)

for obj in moving_objects:
    track_id = obj['track_id']
    class_name = obj['class_name']
    speed = obj['speed']
    drawing_pos = obj['drawing_position']

    print(f"ID:{track_id} {class_name} at ({drawing_pos[0]:.1f}, {drawing_pos[1]:.1f})")
```

## 고급 기능 (선택)

### 구역(Zone) 매칭

도면 좌표를 기반으로 객체가 어느 구역에 있는지 판단할 수 있습니다.

**구현됨**: `--zones` 파라미터로 JSON 파일 지정 가능

**구역 정의 JSON 예시**:
```json
{
  "zones": [
    {
      "id": "zone_A",
      "name": "작업구역 A",
      "polygon": [
        [100, 100],
        [500, 100],
        [500, 400],
        [100, 400]
      ]
    },
    {
      "id": "danger_zone",
      "name": "위험구역",
      "polygon": [
        [1100, 100],
        [1500, 100],
        [1500, 500],
        [1100, 500]
      ]
    }
  ]
}
```

**사용법**:
```bash
python tracker.py \
    --calibration ../camera_calibration/output/calibration_data.pkl \
    --homography ../homography/data/homography.json \
    --input video.mp4 \
    --zones zones.json
```

**출력**:
```
[Frame 150] Moving objects: 2
  - ID:1 person Foot:(640,1020) Drawing:(1245.3,2341.7) Zone:작업구역 A Speed:4.52px/f
  - ID:3 car Foot:(850,980) Drawing:(3421.1,1523.9) Zone:위험구역 Speed:12.34px/f
```

## 다음 단계

1. **구역 기반 알림**: 특정 구역 진입 시 알림
2. **데이터 저장**: 추적 결과를 데이터베이스에 저장
3. **통계 분석**: 구역별 이동 패턴 분석
4. **다중 카메라**: 여러 카메라 통합 추적
5. ~~**디지털 트윈 연동**: 실시간 도면 위 시각화~~ ✅ **구현 완료**
6. **포즈 추정**: 사람의 정확한 발 위치 검출 (Keypoint 기반)
7. **3D 시각화**: 높이 정보를 포함한 3D 도면 위 객체 표시

## 참고 자료

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [ByteTrack Paper](https://arxiv.org/abs/2110.06864)
- [OpenCV Tracking](https://docs.opencv.org/4.x/d9/df8/group__tracking.html)

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.
