## 호모그래피 대응점 수집 도구

CCTV 영상 프레임과 CAD 도면 사이의 호모그래피 변환을 위한 대응점 수집 및 계산 도구입니다.

## 개요

이 도구는 다음 기능을 제공합니다:
- 영상 프레임과 CAD 도면을 나란히 표시
- 마우스 클릭으로 대응점 수집
- 호모그래피 행렬 자동 계산
- 재투영 오차 계산 및 검증
- 대응점 및 호모그래피 행렬 저장/로드
- 버전 관리

## 설치

필요한 패키지를 설치합니다:

```bash
cd homography
pip install -r requirements.txt
```

## 사용 방법

### 1. 프로그램 실행

```bash
python point_collector.py
```

### 2. 카메라 캘리브레이션 로드 (중요!)

**호모그래피를 계산하기 전에 반드시 카메라 캘리브레이션을 먼저 로드해야 합니다.**

1. **캘리브레이션 로드** 버튼 클릭
2. `../camera_calibration/output/calibration_data.pkl` 파일 선택
3. 캘리브레이션 로드 완료 확인
   - 상태 표시: "캘리브레이션: 로드됨 ✓" (녹색)
   - 카메라 내부 파라미터 및 왜곡 계수 정보 표시

**왜 필요한가요?**
- CCTV 카메라의 렌즈 왜곡을 보정하기 위함
- 왜곡이 보정되지 않은 영상으로 호모그래피를 계산하면 부정확한 결과가 나옴
- 캘리브레이션 로드 후 영상 프레임을 로드하면 자동으로 왜곡 보정이 적용됨

### 3. 이미지 로드

1. **영상 프레임 로드**: 상단의 "영상 프레임 로드" 버튼 클릭
   - CCTV에서 캡처한 이미지 선택
   - 권장: `../camera_calibration/images/` 폴더의 이미지 사용
   - **자동 왜곡 보정**: 캘리브레이션이 로드되어 있으면 자동으로 왜곡 보정이 적용됩니다

2. **도면 로드**: 상단의 "도면 로드" 버튼 클릭
   - CAD 도면 이미지 (PNG, JPG 등) 선택
   - **DXF 파일 직접 지원**: DXF 파일을 선택하면 자동으로 이미지로 렌더링됩니다
   - DXF 렌더링 시간: 파일 크기에 따라 수 초~수십 초 소요

### 4. 대응점 수집

**중요**: 영상과 도면에서 **동일한 위치**를 찾아 클릭해야 합니다.

#### 방법 A: 영상 → 도면 순서
1. 영상 프레임에서 특정 지점 클릭 (예: 모서리, 교차점)
2. 도면에서 같은 위치 클릭
3. 대응점 쌍이 자동으로 추가됩니다

#### 방법 B: 도면 → 영상 순서
1. 도면에서 특정 지점 클릭
2. 영상 프레임에서 같은 위치 클릭
3. 대응점 쌍이 자동으로 추가됩니다

**팁**:
- 최소 4개의 대응점이 필요합니다
- 권장: 6~10개의 대응점 (정확도 향상)
- 좋은 대응점 위치:
  - 명확한 모서리
  - 선의 교차점
  - 특징적인 물체
  - 화면 전체에 고르게 분포

### 5. 호모그래피 계산

1. 대응점을 4개 이상 수집한 후
2. "호모그래피 계산" 버튼 클릭
3. 결과 확인:
   - 재투영 오차 (낮을수록 좋음)
   - 호모그래피 행렬
   - Inlier/Outlier 정보

**재투영 오차 기준**:
- 평균 < 2.0 pixels: 우수
- 평균 2.0~5.0 pixels: 양호
- 평균 > 5.0 pixels: 재수집 권장

### 6. 데이터 저장

1. "저장" 버튼 클릭
2. 파일명 입력 (기본: `homography_YYYYMMDD_HHMMSS.json`)
3. `data/` 폴더에 저장됩니다

**저장 내용**:
- 모든 대응점 좌표 (왜곡 보정된 좌표)
- 호모그래피 행렬
- 재투영 오차 통계
- 메타데이터 (생성일시, 이미지 경로 등)

### 7. 데이터 불러오기

1. "불러오기" 버튼 클릭
2. 이전에 저장한 JSON 파일 선택
3. 대응점과 호모그래피 행렬이 복원됩니다

## 주요 기능

### 대응점 관리

- **삭제**: 테이블에서 행 선택 후 "선택한 점 삭제"
- **전체 삭제**: "모든 점 삭제" (확인 필요)
- **자동 번호 매김**: 각 점에 번호가 자동으로 표시됩니다

### 시각화

- 각 대응점은 **녹색 원**으로 표시
- **흰색 테두리**로 강조
- **노란색 숫자**로 인덱스 표시

### 호모그래피 알고리즘

- 기본: **RANSAC** (이상치 제거)
- RANSAC 임계값: 5.0 pixels
- 최소 4개 점 필요
- 더 많은 점 = 더 정확한 결과

## 폴더 구조

```
homography/
├── README.md                      # 이 파일
├── requirements.txt               # 필요한 패키지
├── point_collector.py             # 메인 GUI 프로그램
├── homography_utils.py            # 호모그래피 유틸리티
├── dxf_renderer.py                # DXF → 이미지 렌더러
│
├── data/                          # 저장된 데이터
│   └── homography_*.json
├── frames/                        # 영상 프레임 (선택사항)
└── drawings/                      # 도면 이미지/DXF (선택사항)
```

## 데이터 형식 (JSON)

```json
{
  "metadata": {
    "created_at": "2024-12-24T10:30:00",
    "updated_at": "2024-12-24T10:35:00",
    "version": "1.0",
    "image_source": "/path/to/frame.jpg",
    "drawing_source": "/path/to/drawing.png",
    "num_points": 8
  },
  "image_points": [
    [100.5, 200.3],
    [350.2, 220.1],
    ...
  ],
  "drawing_points": [
    [1500.0, 2000.0],
    [3500.0, 2200.0],
    ...
  ],
  "homography_matrix": [
    [1.234, 0.567, 890.1],
    [0.234, 1.456, 678.9],
    [0.001, 0.002, 1.000]
  ],
  "reprojection_error": {
    "mean": 1.234,
    "max": 3.456,
    "median": 1.123,
    "std": 0.567,
    "errors": [1.2, 0.9, 1.5, ...]
  }
}
```

## Python API 사용 예제

### 왜곡 보정 후 호모그래피 변환

```python
import cv2
import pickle
import numpy as np
from homography_utils import HomographyData

# 1. 카메라 캘리브레이션 로드
with open('../camera_calibration/output/calibration_data.pkl', 'rb') as f:
    calib_data = pickle.load(f)

camera_matrix = calib_data['camera_matrix']
dist_coeffs = calib_data['dist_coeffs']

# 2. 호모그래피 데이터 로드
homography = HomographyData()
homography.load_from_file('data/homography_20241224_103000.json')

# 3. 영상 프레임 로드 및 왜곡 보정
frame = cv2.imread('frames/cctv_frame.jpg')
h, w = frame.shape[:2]

# 왜곡 보정
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix, dist_coeffs, (w, h), 1, (w, h)
)
undistorted_frame = cv2.undistort(
    frame, camera_matrix, dist_coeffs, None, new_camera_matrix
)

# 4. 왜곡 보정된 영상 좌표를 도면 좌표로 변환
image_point = (640, 480)  # 왜곡 보정된 영상에서의 점
drawing_point = homography.transform_point(image_point)
print(f"도면 좌표: {drawing_point}")

# 5. 역변환 (도면 → 영상)
inverse_point = homography.transform_point(drawing_point, inverse=True)
print(f"역변환 결과: {inverse_point}")
```

### 이미지 워핑

```python
import cv2
import pickle
from homography_utils import HomographyData, warp_image

# 1. 캘리브레이션 로드
with open('../camera_calibration/output/calibration_data.pkl', 'rb') as f:
    calib_data = pickle.load(f)

camera_matrix = calib_data['camera_matrix']
dist_coeffs = calib_data['dist_coeffs']

# 2. 호모그래피 로드
homography = HomographyData()
homography.load_from_file('data/homography_20241224_103000.json')

# 3. 영상 프레임 로드 및 왜곡 보정
frame = cv2.imread('frames/cctv_frame.jpg')
h, w = frame.shape[:2]

new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix, dist_coeffs, (w, h), 1, (w, h)
)
undistorted_frame = cv2.undistort(
    frame, camera_matrix, dist_coeffs, None, new_camera_matrix
)

# 4. 도면 크기에 맞춰 워핑
output_size = (4000, 3000)  # 도면 크기
warped = warp_image(undistorted_frame, homography.homography_matrix, output_size)

# 5. 결과 저장
cv2.imwrite('warped_frame.jpg', warped)
```

### OpenCV 형식으로 내보내기

```python
from homography_utils import HomographyData

homography = HomographyData()
homography.load_from_file('data/homography_20241224_103000.json')

# OpenCV 형식으로 변환
cv_data = homography.export_for_opencv()

H = cv_data['H']              # 3x3 행렬
src_pts = cv_data['src_pts']  # Nx2 numpy array
dst_pts = cv_data['dst_pts']  # Nx2 numpy array
```

### DXF 파일을 이미지로 렌더링

```python
from dxf_renderer import render_dxf_to_image, render_dxf_to_file

# 방법 1: numpy 배열로 반환
image = render_dxf_to_image(
    'drawing.dxf',
    output_size=(3000, 3000),
    background_color=(255, 255, 255)  # 흰색 배경
)

# OpenCV로 저장
import cv2
cv2.imwrite('drawing.png', image)

# 방법 2: 파일로 직접 저장
render_dxf_to_file(
    'drawing.dxf',
    'drawing.png',
    output_size=(4000, 4000),
    background_color=(240, 240, 240)  # 연한 회색 배경
)
```

**DXF 렌더링 옵션**:
- `output_size`: 출력 이미지 크기 (width, height)
- `background_color`: (R, G, B) 배경색
  - 흰색: `(255, 255, 255)`
  - 검정색: `(0, 0, 0)`
  - 회색: `(200, 200, 200)`

## 문제 해결

### 1. "캘리브레이션: 미로드" 상태에서 호모그래피 계산

**문제**: 왜곡이 보정되지 않은 영상으로 호모그래피를 계산하면 부정확한 결과가 나옴

**해결**:
1. 먼저 "캘리브레이션 로드" 버튼 클릭
2. `../camera_calibration/output/calibration_data.pkl` 파일 선택
3. "캘리브레이션: 로드됨 ✓" 상태 확인
4. 그 후 영상 프레임 로드 (자동 왜곡 보정 적용됨)

### 2. "최소 4개의 대응점이 필요합니다"

- 대응점을 4개 이상 수집하세요
- 각 점은 영상과 도면 양쪽에 찍혀야 합니다

### 3. "호모그래피 행렬 계산에 실패했습니다"

**원인**:
- 대응점이 일직선상에 있음
- 대응점이 너무 가까움
- 잘못된 대응점 포함

**해결**:
- 화면 전체에 고르게 대응점 배치
- 명확한 특징점 선택
- 잘못 찍은 점 삭제 후 재시도

### 4. 재투영 오차가 큼 (> 5.0 pixels)

**원인**:
- 부정확한 대응점
- 캘리브레이션 미로드 (왜곡이 보정되지 않음)
- 잘못된 캘리브레이션 파일 사용

**해결**:
1. **먼저 캘리브레이션 로드 확인**
   - "캘리브레이션: 로드됨 ✓" 상태 확인
   - 올바른 calibration_data.pkl 파일 사용
2. **대응점 재확인**
   - 모든 대응점 재확인
   - 불량 대응점 삭제 후 재수집
3. **캘리브레이션 재수행**
   - fx/fy 비율이 0.95~1.05 범위인지 확인
   - 필요시 camera_calibration부터 다시 수행

### 5. 이미지가 표시되지 않음

- 지원 형식 확인: JPG, PNG, BMP, **DXF**
- 파일 경로에 한글이 있으면 문제 발생 가능
- 이미지 파일이 손상되었는지 확인

### 6. DXF 렌더링이 너무 느림

**원인**:
- DXF 파일이 매우 복잡함 (엔티티가 많음)
- 큰 출력 해상도 (기본 3000x3000)

**해결**:
- `dxf_renderer.py`의 `output_size` 조정
- 불필요한 레이어 제거
- 간단한 도면 사용

## 워크플로우 권장사항

### 전체 프로세스

1. **카메라 캘리브레이션** (`camera_calibration/`)
   - 체커보드 패턴으로 여러 각도에서 20장 이상 촬영
   - 내부 파라미터 및 왜곡 계수 계산
   - `calibration_data.pkl` 파일 생성

2. **영상 캡처**
   - RTSP 스트림에서 프레임 저장
   - 또는 기존 이미지 사용
   - 체커보드가 아닌 실제 작업 환경 촬영

3. **호모그래피 수집** (`homography/`)
   - **중요**: 먼저 캘리브레이션 데이터 로드
   - 영상 프레임 로드 (자동 왜곡 보정 적용됨)
   - 도면 로드 (DXF 또는 이미지)
   - 대응점 6~10개 수집
   - 호모그래피 행렬 계산 및 저장

4. **실시간 적용**
   - 저장된 호모그래피 행렬 로드
   - 실시간 CCTV 영상에 왜곡 보정 적용
   - 왜곡 보정된 영상을 도면 좌표로 변환

**중요**: 호모그래피를 계산할 때 사용한 영상과 실시간 영상 모두 **동일한 캘리브레이션 파라미터**로 왜곡 보정되어야 합니다.

## 고급 사용법

### 여러 개의 호모그래피

다른 카메라 각도나 위치마다 별도의 호모그래피를 저장:

```
data/
├── homography_camera1_angle1.json
├── homography_camera1_angle2.json
├── homography_camera2.json
└── ...
```

### 버전 관리

메타데이터의 `version` 필드로 호모그래피 버전 관리:
- 카메라 위치 변경 시 새 버전 생성
- 날짜/시간 기반 파일명 사용

## 참고 자료

- [OpenCV Perspective Transformation](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html)
- [호모그래피 개념](https://darkpgmr.tistory.com/79)
- [RANSAC 알고리즘](https://en.wikipedia.org/wiki/Random_sample_consensus)

## 다음 단계

호모그래피 행렬을 얻은 후:
1. 실시간 CCTV 영상에 적용
2. 객체 검출 결과를 도면 좌표로 변환
3. 디지털 트윈 시스템에 통합
