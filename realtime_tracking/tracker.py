"""
실시간 객체 검출 및 추적 시스템

CCTV 영상에서 객체를 검출하고 추적하여 도면 좌표로 변환합니다.

기능:
1. 왜곡 보정된 영상에서 YOLO 검출
2. ByteTrack으로 객체 추적
3. 속도 추정 및 정적/동적 분리
4. 호모그래피로 도면 좌표 변환
5. 실시간 시각화

필요한 패키지: ultralytics, opencv-python, numpy
"""

import cv2
import numpy as np
import pickle
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# homography_utils import를 위한 경로 추가
sys.path.append(str(Path(__file__).parent.parent / 'homography'))

from ultralytics import YOLO
from homography_utils import HomographyData
from dxf_renderer import render_dxf_to_image, DXFRenderer
import ezdxf
from PyQt6.QtWidgets import QApplication


class RealtimeTracker:
    """
    실시간 객체 검출 및 추적 클래스
    """

    def __init__(
        self,
        calibration_path,
        homography_path,
        model_path="yolov8n.pt",
        conf_threshold=0.3,
        iou_threshold=0.5,
        target_classes=None,
        speed_threshold=1.0,  # pixels/frame
        stationary_frames=30,  # 정지 판단 프레임 수
        zones_path=None,  # 구역 정의 파일 경로 (선택)
        drawing_path=None,  # DXF 도면 파일 경로 (선택)
        drawing_size=(1500, 1500),  # 도면 렌더링 크기
    ):
        """
        Args:
            calibration_path: 카메라 캘리브레이션 파일 경로 (.pkl)
            homography_path: 호모그래피 데이터 파일 경로 (.json)
            model_path: YOLO 모델 경로
            conf_threshold: 검출 신뢰도 임계값
            iou_threshold: NMS IoU 임계값
            target_classes: 검출할 클래스 리스트 (None이면 전체)
            speed_threshold: 동적 판단 속도 임계값 (pixels/frame)
            stationary_frames: 정지 판단 프레임 수
            zones_path: 구역 정의 JSON 파일 경로 (선택)
            drawing_path: DXF 도면 파일 경로 (선택)
            drawing_size: 도면 렌더링 크기 (width, height)
        """
        # 카메라 캘리브레이션 로드
        print(f"카메라 캘리브레이션 로드: {calibration_path}")
        with open(calibration_path, 'rb') as f:
            calib_data = pickle.load(f)

        self.camera_matrix = calib_data['camera_matrix']
        self.dist_coeffs = calib_data['dist_coeffs']

        # 호모그래피 로드
        print(f"호모그래피 데이터 로드: {homography_path}")
        self.homography_data = HomographyData()
        self.homography_data.load_from_file(homography_path)

        # DXF 파일 자동 감지 (drawing_path가 None인 경우)
        if drawing_path is None:
            # homography 파일과 같은 디렉토리에서 .dxf 파일 찾기
            homography_dir = os.path.dirname(homography_path)
            if homography_dir:
                dxf_files = [f for f in os.listdir(homography_dir) if f.lower().endswith('.dxf')]
                if dxf_files:
                    drawing_path = os.path.join(homography_dir, dxf_files[0])
                    print(f"DXF 파일 자동 감지: {drawing_path}")

        # YOLO 모델 로드
        print(f"YOLO 모델 로드: {model_path}")
        self.model = YOLO(model_path)

        # 파라미터 설정
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes
        self.speed_threshold = speed_threshold
        self.stationary_frames = stationary_frames

        # 추적 데이터
        self.track_history = defaultdict(lambda: {
            'positions': [],
            'timestamps': [],
            'speeds': [],
            'stationary_count': 0,
            'is_moving': True,
            'drawing_positions': []
        })

        # 통계
        self.stats = {
            'total_detections': 0,
            'active_tracks': 0,
            'moving_objects': 0,
            'stationary_objects': 0,
            'frame_count': 0,
            'fps': 0.0
        }

        # 구역 정의 로드 (선택)
        self.zones = []
        if zones_path and os.path.exists(zones_path):
            print(f"구역 정의 로드: {zones_path}")
            with open(zones_path, 'r', encoding='utf-8') as f:
                zones_data = json.load(f)
                self.zones = zones_data.get('zones', [])
            print(f"  - {len(self.zones)}개 구역 로드됨")
        else:
            if zones_path:
                print(f"경고: 구역 파일을 찾을 수 없음: {zones_path}")

        # DXF 도면 렌더링 (선택)
        self.drawing_image = None
        self.drawing_bounds = None  # (min_x, min_y, max_x, max_y)
        self.drawing_size = drawing_size

        if drawing_path and os.path.exists(drawing_path):
            print(f"DXF 도면 렌더링 중: {drawing_path}")
            self._load_drawing(drawing_path)
        else:
            if drawing_path:
                print(f"경고: 도면 파일을 찾을 수 없음: {drawing_path}")

        print("초기화 완료")

    def undistort_frame(self, frame):
        """
        프레임 왜곡 보정
        """
        h, w = frame.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )
        undistorted = cv2.undistort(
            frame, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix
        )
        return undistorted

    def get_bbox_center(self, bbox):
        """
        bbox의 중심점 계산

        Args:
            bbox: (x1, y1, x2, y2)

        Returns:
            (cx, cy)
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def get_foot_position(self, bbox, class_id):
        """
        bbox의 바닥 접점 계산 (호모그래피용)

        바닥 평면의 점만 호모그래피 변환에 의미가 있음

        Args:
            bbox: (x1, y1, x2, y2)
            class_id: YOLO 클래스 ID

        Returns:
            (foot_x, foot_y): 바닥 접점 좌표
        """
        x1, y1, x2, y2 = bbox

        # 클래스별 바닥 접점 정의
        # COCO 데이터셋 기준:
        # 0: person, 1: bicycle, 2: car, 3: motorcycle,
        # 5: bus, 7: truck, ...

        if class_id == 0:  # person
            # 사람: bbox 하단 중앙 (발 위치)
            foot_x = (x1 + x2) / 2
            foot_y = y2  # 하단

        elif class_id in [1, 2, 3, 5, 7]:  # 차량류 (bicycle, car, motorcycle, bus, truck)
            # 차량: bbox 하단 중앙 (바퀴/접지점)
            foot_x = (x1 + x2) / 2
            foot_y = y2  # 하단

        else:
            # 기타: bbox 하단 중앙
            foot_x = (x1 + x2) / 2
            foot_y = y2

        return (foot_x, foot_y)

    def calculate_speed(self, track_id, current_pos, current_time):
        """
        트랙의 속도 계산

        Args:
            track_id: 트랙 ID
            current_pos: 현재 위치 (x, y)
            current_time: 현재 시간

        Returns:
            speed: 속도 (pixels/frame)
        """
        history = self.track_history[track_id]

        # 위치 히스토리 업데이트
        history['positions'].append(current_pos)
        history['timestamps'].append(current_time)

        # 최근 10프레임만 유지
        if len(history['positions']) > 10:
            history['positions'] = history['positions'][-10:]
            history['timestamps'] = history['timestamps'][-10:]

        # 속도 계산 (최소 2개 프레임 필요)
        if len(history['positions']) < 2:
            return 0.0

        # 최근 5프레임 평균 속도
        recent_positions = history['positions'][-5:]
        if len(recent_positions) < 2:
            return 0.0

        distances = []
        for i in range(1, len(recent_positions)):
            prev_pos = recent_positions[i - 1]
            curr_pos = recent_positions[i]
            dist = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
            distances.append(dist)

        speed = np.mean(distances) if distances else 0.0
        history['speeds'].append(speed)

        # 최근 속도만 유지
        if len(history['speeds']) > 10:
            history['speeds'] = history['speeds'][-10:]

        return speed

    def update_movement_status(self, track_id, speed):
        """
        트랙의 움직임 상태 업데이트

        Args:
            track_id: 트랙 ID
            speed: 현재 속도
        """
        history = self.track_history[track_id]

        if speed < self.speed_threshold:
            history['stationary_count'] += 1
        else:
            history['stationary_count'] = 0

        # 일정 프레임 이상 정지하면 정적 객체로 분류
        if history['stationary_count'] >= self.stationary_frames:
            history['is_moving'] = False
        else:
            history['is_moving'] = True

    def transform_to_drawing(self, image_point):
        """
        영상 좌표를 도면 좌표로 변환

        Args:
            image_point: (x, y) 영상 좌표

        Returns:
            (x, y) 도면 좌표
        """
        return self.homography_data.transform_point(image_point)

    def point_in_polygon(self, point, polygon):
        """
        점이 폴리곤 내부에 있는지 판단 (Ray-casting algorithm)

        Args:
            point: (x, y) 점 좌표
            polygon: [(x1, y1), (x2, y2), ...] 폴리곤 꼭짓점 리스트

        Returns:
            bool: 내부에 있으면 True
        """
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def get_zone(self, drawing_point):
        """
        도면 좌표가 어느 구역에 속하는지 판단

        Args:
            drawing_point: (x, y) 도면 좌표

        Returns:
            zone_info: {'id': str, 'name': str} 또는 None
        """
        if not self.zones:
            return None

        for zone in self.zones:
            polygon = zone.get('polygon', [])
            if self.point_in_polygon(drawing_point, polygon):
                return {
                    'id': zone.get('id', ''),
                    'name': zone.get('name', '')
                }

        return None  # 어떤 구역에도 속하지 않음

    def _load_drawing(self, dxf_path):
        """
        DXF 파일을 렌더링하고 좌표 변환 정보를 설정합니다.
        """
        try:
            # QApplication 확인/생성 (PyQt6 필요)
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)

            # DXF 파일 읽기
            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()

            # DXF 도면의 실제 좌표 범위 계산
            min_x, min_y = float('inf'), float('inf')
            max_x, max_y = float('-inf'), float('-inf')

            for entity in msp:
                if entity.dxftype() == 'LINE':
                    start = entity.dxf.start
                    end = entity.dxf.end
                    min_x = min(min_x, start.x, end.x)
                    min_y = min(min_y, start.y, end.y)
                    max_x = max(max_x, start.x, end.x)
                    max_y = max(max_y, start.y, end.y)
                elif entity.dxftype() in ['LWPOLYLINE', 'POLYLINE']:
                    try:
                        if entity.dxftype() == 'LWPOLYLINE':
                            points = list(entity.get_points('xy'))
                        else:
                            points = [(p.dxf.location.x, p.dxf.location.y) for p in entity.points()]
                        for x, y in points:
                            min_x = min(min_x, x)
                            min_y = min(min_y, y)
                            max_x = max(max_x, x)
                            max_y = max(max_y, y)
                    except:
                        pass
                elif entity.dxftype() == 'CIRCLE':
                    center = entity.dxf.center
                    radius = entity.dxf.radius
                    min_x = min(min_x, center.x - radius)
                    min_y = min(min_y, center.y - radius)
                    max_x = max(max_x, center.x + radius)
                    max_y = max(max_y, center.y + radius)

            # 유효한 범위가 있는지 확인
            if min_x == float('inf') or min_y == float('inf'):
                print("경고: DXF 파일에서 유효한 엔티티를 찾을 수 없습니다.")
                return

            # 여백 추가 (5%)
            width = max_x - min_x
            height = max_y - min_y
            margin = max(width, height) * 0.05
            min_x -= margin
            min_y -= margin
            max_x += margin
            max_y += margin

            self.drawing_bounds = (min_x, min_y, max_x, max_y)
            print(f"  - 도면 좌표 범위: X({min_x:.1f} ~ {max_x:.1f}), Y({min_y:.1f} ~ {max_y:.1f})")

            # DXF를 이미지로 렌더링
            self.drawing_image = render_dxf_to_image(
                dxf_path,
                output_size=self.drawing_size,
                background_color=(255, 255, 255)
            )

            if self.drawing_image is not None:
                print(f"  - 도면 렌더링 완료: {self.drawing_size[0]}x{self.drawing_size[1]}")
            else:
                print("  - 경고: 도면 렌더링 실패")

        except Exception as e:
            print(f"DXF 로드 오류: {e}")
            import traceback
            traceback.print_exc()

    def drawing_coord_to_pixel(self, drawing_point):
        """
        도면 좌표를 이미지 픽셀 좌표로 변환합니다.

        Args:
            drawing_point: (x, y) 도면 좌표

        Returns:
            (px, py): 픽셀 좌표, 실패 시 None
        """
        if self.drawing_bounds is None or self.drawing_image is None:
            return None

        min_x, min_y, max_x, max_y = self.drawing_bounds
        img_h, img_w = self.drawing_image.shape[:2]

        x, y = drawing_point

        # 도면 좌표 → 정규화 좌표 (0~1)
        norm_x = (x - min_x) / (max_x - min_x)
        norm_y = (y - min_y) / (max_y - min_y)

        # 정규화 좌표 → 픽셀 좌표
        # Y축 반전 (DXF는 Y축이 위로 증가, 이미지는 아래로 증가)
        px = int(norm_x * img_w)
        py = int((1 - norm_y) * img_h)

        # 범위 체크
        px = max(0, min(px, img_w - 1))
        py = max(0, min(py, img_h - 1))

        return (px, py)

    def draw_on_floorplan(self, detections):
        """
        도면 위에 객체들을 시각화합니다.

        Args:
            detections: 검출 결과 리스트

        Returns:
            numpy.ndarray: 시각화된 도면 이미지, 실패 시 None
        """
        if self.drawing_image is None:
            return None

        # 도면 이미지 복사
        display_image = self.drawing_image.copy()

        # 모든 활성 트랙의 궤적 그리기
        for track_id, history in self.track_history.items():
            drawing_positions = history.get('drawing_positions', [])
            if len(drawing_positions) < 2:
                continue

            # 궤적 색상 (트랙 ID 기반)
            is_moving = history.get('is_moving', True)
            if is_moving:
                color = (0, 200, 0)  # 초록 (동적)
            else:
                color = (200, 200, 200)  # 회색 (정적)

            # 궤적 그리기
            for i in range(1, len(drawing_positions)):
                pt1_drawing = drawing_positions[i - 1]
                pt2_drawing = drawing_positions[i]

                pt1_px = self.drawing_coord_to_pixel(pt1_drawing)
                pt2_px = self.drawing_coord_to_pixel(pt2_drawing)

                if pt1_px and pt2_px:
                    cv2.line(display_image, pt1_px, pt2_px, color, 2)

        # 현재 검출된 객체들 그리기
        for det in detections:
            drawing_pos = det['drawing_position']
            track_id = det['track_id']
            class_name = det['class_name']
            is_moving = det['is_moving']

            # 픽셀 좌표로 변환
            pixel_pos = self.drawing_coord_to_pixel(drawing_pos)
            if pixel_pos is None:
                continue

            # 색상
            if is_moving:
                color = (0, 255, 0)  # 초록 (동적)
            else:
                color = (0, 0, 255)  # 빨강 (정적)

            # 객체 위치 표시 (원)
            cv2.circle(display_image, pixel_pos, 8, color, -1)
            cv2.circle(display_image, pixel_pos, 10, (0, 0, 0), 2)

            # 라벨
            label = f"ID:{track_id} {class_name}"
            font_scale = 0.5
            thickness = 1

            # 텍스트 크기 계산
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            # 배경 사각형
            label_y = pixel_pos[1] - 15
            cv2.rectangle(
                display_image,
                (pixel_pos[0] - 2, label_y - text_h - 2),
                (pixel_pos[0] + text_w + 2, label_y + 2),
                (255, 255, 255),
                -1
            )

            # 텍스트
            cv2.putText(
                display_image, label,
                (pixel_pos[0], label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness
            )

        return display_image

    def process_frame(self, frame):
        """
        단일 프레임 처리

        Args:
            frame: 입력 프레임

        Returns:
            undistorted_frame: 왜곡 보정된 프레임
            annotated_frame: 검출/추적 결과가 그려진 프레임
            detections: 검출 결과 리스트
        """
        start_time = time.time()

        # 1. 왜곡 보정
        undistorted = self.undistort_frame(frame)

        # 2. YOLO 검출 + ByteTrack 추적
        results = self.model.track(
            undistorted,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.target_classes,
            persist=True,  # 추적 유지
            tracker="bytetrack.yaml",  # ByteTrack 사용
            verbose=False
        )

        # 3. 결과 처리
        detections = []
        annotated_frame = undistorted.copy()

        if results and len(results) > 0:
            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes

                for i in range(len(boxes)):
                    # bbox, confidence, class, track_id
                    bbox = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls = int(boxes.cls[i].cpu().numpy())

                    # track_id 가져오기
                    if boxes.id is not None:
                        track_id = int(boxes.id[i].cpu().numpy())
                    else:
                        continue  # 추적 실패 시 스킵

                    # bbox 중심점 (시각화 및 궤적용)
                    center = self.get_bbox_center(bbox)

                    # 바닥 접점 (호모그래피 변환용)
                    foot_pos = self.get_foot_position(bbox, cls)

                    # 속도 계산 (바닥 접점 기준)
                    speed = self.calculate_speed(track_id, foot_pos, time.time())

                    # 움직임 상태 업데이트
                    self.update_movement_status(track_id, speed)

                    # 도면 좌표 변환 (바닥 접점 사용)
                    drawing_pos = self.transform_to_drawing(foot_pos)
                    self.track_history[track_id]['drawing_positions'].append(drawing_pos)

                    # 최근 위치만 유지
                    if len(self.track_history[track_id]['drawing_positions']) > 30:
                        self.track_history[track_id]['drawing_positions'] = \
                            self.track_history[track_id]['drawing_positions'][-30:]

                    # 구역 판단
                    zone_info = self.get_zone(drawing_pos)

                    # 검출 정보 저장
                    detection = {
                        'track_id': track_id,
                        'bbox': bbox.tolist(),
                        'center': center,
                        'foot_position': foot_pos,  # 바닥 접점 (영상 좌표)
                        'confidence': conf,
                        'class': cls,
                        'class_name': self.model.names[cls],
                        'speed': speed,
                        'is_moving': self.track_history[track_id]['is_moving'],
                        'drawing_position': drawing_pos,  # 도면 좌표
                        'zone': zone_info  # 구역 정보
                    }
                    detections.append(detection)

                    # 시각화 (동적 객체만 또는 전체)
                    is_moving = self.track_history[track_id]['is_moving']

                    # 색상: 초록(동적), 빨강(정적)
                    color = (0, 255, 0) if is_moving else (0, 0, 255)

                    # bbox 그리기
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                    # 라벨
                    label = f"ID:{track_id} {self.model.names[cls]} {conf:.2f}"
                    status = "MOVING" if is_moving else "STATIC"
                    label += f" {status} {speed:.1f}px/f"

                    cv2.putText(
                        annotated_frame, label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2
                    )

                    # 중심점 표시 (참고용)
                    cv2.circle(annotated_frame, (int(center[0]), int(center[1])), 3, color, -1)

                    # 바닥 접점 표시 (호모그래피 변환 기준점)
                    cv2.circle(annotated_frame, (int(foot_pos[0]), int(foot_pos[1])), 6, (255, 255, 0), -1)
                    cv2.circle(annotated_frame, (int(foot_pos[0]), int(foot_pos[1])), 8, color, 2)

                    # 궤적 그리기 (최근 10개, 바닥 접점 기준)
                    positions = self.track_history[track_id]['positions'][-10:]
                    for j in range(1, len(positions)):
                        pt1 = (int(positions[j-1][0]), int(positions[j-1][1]))
                        pt2 = (int(positions[j][0]), int(positions[j][1]))
                        cv2.line(annotated_frame, pt1, pt2, color, 2)

        # 4. 통계 업데이트
        self.stats['total_detections'] = len(detections)
        self.stats['active_tracks'] = len(self.track_history)
        self.stats['moving_objects'] = sum(1 for d in detections if d['is_moving'])
        self.stats['stationary_objects'] = len(detections) - self.stats['moving_objects']
        self.stats['frame_count'] += 1

        # FPS 계산
        elapsed = time.time() - start_time
        if elapsed > 0:
            self.stats['fps'] = 1.0 / elapsed

        return undistorted, annotated_frame, detections

    def draw_stats(self, frame):
        """
        통계 정보를 프레임에 그리기
        """
        y_offset = 30
        line_height = 25

        stats_text = [
            f"FPS: {self.stats['fps']:.1f}",
            f"Frame: {self.stats['frame_count']}",
            f"Detections: {self.stats['total_detections']}",
            f"Active Tracks: {self.stats['active_tracks']}",
            f"Moving: {self.stats['moving_objects']}",
            f"Stationary: {self.stats['stationary_objects']}"
        ]

        for i, text in enumerate(stats_text):
            y = y_offset + i * line_height
            cv2.putText(
                frame, text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2
            )
            cv2.putText(
                frame, text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 1
            )

    def get_moving_objects(self, detections):
        """
        동적 객체만 필터링

        Args:
            detections: 전체 검출 결과

        Returns:
            moving_detections: 동적 객체만 포함
        """
        return [d for d in detections if d['is_moving']]

    def run_video(
        self,
        video_source,
        display=True,
        save_output=False,
        output_path="output.mp4",
        skip_frames=5
    ):
        """
        비디오 소스(RTSP 스트림 또는 파일)에서 검출/추적 실행

        Args:
            video_source: RTSP URL 또는 비디오 파일 경로
            display: 화면 표시 여부
            save_output: 영상 저장 여부
            output_path: 출력 영상 경로
            skip_frames: 프레임 스킵 수 (RTSP 지연 감소용, 파일은 0 권장)
        """
        # 입력 소스 타입 감지
        is_rtsp = video_source.startswith('rtsp://') or video_source.startswith('http://')
        is_webcam = video_source.isdigit()

        if is_rtsp:
            # RTSP 설정
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|"
                "fflags;nobuffer|"
                "flags;low_delay|"
                "max_delay;0"
            )
            print(f"RTSP 연결 중: {video_source}")
        elif is_webcam:
            print(f"웹캠 연결 중: 장치 {video_source}")
            video_source = int(video_source)
        else:
            # 파일 경로 확인
            if not os.path.exists(video_source):
                print(f"오류: 파일을 찾을 수 없습니다: {video_source}")
                return
            print(f"비디오 파일 로드 중: {video_source}")
            # 파일인 경우 프레임 스킵 기본값 0
            if skip_frames == 5:  # 기본값인 경우
                skip_frames = 0
                print("비디오 파일 모드: 프레임 스킵 비활성화")

        # 비디오 캡처 열기
        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            if is_rtsp:
                print("RTSP 연결 실패")
            else:
                print("비디오 소스 열기 실패")
            return

        if is_rtsp:
            print("RTSP 연결 성공")
        else:
            print("비디오 소스 열기 성공")

        # 비디오 정보
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # FPS 값 검증 및 기본값 설정
        if fps <= 0 or fps > 120:
            print(f"경고: 비정상적인 FPS 값 ({fps}), 기본값 30 사용")
            fps = 30

        print(f"영상 정보: {width}x{height} @ {fps}FPS")

        # 비디오 라이터 (저장 시)
        writer = None
        floorplan_writer = None
        floorplan_output_path = None

        if save_output:
            # 출력 경로 검증
            if os.path.isdir(output_path):
                # 디렉토리만 지정된 경우 기본 파일명 추가
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = os.path.join(output_path, f"tracked_{timestamp}.mp4")
                print(f"출력 파일명 자동 생성: {output_path}")

            # 파일 확장자 확인
            if not output_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                output_path += '.mp4'
                print(f"확장자 추가: {output_path}")

            # 출력 디렉토리 생성
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"출력 디렉토리 생성: {output_dir}")

            # VideoWriter 초기화 (영상)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # 초기화 확인
            if not writer.isOpened():
                print(f"경고: VideoWriter 초기화 실패 - {output_path}")
                print("다른 코덱으로 재시도 중...")
                # XVID 코덱으로 재시도
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                output_path_avi = output_path.replace('.mp4', '.avi')
                writer = cv2.VideoWriter(output_path_avi, fourcc, fps, (width, height))
                if writer.isOpened():
                    print(f"출력 저장: {output_path_avi} (AVI 형식)")
                    output_path = output_path_avi
                else:
                    print("오류: VideoWriter 초기화 실패. 저장이 불가능합니다.")
                    writer = None
            else:
                print(f"영상 출력 저장: {output_path}")

            # Floor Plan VideoWriter 초기화 (도면이 있는 경우)
            if self.drawing_image is not None:
                # 도면 출력 경로 생성
                base_path = os.path.splitext(output_path)[0]
                ext = os.path.splitext(output_path)[1]
                floorplan_output_path = f"{base_path}_floorplan{ext}"

                # 도면 이미지 크기
                fp_height, fp_width = self.drawing_image.shape[:2]

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                floorplan_writer = cv2.VideoWriter(
                    floorplan_output_path, fourcc, fps, (fp_width, fp_height)
                )

                if not floorplan_writer.isOpened():
                    print(f"경고: Floor Plan VideoWriter 초기화 실패")
                    # XVID 코덱으로 재시도
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    floorplan_output_path_avi = floorplan_output_path.replace('.mp4', '.avi')
                    floorplan_writer = cv2.VideoWriter(
                        floorplan_output_path_avi, fourcc, fps, (fp_width, fp_height)
                    )
                    if floorplan_writer.isOpened():
                        print(f"도면 출력 저장: {floorplan_output_path_avi} (AVI 형식)")
                        floorplan_output_path = floorplan_output_path_avi
                    else:
                        print("경고: Floor Plan VideoWriter 초기화 실패. 도면 영상 저장 불가.")
                        floorplan_writer = None
                else:
                    print(f"도면 출력 저장: {floorplan_output_path}")

        try:
            while True:
                # 프레임 스킵 (RTSP 지연 감소용)
                if skip_frames > 0:
                    for _ in range(skip_frames):
                        cap.grab()

                # 프레임 읽기
                ret, frame = cap.read()
                if not ret:
                    if is_rtsp:
                        print("프레임 읽기 실패")
                    else:
                        print("비디오 종료")
                    break

                # 프레임 처리
                undistorted, annotated, detections = self.process_frame(frame)

                # 통계 표시
                self.draw_stats(annotated)

                # 동적 객체만 출력 (옵션)
                moving_objects = self.get_moving_objects(detections)
                if moving_objects:
                    print(f"[Frame {self.stats['frame_count']}] "
                          f"Moving objects: {len(moving_objects)}")
                    for obj in moving_objects:
                        zone_str = ""
                        if obj['zone']:
                            zone_str = f" Zone:{obj['zone']['name']}"

                        print(f"  - ID:{obj['track_id']} {obj['class_name']} "
                              f"Foot:({obj['foot_position'][0]:.0f},{obj['foot_position'][1]:.0f}) "
                              f"Drawing:({obj['drawing_position'][0]:.1f},{obj['drawing_position'][1]:.1f})"
                              f"{zone_str} "
                              f"Speed:{obj['speed']:.2f}px/f")

                # 도면 위에 객체 시각화
                floorplan_display = None
                if self.drawing_image is not None:
                    floorplan_display = self.draw_on_floorplan(detections)

                # 화면 표시
                if display:
                    cv2.imshow('Realtime Tracker', annotated)

                    # 도면도 함께 표시
                    if floorplan_display is not None:
                        cv2.imshow('Floor Plan', floorplan_display)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("사용자 종료")
                        break
                    elif key == ord('s'):
                        # 스크린샷 저장
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        screenshot_path = f"screenshot_{timestamp}.jpg"
                        cv2.imwrite(screenshot_path, annotated)
                        print(f"스크린샷 저장: {screenshot_path}")

                        # 도면 스크린샷도 저장
                        if floorplan_display is not None:
                            floorplan_screenshot = f"screenshot_floorplan_{timestamp}.jpg"
                            cv2.imwrite(floorplan_screenshot, floorplan_display)
                            print(f"도면 스크린샷 저장: {floorplan_screenshot}")

                # 영상 저장
                if writer is not None:
                    writer.write(annotated)

                # 도면 영상 저장
                if floorplan_writer is not None and floorplan_display is not None:
                    floorplan_writer.write(floorplan_display)

        except KeyboardInterrupt:
            print("\n키보드 인터럽트")

        finally:
            # 정리
            cap.release()

            # 영상 저장 완료
            if writer is not None:
                writer.release()
                print(f"\n영상 저장 완료: {output_path}")
                # 파일 크기 확인
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                    print(f"  파일 크기: {file_size:.2f} MB")
                else:
                    print("  경고: 저장된 파일을 찾을 수 없습니다.")

            # 도면 영상 저장 완료
            if floorplan_writer is not None:
                floorplan_writer.release()
                print(f"\n도면 영상 저장 완료: {floorplan_output_path}")
                # 파일 크기 확인
                if os.path.exists(floorplan_output_path):
                    file_size = os.path.getsize(floorplan_output_path) / (1024 * 1024)  # MB
                    print(f"  파일 크기: {file_size:.2f} MB")
                else:
                    print("  경고: 저장된 파일을 찾을 수 없습니다.")

            if display:
                cv2.destroyAllWindows()

            print("\n=== 최종 통계 ===")
            print(f"총 프레임: {self.stats['frame_count']}")
            print(f"활성 트랙: {self.stats['active_tracks']}")
            print(f"평균 FPS: {self.stats['fps']:.1f}")


def main():
    """
    메인 실행 함수
    """
    import argparse

    parser = argparse.ArgumentParser(description="실시간 객체 검출 및 추적")
    parser.add_argument(
        '--calibration',
        type=str,
        default='../camera_calibration/output/calibration_data.pkl',
        help='카메라 캘리브레이션 파일 경로'
    )
    parser.add_argument(
        '--homography',
        type=str,
        required=True,
        help='호모그래피 데이터 파일 경로 (.json)'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='입력 소스 (RTSP URL, 비디오 파일 경로, 또는 웹캠 번호)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        help='YOLO 모델 경로'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.3,
        help='검출 신뢰도 임계값'
    )
    parser.add_argument(
        '--classes',
        type=int,
        nargs='+',
        default=None,
        help='검출할 클래스 ID (예: 0=person, 2=car)'
    )
    parser.add_argument(
        '--speed-threshold',
        type=float,
        default=1.0,
        help='동적 판단 속도 임계값 (pixels/frame)'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='화면 표시 안 함'
    )
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='출력 영상 저장 경로'
    )
    parser.add_argument(
        '--skip-frames',
        type=int,
        default=5,
        help='프레임 스킵 수 (RTSP 지연 감소용, 파일은 자동 0)'
    )
    parser.add_argument(
        '--zones',
        type=str,
        default=None,
        help='구역 정의 JSON 파일 경로 (선택)'
    )
    parser.add_argument(
        '--drawing',
        type=str,
        default=None,
        help='DXF 도면 파일 경로 (선택, 도면 위에 객체 움직임 표시)'
    )
    parser.add_argument(
        '--drawing-size',
        type=int,
        nargs=2,
        default=[1500, 1500],
        help='도면 렌더링 크기 (width height)'
    )

    args = parser.parse_args()

    # 트래커 초기화
    tracker = RealtimeTracker(
        calibration_path=args.calibration,
        homography_path=args.homography,
        model_path=args.model,
        conf_threshold=args.conf,
        target_classes=args.classes,
        speed_threshold=args.speed_threshold,
        zones_path=args.zones,
        drawing_path=args.drawing,
        drawing_size=tuple(args.drawing_size)
    )

    # 실행
    tracker.run_video(
        video_source=args.input,
        display=not args.no_display,
        save_output=args.save is not None,
        output_path=args.save if args.save else "output.mp4",
        skip_frames=args.skip_frames
    )


if __name__ == "__main__":
    main()
