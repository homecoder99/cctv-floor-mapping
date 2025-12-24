"""
정밀 호모그래피 대응점 수집 도구 (QGraphicsView 기반)

주요 기능:
- 마우스 휠 줌/팬
- Ctrl+마우스로 돋보기 (확대경)
- 점 드래그로 미세 조정
- 좌→우 페어링 강제 (상태 머신)
- 실시간 H 계산 및 점별 오차 표시
- 테이블 클릭 시 자동 이동/하이라이트
- Undo/Redo
- 점 활성화/비활성화
- DXF 정점/교차점 스냅
- 점 분포 품질 경고
- 마우스 십자선 (직선 정렬 확인)
"""

import sys
import json
import numpy as np
import cv2
import ezdxf
import pickle
from pathlib import Path
from datetime import datetime
from enum import Enum
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsPixmapItem,
    QTableWidget, QTableWidgetItem, QPushButton, QLabel, QFileDialog,
    QMessageBox, QHeaderView, QCheckBox, QGroupBox, QSplitter, QTextEdit
)
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal, QTimer
from PyQt6.QtGui import (
    QPen, QColor, QBrush, QPixmap, QImage, QPainter,
    QTransform, QWheelEvent, QMouseEvent, QPainterPath, QCursor
)
from scipy.spatial import ConvexHull

# DXF 렌더링
sys.path.append(str(Path(__file__).parent.parent))
from homography.dxf_renderer import render_dxf_to_image


# =============================================================================
# 상태 정의
# =============================================================================

class PairState(Enum):
    """대응점 페어링 상태"""
    WAITING_LEFT = "LEFT 이미지에서 점 선택 대기 중"
    WAITING_RIGHT = "RIGHT 도면에서 점 선택 대기 중"


# =============================================================================
# DXF 정점 추출기
# =============================================================================

class DXFVertexExtractor:
    """DXF 파일에서 모든 정점과 교차점 추출"""

    def __init__(self, dxf_path):
        self.dxf_path = dxf_path
        self.vertices = []  # [(x, y), ...]
        self.extract_vertices()

    def extract_vertices(self):
        """DXF에서 모든 정점 추출"""
        try:
            doc = ezdxf.readfile(self.dxf_path)
            msp = doc.modelspace()

            for entity in msp:
                entity_type = entity.dxftype()

                if entity_type == 'LINE':
                    start = entity.dxf.start
                    end = entity.dxf.end
                    self.vertices.append((start.x, start.y))
                    self.vertices.append((end.x, end.y))

                elif entity_type == 'LWPOLYLINE':
                    points = list(entity.get_points('xy'))
                    for x, y in points:
                        self.vertices.append((x, y))

                elif entity_type == 'POLYLINE':
                    points = list(entity.points())
                    for point in points:
                        loc = point.dxf.location
                        self.vertices.append((loc.x, loc.y))

                elif entity_type == 'CIRCLE':
                    center = entity.dxf.center
                    # 원의 중심만 추출 (원 위의 점은 무한하므로)
                    self.vertices.append((center.x, center.y))

                elif entity_type == 'ARC':
                    # 호의 시작점, 끝점
                    center = entity.dxf.center
                    radius = entity.dxf.radius
                    start_angle = entity.dxf.start_angle
                    end_angle = entity.dxf.end_angle

                    import math
                    start_x = center.x + radius * math.cos(math.radians(start_angle))
                    start_y = center.y + radius * math.sin(math.radians(start_angle))
                    end_x = center.x + radius * math.cos(math.radians(end_angle))
                    end_y = center.y + radius * math.sin(math.radians(end_angle))

                    self.vertices.append((start_x, start_y))
                    self.vertices.append((end_x, end_y))

            # 중복 제거
            self.vertices = list(set(self.vertices))

            print(f"DXF에서 {len(self.vertices)}개 정점 추출")

        except Exception as e:
            print(f"DXF 정점 추출 오류: {e}")
            self.vertices = []

    def find_nearest(self, x, y, max_distance=50):
        """
        (x, y)에서 가장 가까운 정점 찾기

        Args:
            x, y: 클릭 좌표
            max_distance: 최대 스냅 거리 (픽셀)

        Returns:
            (snap_x, snap_y) 또는 None
        """
        if not self.vertices:
            return None

        # 거리 계산
        distances = [np.sqrt((vx - x)**2 + (vy - y)**2) for vx, vy in self.vertices]
        min_dist = min(distances)
        min_idx = distances.index(min_dist)

        if min_dist <= max_distance:
            return self.vertices[min_idx]
        else:
            return None


# =============================================================================
# 드래그 가능한 점 아이템
# =============================================================================

class DraggablePointItem(QGraphicsEllipseItem):
    """드래그 가능한 대응점"""

    def __init__(self, x, y, index, is_left, radius=8):
        super().__init__(-radius, -radius, radius*2, radius*2)
        self.index = index
        self.is_left = is_left
        self.radius = radius

        # 위치 설정
        self.setPos(x, y)

        # 드래그 가능 설정
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)

        # 초기 스타일
        self.active = True
        self.error = 0.0
        self.update_style()

        # 라벨
        from PyQt6.QtWidgets import QGraphicsTextItem
        self.label = QGraphicsTextItem(str(index), self)
        self.label.setDefaultTextColor(Qt.GlobalColor.white)
        self.label.setPos(radius + 2, -radius - 2)

    def update_style(self):
        """스타일 업데이트 (오차에 따라 색상 변경)"""
        if not self.active:
            # 비활성: 회색
            pen = QPen(QColor(128, 128, 128), 2)
            brush = QBrush(QColor(200, 200, 200, 100))
        elif self.error == 0:
            # 오차 없음: 녹색
            pen = QPen(QColor(0, 255, 0), 2)
            brush = QBrush(QColor(0, 255, 0, 150))
        elif self.error < 5:
            # 좋음: 연두색
            pen = QPen(QColor(100, 255, 100), 2)
            brush = QBrush(QColor(100, 255, 100, 150))
        elif self.error < 10:
            # 보통: 노란색
            pen = QPen(QColor(255, 255, 0), 2)
            brush = QBrush(QColor(255, 255, 0, 150))
        elif self.error < 20:
            # 나쁨: 주황색
            pen = QPen(QColor(255, 165, 0), 2)
            brush = QBrush(QColor(255, 165, 0, 150))
        else:
            # 매우 나쁨: 빨간색
            pen = QPen(QColor(255, 0, 0), 3)
            brush = QBrush(QColor(255, 0, 0, 150))

        self.setPen(pen)
        self.setBrush(brush)

    def set_error(self, error):
        """오차 설정 및 스타일 업데이트"""
        self.error = error
        self.update_style()

    def set_active(self, active):
        """활성화 상태 설정"""
        self.active = active
        self.update_style()
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsMovable, active)


# =============================================================================
# 줌/팬/돋보기 지원 QGraphicsView
# =============================================================================

class ZoomablePannableGraphicsView(QGraphicsView):
    """줌/팬/돋보기 기능이 있는 QGraphicsView"""

    point_moved = pyqtSignal(int, float, float)  # index, x, y

    def __init__(self, parent=None, enable_snap=False):
        super().__init__(parent)

        # Scene 설정
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # 렌더링 설정
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # 스크롤 설정
        self.setDragMode(QGraphicsView.DragMode.NoDrag)  # 손바닥 → 십자선으로 변경
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        # 커서 설정
        self.setCursor(Qt.CursorShape.CrossCursor)

        # 이미지 아이템
        self.image_item = None

        # 줌 설정
        self.zoom_factor = 1.15
        self.min_zoom = 0.1
        self.max_zoom = 20.0
        self.current_zoom = 1.0

        # 돋보기 설정
        self.magnifier_enabled = False
        self.magnifier_radius = 100
        self.magnifier_zoom = 3.0

        # 십자선 설정
        self.crosshair_enabled = True
        self.mouse_scene_pos = None

        # 점들
        self.points = {}  # index -> DraggablePointItem

        # 팬 모드
        self.pan_mode = False
        self.pan_start = None

        # 스냅 설정
        self.enable_snap = enable_snap
        self.vertex_extractor = None

    def set_vertex_extractor(self, extractor):
        """DXF 정점 추출기 설정"""
        self.vertex_extractor = extractor

    def set_image(self, image):
        """이미지 설정 (numpy array)"""
        h, w = image.shape[:2]

        # numpy -> QPixmap
        if len(image.shape) == 2:
            # Grayscale
            qimage = QImage(image.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            # BGR -> RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            qimage = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(qimage)

        # 기존 이미지 제거
        if self.image_item:
            self.scene.removeItem(self.image_item)

        # 새 이미지 추가
        self.image_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.image_item)

        # Scene 크기 설정
        self.scene.setSceneRect(0, 0, w, h)

        # Fit in view
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.current_zoom = 1.0

    def add_point(self, x, y, index, is_left):
        """점 추가"""
        point = DraggablePointItem(x, y, index, is_left)
        self.scene.addItem(point)
        self.points[index] = point
        return point

    def remove_point(self, index):
        """점 제거"""
        if index in self.points:
            self.scene.removeItem(self.points[index])
            del self.points[index]

    def clear_points(self):
        """모든 점 제거"""
        for point in list(self.points.values()):
            self.scene.removeItem(point)
        self.points.clear()

    def highlight_point(self, index):
        """점 하이라이트 및 화면 이동"""
        if index in self.points:
            point = self.points[index]

            # 해당 점으로 이동
            self.centerOn(point)

            # 하이라이트 애니메이션 (깜빡임)
            original_pen = point.pen()
            highlight_pen = QPen(Qt.GlobalColor.cyan, 5)

            def flash(count=0):
                if count < 6:  # 3번 깜빡임
                    if count % 2 == 0:
                        point.setPen(highlight_pen)
                    else:
                        point.setPen(original_pen)
                    QTimer.singleShot(200, lambda: flash(count + 1))
                else:
                    point.update_style()

            flash()

    def wheelEvent(self, event: QWheelEvent):
        """마우스 휠로 줌"""
        if event.angleDelta().y() > 0:
            factor = self.zoom_factor
        else:
            factor = 1.0 / self.zoom_factor

        new_zoom = self.current_zoom * factor

        if self.min_zoom <= new_zoom <= self.max_zoom:
            self.scale(factor, factor)
            self.current_zoom = new_zoom

    def mousePressEvent(self, event: QMouseEvent):
        """마우스 클릭"""
        if event.button() == Qt.MouseButton.LeftButton:
            # Ctrl 키로 돋보기 활성화
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.magnifier_enabled = True
                self.viewport().update()
                return

            # 중간 버튼이나 Space로 팬 모드
        if event.button() == Qt.MouseButton.MiddleButton or \
           (event.button() == Qt.MouseButton.LeftButton and
            event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self.pan_mode = True
            self.pan_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """마우스 이동"""
        # Scene 좌표 저장 (십자선용)
        self.mouse_scene_pos = self.mapToScene(event.pos())

        # 십자선 다시 그리기
        if self.crosshair_enabled:
            self.viewport().update()

        # 돋보기
        if self.magnifier_enabled:
            self.viewport().update()

        # 팬 모드
        if self.pan_mode and self.pan_start:
            delta = event.pos() - self.pan_start
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self.pan_start = event.pos()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """마우스 릴리즈"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.magnifier_enabled = False
            self.viewport().update()

        if event.button() == Qt.MouseButton.MiddleButton or \
           (event.button() == Qt.MouseButton.LeftButton and self.pan_mode):
            self.pan_mode = False
            self.pan_start = None
            self.setCursor(Qt.CursorShape.CrossCursor)

        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        """페인트 이벤트 (돋보기 + 십자선 그리기)"""
        super().paintEvent(event)

        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 십자선 그리기
        if self.crosshair_enabled and self.mouse_scene_pos:
            # Scene 좌표 → Viewport 좌표
            view_pos = self.mapFromScene(self.mouse_scene_pos)

            # 반투명 빨간색 십자선
            pen = QPen(QColor(255, 0, 0, 150), 1, Qt.PenStyle.DashLine)
            painter.setPen(pen)

            # 가로선 (전체 너비)
            painter.drawLine(0, view_pos.y(), self.viewport().width(), view_pos.y())

            # 세로선 (전체 높이)
            painter.drawLine(view_pos.x(), 0, view_pos.x(), self.viewport().height())

            # 좌표 표시
            if self.image_item:
                scene_rect = self.scene.sceneRect()
                if scene_rect.contains(self.mouse_scene_pos):
                    coord_text = f"({self.mouse_scene_pos.x():.0f}, {self.mouse_scene_pos.y():.0f})"
                    painter.setPen(QPen(Qt.GlobalColor.white, 1))
                    painter.setBrush(QBrush(QColor(0, 0, 0, 180)))

                    # 텍스트 박스
                    from PyQt6.QtGui import QFont, QFontMetrics
                    font = QFont("Arial", 10)
                    painter.setFont(font)
                    metrics = QFontMetrics(font)
                    text_rect = metrics.boundingRect(coord_text)

                    box_x = view_pos.x() + 10
                    box_y = view_pos.y() + 10

                    painter.drawRect(box_x, box_y, text_rect.width() + 10, text_rect.height() + 6)
                    painter.drawText(box_x + 5, box_y + text_rect.height() + 2, coord_text)

        # 돋보기 그리기
        if self.magnifier_enabled:
            # 마우스 위치
            mouse_pos = self.mapFromGlobal(self.cursor().pos())

            # 돋보기 영역
            mag_rect = QRectF(
                mouse_pos.x() - self.magnifier_radius,
                mouse_pos.y() - self.magnifier_radius,
                self.magnifier_radius * 2,
                self.magnifier_radius * 2
            )

            # Scene 좌표
            scene_pos = self.mapToScene(mouse_pos.toPoint())

            # 확대할 Scene 영역
            source_size = self.magnifier_radius / self.magnifier_zoom
            source_rect = QRectF(
                scene_pos.x() - source_size,
                scene_pos.y() - source_size,
                source_size * 2,
                source_size * 2
            )

            # 돋보기 배경
            painter.setBrush(Qt.GlobalColor.white)
            painter.setPen(QPen(Qt.GlobalColor.black, 3))
            painter.drawEllipse(mag_rect)

            # 클리핑
            path = QPainterPath()
            path.addEllipse(mag_rect)
            painter.setClipPath(path)

            # Scene 렌더링 (확대)
            self.scene.render(painter, mag_rect, source_rect)

            # 십자선
            painter.setClipping(False)
            painter.setPen(QPen(Qt.GlobalColor.red, 2))
            painter.drawLine(mouse_pos.x() - 15, mouse_pos.y(), mouse_pos.x() + 15, mouse_pos.y())
            painter.drawLine(mouse_pos.x(), mouse_pos.y() - 15, mouse_pos.x(), mouse_pos.y() + 15)

        painter.end()


# =============================================================================
# 이미지 패널 (왼쪽/오른쪽)
# =============================================================================

class ImagePanel(QWidget):
    """이미지 뷰어 패널"""

    point_clicked = pyqtSignal(float, float)  # x, y in scene coordinates

    def __init__(self, title, enable_snap=False, parent=None):
        super().__init__(parent)

        self.title = title
        self.enable_snap = enable_snap

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Title
        title_label = QLabel(f"<b>{title}</b>")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # Status
        self.status_label = QLabel("이미지 로드 대기 중")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        # View
        self.view = ZoomablePannableGraphicsView(self, enable_snap=enable_snap)
        layout.addWidget(self.view, 1)

        # 클릭 이벤트 연결
        self.view.scene.mousePressEvent = self._on_scene_click

    def _on_scene_click(self, event):
        """Scene 클릭 이벤트"""
        if event.button() == Qt.MouseButton.LeftButton:
            # Ctrl 키가 눌려있으면 돋보기 모드 (점 추가 안 함)
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                return

            # Shift 키가 눌려있으면 팬 모드 (점 추가 안 함)
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                return

            # 점이 클릭되지 않은 경우에만 새 점 추가
            item = self.view.scene.itemAt(event.scenePos(), QTransform())
            if not isinstance(item, DraggablePointItem):
                pos = event.scenePos()
                x, y = pos.x(), pos.y()

                # 스냅 적용 (도면 패널만)
                if self.enable_snap and self.view.vertex_extractor:
                    snapped = self.view.vertex_extractor.find_nearest(x, y, max_distance=30)
                    if snapped:
                        x, y = snapped
                        print(f"스냅: ({pos.x():.1f}, {pos.y():.1f}) → ({x:.1f}, {y:.1f})")

                self.point_clicked.emit(x, y)

    def set_image(self, image):
        """이미지 설정"""
        self.view.set_image(image)
        h, w = image.shape[:2]
        self.status_label.setText(
            f"해상도: {w}×{h} | 휠:줌 | Ctrl+클릭:돋보기 | Shift+드래그:팬"
        )

    def set_status(self, text, color="black"):
        """상태 메시지 설정"""
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")


# =============================================================================
# 점 분포 품질 분석기
# =============================================================================

class PointDistributionAnalyzer:
    """점 분포 품질 분석"""

    @staticmethod
    def analyze(points):
        """
        점 분포 분석

        Args:
            points: [(x, y), ...] 리스트

        Returns:
            {
                'convex_hull_area': float,
                'bounding_box_area': float,
                'coverage_ratio': float,  # convex_hull / bounding_box
                'x_range': float,
                'y_range': float,
                'aspect_ratio': float,  # x_range / y_range
                'warnings': [str, ...]
            }
        """
        if len(points) < 3:
            return {
                'warnings': ['최소 3개의 점이 필요합니다.']
            }

        points = np.array(points)

        # Bounding box
        min_x, min_y = points.min(axis=0)
        max_x, max_y = points.max(axis=0)
        x_range = max_x - min_x
        y_range = max_y - min_y
        bbox_area = x_range * y_range

        # Convex hull
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume  # 2D에서는 area
        except:
            hull_area = 0

        # 비율
        coverage_ratio = hull_area / bbox_area if bbox_area > 0 else 0
        aspect_ratio = x_range / y_range if y_range > 0 else 0

        # 경고 생성
        warnings = []

        if len(points) < 8:
            warnings.append(f"⚠ 점이 {len(points)}개뿐입니다. 최소 12개 이상 권장합니다.")

        if coverage_ratio < 0.3:
            warnings.append(f"⚠ 점들이 한쪽에 몰려있습니다 (커버리지: {coverage_ratio*100:.1f}%). 넓게 분포시키세요.")

        if aspect_ratio < 0.3 or aspect_ratio > 3.0:
            warnings.append(f"⚠ 점 분포가 한 방향으로 치우쳤습니다 (가로:세로 = {aspect_ratio:.2f}:1).")

        # 공선성 체크 (모든 점이 거의 직선 위에 있는지)
        if len(points) >= 3:
            # 첫 두 점으로 직선 방정식 생성
            p1, p2 = points[0], points[1]

            # 나머지 점들의 거리 계산
            if np.linalg.norm(p2 - p1) > 1e-6:
                line_vec = p2 - p1
                line_vec = line_vec / np.linalg.norm(line_vec)

                max_dist = 0
                for p in points[2:]:
                    vec = p - p1
                    cross = abs(np.cross(line_vec, vec))
                    max_dist = max(max_dist, cross)

                if max_dist < 50:  # 50픽셀 이내
                    warnings.append(f"⚠ 모든 점이 거의 직선 위에 있습니다 (최대 편차: {max_dist:.1f}px). 2D 영역을 커버하세요!")

        if not warnings:
            warnings.append("✓ 점 분포가 양호합니다.")

        return {
            'convex_hull_area': hull_area,
            'bounding_box_area': bbox_area,
            'coverage_ratio': coverage_ratio,
            'x_range': x_range,
            'y_range': y_range,
            'aspect_ratio': aspect_ratio,
            'warnings': warnings
        }


# =============================================================================
# 메인 윈도우
# =============================================================================

class PointCollectorWindow(QMainWindow):
    """메인 윈도우"""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("정밀 호모그래피 대응점 수집 도구 v2")
        self.setGeometry(100, 100, 1800, 1000)

        # 데이터
        self.image_path = None
        self.drawing_path = None
        self.calibration_path = None
        self.image = None
        self.image_original = None  # 원본 (왜곡 보정 전)
        self.drawing_image = None
        self.dxf_vertex_extractor = None

        # 캘리브레이션
        self.camera_matrix = None
        self.dist_coeffs = None
        self.undistort_enabled = True  # 왜곡 보정 활성화

        self.point_pairs = []  # [(img_x, img_y, draw_x, draw_y, active), ...]
        self.history = []  # Undo/Redo
        self.history_index = -1

        # 상태
        self.pair_state = PairState.WAITING_LEFT
        self.pending_left_point = None  # (x, y)

        # DXF 메타데이터 (픽셀→CAD 변환용)
        self.dxf_bounds = None  # (min_x, min_y, max_x, max_y)
        self.dxf_y_inverted = True  # 렌더링 시 Y축 반전 여부
        self.dxf_output_size = None  # (width, height)

        # UI 초기화
        self.init_ui()

        # 타이머로 실시간 H 계산 및 분포 분석
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_homography)
        self.update_timer.start(500)  # 0.5초마다

    def init_ui(self):
        """UI 초기화"""
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)

        # =============================================================================
        # 상단: 파일 선택
        # =============================================================================
        file_group = QGroupBox("1. 파일 선택")
        file_layout = QVBoxLayout()

        # 첫 번째 행: 이미지 + DXF
        first_row = QHBoxLayout()

        self.image_btn = QPushButton("이미지 선택...")
        self.image_btn.clicked.connect(self.select_image)
        first_row.addWidget(self.image_btn)

        self.image_label = QLabel("선택 안 됨")
        first_row.addWidget(self.image_label, 1)

        self.drawing_btn = QPushButton("DXF 선택...")
        self.drawing_btn.clicked.connect(self.select_drawing)
        first_row.addWidget(self.drawing_btn)

        self.drawing_label = QLabel("선택 안 됨")
        first_row.addWidget(self.drawing_label, 1)

        file_layout.addLayout(first_row)

        # 두 번째 행: 캘리브레이션 + 왜곡 보정 토글
        second_row = QHBoxLayout()

        self.calibration_btn = QPushButton("캘리브레이션 선택... (선택사항)")
        self.calibration_btn.clicked.connect(self.select_calibration)
        second_row.addWidget(self.calibration_btn)

        self.calibration_label = QLabel("선택 안 됨")
        second_row.addWidget(self.calibration_label, 1)

        self.undistort_checkbox = QCheckBox("왜곡 보정 활성화")
        self.undistort_checkbox.setChecked(True)
        self.undistort_checkbox.setEnabled(False)  # 캘리브레이션 로드 전엔 비활성화
        self.undistort_checkbox.stateChanged.connect(self.toggle_undistort)
        second_row.addWidget(self.undistort_checkbox)

        file_layout.addLayout(second_row)

        file_group.setLayout(file_layout)
        main_layout.addWidget(file_group)

        # =============================================================================
        # 중단: 이미지 뷰어 (좌우 분할)
        # =============================================================================
        view_group = QGroupBox("2. 대응점 선택")
        view_layout = QHBoxLayout()

        # 왼쪽: 이미지
        self.left_panel = ImagePanel("이미지 (Camera View)", enable_snap=False)
        self.left_panel.point_clicked.connect(self.on_left_clicked)
        view_layout.addWidget(self.left_panel, 1)

        # 오른쪽: 도면 (스냅 활성화)
        self.right_panel = ImagePanel("도면 (DXF Floor Plan) - 자동 스냅", enable_snap=True)
        self.right_panel.point_clicked.connect(self.on_right_clicked)
        view_layout.addWidget(self.right_panel, 1)

        view_group.setLayout(view_layout)
        main_layout.addWidget(view_group, 1)

        # =============================================================================
        # 하단: 테이블 및 컨트롤
        # =============================================================================
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)

        # 왼쪽: 테이블
        table_group = QGroupBox("3. 대응점 목록")
        table_layout = QVBoxLayout()

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels([
            "번호", "이미지 (x, y)", "도면 (x, y)", "오차 (px)", "활성", "삭제"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.cellClicked.connect(self.on_table_clicked)
        table_layout.addWidget(self.table)

        table_group.setLayout(table_layout)
        bottom_splitter.addWidget(table_group)

        # 오른쪽: 상태 및 컨트롤
        control_group = QGroupBox("4. 컨트롤 & 품질")
        control_layout = QVBoxLayout()

        # 상태
        self.state_label = QLabel()
        self.update_state_label()
        control_layout.addWidget(self.state_label)

        # 통계
        self.stats_label = QLabel("대응점: 0개")
        control_layout.addWidget(self.stats_label)

        # 분포 품질 경고
        self.quality_text = QTextEdit()
        self.quality_text.setReadOnly(True)
        self.quality_text.setMaximumHeight(120)
        self.quality_text.setPlaceholderText("점 분포 품질 분석 결과가 여기에 표시됩니다.")
        control_layout.addWidget(self.quality_text)

        # Undo/Redo
        undo_redo_layout = QHBoxLayout()
        self.undo_btn = QPushButton("↶ Undo")
        self.undo_btn.clicked.connect(self.undo)
        undo_redo_layout.addWidget(self.undo_btn)

        self.redo_btn = QPushButton("↷ Redo")
        self.redo_btn.clicked.connect(self.redo)
        undo_redo_layout.addWidget(self.redo_btn)
        control_layout.addLayout(undo_redo_layout)

        # 전체 삭제
        self.clear_btn = QPushButton("전체 삭제")
        self.clear_btn.clicked.connect(self.clear_all_points)
        control_layout.addWidget(self.clear_btn)

        # 저장
        self.save_btn = QPushButton("💾 저장")
        self.save_btn.clicked.connect(self.save_homography)
        self.save_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        control_layout.addWidget(self.save_btn)

        control_layout.addStretch()

        control_group.setLayout(control_layout)
        bottom_splitter.addWidget(control_group)

        bottom_splitter.setSizes([1200, 600])
        main_layout.addWidget(bottom_splitter)

    def update_state_label(self):
        """상태 라벨 업데이트"""
        if self.pair_state == PairState.WAITING_LEFT:
            self.state_label.setText(
                f"<h3 style='color: blue;'>⬅ 왼쪽 이미지에서 점을 선택하세요</h3>"
            )
            self.left_panel.set_status("여기를 클릭하세요!", "blue")
            self.right_panel.set_status("대기 중...", "gray")
        else:
            self.state_label.setText(
                f"<h3 style='color: green;'>➡ 오른쪽 도면에서 대응점을 선택하세요</h3>"
            )
            self.left_panel.set_status("대기 중...", "gray")
            self.right_panel.set_status("여기를 클릭하세요! (자동 스냅)", "green")

    def select_calibration(self):
        """캘리브레이션 파일 선택"""
        path, _ = QFileDialog.getOpenFileName(
            self, "캘리브레이션 파일 선택", "",
            "Pickle Files (*.pkl)"
        )

        if path:
            try:
                with open(path, 'rb') as f:
                    calib_data = pickle.load(f)

                self.camera_matrix = calib_data['camera_matrix']
                self.dist_coeffs = calib_data['dist_coeffs']
                self.calibration_path = path

                self.calibration_label.setText(f"{Path(path).name} ✓")
                self.undistort_checkbox.setEnabled(True)

                print(f"캘리브레이션 로드 성공: {path}")
                print(f"Camera matrix:\n{self.camera_matrix}")
                print(f"Distortion coeffs: {self.dist_coeffs.ravel()}")

                # 이미지가 이미 로드되어 있으면 다시 로드 (왜곡 보정 적용)
                if self.image_path:
                    self.reload_image()

            except Exception as e:
                QMessageBox.critical(self, "오류", f"캘리브레이션 파일 로드 실패:\n{e}")
                self.camera_matrix = None
                self.dist_coeffs = None
                self.calibration_label.setText("선택 안 됨")
                self.undistort_checkbox.setEnabled(False)

    def toggle_undistort(self, state):
        """왜곡 보정 토글"""
        self.undistort_enabled = (state == Qt.CheckState.Checked.value)

        # 이미지 다시 로드
        if self.image_path:
            self.reload_image()

    def reload_image(self):
        """이미지 다시 로드 (왜곡 보정 적용/미적용)"""
        if self.image_original is None:
            return

        if self.undistort_enabled and self.camera_matrix is not None and self.dist_coeffs is not None:
            # 왜곡 보정 적용
            h, w = self.image_original.shape[:2]
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
            )
            self.image = cv2.undistort(
                self.image_original, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix
            )
            print(f"왜곡 보정 적용: {w}x{h}")
        else:
            # 원본 사용
            self.image = self.image_original.copy()
            print("원본 이미지 사용 (왜곡 보정 없음)")

        # 화면 업데이트
        self.left_panel.set_image(self.image)

        # 상태 메시지
        if self.undistort_enabled and self.camera_matrix is not None:
            self.image_label.setText(f"{Path(self.image_path).name} (왜곡 보정 ✓)")
        else:
            self.image_label.setText(Path(self.image_path).name)

    def select_image(self):
        """이미지 파일 선택"""
        path, _ = QFileDialog.getOpenFileName(
            self, "이미지 선택", "",
            "Images (*.jpg *.jpeg *.png *.bmp)"
        )

        if path:
            self.image_path = path
            self.image_original = cv2.imread(path)

            if self.image_original is not None:
                # 왜곡 보정 적용 여부에 따라 이미지 설정
                self.reload_image()
            else:
                QMessageBox.critical(self, "오류", "이미지를 읽을 수 없습니다.")

    def select_drawing(self):
        """DXF 파일 선택"""
        path, _ = QFileDialog.getOpenFileName(
            self, "DXF 선택", "",
            "DXF Files (*.dxf)"
        )

        if path:
            self.drawing_path = path

            # DXF 정점 추출
            self.dxf_vertex_extractor = DXFVertexExtractor(path)
            self.right_panel.view.set_vertex_extractor(self.dxf_vertex_extractor)

            # DXF 렌더링 (메타데이터 포함)
            print(f"DXF 렌더링 중: {path}")
            result = render_dxf_to_image(path, output_size=(3000, 3000), return_metadata=True)

            if result is not None:
                self.drawing_image = result['image']
                self.dxf_bounds = result['bounds']
                self.dxf_y_inverted = result['y_inverted']
                self.dxf_output_size = result['output_size']

                print(f"  DXF bounds (CAD 좌표): {self.dxf_bounds}")
                print(f"  Y inverted: {self.dxf_y_inverted}")
                print(f"  Output size: {self.dxf_output_size}")

                self.right_panel.set_image(self.drawing_image)
                self.drawing_label.setText(f"{Path(path).name} ({len(self.dxf_vertex_extractor.vertices)}개 정점)")
            else:
                QMessageBox.critical(self, "오류", "DXF 렌더링 실패.")

    def on_left_clicked(self, x, y):
        """왼쪽 이미지 클릭"""
        if self.pair_state == PairState.WAITING_LEFT:
            # 왼쪽 점 저장
            self.pending_left_point = (x, y)

            # 임시 점 표시
            self.left_panel.view.add_point(x, y, len(self.point_pairs), is_left=True)

            # 상태 전환
            self.pair_state = PairState.WAITING_RIGHT
            self.update_state_label()

    def pixel_to_cad(self, px, py):
        """
        도면 이미지의 픽셀 좌표를 DXF CAD 좌표로 역변환

        Args:
            px, py: 픽셀 좌표 (렌더링된 이미지 상의 좌표)

        Returns:
            (cad_x, cad_y): DXF CAD 좌표, 실패 시 None
        """
        if self.dxf_bounds is None or self.dxf_output_size is None:
            return None

        min_x, min_y, max_x, max_y = self.dxf_bounds
        W, H = self.dxf_output_size

        # 픽셀 → 정규화 좌표 (0~1)
        norm_x = px / W
        norm_y = py / H

        # Y축 반전 처리
        if self.dxf_y_inverted:
            # 렌더러가 -y를 적용했으므로 (Y↓ 픽셀 좌표)
            # CAD 좌표(Y↑)로 복원: y_cad = max_y - norm_y * (max_y - min_y)
            norm_y_inverted = 1.0 - norm_y
        else:
            norm_y_inverted = norm_y

        # 정규화 좌표 → CAD 좌표
        cad_x = min_x + norm_x * (max_x - min_x)
        cad_y = min_y + norm_y_inverted * (max_y - min_y)

        return (cad_x, cad_y)

    def cad_to_pixel(self, cad_x, cad_y):
        """
        DXF CAD 좌표를 도면 이미지의 픽셀 좌표로 변환 (역변환)

        Args:
            cad_x, cad_y: DXF CAD 좌표

        Returns:
            (px, py): 픽셀 좌표, 실패 시 None
        """
        if self.dxf_bounds is None or self.dxf_output_size is None:
            return None

        min_x, min_y, max_x, max_y = self.dxf_bounds
        W, H = self.dxf_output_size

        # CAD 좌표 → 정규화 좌표 (0~1)
        norm_x = (cad_x - min_x) / (max_x - min_x)
        norm_y = (cad_y - min_y) / (max_y - min_y)

        # Y축 반전 처리
        if self.dxf_y_inverted:
            # CAD 좌표(Y↑) → 픽셀 좌표(Y↓)
            norm_y = 1.0 - norm_y

        # 정규화 좌표 → 픽셀 좌표
        px = norm_x * W
        py = norm_y * H

        return (px, py)

    def on_right_clicked(self, x, y):
        """오른쪽 도면 클릭"""
        if self.pair_state == PairState.WAITING_RIGHT and self.pending_left_point:
            # 페어 완성
            img_x, img_y = self.pending_left_point

            # 픽셀 좌표를 CAD 좌표로 변환
            cad_coords = self.pixel_to_cad(x, y)
            if cad_coords is None:
                QMessageBox.warning(self, "경고", "DXF 메타데이터가 없어 좌표 변환 불가")
                return

            draw_x, draw_y = cad_coords

            print(f"  도면 클릭: 픽셀({x:.1f}, {y:.1f}) → CAD({draw_x:.1f}, {draw_y:.1f})")

            # 저장 (CAD 좌표로)
            self.add_point_pair(img_x, img_y, draw_x, draw_y)

            # 오른쪽 점 표시 (픽셀 좌표로 - 화면 표시용)
            index = len(self.point_pairs) - 1
            self.right_panel.view.add_point(x, y, index, is_left=False)

            # 상태 초기화
            self.pending_left_point = None
            self.pair_state = PairState.WAITING_LEFT
            self.update_state_label()

    def add_point_pair(self, img_x, img_y, draw_x, draw_y, active=True):
        """대응점 페어 추가"""
        self.point_pairs.append([img_x, img_y, draw_x, draw_y, active])
        self.update_table()
        self.save_history()

    def update_table(self):
        """테이블 업데이트"""
        self.table.setRowCount(len(self.point_pairs))

        for i, (img_x, img_y, draw_x, draw_y, active) in enumerate(self.point_pairs):
            # 번호
            self.table.setItem(i, 0, QTableWidgetItem(str(i)))

            # 이미지 좌표
            self.table.setItem(i, 1, QTableWidgetItem(f"({img_x:.1f}, {img_y:.1f})"))

            # 도면 좌표
            self.table.setItem(i, 2, QTableWidgetItem(f"({draw_x:.1f}, {draw_y:.1f})"))

            # 오차 (나중에 업데이트)
            self.table.setItem(i, 3, QTableWidgetItem("-"))

            # 활성화 체크박스
            checkbox = QCheckBox()
            checkbox.setChecked(active)
            checkbox.stateChanged.connect(lambda state, idx=i: self.toggle_point(idx, state))
            self.table.setCellWidget(i, 4, checkbox)

            # 삭제 버튼
            delete_btn = QPushButton("🗑")
            delete_btn.clicked.connect(lambda _, idx=i: self.delete_point(idx))
            self.table.setCellWidget(i, 5, delete_btn)

        # 통계 업데이트
        active_count = sum(1 for p in self.point_pairs if p[4])
        self.stats_label.setText(f"대응점: {len(self.point_pairs)}개 (활성: {active_count}개)")

    def toggle_point(self, index, state):
        """점 활성화/비활성화"""
        self.point_pairs[index][4] = (state == Qt.CheckState.Checked.value)

        # 점 스타일 업데이트
        if index in self.left_panel.view.points:
            self.left_panel.view.points[index].set_active(self.point_pairs[index][4])
        if index in self.right_panel.view.points:
            self.right_panel.view.points[index].set_active(self.point_pairs[index][4])

        self.update_table()

    def delete_point(self, index):
        """점 삭제"""
        # 점 제거
        self.left_panel.view.remove_point(index)
        self.right_panel.view.remove_point(index)

        # 데이터 제거
        del self.point_pairs[index]

        # 재인덱싱
        self.reindex_points()

        self.update_table()
        self.save_history()

    def reindex_points(self):
        """점 재인덱싱"""
        # 모든 점 제거
        self.left_panel.view.clear_points()
        self.right_panel.view.clear_points()

        # 다시 추가
        for i, (img_x, img_y, draw_x, draw_y, active) in enumerate(self.point_pairs):
            # 왼쪽: 영상 좌표 (픽셀)
            left_pt = self.left_panel.view.add_point(img_x, img_y, i, is_left=True)

            # 오른쪽: CAD 좌표 → 픽셀 좌표로 변환
            pixel_coords = self.cad_to_pixel(draw_x, draw_y)
            if pixel_coords:
                px, py = pixel_coords
                right_pt = self.right_panel.view.add_point(px, py, i, is_left=False)
                right_pt.set_active(active)
            else:
                # CAD→픽셀 변환 실패 시 (메타데이터 없음)
                print(f"경고: CAD→픽셀 변환 실패 - 점 {i}")

            left_pt.set_active(active)

    def clear_all_points(self):
        """전체 삭제"""
        reply = QMessageBox.question(
            self, "확인", "모든 대응점을 삭제하시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.point_pairs.clear()
            self.left_panel.view.clear_points()
            self.right_panel.view.clear_points()
            self.update_table()
            self.save_history()

    def on_table_clicked(self, row, _):
        """테이블 클릭 시 해당 점으로 이동"""
        if row < len(self.point_pairs):
            # 왼쪽과 오른쪽 모두 하이라이트
            self.left_panel.view.highlight_point(row)
            self.right_panel.view.highlight_point(row)

    def update_homography(self):
        """실시간 호모그래피 계산 및 오차 업데이트"""
        # 활성 점만 추출
        active_points = [(img_x, img_y, draw_x, draw_y)
                        for img_x, img_y, draw_x, draw_y, active in self.point_pairs
                        if active]

        if len(active_points) < 4:
            return

        # numpy 배열로 변환
        img_pts = np.array([[p[0], p[1]] for p in active_points], dtype=np.float32)
        draw_pts = np.array([[p[2], p[3]] for p in active_points], dtype=np.float32)

        # H 계산
        try:
            H, _ = cv2.findHomography(img_pts, draw_pts, cv2.RANSAC, 5.0)
        except:
            return

        if H is None:
            return

        # 재투영 오차 계산
        img_pts_h = np.hstack([img_pts, np.ones((len(img_pts), 1))])
        transformed_h = (H @ img_pts_h.T).T
        transformed = transformed_h[:, :2] / transformed_h[:, 2:3]

        errors = np.linalg.norm(transformed - draw_pts, axis=1)

        # 점별 오차 업데이트
        active_idx = 0
        for i, (_, _, _, _, active) in enumerate(self.point_pairs):
            if active:
                error = errors[active_idx]
                active_idx += 1

                # 테이블 업데이트
                self.table.setItem(i, 3, QTableWidgetItem(f"{error:.1f}"))

                # 점 색상 업데이트
                if i in self.left_panel.view.points:
                    self.left_panel.view.points[i].set_error(error)
                if i in self.right_panel.view.points:
                    self.right_panel.view.points[i].set_error(error)

        # 점 분포 품질 분석
        self.analyze_distribution()

    def analyze_distribution(self):
        """점 분포 품질 분석 및 표시"""
        if len(self.point_pairs) < 3:
            self.quality_text.clear()
            return

        # 활성 점만
        active_img_pts = [(img_x, img_y)
                         for img_x, img_y, _, _, active in self.point_pairs
                         if active]

        if len(active_img_pts) < 3:
            return

        # 분석
        result = PointDistributionAnalyzer.analyze(active_img_pts)

        # 결과 표시
        text = "<b>점 분포 품질 분석:</b><br>"
        text += f"커버리지: {result.get('coverage_ratio', 0)*100:.1f}%<br>"
        text += f"가로:세로 비율: {result.get('aspect_ratio', 0):.2f}:1<br>"
        text += "<br><b>경고:</b><br>"

        for warning in result.get('warnings', []):
            if warning.startswith('✓'):
                text += f"<span style='color: green;'>{warning}</span><br>"
            else:
                text += f"<span style='color: red;'>{warning}</span><br>"

        self.quality_text.setHtml(text)

    def save_history(self):
        """히스토리 저장 (Undo/Redo용)"""
        # 현재 상태 저장
        state = [p.copy() for p in self.point_pairs]

        # 히스토리 인덱스 이후 제거
        self.history = self.history[:self.history_index + 1]

        # 새 상태 추가
        self.history.append(state)
        self.history_index += 1

        # 최대 100개까지만
        if len(self.history) > 100:
            self.history.pop(0)
            self.history_index -= 1

    def undo(self):
        """실행 취소"""
        if self.history_index > 0:
            self.history_index -= 1
            self.point_pairs = [p.copy() for p in self.history[self.history_index]]
            self.reindex_points()
            self.update_table()

    def redo(self):
        """다시 실행"""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.point_pairs = [p.copy() for p in self.history[self.history_index]]
            self.reindex_points()
            self.update_table()

    def save_homography(self):
        """호모그래피 저장"""
        if len(self.point_pairs) < 4:
            QMessageBox.warning(self, "경고", "최소 4개의 대응점이 필요합니다.")
            return

        # 활성 점만
        active_points = [(img_x, img_y, draw_x, draw_y)
                        for img_x, img_y, draw_x, draw_y, active in self.point_pairs
                        if active]

        if len(active_points) < 4:
            QMessageBox.warning(self, "경고", "최소 4개의 활성 대응점이 필요합니다.")
            return

        # numpy 배열
        img_pts = np.array([[p[0], p[1]] for p in active_points], dtype=np.float32)
        draw_pts = np.array([[p[2], p[3]] for p in active_points], dtype=np.float32)

        # H 계산
        H, _ = cv2.findHomography(img_pts, draw_pts, cv2.RANSAC, 5.0)

        if H is None:
            QMessageBox.critical(self, "오류", "호모그래피 계산 실패.")
            return

        # 오차 계산
        img_pts_h = np.hstack([img_pts, np.ones((len(img_pts), 1))])
        transformed_h = (H @ img_pts_h.T).T
        transformed = transformed_h[:, :2] / transformed_h[:, 2:3]
        errors = np.linalg.norm(transformed - draw_pts, axis=1)

        # JSON 생성
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"homography/data/homography_{timestamp}.json"

        data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'version': '3.0',  # CAD 좌표 저장 버전
                'image_source': self.image_path,
                'drawing_source': self.drawing_path,
                'calibration_source': self.calibration_path,
                'undistort_applied': self.undistort_enabled and self.camera_matrix is not None,
                'num_points': len(active_points),
                'total_points': len(self.point_pairs),
                'active_points': len(active_points),
                # DXF 메타데이터 추가
                'dxf_bounds': list(self.dxf_bounds) if self.dxf_bounds else None,
                'dxf_y_inverted': self.dxf_y_inverted,
                'dxf_output_size': list(self.dxf_output_size) if self.dxf_output_size else None,
                'coordinate_system': 'CAD'  # drawing_points가 CAD 좌표임을 명시
            },
            'image_points': [[float(x), float(y)] for x, y in img_pts],
            'drawing_points': [[float(x), float(y)] for x, y in draw_pts],  # CAD 좌표
            'homography_matrix': H.tolist(),
            'reprojection_error': {
                'mean': float(np.mean(errors)),
                'max': float(np.max(errors)),
                'std': float(np.std(errors)),
                'median': float(np.median(errors)),
                'errors': errors.tolist()
            }
        }

        # 저장
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        # 결과 표시
        undistort_msg = ""
        if self.camera_matrix is not None:
            if self.undistort_enabled:
                undistort_msg = "\n왜곡 보정: ✓ 적용됨"
            else:
                undistort_msg = "\n왜곡 보정: ✗ 비활성화됨"

        msg = f"""호모그래피 저장 완료!

파일: {output_path}

대응점: {len(active_points)}개 (전체: {len(self.point_pairs)}개){undistort_msg}
재투영 오차:
  - 평균: {np.mean(errors):.2f} pixels
  - 최대: {np.max(errors):.2f} pixels
  - 중앙값: {np.median(errors):.2f} pixels

{"✓ 우수한 정확도!" if np.mean(errors) < 5 else "⚠ 오차가 큽니다. 대응점을 재확인하세요."}
"""

        QMessageBox.information(self, "저장 완료", msg)


# =============================================================================
# 메인
# =============================================================================

def main():
    app = QApplication(sys.argv)

    window = PointCollectorWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
