"""
DXF 파일을 이미지로 렌더링하는 유틸리티

필요한 패키지: ezdxf, PyQt6, opencv-python, numpy
"""

import sys
import math
import cv2
import numpy as np
import ezdxf
from PyQt6.QtWidgets import QApplication, QGraphicsScene
from PyQt6.QtGui import QPen, QColor, QPainterPath, QPixmap, QPainter, QImage
from PyQt6.QtCore import Qt, QRectF


class DXFRenderer:
    """
    DXF 파일을 이미지로 렌더링하는 클래스
    """

    def __init__(self):
        self.scene = None
        self.layer_colors = {}

    def render_to_image(self, dxf_path, output_size=(2000, 2000), background_color=(255, 255, 255)):
        """
        DXF 파일을 이미지로 렌더링합니다.

        Args:
            dxf_path: DXF 파일 경로
            output_size: (width, height) 출력 이미지 크기
            background_color: (R, G, B) 배경색

        Returns:
            numpy.ndarray: BGR 형식의 OpenCV 이미지, 실패 시 None
        """
        try:
            # DXF 파일 로드
            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()

            # QGraphicsScene 생성
            self.scene = QGraphicsScene()

            # 레이어 색상 설정
            self._setup_layer_colors(doc)

            # 모든 엔티티 렌더링
            for entity in msp:
                self._render_entity(entity)

            # Scene의 bounding rect 가져오기
            scene_rect = self.scene.itemsBoundingRect()

            if scene_rect.isNull() or scene_rect.isEmpty():
                print("DXF 파일이 비어있습니다.")
                return None

            # 여백 추가 (5%)
            margin = max(scene_rect.width(), scene_rect.height()) * 0.05
            scene_rect.adjust(-margin, -margin, margin, margin)

            # QPixmap 생성
            pixmap = QPixmap(output_size[0], output_size[1])
            pixmap.fill(QColor(*background_color))

            # QPainter로 렌더링
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Scene을 pixmap에 렌더링
            self.scene.render(painter, QRectF(pixmap.rect()), scene_rect, Qt.AspectRatioMode.KeepAspectRatio)
            painter.end()

            # QPixmap -> QImage -> numpy array 변환
            qimage = pixmap.toImage()
            qimage = qimage.convertToFormat(QImage.Format.Format_RGB888)

            width = qimage.width()
            height = qimage.height()

            ptr = qimage.bits()
            ptr.setsize(qimage.sizeInBytes())

            # numpy array로 변환
            arr = np.array(ptr).reshape(height, width, 3)

            # RGB -> BGR 변환 (OpenCV 형식)
            bgr_image = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

            return bgr_image

        except Exception as e:
            print(f"DXF 렌더링 오류: {e}")
            import traceback
            traceback.print_exc()
            return None

    def render_to_file(self, dxf_path, output_path, output_size=(2000, 2000), background_color=(255, 255, 255)):
        """
        DXF 파일을 이미지 파일로 렌더링하여 저장합니다.

        Args:
            dxf_path: DXF 파일 경로
            output_path: 출력 이미지 경로 (PNG, JPG 등)
            output_size: (width, height) 출력 이미지 크기
            background_color: (R, G, B) 배경색

        Returns:
            bool: 성공 여부
        """
        image = self.render_to_image(dxf_path, output_size, background_color)

        if image is None:
            return False

        try:
            cv2.imwrite(output_path, image)
            return True
        except Exception as e:
            print(f"이미지 저장 오류: {e}")
            return False

    def _setup_layer_colors(self, doc):
        """레이어 색상 설정"""
        default_colors = [
            QColor(0, 0, 0),        # Black (도면용)
            QColor(255, 0, 0),      # Red
            QColor(255, 255, 0),    # Yellow
            QColor(0, 255, 0),      # Green
            QColor(0, 255, 255),    # Cyan
            QColor(0, 0, 255),      # Blue
            QColor(255, 0, 255),    # Magenta
        ]

        for i, layer in enumerate(doc.layers):
            color_index = i % len(default_colors)
            self.layer_colors[layer.dxf.name] = default_colors[color_index]

    def _get_pen(self, entity):
        """엔티티 펜 가져오기"""
        layer_name = entity.dxf.layer
        color = self.layer_colors.get(layer_name, QColor(0, 0, 0))
        pen = QPen(color)
        pen.setWidth(2)  # 선 두께
        pen.setCosmetic(True)
        return pen

    def _render_entity(self, entity):
        """단일 DXF 엔티티 렌더링"""
        entity_type = entity.dxftype()

        if entity_type == 'LINE':
            self._render_line(entity)
        elif entity_type == 'LWPOLYLINE':
            self._render_lwpolyline(entity)
        elif entity_type == 'POLYLINE':
            self._render_polyline(entity)
        elif entity_type == 'CIRCLE':
            self._render_circle(entity)
        elif entity_type == 'ARC':
            self._render_arc(entity)
        elif entity_type == 'ELLIPSE':
            self._render_ellipse(entity)
        elif entity_type == 'SPLINE':
            self._render_spline(entity)
        elif entity_type == 'TEXT':
            self._render_text(entity)
        elif entity_type == 'MTEXT':
            self._render_mtext(entity)

    def _render_line(self, entity):
        """LINE 렌더링"""
        start = entity.dxf.start
        end = entity.dxf.end
        pen = self._get_pen(entity)
        self.scene.addLine(start.x, -start.y, end.x, -end.y, pen)

    def _render_lwpolyline(self, entity):
        """LWPOLYLINE 렌더링"""
        pen = self._get_pen(entity)
        path = QPainterPath()

        points = list(entity.get_points('xy'))
        if not points:
            return

        path.moveTo(points[0][0], -points[0][1])
        for x, y in points[1:]:
            path.lineTo(x, -y)

        if entity.closed:
            path.closeSubpath()

        self.scene.addPath(path, pen)

    def _render_polyline(self, entity):
        """POLYLINE 렌더링"""
        pen = self._get_pen(entity)
        path = QPainterPath()

        points = list(entity.points())
        if not points:
            return

        first_point = points[0]
        path.moveTo(first_point.dxf.location.x, -first_point.dxf.location.y)

        for point in points[1:]:
            loc = point.dxf.location
            path.lineTo(loc.x, -loc.y)

        if entity.is_closed:
            path.closeSubpath()

        self.scene.addPath(path, pen)

    def _render_circle(self, entity):
        """CIRCLE 렌더링"""
        center = entity.dxf.center
        radius = entity.dxf.radius
        pen = self._get_pen(entity)

        self.scene.addEllipse(
            center.x - radius, -center.y - radius,
            radius * 2, radius * 2,
            pen
        )

    def _render_arc(self, entity):
        """ARC 렌더링"""
        center = entity.dxf.center
        radius = entity.dxf.radius
        start_angle = entity.dxf.start_angle
        end_angle = entity.dxf.end_angle

        pen = self._get_pen(entity)
        path = QPainterPath()

        start_rad = math.radians(start_angle)
        end_rad = math.radians(end_angle)

        start_x = center.x + radius * math.cos(start_rad)
        start_y = center.y + radius * math.sin(start_rad)

        path.moveTo(start_x, -start_y)

        rect = QRectF(center.x - radius, -center.y - radius, radius * 2, radius * 2)
        span_angle = end_angle - start_angle
        if span_angle < 0:
            span_angle += 360

        path.arcTo(rect, -start_angle, -span_angle)
        self.scene.addPath(path, pen)

    def _render_ellipse(self, entity):
        """ELLIPSE 렌더링"""
        center = entity.dxf.center
        major_axis = entity.dxf.major_axis
        ratio = entity.dxf.ratio

        pen = self._get_pen(entity)

        major_length = math.sqrt(major_axis.x**2 + major_axis.y**2)
        minor_length = major_length * ratio

        self.scene.addEllipse(
            center.x - major_length, -center.y - minor_length,
            major_length * 2, minor_length * 2,
            pen
        )

    def _render_spline(self, entity):
        """SPLINE 렌더링"""
        pen = self._get_pen(entity)
        path = QPainterPath()

        points = list(entity.control_points)
        if not points:
            return

        path.moveTo(points[0].x, -points[0].y)
        for point in points[1:]:
            path.lineTo(point.x, -point.y)

        self.scene.addPath(path, pen)

    def _render_text(self, entity):
        """TEXT 렌더링"""
        text = entity.dxf.text
        insert = entity.dxf.insert
        height = entity.dxf.height

        pen = self._get_pen(entity)
        text_item = self.scene.addText(text)
        text_item.setPos(insert.x, -insert.y)
        text_item.setDefaultTextColor(pen.color())

        scale_factor = height / text_item.boundingRect().height() if text_item.boundingRect().height() > 0 else 1
        text_item.setScale(scale_factor)

    def _render_mtext(self, entity):
        """MTEXT 렌더링"""
        text = entity.text
        insert = entity.dxf.insert

        pen = self._get_pen(entity)
        text_item = self.scene.addText(text)
        text_item.setPos(insert.x, -insert.y)
        text_item.setDefaultTextColor(pen.color())


def render_dxf_to_image(dxf_path, output_size=(2000, 2000), background_color=(255, 255, 255)):
    """
    DXF 파일을 이미지로 렌더링합니다 (편의 함수).

    Args:
        dxf_path: DXF 파일 경로
        output_size: (width, height) 출력 크기
        background_color: (R, G, B) 배경색

    Returns:
        numpy.ndarray: OpenCV 이미지 (BGR), 실패 시 None
    """
    # QApplication이 없으면 생성
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    renderer = DXFRenderer()
    return renderer.render_to_image(dxf_path, output_size, background_color)


def render_dxf_to_file(dxf_path, output_path, output_size=(2000, 2000), background_color=(255, 255, 255)):
    """
    DXF 파일을 이미지 파일로 저장합니다 (편의 함수).

    Args:
        dxf_path: DXF 파일 경로
        output_path: 출력 이미지 경로
        output_size: (width, height) 출력 크기
        background_color: (R, G, B) 배경색

    Returns:
        bool: 성공 여부
    """
    # QApplication이 없으면 생성
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    renderer = DXFRenderer()
    return renderer.render_to_file(dxf_path, output_path, output_size, background_color)


if __name__ == "__main__":
    # 테스트
    if len(sys.argv) > 1:
        dxf_file = sys.argv[1]
        output_file = dxf_file.replace('.dxf', '.png')

        print(f"DXF 렌더링 중: {dxf_file}")
        success = render_dxf_to_file(dxf_file, output_file, output_size=(3000, 3000))

        if success:
            print(f"저장 완료: {output_file}")
        else:
            print("렌더링 실패")
    else:
        print("사용법: python dxf_renderer.py <dxf_file>")
