"""DXF Viewer Widget for PyQt6"""
import ezdxf
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene
from PyQt6.QtGui import QPen, QColor, QPainterPath, QPainter
from PyQt6.QtCore import Qt, QRectF
import math


class DXFViewer(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        # Setup view properties
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        # Color mapping for layers
        self.layer_colors = {}

    def load_dxf(self, filepath):
        """Load and render DXF file"""
        try:
            doc = ezdxf.readfile(filepath)
            msp = doc.modelspace()

            # Clear previous scene
            self.scene.clear()

            # Setup layer colors
            self._setup_layer_colors(doc)

            # Render all entities
            for entity in msp:
                self._render_entity(entity)

            # Fit view to content
            self.fitInView(self.scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

            return True
        except Exception as e:
            print(f"Error loading DXF: {e}")
            return False

    def _setup_layer_colors(self, doc):
        """Setup color mapping for layers"""
        default_colors = [
            QColor(255, 255, 255),  # White
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
        """Get pen for entity based on layer"""
        layer_name = entity.dxf.layer
        color = self.layer_colors.get(layer_name, QColor(255, 255, 255))
        pen = QPen(color)
        pen.setWidth(0)  # Cosmetic pen
        return pen

    def _render_entity(self, entity):
        """Render a single DXF entity"""
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
        """Render LINE entity"""
        start = entity.dxf.start
        end = entity.dxf.end
        pen = self._get_pen(entity)
        self.scene.addLine(start.x, -start.y, end.x, -end.y, pen)

    def _render_lwpolyline(self, entity):
        """Render LWPOLYLINE entity"""
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
        """Render POLYLINE entity"""
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
        """Render CIRCLE entity"""
        center = entity.dxf.center
        radius = entity.dxf.radius
        pen = self._get_pen(entity)

        self.scene.addEllipse(
            center.x - radius, -center.y - radius,
            radius * 2, radius * 2,
            pen
        )

    def _render_arc(self, entity):
        """Render ARC entity"""
        center = entity.dxf.center
        radius = entity.dxf.radius
        start_angle = entity.dxf.start_angle
        end_angle = entity.dxf.end_angle

        pen = self._get_pen(entity)
        path = QPainterPath()

        # Convert angles to radians
        start_rad = math.radians(start_angle)
        end_rad = math.radians(end_angle)

        # Calculate start point
        start_x = center.x + radius * math.cos(start_rad)
        start_y = center.y + radius * math.sin(start_rad)

        path.moveTo(start_x, -start_y)

        # Create arc
        rect = QRectF(center.x - radius, -center.y - radius, radius * 2, radius * 2)
        span_angle = end_angle - start_angle
        if span_angle < 0:
            span_angle += 360

        path.arcTo(rect, -start_angle, -span_angle)
        self.scene.addPath(path, pen)

    def _render_ellipse(self, entity):
        """Render ELLIPSE entity"""
        center = entity.dxf.center
        major_axis = entity.dxf.major_axis
        ratio = entity.dxf.ratio

        pen = self._get_pen(entity)

        # Calculate ellipse dimensions
        major_length = math.sqrt(major_axis.x**2 + major_axis.y**2)
        minor_length = major_length * ratio

        self.scene.addEllipse(
            center.x - major_length, -center.y - minor_length,
            major_length * 2, minor_length * 2,
            pen
        )

    def _render_spline(self, entity):
        """Render SPLINE entity"""
        pen = self._get_pen(entity)
        path = QPainterPath()

        # Get spline points
        points = list(entity.control_points)
        if not points:
            return

        path.moveTo(points[0].x, -points[0].y)
        for point in points[1:]:
            path.lineTo(point.x, -point.y)

        self.scene.addPath(path, pen)

    def _render_text(self, entity):
        """Render TEXT entity"""
        text = entity.dxf.text
        insert = entity.dxf.insert
        height = entity.dxf.height

        pen = self._get_pen(entity)
        text_item = self.scene.addText(text)
        text_item.setPos(insert.x, -insert.y)
        text_item.setDefaultTextColor(pen.color())

        # Scale text based on height
        scale_factor = height / text_item.boundingRect().height()
        text_item.setScale(scale_factor)

    def _render_mtext(self, entity):
        """Render MTEXT entity"""
        text = entity.text
        insert = entity.dxf.insert

        pen = self._get_pen(entity)
        text_item = self.scene.addText(text)
        text_item.setPos(insert.x, -insert.y)
        text_item.setDefaultTextColor(pen.color())

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        zoom_factor = 1.15

        if event.angleDelta().y() > 0:
            self.scale(zoom_factor, zoom_factor)
        else:
            self.scale(1 / zoom_factor, 1 / zoom_factor)
