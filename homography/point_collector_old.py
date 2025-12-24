"""
호모그래피 대응점 수집 GUI 프로그램

영상 프레임과 CAD 도면에서 대응점을 수집하고 호모그래피 행렬을 계산합니다.

필요한 패키지: PyQt6, opencv-python, numpy
"""

import sys
import os
import cv2
import numpy as np
import pickle
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem,
    QMessageBox, QSplitter, QGroupBox, QTextEdit, QLineEdit
)
from PyQt6.QtCore import Qt, QPoint, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor

from homography_utils import HomographyData
from dxf_renderer import render_dxf_to_image


class ImageViewer(QLabel):
    """
    클릭 가능한 이미지 뷰어 위젯
    """
    point_clicked = pyqtSignal(int, int)  # (x, y) 좌표

    def __init__(self, title="Image"):
        super().__init__()
        self.title = title
        self.image = None
        self.pixmap = None
        self.points = []  # [(x, y), ...]
        self.scale = 1.0
        self.camera_matrix = None
        self.dist_coeffs = None

        self.setMinimumSize(400, 300)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("QLabel { background-color: #2b2b2b; border: 2px solid #555; }")
        self.setText(f"{title}\n\n클릭하여 이미지를 로드하세요")
        self.setWordWrap(True)

    def set_camera_calibration(self, camera_matrix, dist_coeffs):
        """
        카메라 캘리브레이션 파라미터를 설정합니다.
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def load_image(self, image_path):
        """
        이미지를 로드합니다. DXF 파일도 지원합니다.
        카메라 캘리브레이션이 설정되어 있으면 왜곡 보정을 적용합니다.
        """
        # DXF 파일인 경우
        if image_path.lower().endswith('.dxf'):
            return self.load_dxf(image_path)

        # 일반 이미지 파일
        self.image = cv2.imread(image_path)
        if self.image is None:
            QMessageBox.warning(self, "오류", "이미지를 로드할 수 없습니다.")
            return False

        # 카메라 캘리브레이션이 있으면 왜곡 보정 적용
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            h, w = self.image.shape[:2]
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
            )
            self.image = cv2.undistort(
                self.image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix
            )
            print(f"왜곡 보정 적용됨 - 이미지 크기: {w}x{h}")

        self.points = []
        self.update_display()
        return True

    def load_dxf(self, dxf_path):
        """
        DXF 파일을 이미지로 렌더링하여 로드합니다.
        """
        try:
            # DXF를 이미지로 렌더링
            print(f"DXF 렌더링 중: {dxf_path}")
            self.image = render_dxf_to_image(dxf_path, output_size=(3000, 3000), background_color=(255, 255, 255))

            if self.image is None:
                QMessageBox.warning(self, "오류", "DXF 파일을 렌더링할 수 없습니다.")
                return False

            self.points = []
            self.update_display()
            print("DXF 렌더링 완료")
            return True

        except Exception as e:
            QMessageBox.critical(self, "오류", f"DXF 로드 실패:\n{str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def set_points(self, points):
        """
        표시할 점들을 설정합니다.
        """
        self.points = points
        self.update_display()

    def update_display(self):
        """
        이미지와 점들을 표시합니다.
        """
        if self.image is None:
            return

        # 이미지 복사
        display_image = self.image.copy()

        # 점들 그리기
        for idx, (x, y) in enumerate(self.points):
            # 점 원으로 표시
            cv2.circle(display_image, (int(x), int(y)), 8, (0, 255, 0), -1)
            cv2.circle(display_image, (int(x), int(y)), 10, (255, 255, 255), 2)

            # 인덱스 번호 표시
            cv2.putText(
                display_image,
                str(idx + 1),
                (int(x) + 15, int(y) - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

        # OpenCV BGR -> RGB 변환
        rgb_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)

        # QPixmap으로 변환
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # 크기 조정
        label_size = self.size()
        scaled_pixmap = QPixmap.fromImage(q_image).scaled(
            label_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.pixmap = scaled_pixmap
        self.setPixmap(scaled_pixmap)

    def mousePressEvent(self, event):
        """
        마우스 클릭 이벤트 처리
        """
        if self.image is None or self.pixmap is None:
            return

        # 클릭 위치를 원본 이미지 좌표로 변환
        click_pos = event.pos()
        pixmap_rect = self.pixmap.rect()

        # pixmap이 label 중앙에 위치하므로 오프셋 계산
        label_rect = self.rect()
        x_offset = (label_rect.width() - pixmap_rect.width()) // 2
        y_offset = (label_rect.height() - pixmap_rect.height()) // 2

        # pixmap 내부 클릭인지 확인
        pixmap_x = click_pos.x() - x_offset
        pixmap_y = click_pos.y() - y_offset

        if pixmap_x < 0 or pixmap_y < 0 or pixmap_x >= pixmap_rect.width() or pixmap_y >= pixmap_rect.height():
            return

        # 픽셀 좌표를 원본 이미지 좌표로 변환
        scale_x = self.image.shape[1] / pixmap_rect.width()
        scale_y = self.image.shape[0] / pixmap_rect.height()

        image_x = int(pixmap_x * scale_x)
        image_y = int(pixmap_y * scale_y)

        # 시그널 발생
        self.point_clicked.emit(image_x, image_y)

    def resizeEvent(self, event):
        """
        크기 변경 이벤트 처리
        """
        super().resizeEvent(event)
        self.update_display()


class PointCollectorWindow(QMainWindow):
    """
    대응점 수집 메인 윈도우
    """

    def __init__(self):
        super().__init__()
        self.homography_data = HomographyData()
        self.current_save_path = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_loaded = False

        self.init_ui()

    def init_ui(self):
        """
        UI 초기화
        """
        self.setWindowTitle("호모그래피 대응점 수집 도구")
        self.setGeometry(100, 100, 1400, 800)

        # 중앙 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 메인 레이아웃
        main_layout = QVBoxLayout(central_widget)

        # 상단: 제목 및 파일 로드 버튼
        top_layout = QHBoxLayout()

        title_label = QLabel("호모그래피 대응점 수집 도구")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        top_layout.addWidget(title_label)

        top_layout.addStretch()

        # 캘리브레이션 로드 버튼
        load_calib_btn = QPushButton("캘리브레이션 로드")
        load_calib_btn.clicked.connect(self.load_calibration)
        load_calib_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; }")
        top_layout.addWidget(load_calib_btn)

        # 캘리브레이션 상태 표시
        self.calib_status_label = QLabel("캘리브레이션: 미로드")
        self.calib_status_label.setStyleSheet("color: #ff9800; font-weight: bold;")
        top_layout.addWidget(self.calib_status_label)

        top_layout.addStretch()

        load_frame_btn = QPushButton("영상 프레임 로드")
        load_frame_btn.clicked.connect(self.load_frame_image)
        top_layout.addWidget(load_frame_btn)

        load_drawing_btn = QPushButton("도면 로드")
        load_drawing_btn.clicked.connect(self.load_drawing_image)
        top_layout.addWidget(load_drawing_btn)

        main_layout.addLayout(top_layout)

        # 중앙: 이미지 뷰어들
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 왼쪽: 영상 프레임
        self.frame_viewer = ImageViewer("영상 프레임")
        self.frame_viewer.point_clicked.connect(self.on_frame_clicked)
        splitter.addWidget(self.frame_viewer)

        # 오른쪽: 도면
        self.drawing_viewer = ImageViewer("CAD 도면")
        self.drawing_viewer.point_clicked.connect(self.on_drawing_clicked)
        splitter.addWidget(self.drawing_viewer)

        splitter.setSizes([700, 700])
        main_layout.addWidget(splitter, stretch=2)

        # 하단: 제어 패널 및 대응점 테이블
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)

        # 제어 패널
        control_group = QGroupBox("제어")
        control_layout = QVBoxLayout()

        # 대응점 관련 버튼
        points_layout = QHBoxLayout()

        self.delete_btn = QPushButton("선택한 점 삭제")
        self.delete_btn.clicked.connect(self.delete_selected_point)
        points_layout.addWidget(self.delete_btn)

        self.clear_btn = QPushButton("모든 점 삭제")
        self.clear_btn.clicked.connect(self.clear_all_points)
        points_layout.addWidget(self.clear_btn)

        control_layout.addLayout(points_layout)

        # 호모그래피 계산 버튼
        self.calc_btn = QPushButton("호모그래피 계산")
        self.calc_btn.clicked.connect(self.calculate_homography)
        self.calc_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }")
        control_layout.addWidget(self.calc_btn)

        # 저장/로드 버튼
        save_load_layout = QHBoxLayout()

        self.save_btn = QPushButton("저장")
        self.save_btn.clicked.connect(self.save_data)
        save_load_layout.addWidget(self.save_btn)

        self.load_btn = QPushButton("불러오기")
        self.load_btn.clicked.connect(self.load_data)
        save_load_layout.addWidget(self.load_btn)

        control_layout.addLayout(save_load_layout)

        # 결과 표시
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(150)
        self.result_text.setPlaceholderText("호모그래피 계산 결과가 여기에 표시됩니다...")
        control_layout.addWidget(QLabel("결과:"))
        control_layout.addWidget(self.result_text)

        control_group.setLayout(control_layout)
        bottom_splitter.addWidget(control_group)

        # 대응점 테이블
        table_group = QGroupBox("대응점 목록")
        table_layout = QVBoxLayout()

        self.points_table = QTableWidget()
        self.points_table.setColumnCount(5)
        self.points_table.setHorizontalHeaderLabels(["#", "영상 X", "영상 Y", "도면 X", "도면 Y"])
        self.points_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table_layout.addWidget(self.points_table)

        table_group.setLayout(table_layout)
        bottom_splitter.addWidget(table_group)

        bottom_splitter.setSizes([400, 800])
        main_layout.addWidget(bottom_splitter, stretch=1)

        # 상태바
        self.statusBar().showMessage("준비")

    def load_frame_image(self):
        """
        영상 프레임 이미지를 로드합니다.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "영상 프레임 선택",
            "../camera_calibration/images",
            "이미지 파일 (*.jpg *.jpeg *.png *.bmp)"
        )

        if file_path:
            if self.frame_viewer.load_image(file_path):
                self.homography_data.metadata['image_source'] = file_path
                self.statusBar().showMessage(f"영상 프레임 로드: {os.path.basename(file_path)}")

    def load_drawing_image(self):
        """
        도면 이미지를 로드합니다. DXF 파일도 지원합니다.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "도면 선택",
            "../data",
            "이미지 및 CAD 파일 (*.jpg *.jpeg *.png *.bmp *.dxf);;DXF 파일 (*.dxf);;이미지 파일 (*.jpg *.jpeg *.png *.bmp)"
        )

        if file_path:
            # DXF 파일이면 렌더링 진행
            if file_path.lower().endswith('.dxf'):
                # 진행 상태 표시
                self.statusBar().showMessage("DXF 파일 렌더링 중... (시간이 걸릴 수 있습니다)")
                QApplication.processEvents()  # UI 업데이트

            if self.drawing_viewer.load_image(file_path):
                self.homography_data.metadata['drawing_source'] = file_path
                self.statusBar().showMessage(f"도면 로드: {os.path.basename(file_path)}")
            else:
                self.statusBar().showMessage("도면 로드 실패")

    def on_frame_clicked(self, x, y):
        """
        영상 프레임이 클릭되었을 때
        """
        # 대응하는 도면 점이 있는지 확인
        num_frame_points = len([p for p in self.homography_data.image_points])
        num_drawing_points = len([p for p in self.homography_data.drawing_points])

        if num_drawing_points > num_frame_points:
            # 도면에서 먼저 점이 찍혀있으면, 프레임 점 추가
            drawing_point = self.homography_data.drawing_points[num_frame_points]
            self.homography_data.add_point_pair((x, y), drawing_point)
            self.update_views()
            self.statusBar().showMessage(f"대응점 쌍 {num_frame_points + 1} 추가됨")
        else:
            # 임시로 프레임 점만 저장 (도면 점 대기 중)
            self.homography_data.image_points.append((x, y))
            self.frame_viewer.set_points(self.homography_data.image_points)
            self.statusBar().showMessage(f"영상 점 {num_frame_points + 1} 추가 - 도면에서 대응점을 클릭하세요")

    def on_drawing_clicked(self, x, y):
        """
        도면이 클릭되었을 때
        """
        num_frame_points = len([p for p in self.homography_data.image_points])
        num_drawing_points = len([p for p in self.homography_data.drawing_points])

        if num_frame_points > num_drawing_points:
            # 프레임에서 먼저 점이 찍혀있으면, 도면 점 추가하여 쌍 완성
            frame_point = self.homography_data.image_points[num_drawing_points]

            # 기존의 임시 점 제거하고 정식 쌍으로 추가
            if len(self.homography_data.image_points) > len(self.homography_data.drawing_points):
                self.homography_data.image_points.pop()

            self.homography_data.add_point_pair(frame_point, (x, y))
            self.update_views()
            self.statusBar().showMessage(f"대응점 쌍 {num_drawing_points + 1} 추가됨")
        else:
            # 임시로 도면 점만 저장 (프레임 점 대기 중)
            self.homography_data.drawing_points.append((x, y))
            self.drawing_viewer.set_points(self.homography_data.drawing_points)
            self.statusBar().showMessage(f"도면 점 {num_drawing_points + 1} 추가 - 영상에서 대응점을 클릭하세요")

    def update_views(self):
        """
        모든 뷰를 업데이트합니다.
        """
        # 이미지 뷰어 업데이트
        self.frame_viewer.set_points(self.homography_data.image_points)
        self.drawing_viewer.set_points(self.homography_data.drawing_points)

        # 테이블 업데이트
        self.points_table.setRowCount(len(self.homography_data.image_points))

        for idx, (img_pt, draw_pt) in enumerate(zip(
            self.homography_data.image_points,
            self.homography_data.drawing_points
        )):
            self.points_table.setItem(idx, 0, QTableWidgetItem(str(idx + 1)))
            self.points_table.setItem(idx, 1, QTableWidgetItem(f"{img_pt[0]:.1f}"))
            self.points_table.setItem(idx, 2, QTableWidgetItem(f"{img_pt[1]:.1f}"))
            self.points_table.setItem(idx, 3, QTableWidgetItem(f"{draw_pt[0]:.1f}"))
            self.points_table.setItem(idx, 4, QTableWidgetItem(f"{draw_pt[1]:.1f}"))

    def delete_selected_point(self):
        """
        선택한 대응점을 삭제합니다.
        """
        selected_rows = self.points_table.selectedIndexes()
        if not selected_rows:
            QMessageBox.warning(self, "경고", "삭제할 점을 선택하세요.")
            return

        row = selected_rows[0].row()
        self.homography_data.remove_point_pair(row)
        self.update_views()
        self.statusBar().showMessage(f"대응점 {row + 1} 삭제됨")

    def clear_all_points(self):
        """
        모든 대응점을 삭제합니다.
        """
        reply = QMessageBox.question(
            self,
            "확인",
            "모든 대응점을 삭제하시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.homography_data.clear_all_points()
            self.update_views()
            self.result_text.clear()
            self.statusBar().showMessage("모든 대응점 삭제됨")

    def calculate_homography(self):
        """
        호모그래피 행렬을 계산합니다.
        """
        success, message = self.homography_data.calculate_homography()

        if success:
            # 재투영 오차 계산
            error_stats = self.homography_data.calculate_reprojection_error()

            result_text = f"=== 호모그래피 계산 완료 ===\n\n"
            result_text += f"상태: {message}\n"
            result_text += f"대응점 개수: {len(self.homography_data.image_points)}개\n\n"

            if error_stats:
                result_text += f"재투영 오차:\n"
                result_text += f"  - 평균: {error_stats['mean']:.4f} pixels\n"
                result_text += f"  - 최대: {error_stats['max']:.4f} pixels\n"
                result_text += f"  - 중간값: {error_stats['median']:.4f} pixels\n"
                result_text += f"  - 표준편차: {error_stats['std']:.4f} pixels\n\n"

            result_text += f"호모그래피 행렬:\n"
            H = self.homography_data.homography_matrix
            for row in H:
                result_text += f"  [{row[0]:12.6f}  {row[1]:12.6f}  {row[2]:12.6f}]\n"

            self.result_text.setText(result_text)
            self.statusBar().showMessage("호모그래피 계산 완료")

            QMessageBox.information(self, "성공", message)
        else:
            self.result_text.setText(f"오류: {message}")
            self.statusBar().showMessage("호모그래피 계산 실패")
            QMessageBox.warning(self, "오류", message)

    def save_data(self):
        """
        대응점과 호모그래피 데이터를 저장합니다.
        """
        if len(self.homography_data.image_points) == 0:
            QMessageBox.warning(self, "경고", "저장할 데이터가 없습니다.")
            return

        default_filename = f"homography_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "데이터 저장",
            os.path.join("data", default_filename),
            "JSON 파일 (*.json)"
        )

        if file_path:
            try:
                self.homography_data.save_to_file(file_path)
                self.current_save_path = file_path
                self.statusBar().showMessage(f"저장 완료: {os.path.basename(file_path)}")
                QMessageBox.information(self, "성공", "데이터가 저장되었습니다.")
            except Exception as e:
                QMessageBox.critical(self, "오류", f"저장 실패: {str(e)}")

    def load_data(self):
        """
        저장된 대응점과 호모그래피 데이터를 로드합니다.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "데이터 불러오기",
            "data",
            "JSON 파일 (*.json)"
        )

        if file_path:
            if self.homography_data.load_from_file(file_path):
                self.current_save_path = file_path
                self.update_views()

                # 결과 표시
                if self.homography_data.homography_matrix is not None:
                    self.calculate_homography()  # 결과 재표시

                self.statusBar().showMessage(f"불러오기 완료: {os.path.basename(file_path)}")
                QMessageBox.information(self, "성공", "데이터가 로드되었습니다.")
            else:
                QMessageBox.critical(self, "오류", "파일 로드에 실패했습니다.")

    def load_calibration(self):
        """
        카메라 캘리브레이션 데이터를 로드합니다.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "캘리브레이션 파일 선택",
            "../camera_calibration/output",
            "Calibration 파일 (*.pkl);;모든 파일 (*)"
        )

        if file_path:
            try:
                with open(file_path, 'rb') as f:
                    calib_data = pickle.load(f)

                self.camera_matrix = calib_data['camera_matrix']
                self.dist_coeffs = calib_data['dist_coeffs']
                self.calibration_loaded = True

                # 영상 프레임 뷰어에 캘리브레이션 설정
                self.frame_viewer.set_camera_calibration(self.camera_matrix, self.dist_coeffs)

                # UI 업데이트
                self.calib_status_label.setText("캘리브레이션: 로드됨 ✓")
                self.calib_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")

                # 결과 표시
                fx = self.camera_matrix[0, 0]
                fy = self.camera_matrix[1, 1]
                cx = self.camera_matrix[0, 2]
                cy = self.camera_matrix[1, 2]

                info_text = f"카메라 캘리브레이션 로드 완료\n\n"
                info_text += f"내부 파라미터:\n"
                info_text += f"  fx = {fx:.2f}\n"
                info_text += f"  fy = {fy:.2f}\n"
                info_text += f"  cx = {cx:.2f}\n"
                info_text += f"  cy = {cy:.2f}\n\n"
                info_text += f"왜곡 계수:\n"
                info_text += f"  k1 = {self.dist_coeffs[0][0]:.6f}\n"
                info_text += f"  k2 = {self.dist_coeffs[0][1]:.6f}\n"
                info_text += f"  p1 = {self.dist_coeffs[0][2]:.6f}\n"
                info_text += f"  p2 = {self.dist_coeffs[0][3]:.6f}\n"
                info_text += f"  k3 = {self.dist_coeffs[0][4]:.6f}\n\n"
                info_text += "영상 프레임 로드 시 자동으로 왜곡 보정이 적용됩니다."

                self.statusBar().showMessage(f"캘리브레이션 로드: {os.path.basename(file_path)}")
                QMessageBox.information(self, "성공", info_text)

            except Exception as e:
                QMessageBox.critical(self, "오류", f"캘리브레이션 로드 실패:\n{str(e)}")
                import traceback
                traceback.print_exc()


def main():
    """
    프로그램 실행
    """
    app = QApplication(sys.argv)

    # 다크 테마 적용 (선택사항)
    app.setStyle("Fusion")

    window = PointCollectorWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
