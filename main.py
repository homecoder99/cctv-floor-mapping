"""Main application for DXF viewer"""
import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QPushButton, QFileDialog, QHBoxLayout, QLabel, QMessageBox
)
from PyQt6.QtCore import Qt
from dxf_viewer import DXFViewer


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DXF 평면도 뷰어")
        self.setGeometry(100, 100, 1200, 800)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Create toolbar
        toolbar = QHBoxLayout()

        # Open button
        self.open_button = QPushButton("DXF 파일 열기")
        self.open_button.clicked.connect(self.open_file)
        toolbar.addWidget(self.open_button)

        # Fit view button
        self.fit_button = QPushButton("화면 맞춤")
        self.fit_button.clicked.connect(self.fit_view)
        toolbar.addWidget(self.fit_button)

        # Status label
        self.status_label = QLabel("파일을 열어주세요")
        toolbar.addWidget(self.status_label)
        toolbar.addStretch()

        # Info label
        info_label = QLabel("마우스: 드래그-이동 | 휠-확대/축소")
        info_label.setStyleSheet("color: gray;")
        toolbar.addWidget(info_label)

        layout.addLayout(toolbar)

        # Create DXF viewer
        self.viewer = DXFViewer()
        layout.addWidget(self.viewer)

        # Load default file if exists
        self.load_default_file()

    def load_default_file(self):
        """Load data/house.dxf if exists"""
        default_path = "data/house.dxf"
        try:
            if self.viewer.load_dxf(default_path):
                self.status_label.setText(f"로드됨: {default_path}")
                self.status_label.setStyleSheet("color: green;")
            else:
                self.status_label.setText("기본 파일을 찾을 수 없습니다")
                self.status_label.setStyleSheet("color: orange;")
        except Exception as e:
            self.status_label.setText("파일을 열어주세요")
            self.status_label.setStyleSheet("color: gray;")

    def open_file(self):
        """Open DXF file dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "DXF 파일 선택",
            "",
            "DXF Files (*.dxf);;All Files (*)"
        )

        if file_path:
            if self.viewer.load_dxf(file_path):
                self.status_label.setText(f"로드됨: {file_path}")
                self.status_label.setStyleSheet("color: green;")
            else:
                self.status_label.setText("파일 로드 실패")
                self.status_label.setStyleSheet("color: red;")
                QMessageBox.critical(
                    self,
                    "오류",
                    "DXF 파일을 로드할 수 없습니다."
                )

    def fit_view(self):
        """Fit view to content"""
        self.viewer.fitInView(
            self.viewer.scene.itemsBoundingRect(),
            Qt.AspectRatioMode.KeepAspectRatio
        )


def main():
    app = QApplication(sys.argv)

    # Set dark theme
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
