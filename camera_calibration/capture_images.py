"""
카메라 캘리브레이션을 위한 체커보드 이미지 캡처 스크립트
필요한 패키지: opencv-python, numpy
설치: pip install opencv-python numpy
"""

import cv2
import os
from datetime import datetime

# RTSP 프로토콜 설정
# 'tcp' 또는 'udp' 선택 (일반적으로 tcp가 더 안정적)
RTSP_TRANSPORT = 'tcp'  # 'tcp' 또는 'udp'

# RTSP 프레임 건너뛰기 설정 (지연 감소용)
# 높은 값 = 더 최신 프레임, 낮은 값 = 더 부드러운 영상
RTSP_FRAME_SKIP = 5  # 0-10 권장 (0은 건너뛰기 없음)

# 체커보드 설정
CHECKERBOARD_SIZE = (9, 6)  # 내부 코너 개수 (가로, 세로)

# 카메라 설정
# 옵션 1: 로컬 웹캠 사용 (기본값)
# CAMERA_SOURCE = 0
CAMERA_SOURCE = "..."

# 옵션 2: RTSP 스트림 사용 (Tapo C200 등)
# RTSP URL 형식: rtsp://username:password@ip_address:port/stream1
# 예시: CAMERA_SOURCE = "rtsp://admin:password@192.168.1.100:554/stream1"
#
# Tapo C200의 경우:
# - stream1: 고화질 스트림 (1080p)
# - stream2: 저화질 스트림 (360p)
# CAMERA_SOURCE = "rtsp://username:password@192.168.x.x:554/stream1"

# 이미지 저장 경로
IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

def capture_checkerboard_images():
    """
    카메라로부터 체커보드 이미지를 캡처하는 함수
    스페이스바를 눌러 이미지 저장, ESC로 종료
    """
    # 카메라 초기화
    print("카메라에 연결 중...")
    print(f"카메라 소스: {CAMERA_SOURCE}")

    # RTSP 스트림인 경우 환경 변수 설정
    if isinstance(CAMERA_SOURCE, str) and CAMERA_SOURCE.startswith('rtsp'):
        print(f"RTSP 프로토콜: {RTSP_TRANSPORT}")
        # FFmpeg 옵션 설정 (OpenCV 백엔드용) - 지연 최소화
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            f"rtsp_transport;{RTSP_TRANSPORT}|"
            "fflags;nobuffer|"  # 버퍼링 비활성화
            "flags;low_delay|"   # 저지연 모드
            "max_delay;0"        # 최대 지연 0
        )

    # VideoCapture 생성 - CAP_FFMPEG 백엔드 명시
    if isinstance(CAMERA_SOURCE, str) and CAMERA_SOURCE.startswith('rtsp'):
        cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(CAMERA_SOURCE)

    # RTSP 스트림 안정성 설정
    if isinstance(CAMERA_SOURCE, str) and CAMERA_SOURCE.startswith('rtsp'):
        # 버퍼 크기 줄이기 (레이턴시 감소)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        print("\nRTSP 스트림 사용 시 확인사항:")
        print("1. RTSP URL이 올바른지 확인")
        print("2. 사용자명과 비밀번호가 정확한지 확인")
        print("3. IP 주소와 포트가 정확한지 확인")
        print("4. 네트워크 연결 상태 확인")
        print("5. 카메라의 RTSP 기능이 활성화되어 있는지 확인")
        print("\n추가 해결 방법:")
        print(f"- 현재 RTSP_TRANSPORT = '{RTSP_TRANSPORT}'")
        print("  파일 상단에서 RTSP_TRANSPORT를 'udp'로 변경해보세요")
        print("- stream1 대신 stream2 (저화질)를 시도해보세요")
        return

    # 첫 프레임 읽기 테스트
    print("첫 프레임 읽는 중...")
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print("첫 프레임을 읽을 수 없습니다. 연결을 확인하세요.")
        cap.release()
        return

    print("연결 성공!")

    # 카메라 해상도 설정 (RTSP의 경우 스트림 자체 해상도 사용)
    if isinstance(CAMERA_SOURCE, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    captured_count = 0

    # RTSP 스트림 사용 여부 확인
    is_rtsp = isinstance(CAMERA_SOURCE, str) and CAMERA_SOURCE.startswith('rtsp')

    # 실제 해상도 가져오기
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("=" * 60)
    print("체커보드 이미지 캡처 프로그램")
    print("=" * 60)
    print(f"체커보드 크기: {CHECKERBOARD_SIZE[0]} x {CHECKERBOARD_SIZE[1]} (내부 코너)")
    print(f"영상 해상도: {actual_width} x {actual_height}")
    if is_rtsp:
        print(f"RTSP 설정:")
        print(f"  - 프로토콜: {RTSP_TRANSPORT}")
        print(f"  - 프레임 건너뛰기: {RTSP_FRAME_SKIP} (지연 감소 모드)")
    print("\n사용법:")
    print("  - 스페이스바: 이미지 저장 (체커보드가 검출되었을 때만)")
    print("  - ESC: 종료")
    print("\n팁:")
    print("  - 체커보드를 다양한 각도로 촬영하세요")
    print("  - 화면 중앙, 모서리 등 다양한 위치에서 촬영하세요")
    print("  - 총 15-20장 이상 촬영을 권장합니다")
    if is_rtsp:
        print("\nRTSP 사용 시:")
        print("  - 영상이 끊기면 RTSP_FRAME_SKIP을 낮추세요")
        print("  - 지연이 크면 RTSP_FRAME_SKIP을 높이세요 (최대 10)")
    print("=" * 60)

    while True:
        # RTSP 스트림의 경우 버퍼를 비워서 최신 프레임 가져오기
        if is_rtsp and RTSP_FRAME_SKIP > 0:
            # 버퍼에 쌓인 오래된 프레임들을 건너뛰기
            for _ in range(RTSP_FRAME_SKIP):
                cap.grab()

        ret, frame = cap.read()

        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        # 체커보드 코너 검출 시도
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_corners, corners = cv2.findChessboardCorners(
            gray,
            CHECKERBOARD_SIZE,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        # 코너가 검출되면 그려서 표시
        display_frame = frame.copy()
        if ret_corners:
            cv2.drawChessboardCorners(display_frame, CHECKERBOARD_SIZE, corners, ret_corners)
            cv2.putText(
                display_frame,
                "Checkerboard detected! Press SPACE to capture",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        else:
            cv2.putText(
                display_frame,
                "No checkerboard detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        # 캡처된 이미지 개수 표시
        cv2.putText(
            display_frame,
            f"Captured: {captured_count} images",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.imshow('Camera Calibration - Capture Images', display_frame)

        key = cv2.waitKey(1) & 0xFF

        # 스페이스바: 이미지 저장
        if key == ord(' '):
            if ret_corners:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"checkerboard_{captured_count:02d}_{timestamp}.jpg"
                filepath = os.path.join(IMAGES_DIR, filename)
                cv2.imwrite(filepath, frame)
                captured_count += 1
                print(f"이미지 저장: {filename}")
            else:
                print("체커보드가 검출되지 않았습니다. 다시 시도해주세요.")

        # ESC: 종료
        elif key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print(f"총 {captured_count}개의 이미지가 캡처되었습니다.")
    print(f"저장 위치: {IMAGES_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    capture_checkerboard_images()
