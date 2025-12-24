"""
카메라/비디오에서 프레임 캡처

RTSP 스트림, 비디오 파일, 웹캠에서 프레임을 캡처하여 이미지로 저장합니다.
왜곡 보정을 선택적으로 적용할 수 있습니다.
"""

import cv2
import numpy as np
import pickle
import os
import sys
from datetime import datetime
import argparse


def capture_frame(
    video_source,
    output_path=None,
    calibration_path=None,
    skip_frames=0,
    preview=True
):
    """
    비디오 소스에서 프레임 캡처

    Args:
        video_source: RTSP URL, 비디오 파일 경로, 또는 웹캠 번호
        output_path: 출력 이미지 경로 (None이면 자동 생성)
        calibration_path: 카메라 캘리브레이션 파일 경로 (None이면 보정 안 함)
        skip_frames: 캡처 전 스킵할 프레임 수 (RTSP 지연 감소용)
        preview: 미리보기 표시 여부
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
        if not os.path.exists(video_source):
            print(f"오류: 파일을 찾을 수 없습니다: {video_source}")
            return False
        print(f"비디오 파일 열기: {video_source}")

    # 캘리브레이션 로드 (선택)
    camera_matrix = None
    dist_coeffs = None

    if calibration_path and os.path.exists(calibration_path):
        print(f"캘리브레이션 로드: {calibration_path}")
        try:
            with open(calibration_path, 'rb') as f:
                calib_data = pickle.load(f)
            camera_matrix = calib_data['camera_matrix']
            dist_coeffs = calib_data['dist_coeffs']
            print("왜곡 보정 활성화")
        except Exception as e:
            print(f"경고: 캘리브레이션 로드 실패: {e}")
            print("왜곡 보정 없이 진행")

    # 비디오 캡처 열기
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("오류: 비디오 소스 열기 실패")
        return False

    print("비디오 소스 열기 성공")

    try:
        # 프레임 스킵 (RTSP 버퍼 비우기)
        if skip_frames > 0:
            print(f"{skip_frames}개 프레임 스킵 중...")
            for _ in range(skip_frames):
                cap.grab()

        print(f"연결 성공!")

        # 미리보기 모드: 라이브 스트림 표시
        if preview:
            print("\n라이브 미리보기:")
            print("  - 's' 키: 현재 프레임 저장")
            print("  - 'q' 키: 취소")
            print("  - ESC 키: 취소")

            frame_to_save = None

            while True:
                # 프레임 읽기
                ret, frame = cap.read()
                if not ret:
                    print("\n프레임 읽기 실패")
                    break

                # 왜곡 보정 (선택)
                if camera_matrix is not None and dist_coeffs is not None:
                    h, w = frame.shape[:2]
                    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
                    )
                    frame = cv2.undistort(
                        frame, camera_matrix, dist_coeffs, None, new_camera_matrix
                    )

                # 라이브 스트림 표시
                display_frame = frame.copy()
                cv2.putText(
                    display_frame,
                    "Press 's' to save, 'q' to quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2
                )
                cv2.imshow('Live Preview - Press S to Capture', display_frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('s'):
                    frame_to_save = frame.copy()
                    print(f"\n프레임 캡처됨: {frame.shape[1]}x{frame.shape[0]}")
                    break
                elif key == ord('q') or key == 27:  # ESC
                    print("\n취소됨")
                    cv2.destroyAllWindows()
                    return False

            cv2.destroyAllWindows()

            if frame_to_save is None:
                print("캡처된 프레임이 없습니다")
                return False

            frame = frame_to_save

        else:
            # 미리보기 없이 바로 캡처
            ret, frame = cap.read()
            if not ret:
                print("오류: 프레임 읽기 실패")
                return False

            print(f"프레임 캡처 성공: {frame.shape[1]}x{frame.shape[0]}")

            # 왜곡 보정 (선택)
            if camera_matrix is not None and dist_coeffs is not None:
                h, w = frame.shape[:2]
                new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                    camera_matrix, dist_coeffs, (w, h), 1, (w, h)
                )
                frame = cv2.undistort(
                    frame, camera_matrix, dist_coeffs, None, new_camera_matrix
                )
                print("왜곡 보정 적용됨")

        # 출력 경로 생성
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"captured_frame_{timestamp}.jpg"

        # 디렉토리 생성
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 이미지 저장
        cv2.imwrite(output_path, frame)
        print(f"\n저장 완료: {output_path}")

        # 파일 크기 확인
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / 1024  # KB
            print(f"파일 크기: {file_size:.2f} KB")

        return True

    except KeyboardInterrupt:
        print("\n키보드 인터럽트")
        return False

    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="카메라/비디오에서 프레임 캡처"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='입력 소스 (RTSP URL, 비디오 파일 경로, 또는 웹캠 번호)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='출력 이미지 경로 (기본: captured_frame_YYYYMMDD_HHMMSS.jpg)'
    )
    parser.add_argument(
        '--calibration',
        type=str,
        default=None,
        help='카메라 캘리브레이션 파일 경로 (.pkl)'
    )
    parser.add_argument(
        '--skip-frames',
        type=int,
        default=10,
        help='캡처 전 스킵할 프레임 수 (RTSP 지연 감소용, 기본: 10)'
    )
    parser.add_argument(
        '--no-preview',
        action='store_true',
        help='미리보기 없이 바로 저장'
    )

    args = parser.parse_args()

    # 프레임 캡처
    success = capture_frame(
        video_source=args.input,
        output_path=args.output,
        calibration_path=args.calibration,
        skip_frames=args.skip_frames,
        preview=not args.no_preview
    )

    if success:
        print("\n✓ 캡처 성공")
        sys.exit(0)
    else:
        print("\n✗ 캡처 실패")
        sys.exit(1)


if __name__ == "__main__":
    main()
