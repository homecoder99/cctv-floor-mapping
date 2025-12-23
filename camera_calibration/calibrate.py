"""
카메라 캘리브레이션 수행 스크립트
저장된 체커보드 이미지들로부터 카메라 내부 파라미터를 계산합니다.

필요한 패키지: opencv-python, numpy
설치: pip install opencv-python numpy
"""

import cv2
import numpy as np
import os
import glob
import pickle

# 체커보드 설정
CHECKERBOARD_SIZE = (9, 6)  # 내부 코너 개수 (가로, 세로)
SQUARE_SIZE = 25.0  # 체커보드 정사각형 한 변의 길이 (mm) - 측정 후 수정하세요!

# 경로 설정
SCRIPT_DIR = os.path.dirname(__file__)
IMAGES_DIR = os.path.join(SCRIPT_DIR, 'images')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def prepare_object_points():
    """
    3D 공간에서 체커보드 코너의 좌표를 준비합니다.
    Z=0인 평면에 체커보드가 놓여있다고 가정합니다.
    """
    objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE
    return objp


def find_checkerboard_corners(images_path):
    """
    이미지들에서 체커보드 코너를 검출합니다.

    Returns:
        objpoints: 3D 공간에서의 체커보드 포인트 리스트
        imgpoints: 2D 이미지 평면에서의 체커보드 포인트 리스트
        image_size: 이미지 크기 (width, height)
    """
    # 3D 포인트 준비
    objp = prepare_object_points()

    # 3D 포인트와 2D 포인트를 저장할 배열
    objpoints = []  # 3D 포인트
    imgpoints = []  # 2D 포인트

    # 이미지 파일 목록 가져오기
    image_files = glob.glob(os.path.join(images_path, '*.jpg'))
    image_files.extend(glob.glob(os.path.join(images_path, '*.png')))

    if not image_files:
        raise FileNotFoundError(f"'{images_path}' 디렉토리에 이미지가 없습니다.")

    print("=" * 60)
    print(f"총 {len(image_files)}개의 이미지를 찾았습니다.")
    print("=" * 60)

    successful_images = []
    image_size = None

    for idx, fname in enumerate(image_files, 1):
        img = cv2.imread(fname)
        if img is None:
            print(f"[{idx}/{len(image_files)}] 읽기 실패: {os.path.basename(fname)}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = gray.shape[::-1]

        # 체커보드 코너 찾기
        ret, corners = cv2.findChessboardCorners(
            gray,
            CHECKERBOARD_SIZE,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            # 코너 위치를 서브픽셀 단위로 정밀화
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints.append(corners_refined)
            successful_images.append(fname)

            print(f"[{idx}/{len(image_files)}] ✓ 성공: {os.path.basename(fname)}")

            # 코너가 그려진 이미지 저장 (선택사항)
            img_with_corners = cv2.drawChessboardCorners(img, CHECKERBOARD_SIZE, corners_refined, ret)
            output_path = os.path.join(
                OUTPUT_DIR,
                f"corners_{os.path.basename(fname)}"
            )
            cv2.imwrite(output_path, img_with_corners)
        else:
            print(f"[{idx}/{len(image_files)}] ✗ 실패: {os.path.basename(fname)}")

    print("=" * 60)
    print(f"캘리브레이션에 사용할 이미지: {len(successful_images)}/{len(image_files)}")
    print("=" * 60)

    if len(successful_images) < 3:
        raise ValueError("캘리브레이션에 최소 3개 이상의 이미지가 필요합니다.")

    return objpoints, imgpoints, image_size


def calibrate_camera(objpoints, imgpoints, image_size):
    """
    카메라 캘리브레이션을 수행하여 내부 파라미터를 계산합니다.

    Returns:
        camera_matrix: 카메라 내부 파라미터 행렬 (3x3)
        dist_coeffs: 왜곡 계수 (5x1)
        rvecs: 회전 벡터들
        tvecs: 이동 벡터들
        rms_error: RMS 재투영 오차
    """
    print("\n카메라 캘리브레이션을 수행 중...")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None
    )

    return camera_matrix, dist_coeffs, rvecs, tvecs, ret


def calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs):
    """
    재투영 오차를 계산합니다.
    """
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    return mean_error / len(objpoints)


def save_calibration_results(camera_matrix, dist_coeffs, rms_error, mean_error):
    """
    캘리브레이션 결과를 저장합니다.
    """
    # 결과를 딕셔너리로 저장
    calibration_data = {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'rms_error': rms_error,
        'mean_reprojection_error': mean_error,
        'checkerboard_size': CHECKERBOARD_SIZE,
        'square_size': SQUARE_SIZE
    }

    # Pickle 파일로 저장
    pickle_path = os.path.join(OUTPUT_DIR, 'calibration_data.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(calibration_data, f)

    # 텍스트 파일로도 저장
    txt_path = os.path.join(OUTPUT_DIR, 'calibration_results.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("카메라 캘리브레이션 결과\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"체커보드 크기: {CHECKERBOARD_SIZE[0]} x {CHECKERBOARD_SIZE[1]} (내부 코너)\n")
        f.write(f"정사각형 크기: {SQUARE_SIZE} mm\n\n")

        f.write("카메라 내부 파라미터 행렬 (Camera Matrix):\n")
        f.write(f"{camera_matrix}\n\n")

        f.write("왜곡 계수 (Distortion Coefficients):\n")
        f.write(f"{dist_coeffs.ravel()}\n\n")

        f.write(f"RMS 재투영 오차: {rms_error:.6f} pixels\n")
        f.write(f"평균 재투영 오차: {mean_error:.6f} pixels\n\n")

        # 주요 파라미터 추출
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]

        f.write("주요 파라미터:\n")
        f.write(f"  - 초점 거리 (fx): {fx:.2f} pixels\n")
        f.write(f"  - 초점 거리 (fy): {fy:.2f} pixels\n")
        f.write(f"  - 주점 (cx): {cx:.2f} pixels\n")
        f.write(f"  - 주점 (cy): {cy:.2f} pixels\n\n")

        k1, k2, p1, p2, k3 = dist_coeffs.ravel()
        f.write("왜곡 계수 상세:\n")
        f.write(f"  - k1 (방사 왜곡): {k1:.6f}\n")
        f.write(f"  - k2 (방사 왜곡): {k2:.6f}\n")
        f.write(f"  - p1 (접선 왜곡): {p1:.6f}\n")
        f.write(f"  - p2 (접선 왜곡): {p2:.6f}\n")
        f.write(f"  - k3 (방사 왜곡): {k3:.6f}\n")

    print(f"\n결과 저장 완료:")
    print(f"  - {pickle_path}")
    print(f"  - {txt_path}")


def print_calibration_results(camera_matrix, dist_coeffs, rms_error, mean_error):
    """
    캘리브레이션 결과를 화면에 출력합니다.
    """
    print("\n" + "=" * 60)
    print("카메라 캘리브레이션 완료!")
    print("=" * 60)

    print("\n카메라 내부 파라미터 행렬 (Camera Matrix):")
    print(camera_matrix)

    print("\n왜곡 계수 (Distortion Coefficients):")
    print(dist_coeffs.ravel())

    print(f"\nRMS 재투영 오차: {rms_error:.6f} pixels")
    print(f"평균 재투영 오차: {mean_error:.6f} pixels")

    # 주요 파라미터 추출
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    print("\n주요 파라미터:")
    print(f"  - 초점 거리 (fx): {fx:.2f} pixels")
    print(f"  - 초점 거리 (fy): {fy:.2f} pixels")
    print(f"  - 주점 (cx): {cx:.2f} pixels")
    print(f"  - 주점 (cy): {cy:.2f} pixels")

    print("=" * 60)


def main():
    """
    메인 함수
    """
    print("\n" + "=" * 60)
    print("카메라 캘리브레이션 프로그램")
    print("=" * 60)
    print(f"체커보드 크기: {CHECKERBOARD_SIZE[0]} x {CHECKERBOARD_SIZE[1]} (내부 코너)")
    print(f"정사각형 크기: {SQUARE_SIZE} mm")
    print(f"이미지 디렉토리: {IMAGES_DIR}")
    print("=" * 60)

    try:
        # 1. 체커보드 코너 검출
        objpoints, imgpoints, image_size = find_checkerboard_corners(IMAGES_DIR)

        # 2. 카메라 캘리브레이션 수행
        camera_matrix, dist_coeffs, rvecs, tvecs, rms_error = calibrate_camera(
            objpoints, imgpoints, image_size
        )

        # 3. 재투영 오차 계산
        mean_error = calculate_reprojection_error(
            objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs
        )

        # 4. 결과 출력
        print_calibration_results(camera_matrix, dist_coeffs, rms_error, mean_error)

        # 5. 결과 저장
        save_calibration_results(camera_matrix, dist_coeffs, rms_error, mean_error)

    except Exception as e:
        print(f"\n오류 발생: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
