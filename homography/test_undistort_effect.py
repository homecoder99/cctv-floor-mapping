"""
테스트 3: undistort 유무에 따른 호모그래피 오차 비교

왜곡 보정이 호모그래피 정확도에 미치는 영향을 측정합니다.
"""

import json
import numpy as np
import cv2
import pickle
from pathlib import Path


def calculate_reprojection_error(H, src_points, dst_points):
    """재투영 오차 계산"""
    # homogeneous 좌표로 변환
    src_h = np.hstack([src_points, np.ones((len(src_points), 1))])

    # 변환
    transformed_h = (H @ src_h.T).T
    transformed = transformed_h[:, :2] / transformed_h[:, 2:3]

    # 오차 계산
    errors = np.linalg.norm(transformed - dst_points, axis=1)

    return {
        'mean': float(np.mean(errors)),
        'max': float(np.max(errors)),
        'std': float(np.std(errors)),
        'median': float(np.median(errors)),
        'errors': errors.tolist()
    }


def test_undistort_effect(image_path, calibration_path, homography_json_path):
    """
    undistort 유무에 따른 호모그래피 정확도 비교

    Args:
        image_path: 원본 이미지 경로
        calibration_path: 캘리브레이션 파일 경로
        homography_json_path: 호모그래피 JSON 파일 경로
    """
    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        print(f"오류: 이미지를 읽을 수 없음: {image_path}")
        return

    h, w = img.shape[:2]
    print(f"원본 이미지: {image_path}")
    print(f"해상도: {w}x{h}")
    print()

    # 캘리브레이션 로드
    if not Path(calibration_path).exists():
        print(f"오류: 캘리브레이션 파일 없음: {calibration_path}")
        return

    with open(calibration_path, 'rb') as f:
        calib_data = pickle.load(f)

    camera_matrix = calib_data['camera_matrix']
    dist_coeffs = calib_data['dist_coeffs']

    print(f"캘리브레이션: {calibration_path}")
    print(f"카메라 매트릭스:\n{camera_matrix}")
    print(f"왜곡 계수: {dist_coeffs.ravel()}")
    print()

    # 왜곡 보정 적용
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    img_undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    print(f"왜곡 보정 완료: {w}x{h}")
    print()

    # 호모그래피 데이터 로드
    with open(homography_json_path, 'r') as f:
        data = json.load(f)

    drawing_points = np.array(data['drawing_points'], dtype=np.float32)

    print(f"호모그래피 JSON: {homography_json_path}")
    print(f"대응점 개수: {len(drawing_points)}")
    print()

    # ========================================
    # 시나리오 1: 원본 이미지에서 점 선택
    # ========================================
    print("=" * 60)
    print("시나리오 1: 원본 이미지 (왜곡 있음)")
    print("=" * 60)
    print("왼쪽 이미지에서 대응점을 클릭하세요 (순서대로)")
    print(f"총 {len(drawing_points)}개 점 필요")
    print()

    # 점 선택 UI
    points_original = []
    current_img = img.copy()

    def mouse_callback_original(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points_original.append([x, y])
            # 점 표시
            cv2.circle(current_img, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(current_img, f"{len(points_original)}", (x+10, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('Original Image (with distortion)', current_img)

    cv2.namedWindow('Original Image (with distortion)')
    cv2.setMouseCallback('Original Image (with distortion)', mouse_callback_original)
    cv2.imshow('Original Image (with distortion)', current_img)

    print("이미지 창에서 대응점 클릭 (모두 선택 후 'Enter')")
    while len(points_original) < len(drawing_points):
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            print("취소됨")
            cv2.destroyAllWindows()
            return
        elif key == 13:  # Enter
            break

    cv2.destroyAllWindows()

    if len(points_original) != len(drawing_points):
        print(f"오류: {len(drawing_points)}개 점이 필요하지만 {len(points_original)}개만 선택됨")
        return

    points_original = np.array(points_original, dtype=np.float32)

    # 원본 이미지에서 H 계산
    H_original, mask = cv2.findHomography(points_original, drawing_points, cv2.RANSAC, 5.0)

    if H_original is None:
        print("오류: 호모그래피 계산 실패 (원본)")
        return

    # 오차 계산
    stats_original = calculate_reprojection_error(H_original, points_original, drawing_points)

    print("\n결과 (원본 이미지):")
    print(f"  평균 오차: {stats_original['mean']:.2f} pixels")
    print(f"  최대 오차: {stats_original['max']:.2f} pixels")
    print(f"  중앙값: {stats_original['median']:.2f} pixels")
    print(f"  Inliers: {np.sum(mask)}/{len(mask)}")
    print()

    # ========================================
    # 시나리오 2: 왜곡 보정된 이미지에서 점 선택
    # ========================================
    print("=" * 60)
    print("시나리오 2: 왜곡 보정된 이미지")
    print("=" * 60)
    print("왼쪽 이미지에서 대응점을 클릭하세요 (순서대로)")
    print(f"총 {len(drawing_points)}개 점 필요")
    print()

    points_undistorted = []
    current_img_undist = img_undistorted.copy()

    def mouse_callback_undistorted(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points_undistorted.append([x, y])
            cv2.circle(current_img_undist, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(current_img_undist, f"{len(points_undistorted)}", (x+10, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('Undistorted Image', current_img_undist)

    cv2.namedWindow('Undistorted Image')
    cv2.setMouseCallback('Undistorted Image', mouse_callback_undistorted)
    cv2.imshow('Undistorted Image', current_img_undist)

    print("이미지 창에서 대응점 클릭 (모두 선택 후 'Enter')")
    while len(points_undistorted) < len(drawing_points):
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            print("취소됨")
            cv2.destroyAllWindows()
            return
        elif key == 13:  # Enter
            break

    cv2.destroyAllWindows()

    if len(points_undistorted) != len(drawing_points):
        print(f"오류: {len(drawing_points)}개 점이 필요하지만 {len(points_undistorted)}개만 선택됨")
        return

    points_undistorted = np.array(points_undistorted, dtype=np.float32)

    # 왜곡 보정 이미지에서 H 계산
    H_undistorted, mask_undist = cv2.findHomography(points_undistorted, drawing_points, cv2.RANSAC, 5.0)

    if H_undistorted is None:
        print("오류: 호모그래피 계산 실패 (왜곡 보정)")
        return

    # 오차 계산
    stats_undistorted = calculate_reprojection_error(H_undistorted, points_undistorted, drawing_points)

    print("\n결과 (왜곡 보정 이미지):")
    print(f"  평균 오차: {stats_undistorted['mean']:.2f} pixels")
    print(f"  최대 오차: {stats_undistorted['max']:.2f} pixels")
    print(f"  중앙값: {stats_undistorted['median']:.2f} pixels")
    print(f"  Inliers: {np.sum(mask_undist)}/{len(mask_undist)}")
    print()

    # ========================================
    # 비교 및 판별
    # ========================================
    print("=" * 60)
    print("비교 결과")
    print("=" * 60)

    improvement = stats_original['mean'] - stats_undistorted['mean']
    improvement_pct = (improvement / stats_original['mean']) * 100 if stats_original['mean'] > 0 else 0

    print(f"원본 이미지 평균 오차:        {stats_original['mean']:>8.2f} pixels")
    print(f"왜곡 보정 이미지 평균 오차:   {stats_undistorted['mean']:>8.2f} pixels")
    print(f"개선량:                      {improvement:>8.2f} pixels ({improvement_pct:+.1f}%)")
    print()

    # 판별
    print("=== 판별 ===")
    if improvement > 20:
        print(f"✓ 왜곡 보정으로 오차가 {improvement:.1f} pixels 감소")
        print("  → 렌즈 왜곡이 호모그래피 정확도에 큰 영향을 미침")
        print("  → 반드시 왜곡 보정된 이미지로 대응점을 찍어야 함")
    elif improvement > 5:
        print(f"△ 왜곡 보정으로 오차가 {improvement:.1f} pixels 소폭 감소")
        print("  → 렌즈 왜곡이 약간 영향을 미침")
        print("  → 왜곡 보정 권장")
    elif abs(improvement) <= 5:
        print(f"- 왜곡 보정 전후 오차 차이가 미미함 ({abs(improvement):.1f} pixels)")
        print("  → 렌즈 왜곡이 호모그래피에 큰 영향을 주지 않음")
        print("  → 왜곡 보정 선택사항")
    else:
        print(f"✗ 왜곡 보정 후 오차가 오히려 증가 ({improvement:.1f} pixels)")
        print("  → 가능한 원인:")
        print("    1. 캘리브레이션이 잘못됨")
        print("    2. 점 선택이 부정확함")
        print("    3. 이 카메라는 왜곡 보정이 필요 없음")

    if stats_undistorted['mean'] < 5.0:
        print("\n✓ 왜곡 보정 후 오차 < 5 pixels → 정상 범위")
    else:
        print(f"\n✗ 왜곡 보정 후에도 오차 > 5 pixels ({stats_undistorted['mean']:.2f})")
        print("  → 왜곡 외에 다른 문제가 있음 (점 입력 오류, 해상도 불일치 등)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="테스트 3: undistort 유무에 따른 호모그래피 오차 비교"
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='원본 이미지 경로'
    )
    parser.add_argument(
        '--calibration',
        type=str,
        default='camera_calibration/output/calibration_data.pkl',
        help='캘리브레이션 파일 경로'
    )
    parser.add_argument(
        '--homography',
        type=str,
        default='homography/data/homography_20251224_162558.json',
        help='호모그래피 JSON 파일 (대응점 개수 참조용)'
    )

    args = parser.parse_args()

    test_undistort_effect(args.image, args.calibration, args.homography)
