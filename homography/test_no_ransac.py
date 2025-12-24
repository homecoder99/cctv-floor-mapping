"""
테스트 2: 선택한 점들만으로 RANSAC 없이 H 계산

RANSAC 없이도 낮은 오차가 나오는지 확인하여
outlier 비율을 진단합니다.
"""

import json
import numpy as np
import cv2


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


def test_no_ransac(homography_json_path, selected_indices=None):
    """
    RANSAC 없이 호모그래피 계산

    Args:
        homography_json_path: 호모그래피 JSON 파일 경로
        selected_indices: 사용할 점의 인덱스 리스트 (None이면 오차 작은 순으로 자동 선택)
    """
    # JSON 로드
    with open(homography_json_path, 'r') as f:
        data = json.load(f)

    # 데이터 추출
    image_points = np.array(data['image_points'], dtype=np.float32)
    drawing_points = np.array(data['drawing_points'], dtype=np.float32)
    original_H = np.array(data['homography_matrix'], dtype=np.float32)
    original_errors = data['reprojection_error']['errors']

    print(f"호모그래피 JSON: {homography_json_path}")
    print(f"전체 대응점 개수: {len(image_points)}")
    print()

    # 원본 H의 오차
    print("=== 원본 RANSAC H ===")
    original_stats = calculate_reprojection_error(original_H, image_points, drawing_points)
    print(f"평균 오차: {original_stats['mean']:.2f} pixels")
    print(f"최대 오차: {original_stats['max']:.2f} pixels")
    print(f"중앙값: {original_stats['median']:.2f} pixels")
    print()

    # 점 선택
    if selected_indices is None:
        # 오차가 작은 순으로 6개 자동 선택
        errors_with_idx = [(i, err) for i, err in enumerate(original_errors)]
        errors_with_idx.sort(key=lambda x: x[1])
        selected_indices = [idx for idx, _ in errors_with_idx[:6]]
        print(f"오차가 작은 점 6개 자동 선택: {selected_indices}")
    else:
        print(f"사용자 지정 점 사용: {selected_indices}")

    print()

    # 선택된 점만 추출
    selected_img_pts = image_points[selected_indices]
    selected_draw_pts = drawing_points[selected_indices]

    print(f"선택된 {len(selected_indices)}개 점의 원본 오차:")
    for i, idx in enumerate(selected_indices):
        print(f"  점 {idx}: {original_errors[idx]:.2f} pixels")
    print()

    # RANSAC 없이 H 계산 (method=0)
    H_no_ransac, _ = cv2.findHomography(
        selected_img_pts,
        selected_draw_pts,
        method=0  # RANSAC 없음
    )

    if H_no_ransac is None:
        print("오류: 호모그래피 계산 실패")
        return

    # 선택된 점들에 대한 오차
    print("=== RANSAC 없이 계산한 H (선택된 점) ===")
    selected_stats = calculate_reprojection_error(H_no_ransac, selected_img_pts, selected_draw_pts)
    print(f"평균 오차: {selected_stats['mean']:.2f} pixels")
    print(f"최대 오차: {selected_stats['max']:.2f} pixels")
    print(f"중앙값: {selected_stats['median']:.2f} pixels")
    print()

    # 전체 점들에 대한 오차 (참고용)
    print("=== RANSAC 없이 계산한 H (전체 점) ===")
    all_stats = calculate_reprojection_error(H_no_ransac, image_points, drawing_points)
    print(f"평균 오차: {all_stats['mean']:.2f} pixels")
    print(f"최대 오차: {all_stats['max']:.2f} pixels")
    print(f"중앙값: {all_stats['median']:.2f} pixels")
    print()

    # 점별 오차 출력
    print("점별 오차 (전체):")
    print(f"{'번호':<6} {'원본 H 오차':<15} {'RANSAC 없는 H 오차':<20}")
    print("-" * 45)
    for i in range(len(image_points)):
        marker = " *" if i in selected_indices else ""
        print(f"{i:<6} {original_stats['errors'][i]:>10.2f} px   {all_stats['errors'][i]:>10.2f} px{marker}")

    print("\n* = RANSAC 없는 H 계산에 사용된 점")
    print()

    # 판별
    print("=== 판별 ===")
    if selected_stats['mean'] < 5.0:
        print("✓ 선택된 점들의 오차 < 5 pixels → 점이 정확하게 입력됨")
        print("  → 기존 전체 점 중 일부가 잘못 입력된 outlier로 추정")
        print(f"  → 오차 큰 점 제거하고 재계산 권장")
    else:
        print("✗ 선택된 점들도 오차 > 5 pixels → 대응점 입력 자체에 문제")
        print("  가능한 원인:")
        print("  1. 점을 잘못 찍음 (클릭 위치 부정확)")
        print("  2. 이미지와 도면의 해상도/좌표계 불일치")
        print("  3. 왜곡 보정 미적용 또는 잘못된 캘리브레이션")

    if all_stats['mean'] > original_stats['mean'] * 1.5:
        print("\n경고: RANSAC 없이 계산한 H의 전체 오차가 원본보다 크게 악화")
        print("  → RANSAC이 일부 outlier를 제외하고 더 나은 H를 찾았음")
        print("  → outlier 비율이 높음 (정상적이지 않음)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="테스트 2: RANSAC 없이 선택한 점으로 H 계산"
    )
    parser.add_argument(
        '--homography',
        type=str,
        default='homography/data/homography_20251224_162558.json',
        help='호모그래피 JSON 파일 경로'
    )
    parser.add_argument(
        '--indices',
        type=int,
        nargs='+',
        default=None,
        help='사용할 점의 인덱스 (예: --indices 0 1 2 3 4 5). 지정 안 하면 오차 작은 6개 자동 선택'
    )

    args = parser.parse_args()

    test_no_ransac(args.homography, args.indices)
