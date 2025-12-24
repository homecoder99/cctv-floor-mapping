"""
테스트 1: H로 변환한 점들을 도면 위에 시각화

호모그래피 행렬로 변환한 점들과 원본 대응점을 비교하여
좌표계/해상도 문제인지 대응점 오입력 문제인지 판별합니다.
"""

import json
import numpy as np
import cv2
import sys
from pathlib import Path

# 상위 디렉토리의 모듈 임포트
sys.path.append(str(Path(__file__).parent.parent))
from homography.dxf_renderer import render_dxf_to_image


def visualize_homography_transform(homography_json_path):
    """
    호모그래피 변환 결과를 시각화

    Args:
        homography_json_path: 호모그래피 JSON 파일 경로
    """
    # JSON 로드
    with open(homography_json_path, 'r') as f:
        data = json.load(f)

    # 데이터 추출
    image_points = np.array(data['image_points'], dtype=np.float32)
    drawing_points = np.array(data['drawing_points'], dtype=np.float32)
    H = np.array(data['homography_matrix'], dtype=np.float32)
    drawing_source = data['metadata']['drawing_source']

    print(f"호모그래피 JSON: {homography_json_path}")
    print(f"DXF 파일: {drawing_source}")
    print(f"대응점 개수: {len(image_points)}")
    print()

    # 이미지 좌표 → 도면 좌표 변환
    # image_points를 homogeneous 좌표로 변환
    image_points_h = np.hstack([image_points, np.ones((len(image_points), 1))])

    # H로 변환
    transformed_points_h = (H @ image_points_h.T).T

    # homogeneous → Cartesian
    transformed_points = transformed_points_h[:, :2] / transformed_points_h[:, 2:3]

    # 오차 계산
    errors = np.linalg.norm(transformed_points - drawing_points, axis=1)

    print("=== 변환 결과 ===")
    print(f"{'번호':<4} {'Image XY':<20} {'→ Drawing XY':<20} {'실제 Drawing XY':<20} {'오차(px)':<10}")
    print("-" * 80)

    for i in range(len(image_points)):
        img_pt = image_points[i]
        trans_pt = transformed_points[i]
        draw_pt = drawing_points[i]
        err = errors[i]

        print(f"{i:<4} ({img_pt[0]:>6.1f},{img_pt[1]:>6.1f}) → "
              f"({trans_pt[0]:>6.1f},{trans_pt[1]:>6.1f}) "
              f"vs ({draw_pt[0]:>6.1f},{draw_pt[1]:>6.1f}) "
              f"{err:>8.1f}")

    print()
    print(f"평균 오차: {np.mean(errors):.1f} pixels")
    print(f"최대 오차: {np.max(errors):.1f} pixels")
    print(f"중앙값: {np.median(errors):.1f} pixels")
    print()

    # DXF 렌더링
    print("DXF 렌더링 중...")
    drawing_img = render_dxf_to_image(drawing_source, output_size=(3000, 3000))

    if drawing_img is None:
        print("오류: DXF 렌더링 실패")
        return

    h, w = drawing_img.shape[:2]
    print(f"도면 이미지 크기: {w}x{h}")

    # 점 그리기
    # 원본 대응점: 녹색 원
    for i, pt in enumerate(drawing_points):
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(drawing_img, (x, y), 8, (0, 255, 0), 2)
            cv2.putText(drawing_img, f"{i}", (x + 10, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 변환된 점: 빨간색 X
    for i, pt in enumerate(transformed_points):
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < w and 0 <= y < h:
            # X 표시
            cv2.line(drawing_img, (x-10, y-10), (x+10, y+10), (0, 0, 255), 2)
            cv2.line(drawing_img, (x-10, y+10), (x+10, y-10), (0, 0, 255), 2)
        else:
            print(f"경고: 변환된 점 {i}가 도면 범위를 벗어남: ({x}, {y})")

    # 오차 연결선: 노란색
    for i in range(len(drawing_points)):
        pt1 = (int(drawing_points[i][0]), int(drawing_points[i][1]))
        pt2 = (int(transformed_points[i][0]), int(transformed_points[i][1]))

        # 둘 다 범위 내에 있을 때만 선 그리기
        if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
            0 <= pt2[0] < w and 0 <= pt2[1] < h):
            cv2.line(drawing_img, pt1, pt2, (0, 255, 255), 1)

    # 범례 추가
    legend_y = 30
    cv2.putText(drawing_img, "Green O: Original points", (10, legend_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(drawing_img, "Red X: Transformed points", (10, legend_y + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(drawing_img, "Yellow line: Error", (10, legend_y + 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 결과 표시
    cv2.imshow('Homography Transform Test 1', drawing_img)
    print("\n이미지 창에서 결과 확인")
    print("  - 녹색 원(O): 원본 대응점")
    print("  - 빨간색 X: H로 변환된 점")
    print("  - 노란색 선: 오차")
    print("\n판별 기준:")
    print("  - 점이 전혀 다른 방향으로 튄다 → 좌표계(축 뒤집힘/해상도 불일치) 문제")
    print("  - 대체로 맞는데 일부만 크게 튄다 → 특정 대응점 오입력(outlier) 문제")
    print("\n아무 키나 누르면 종료")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 결과 이미지 저장
    output_path = homography_json_path.replace('.json', '_test1_visual.jpg')
    cv2.imwrite(output_path, drawing_img)
    print(f"\n결과 이미지 저장: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="테스트 1: H로 변환한 점들을 도면 위에 시각화"
    )
    parser.add_argument(
        '--homography',
        type=str,
        default='homography/data/homography_20251224_162558.json',
        help='호모그래피 JSON 파일 경로'
    )

    args = parser.parse_args()

    visualize_homography_transform(args.homography)
