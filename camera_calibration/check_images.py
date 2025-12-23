"""
캘리브레이션 이미지 품질 검사 스크립트
각 이미지의 품질을 분석하고 문제가 있는 이미지를 찾아냅니다.

필요한 패키지: opencv-python, numpy
설치: pip install opencv-python numpy
"""

import cv2
import numpy as np
import os
import glob


# 체커보드 설정
CHECKERBOARD_SIZE = (9, 6)  # 내부 코너 개수 (가로, 세로)

# 경로 설정
SCRIPT_DIR = os.path.dirname(__file__)
IMAGES_DIR = os.path.join(SCRIPT_DIR, 'images')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output', 'image_quality')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def analyze_image_quality(image_path, checkerboard_size):
    """
    이미지의 품질을 분석합니다.

    Returns:
        dict: 분석 결과
    """
    img = cv2.imread(image_path)
    if img is None:
        return {
            'success': False,
            'error': '이미지를 읽을 수 없습니다'
        }

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 체커보드 코너 검출
    ret, corners = cv2.findChessboardCorners(
        gray,
        checkerboard_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    result = {
        'image_path': image_path,
        'filename': os.path.basename(image_path),
        'success': ret,
        'image_size': gray.shape[::-1],
        'brightness_mean': np.mean(gray),
        'brightness_std': np.std(gray),
        'warnings': []
    }

    if not ret:
        result['error'] = '체커보드를 검출할 수 없습니다'
        result['warnings'].append({
            'severity': 'CRITICAL',
            'message': '체커보드 코너 검출 실패',
            'solution': '이 이미지를 삭제하고 다시 촬영하세요'
        })
        return result

    # 코너 정밀화
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # 코너 품질 분석
    # 1. 코너 간격의 균일성 검사
    distances = []
    for i in range(len(corners_refined) - 1):
        dist = np.linalg.norm(corners_refined[i] - corners_refined[i + 1])
        distances.append(dist)

    distances = np.array(distances)
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    cv_dist = std_dist / mean_dist if mean_dist > 0 else 0  # 변동계수

    result['mean_corner_distance'] = mean_dist
    result['std_corner_distance'] = std_dist
    result['cv_corner_distance'] = cv_dist

    # 변동계수가 크면 왜곡이 심하거나 각도가 극단적
    if cv_dist > 0.3:
        result['warnings'].append({
            'severity': 'WARNING',
            'message': f'코너 간격의 변동이 큽니다 (CV: {cv_dist:.3f})',
            'solution': '체커보드가 너무 기울어졌거나 왜곡이 심합니다'
        })

    # 2. 밝기 검사
    if result['brightness_mean'] < 50:
        result['warnings'].append({
            'severity': 'WARNING',
            'message': f'이미지가 너무 어둡습니다 (평균 밝기: {result["brightness_mean"]:.1f})',
            'solution': '조명을 밝게 하세요'
        })
    elif result['brightness_mean'] > 200:
        result['warnings'].append({
            'severity': 'WARNING',
            'message': f'이미지가 너무 밝습니다 (평균 밝기: {result["brightness_mean"]:.1f})',
            'solution': '조명을 어둡게 하거나 노출을 낮추세요'
        })

    # 3. 체커보드 크기 검사 (이미지 대비)
    x_coords = corners_refined[:, 0, 0]
    y_coords = corners_refined[:, 0, 1]
    board_width = np.max(x_coords) - np.min(x_coords)
    board_height = np.max(y_coords) - np.min(y_coords)
    image_area = gray.shape[0] * gray.shape[1]
    board_area = board_width * board_height
    board_ratio = board_area / image_area

    result['board_area_ratio'] = board_ratio

    if board_ratio < 0.1:
        result['warnings'].append({
            'severity': 'WARNING',
            'message': f'체커보드가 너무 작습니다 (화면 비율: {board_ratio*100:.1f}%)',
            'solution': '체커보드를 카메라에 더 가까이 가져오세요'
        })
    elif board_ratio > 0.8:
        result['warnings'].append({
            'severity': 'WARNING',
            'message': f'체커보드가 너무 큽니다 (화면 비율: {board_ratio*100:.1f}%)',
            'solution': '체커보드를 카메라에서 더 멀리 두세요'
        })

    # 4. 흐림(Blur) 검사
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    result['sharpness'] = laplacian_var

    if laplacian_var < 100:
        result['warnings'].append({
            'severity': 'WARNING',
            'message': f'이미지가 흐릿합니다 (선명도: {laplacian_var:.1f})',
            'solution': '초점을 다시 맞추거나 손떨림 방지'
        })

    return result


def print_quality_report(results):
    """
    품질 검사 리포트를 출력합니다.
    """
    print("\n" + "=" * 80)
    print("이미지 품질 검사 리포트")
    print("=" * 80)

    total_images = len(results)
    successful_images = sum(1 for r in results if r['success'])
    failed_images = total_images - successful_images

    print(f"\n총 이미지 수: {total_images}")
    print(f"체커보드 검출 성공: {successful_images}")
    print(f"체커보드 검출 실패: {failed_images}")

    if failed_images > 0:
        print("\n" + "-" * 80)
        print("❌ 체커보드 검출 실패 이미지:")
        print("-" * 80)
        for r in results:
            if not r['success']:
                print(f"  - {r['filename']}")
                if 'error' in r:
                    print(f"    이유: {r['error']}")

    # 경고가 있는 이미지
    warning_images = [r for r in results if r['success'] and len(r['warnings']) > 0]

    if warning_images:
        print("\n" + "-" * 80)
        print("⚠️  경고가 있는 이미지:")
        print("-" * 80)
        for r in warning_images:
            print(f"\n  📷 {r['filename']}")
            for w in r['warnings']:
                print(f"     {w['severity']}: {w['message']}")
                print(f"     해결: {w['solution']}")

    # 우수한 이미지
    good_images = [r for r in results if r['success'] and len(r['warnings']) == 0]

    if good_images:
        print("\n" + "-" * 80)
        print(f"✅ 우수한 이미지: {len(good_images)}개")
        print("-" * 80)
        for r in good_images[:5]:  # 처음 5개만 표시
            print(f"  ✓ {r['filename']}")
        if len(good_images) > 5:
            print(f"  ... 외 {len(good_images) - 5}개")

    # 권장사항
    print("\n" + "=" * 80)
    print("권장사항")
    print("=" * 80)

    if failed_images > 0:
        print(f"\n❌ {failed_images}개의 이미지에서 체커보드 검출 실패")
        print("   → images/ 폴더에서 해당 이미지들을 삭제하세요")

    if successful_images < 15:
        print(f"\n⚠️  사용 가능한 이미지가 {successful_images}개로 부족합니다")
        print("   → 최소 15개, 권장 20개 이상 촬영하세요")

    if successful_images >= 15 and len(warning_images) == 0:
        print("\n✅ 모든 이미지의 품질이 우수합니다!")
        print("   → calibrate.py를 실행하여 캘리브레이션을 진행하세요")

    print("=" * 80 + "\n")


def save_quality_report(results):
    """
    품질 검사 결과를 파일로 저장합니다.
    """
    report_path = os.path.join(OUTPUT_DIR, 'quality_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("이미지 품질 검사 상세 리포트\n")
        f.write("=" * 80 + "\n\n")

        for r in results:
            f.write(f"이미지: {r['filename']}\n")
            f.write("-" * 80 + "\n")
            f.write(f"  검출 성공: {'✓' if r['success'] else '✗'}\n")

            if r['success']:
                f.write(f"  이미지 크기: {r['image_size'][0]} x {r['image_size'][1]}\n")
                f.write(f"  평균 밝기: {r['brightness_mean']:.1f}\n")
                f.write(f"  밝기 표준편차: {r['brightness_std']:.1f}\n")
                f.write(f"  선명도: {r['sharpness']:.1f}\n")
                f.write(f"  체커보드 화면 비율: {r['board_area_ratio']*100:.1f}%\n")
                f.write(f"  코너 간격 평균: {r['mean_corner_distance']:.2f} pixels\n")
                f.write(f"  코너 간격 변동계수: {r['cv_corner_distance']:.3f}\n")

                if r['warnings']:
                    f.write(f"\n  경고:\n")
                    for w in r['warnings']:
                        f.write(f"    - [{w['severity']}] {w['message']}\n")
                        f.write(f"      해결: {w['solution']}\n")
            else:
                f.write(f"  오류: {r.get('error', '알 수 없는 오류')}\n")

            f.write("\n")

    print(f"상세 리포트 저장: {report_path}")


def main():
    """
    메인 함수
    """
    # 이미지 파일 목록 가져오기
    image_files = glob.glob(os.path.join(IMAGES_DIR, '*.jpg'))
    image_files.extend(glob.glob(os.path.join(IMAGES_DIR, '*.png')))

    if not image_files:
        print(f"'{IMAGES_DIR}' 디렉토리에 이미지가 없습니다.")
        print("먼저 capture_images.py를 실행하여 이미지를 촬영하세요.")
        return 1

    print(f"\n총 {len(image_files)}개의 이미지를 검사합니다...")
    print(f"체커보드 크기: {CHECKERBOARD_SIZE[0]} x {CHECKERBOARD_SIZE[1]} (내부 코너)\n")

    # 각 이미지 분석
    results = []
    for idx, img_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] {os.path.basename(img_path)}...", end=' ')
        result = analyze_image_quality(img_path, CHECKERBOARD_SIZE)
        results.append(result)

        if result['success']:
            if result['warnings']:
                print("⚠️  경고")
            else:
                print("✓")
        else:
            print("✗ 실패")

    # 리포트 출력
    print_quality_report(results)

    # 리포트 저장
    save_quality_report(results)

    return 0


if __name__ == "__main__":
    exit(main())
