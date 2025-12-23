"""
캘리브레이션 결과 검증 스크립트
캘리브레이션이 올바르게 수행되었는지 확인합니다.

필요한 패키지: opencv-python, numpy, matplotlib
설치: pip install opencv-python numpy matplotlib
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os


def load_calibration_results(filepath):
    """
    캘리브레이션 결과를 로드합니다.
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def validate_camera_matrix(camera_matrix):
    """
    카메라 매트릭스를 검증합니다.

    Returns:
        dict: 검증 결과 및 경고 메시지
    """
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    results = {
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
        'ratio': fx / fy if fy > 0 else 0,
        'warnings': [],
        'passed': True
    }

    # 1. fx/fy 비율 검사 (정사각형 픽셀 검증)
    ratio = fx / fy
    if ratio < 0.95 or ratio > 1.05:
        results['warnings'].append({
            'severity': 'CRITICAL',
            'message': f'fx/fy 비율이 비정상입니다: {ratio:.4f}',
            'details': (
                '정상적인 카메라는 fx와 fy가 거의 같아야 합니다 (비율 0.95~1.05).\n'
                '가능한 원인:\n'
                '  1. 체커보드 크기 설정 오류 (CHECKERBOARD_SIZE)\n'
                '  2. 체커보드가 왜곡되어 표시됨 (모니터 비율 설정)\n'
                '  3. 촬영 각도의 다양성 부족\n'
                '  4. 일부 이미지에서 코너 검출 오류'
            )
        })
        results['passed'] = False
    elif ratio < 0.98 or ratio > 1.02:
        results['warnings'].append({
            'severity': 'WARNING',
            'message': f'fx/fy 비율이 약간 벗어났습니다: {ratio:.4f}',
            'details': '정밀한 측정이 필요하다면 재촬영을 권장합니다.'
        })

    # 2. 초점 거리 합리성 검사
    if fx < 100 or fx > 10000:
        results['warnings'].append({
            'severity': 'WARNING',
            'message': f'초점 거리(fx)가 비정상적입니다: {fx:.2f} pixels',
            'details': '일반적으로 100~10000 범위여야 합니다.'
        })

    # 3. 주점(cx, cy) 위치 검사
    # 주점은 이미지 중심 근처에 있어야 함
    # 이미지 크기를 모르므로 절대값만 체크
    if cx < 0 or cy < 0:
        results['warnings'].append({
            'severity': 'CRITICAL',
            'message': '주점(principal point)이 음수입니다.',
            'details': '캘리브레이션이 잘못되었을 가능성이 높습니다.'
        })
        results['passed'] = False

    return results


def validate_distortion_coeffs(dist_coeffs):
    """
    왜곡 계수를 검증합니다.
    """
    k1, k2, p1, p2, k3 = dist_coeffs.ravel()

    results = {
        'k1': k1,
        'k2': k2,
        'k3': k3,
        'p1': p1,
        'p2': p2,
        'warnings': []
    }

    # 왜곡 계수가 너무 크면 경고
    if abs(k1) > 1.0:
        results['warnings'].append({
            'severity': 'WARNING',
            'message': f'방사 왜곡 계수 k1이 큽니다: {k1:.6f}',
            'details': '광각 렌즈이거나 캘리브레이션 오류일 수 있습니다.'
        })

    if abs(k2) > 1.0:
        results['warnings'].append({
            'severity': 'WARNING',
            'message': f'방사 왜곡 계수 k2가 큽니다: {k2:.6f}',
            'details': '심한 왜곡이 있거나 캘리브레이션 오류일 수 있습니다.'
        })

    return results


def validate_reprojection_error(mean_error, rms_error):
    """
    재투영 오차를 검증합니다.
    """
    results = {
        'mean_error': mean_error,
        'rms_error': rms_error,
        'warnings': []
    }

    # 재투영 오차 기준
    if mean_error > 1.0:
        results['warnings'].append({
            'severity': 'CRITICAL',
            'message': f'평균 재투영 오차가 너무 큽니다: {mean_error:.4f} pixels',
            'details': (
                '1.0 pixel 이상의 오차는 캘리브레이션이 실패했음을 의미합니다.\n'
                '가능한 원인:\n'
                '  1. 체커보드 크기(CHECKERBOARD_SIZE) 설정 오류\n'
                '  2. 정사각형 크기(SQUARE_SIZE) 측정 오류\n'
                '  3. 흔들린 이미지 포함\n'
                '  4. 초점이 맞지 않은 이미지'
            )
        })
    elif mean_error > 0.5:
        results['warnings'].append({
            'severity': 'WARNING',
            'message': f'평균 재투영 오차가 높습니다: {mean_error:.4f} pixels',
            'details': '정밀한 측정이 필요하다면 재촬영을 권장합니다 (목표: 0.5 이하).'
        })
    else:
        results['warnings'].append({
            'severity': 'GOOD',
            'message': f'재투영 오차가 우수합니다: {mean_error:.4f} pixels',
            'details': '캘리브레이션 품질이 좋습니다.'
        })

    return results


def print_validation_report(calib_data):
    """
    전체 검증 리포트를 출력합니다.
    """
    camera_matrix = calib_data['camera_matrix']
    dist_coeffs = calib_data['dist_coeffs']
    mean_error = calib_data.get('mean_reprojection_error', 0)
    rms_error = calib_data.get('rms_error', 0)

    print("\n" + "=" * 80)
    print("캘리브레이션 검증 리포트")
    print("=" * 80)

    # 1. 카메라 매트릭스 검증
    print("\n[ 1. 카메라 내부 파라미터 검증 ]")
    print("-" * 80)
    matrix_results = validate_camera_matrix(camera_matrix)

    print(f"초점 거리 (fx): {matrix_results['fx']:.2f} pixels")
    print(f"초점 거리 (fy): {matrix_results['fy']:.2f} pixels")
    print(f"fx/fy 비율: {matrix_results['ratio']:.6f}")
    print(f"주점 (cx, cy): ({matrix_results['cx']:.2f}, {matrix_results['cy']:.2f})")

    # 2. 왜곡 계수 검증
    print("\n[ 2. 왜곡 계수 검증 ]")
    print("-" * 80)
    dist_results = validate_distortion_coeffs(dist_coeffs)
    print(f"방사 왜곡 (k1): {dist_results['k1']:.6f}")
    print(f"방사 왜곡 (k2): {dist_results['k2']:.6f}")
    print(f"방사 왜곡 (k3): {dist_results['k3']:.6f}")
    print(f"접선 왜곡 (p1): {dist_results['p1']:.6f}")
    print(f"접선 왜곡 (p2): {dist_results['p2']:.6f}")

    # 3. 재투영 오차 검증
    print("\n[ 3. 재투영 오차 검증 ]")
    print("-" * 80)
    error_results = validate_reprojection_error(mean_error, rms_error)
    print(f"평균 재투영 오차: {error_results['mean_error']:.6f} pixels")
    print(f"RMS 재투영 오차: {error_results['rms_error']:.6f} pixels")

    # 4. 경고 및 권장사항
    print("\n[ 4. 경고 및 권장사항 ]")
    print("-" * 80)

    all_warnings = (
        matrix_results['warnings'] +
        dist_results['warnings'] +
        error_results['warnings']
    )

    critical_warnings = [w for w in all_warnings if w['severity'] == 'CRITICAL']
    normal_warnings = [w for w in all_warnings if w['severity'] == 'WARNING']
    good_messages = [w for w in all_warnings if w['severity'] == 'GOOD']

    if critical_warnings:
        print("\n🚨 치명적 문제 (CRITICAL):")
        for w in critical_warnings:
            print(f"\n  ⚠️  {w['message']}")
            print(f"      {w['details'].replace(chr(10), chr(10) + '      ')}")

    if normal_warnings:
        print("\n⚠️  경고 (WARNING):")
        for w in normal_warnings:
            print(f"\n  ⚠️  {w['message']}")
            print(f"      {w['details'].replace(chr(10), chr(10) + '      ')}")

    if good_messages:
        print("\n✅ 양호:")
        for w in good_messages:
            print(f"  ✓ {w['message']}")

    if not critical_warnings and not normal_warnings:
        print("\n✅ 모든 검증을 통과했습니다!")

    # 5. 최종 판정
    print("\n" + "=" * 80)
    print("[ 최종 판정 ]")
    print("=" * 80)

    if critical_warnings:
        print("❌ 캘리브레이션 결과가 부적합합니다.")
        print("   재촬영을 강력히 권장합니다.")
    elif normal_warnings:
        print("⚠️  캘리브레이션 결과에 일부 문제가 있습니다.")
        print("   용도에 따라 재촬영을 고려하세요.")
    else:
        print("✅ 캘리브레이션 결과가 우수합니다.")
        print("   사용해도 무방합니다.")

    print("=" * 80 + "\n")

    return matrix_results['passed'] and not critical_warnings


def main():
    """
    메인 함수
    """
    script_dir = os.path.dirname(__file__)
    calib_file = os.path.join(script_dir, 'output', 'calibration_data.pkl')

    if not os.path.exists(calib_file):
        print(f"캘리브레이션 결과 파일을 찾을 수 없습니다: {calib_file}")
        print("먼저 calibrate.py를 실행하세요.")
        return 1

    # 캘리브레이션 데이터 로드
    calib_data = load_calibration_results(calib_file)

    # 검증 리포트 출력
    passed = print_validation_report(calib_data)

    return 0 if passed else 1


if __name__ == "__main__":
    exit(main())
