"""
호모그래피 계산 및 관리 유틸리티

필요한 패키지: opencv-python, numpy
"""

import cv2
import numpy as np
import json
import os
from datetime import datetime


class HomographyData:
    """
    호모그래피 대응점 및 행렬 데이터 관리 클래스
    """

    def __init__(self):
        self.image_points = []  # 영상 프레임의 점들 [(x, y), ...]
        self.drawing_points = []  # 도면의 점들 [(x, y), ...]
        self.homography_matrix = None  # 3x3 호모그래피 행렬
        self.metadata = {
            'created_at': None,
            'updated_at': None,
            'version': '1.0',
            'image_source': '',
            'drawing_source': '',
            'num_points': 0
        }

    def add_point_pair(self, image_point, drawing_point):
        """
        대응점 쌍을 추가합니다.

        Args:
            image_point: (x, y) 영상 프레임의 점
            drawing_point: (x, y) 도면의 점
        """
        self.image_points.append(tuple(image_point))
        self.drawing_points.append(tuple(drawing_point))
        self.metadata['num_points'] = len(self.image_points)
        self.metadata['updated_at'] = datetime.now().isoformat()

        if self.metadata['created_at'] is None:
            self.metadata['created_at'] = self.metadata['updated_at']

    def remove_point_pair(self, index):
        """
        지정된 인덱스의 대응점 쌍을 삭제합니다.

        Args:
            index: 삭제할 점의 인덱스
        """
        if 0 <= index < len(self.image_points):
            self.image_points.pop(index)
            self.drawing_points.pop(index)
            self.metadata['num_points'] = len(self.image_points)
            self.metadata['updated_at'] = datetime.now().isoformat()
            # 점이 변경되었으므로 호모그래피 행렬도 무효화
            self.homography_matrix = None

    def clear_all_points(self):
        """
        모든 대응점을 삭제합니다.
        """
        self.image_points = []
        self.drawing_points = []
        self.homography_matrix = None
        self.metadata['num_points'] = 0
        self.metadata['updated_at'] = datetime.now().isoformat()

    def calculate_homography(self, method=cv2.RANSAC, ransac_threshold=5.0):
        """
        대응점들로부터 호모그래피 행렬을 계산합니다.

        Args:
            method: OpenCV 호모그래피 계산 방법 (RANSAC, LMEDS 등)
            ransac_threshold: RANSAC 임계값 (픽셀 단위)

        Returns:
            bool: 계산 성공 여부
            str: 오류 메시지 (실패 시)
        """
        if len(self.image_points) < 4:
            return False, f"최소 4개의 대응점이 필요합니다 (현재: {len(self.image_points)}개)"

        # numpy 배열로 변환
        src_pts = np.array(self.image_points, dtype=np.float32)
        dst_pts = np.array(self.drawing_points, dtype=np.float32)

        try:
            # 호모그래피 행렬 계산
            H, mask = cv2.findHomography(src_pts, dst_pts, method, ransac_threshold)

            if H is None:
                return False, "호모그래피 행렬 계산에 실패했습니다"

            self.homography_matrix = H
            self.metadata['updated_at'] = datetime.now().isoformat()

            # inlier 개수 확인 (RANSAC 사용 시)
            if mask is not None:
                inliers = np.sum(mask)
                outliers = len(mask) - inliers
                if outliers > 0:
                    return True, f"성공 (Inliers: {inliers}, Outliers: {outliers})"

            return True, "호모그래피 행렬 계산 성공"

        except Exception as e:
            return False, f"오류 발생: {str(e)}"

    def transform_point(self, point, inverse=False):
        """
        호모그래피 변환을 적용하여 점을 변환합니다.

        Args:
            point: (x, y) 변환할 점
            inverse: True면 역변환 적용

        Returns:
            (x, y): 변환된 점, 실패 시 None
        """
        if self.homography_matrix is None:
            return None

        H = self.homography_matrix
        if inverse:
            H = np.linalg.inv(H)

        # 동차 좌표로 변환
        pt = np.array([[point[0], point[1]]], dtype=np.float32)
        pt = pt.reshape(-1, 1, 2)

        # 변환 적용
        transformed = cv2.perspectiveTransform(pt, H)

        return tuple(transformed[0][0])

    def calculate_reprojection_error(self):
        """
        재투영 오차를 계산합니다.

        Returns:
            dict: 오차 통계 {'mean': float, 'max': float, 'std': float, 'errors': list}
        """
        if self.homography_matrix is None or len(self.image_points) == 0:
            return None

        errors = []
        for img_pt, draw_pt in zip(self.image_points, self.drawing_points):
            # 영상 점을 도면으로 변환
            transformed = self.transform_point(img_pt)
            if transformed is None:
                continue

            # 실제 도면 점과의 거리 계산
            error = np.sqrt(
                (transformed[0] - draw_pt[0])**2 +
                (transformed[1] - draw_pt[1])**2
            )
            errors.append(error)

        if not errors:
            return None

        errors = np.array(errors)
        return {
            'mean': float(np.mean(errors)),
            'max': float(np.max(errors)),
            'std': float(np.std(errors)),
            'median': float(np.median(errors)),
            'errors': errors.tolist()
        }

    def save_to_file(self, filepath):
        """
        대응점과 호모그래피 행렬을 JSON 파일로 저장합니다.

        Args:
            filepath: 저장할 파일 경로
        """
        data = {
            'metadata': self.metadata,
            'image_points': self.image_points,
            'drawing_points': self.drawing_points,
            'homography_matrix': self.homography_matrix.tolist() if self.homography_matrix is not None else None
        }

        # 재투영 오차도 저장
        if self.homography_matrix is not None:
            error_stats = self.calculate_reprojection_error()
            if error_stats:
                data['reprojection_error'] = error_stats

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_from_file(self, filepath):
        """
        JSON 파일에서 대응점과 호모그래피 행렬을 로드합니다.

        Args:
            filepath: 로드할 파일 경로

        Returns:
            bool: 로드 성공 여부
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.metadata = data.get('metadata', self.metadata)
            self.image_points = [tuple(pt) for pt in data.get('image_points', [])]
            self.drawing_points = [tuple(pt) for pt in data.get('drawing_points', [])]

            h_matrix = data.get('homography_matrix')
            if h_matrix is not None:
                self.homography_matrix = np.array(h_matrix, dtype=np.float64)
            else:
                self.homography_matrix = None

            return True

        except Exception as e:
            print(f"파일 로드 오류: {e}")
            return False

    def export_for_opencv(self):
        """
        OpenCV에서 직접 사용할 수 있는 형태로 데이터를 내보냅니다.

        Returns:
            dict: {'H': numpy array, 'src_pts': numpy array, 'dst_pts': numpy array}
        """
        return {
            'H': self.homography_matrix,
            'src_pts': np.array(self.image_points, dtype=np.float32),
            'dst_pts': np.array(self.drawing_points, dtype=np.float32)
        }


def warp_image(image, homography_matrix, output_size):
    """
    호모그래피 변환을 적용하여 이미지를 워핑합니다.

    Args:
        image: 입력 이미지
        homography_matrix: 3x3 호모그래피 행렬
        output_size: (width, height) 출력 이미지 크기

    Returns:
        numpy.ndarray: 워핑된 이미지
    """
    return cv2.warpPerspective(image, homography_matrix, output_size)


def overlay_images(background, foreground, alpha=0.5):
    """
    두 이미지를 알파 블렌딩으로 오버레이합니다.

    Args:
        background: 배경 이미지
        foreground: 전경 이미지
        alpha: 전경 이미지 투명도 (0.0~1.0)

    Returns:
        numpy.ndarray: 합성된 이미지
    """
    return cv2.addWeighted(background, 1 - alpha, foreground, alpha, 0)
