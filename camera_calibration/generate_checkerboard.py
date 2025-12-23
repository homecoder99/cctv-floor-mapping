"""
체커보드 패턴 생성 스크립트
모니터에 띄워서 사용할 수 있는 체커보드 이미지를 생성합니다.

필요한 패키지: opencv-python, numpy
설치: pip install opencv-python numpy
"""

import cv2
import numpy as np
import os

# 체커보드 설정
CHECKERBOARD_SIZE = (10, 7)  # 정사각형 개수 (가로, 세로) - 내부 코너는 (9, 6)
SQUARE_SIZE_PIXELS = 100  # 각 정사각형의 크기 (pixels)

# 출력 설정
OUTPUT_DIR = os.path.dirname(__file__)
OUTPUT_FILENAME = "checkerboard_pattern.png"


def generate_checkerboard(rows, cols, square_size):
    """
    체커보드 패턴을 생성합니다.

    Args:
        rows: 정사각형 개수 (세로)
        cols: 정사각형 개수 (가로)
        square_size: 각 정사각형의 크기 (pixels)

    Returns:
        체커보드 이미지
    """
    # 이미지 크기 계산
    height = rows * square_size
    width = cols * square_size

    # 빈 이미지 생성
    checkerboard = np.zeros((height, width), dtype=np.uint8)

    # 체커보드 패턴 생성
    for i in range(rows):
        for j in range(cols):
            # 체스판처럼 교대로 흰색/검은색 칠하기
            if (i + j) % 2 == 0:
                y_start = i * square_size
                y_end = (i + 1) * square_size
                x_start = j * square_size
                x_end = (j + 1) * square_size
                checkerboard[y_start:y_end, x_start:x_end] = 255

    return checkerboard


def add_border(image, border_size=50, border_color=128):
    """
    이미지에 테두리를 추가합니다.

    Args:
        image: 입력 이미지
        border_size: 테두리 크기 (pixels)
        border_color: 테두리 색상 (0-255)

    Returns:
        테두리가 추가된 이미지
    """
    return cv2.copyMakeBorder(
        image,
        border_size,
        border_size,
        border_size,
        border_size,
        cv2.BORDER_CONSTANT,
        value=border_color
    )


def add_info_text(image, checkerboard_size, square_size):
    """
    이미지에 정보 텍스트를 추가합니다.

    Args:
        image: 입력 이미지
        checkerboard_size: 체커보드 크기 (cols, rows)
        square_size: 정사각형 크기 (pixels)

    Returns:
        텍스트가 추가된 이미지
    """
    # 컬러 이미지로 변환 (텍스트를 색상으로 표시하기 위해)
    img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 텍스트 설정
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (0, 0, 255)  # 빨간색
    thickness = 2

    # 정보 텍스트
    cols, rows = checkerboard_size
    inner_corners = (cols - 1, rows - 1)

    texts = [
        f"Checkerboard: {cols} x {rows} squares",
        f"Inner corners: {inner_corners[0]} x {inner_corners[1]}",
        f"Square size: {square_size} pixels"
    ]

    # 텍스트 위치 (상단 왼쪽)
    y_offset = 30
    x_offset = 10

    for i, text in enumerate(texts):
        y = y_offset + i * 25
        cv2.putText(img_color, text, (x_offset, y), font, font_scale, font_color, thickness)

    return img_color


def main():
    """
    메인 함수
    """
    print("=" * 60)
    print("체커보드 패턴 생성 프로그램")
    print("=" * 60)
    print(f"체커보드 크기: {CHECKERBOARD_SIZE[0]} x {CHECKERBOARD_SIZE[1]} (정사각형)")
    print(f"내부 코너: {CHECKERBOARD_SIZE[0]-1} x {CHECKERBOARD_SIZE[1]-1}")
    print(f"정사각형 크기: {SQUARE_SIZE_PIXELS} pixels")
    print("=" * 60)

    # 체커보드 생성
    checkerboard = generate_checkerboard(
        CHECKERBOARD_SIZE[1],  # rows
        CHECKERBOARD_SIZE[0],  # cols
        SQUARE_SIZE_PIXELS
    )

    # 테두리 추가
    checkerboard_with_border = add_border(checkerboard, border_size=50, border_color=200)

    # 정보 텍스트 추가
    final_image = add_info_text(checkerboard_with_border, CHECKERBOARD_SIZE, SQUARE_SIZE_PIXELS)

    # 이미지 저장
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    cv2.imwrite(output_path, final_image)

    print(f"\n체커보드 이미지 생성 완료!")
    print(f"저장 경로: {output_path}")
    print(f"이미지 크기: {final_image.shape[1]} x {final_image.shape[0]} pixels")

    # 이미지 표시 (옵션)
    print("\n이미지를 확인하세요. 아무 키나 누르면 종료됩니다.")

    # 화면에 맞게 리사이즈해서 표시
    display_height = 800
    aspect_ratio = final_image.shape[1] / final_image.shape[0]
    display_width = int(display_height * aspect_ratio)
    display_image = cv2.resize(final_image, (display_width, display_height))

    cv2.imshow('Checkerboard Pattern', display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("사용 방법:")
    print("1. 생성된 이미지를 전체화면으로 모니터에 띄웁니다")
    print("2. capture_images.py를 실행하여 이미지를 캡처합니다")
    print("3. 모니터를 다양한 각도로 회전시키면서 촬영합니다")
    print("=" * 60)


if __name__ == "__main__":
    main()
