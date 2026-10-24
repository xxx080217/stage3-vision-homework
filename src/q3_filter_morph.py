"""第三问：比较 滤波(前处理) 与 开/闭运算(后处理) 对掩膜质量的影响。

运行（在仓库根目录）：python src/q3_filter_morph.py
输出：outputs/q3/ 下每帧一张 2x2 对比图（原始掩膜 / 高斯滤波 / 中值滤波 / 无此问无关项），
     另有 morphology_open_3x3.png 与 morphology_open_7x7.png 的核尺寸对比图。
"""
import os

import cv2
import numpy as np

FRAMES = [0, 109, 217]
FRAME_DIR = "outputs/frames"
OUT_DIR = "outputs/q3"

HSV_LOWER = np.array([3, 120, 150])
HSV_UPPER = np.array([14, 255, 255])

LABEL = {"fontFace": cv2.FONT_HERSHEY_SIMPLEX, "fontScale": 0.9,
         "color": (0, 0, 0), "thickness": 2}


def segment(hsv):
    return cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)


def tile_with_label(mask, text):
    """单通道掩膜 -> 带文字标签的 BGR 小图。"""
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(vis, (0, 0), (280, 40), (255, 255, 255), -1)
    cv2.putText(vis, text, (10, 30), **LABEL)
    return vis


def make_grid(tiles):
    """4 张图拼成 2x2，统一缩放到 480 宽。"""
    tiles = [cv2.resize(t, (480, 270)) for t in tiles]
    top = np.hstack(tiles[:2])
    bottom = np.hstack(tiles[2:])
    return np.vstack([top, bottom])


def compare_for_frame(idx):
    bgr = cv2.imread(os.path.join(FRAME_DIR, f"frame_{idx:06d}.png"))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # 方法1：原始掩膜（不处理）
    m_raw = segment(hsv)

    # 方法2：先对原图滤波，再做同样的 HSV 分割
    blur_g = cv2.GaussianBlur(bgr, (3, 3), 0)
    m_gauss = segment(cv2.cvtColor(blur_g, cv2.COLOR_BGR2HSV))
    blur_m = cv2.medianBlur(bgr, 3)
    m_median = segment(cv2.cvtColor(blur_m, cv2.COLOR_BGR2HSV))

    # 方法3/4：对原始掩膜做开运算 / 闭运算（后处理）
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    m_open = cv2.morphologyEx(m_raw, cv2.MORPH_OPEN, k3)
    m_close = cv2.morphologyEx(m_raw, cv2.MORPH_CLOSE, k3)

    grid1 = make_grid([tile_with_label(m_raw, "1 raw"),
                       tile_with_label(m_gauss, "2 gaussian 3x3"),
                       tile_with_label(m_median, "3 median 3x3"),
                       tile_with_label(m_open, "4 open 3x3 (raw)")])
    grid2 = make_grid([tile_with_label(m_raw, "1 raw"),
                       tile_with_label(m_close, "5 close 3x3 (raw)"),
                       tile_with_label(
                           cv2.morphologyEx(m_raw, cv2.MORPH_OPEN,
                                            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))),
                           "6 open 7x7 (raw)"),
                       tile_with_label(
                           cv2.morphologyEx(m_raw, cv2.MORPH_CLOSE,
                                            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))),
                           "7 close 7x7 (raw)")])

    cv2.imwrite(os.path.join(OUT_DIR, f"frame_{idx:06d}_compare.png"), grid1)
    cv2.imwrite(os.path.join(OUT_DIR, f"frame_{idx:06d}_kernel_compare.png"), grid2)

    # 打印一些量化线索，方便写报告
    n_labels_raw, _ = cv2.connectedComponents(m_raw)
    n_labels_open, _ = cv2.connectedComponents(m_open)
    holes = cv2.bitwise_not(m_close)
    n_holes, _ = cv2.connectedComponents(holes[3:-3, 3:-3])
    print(f"frame {idx}: 原始连通域 {n_labels_raw - 1}, 开运算后 {n_labels_open - 1}, "
          f"闭运算可填充的黑区 {n_holes - 1}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for idx in FRAMES:
        compare_for_frame(idx)


if __name__ == "__main__":
    main()
