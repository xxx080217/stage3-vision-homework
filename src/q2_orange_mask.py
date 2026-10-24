"""第二问：HSV 筛橙色，生成二值掩膜与保色图。

运行（在仓库根目录）：python src/q2_orange_mask.py
输出：outputs/q2/frame_编号_mask.png、frame_编号_orange.png
"""
import os

import cv2
import numpy as np

FRAMES = [0, 109, 217]
FRAME_DIR = "outputs/frames"
OUT_DIR = "outputs/q2"

# 阈值的选择依据：用 cv2.cvtColor 把样例帧转 HSV 后，
# 在橙色板区域取样，发现 H 集中在 10~20 附近，故取 5~25 留余量；
# S、V 下限用来排除发灰、发暗的背景。
HSV_LOWER = np.array([3, 120, 150])
HSV_UPPER = np.array([14, 255, 255])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for idx in FRAMES:
        bgr = cv2.imread(os.path.join(FRAME_DIR, f"frame_{idx:06d}.png"))
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)   # 白=橙色范围内, 黑=范围外
        orange = cv2.bitwise_and(bgr, bgr, mask=mask)   # 只保留掩膜为白处的原色

        cv2.imwrite(os.path.join(OUT_DIR, f"frame_{idx:06d}_mask.png"), mask)
        cv2.imwrite(os.path.join(OUT_DIR, f"frame_{idx:06d}_orange.png"), orange)
        print(f"frame {idx}: 掩膜白色像素占比 {mask.mean() / 255:.2%}")


if __name__ == "__main__":
    main()
