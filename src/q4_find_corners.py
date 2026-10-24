"""第四问：从单帧掩膜中找出橙色板四个角点并画标注图。

运行（在仓库根目录）：python src/q4_find_corners.py
输出：outputs/q4/frame_编号_annotated.png 与 frame_编号_corners.json
"""
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(__file__))
from detector import detect_corners  # noqa: E402

FRAMES = [0, 109, 217]
FRAME_DIR = "outputs/frames"
OUT_DIR = "outputs/q4"

COLORS = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 255)]


def annotate(frame, corners):
    """画角点、编号（0~3）和四边形（连线顺序 0->1->2->3->0）。"""
    vis = frame.copy()
    pts = [tuple(p) for p in corners]
    for i in range(4):
        cv2.line(vis, pts[i], pts[(i + 1) % 4], (0, 255, 0), 3)   # 四边形
    for i, p in enumerate(pts):
        cv2.circle(vis, p, 7, COLORS[i], -1)                       # 角点
        cv2.putText(vis, str(i), (p[0] + 10, p[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
    return vis


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for idx in FRAMES:
        frame = cv2.imread(os.path.join(FRAME_DIR, f"frame_{idx:06d}.png"))
        valid, corners = detect_corners(frame)

        if valid:
            vis = annotate(frame, corners)
            print(f"frame {idx}: valid=True  corners={corners}")
        else:
            vis = frame.copy()
            cv2.putText(vis, "NOT FOUND", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            print(f"frame {idx}: valid=False")

        cv2.imwrite(os.path.join(OUT_DIR, f"frame_{idx:06d}_annotated.png"), vis)
        with open(os.path.join(OUT_DIR, f"frame_{idx:06d}_corners.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"frame_index": idx, "valid": valid, "corners": corners}, f)


if __name__ == "__main__":
    main()
