"""第五问：复用单帧流程，标注整个视频。

运行（在仓库根目录）：python src/q5_annotate_video.py
输出：outputs/annotated.mp4（960x540, 30FPS, 218 帧）和 outputs/corners.jsonl（218 行）
"""
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(__file__))
from detector import detect_corners  # noqa: E402
from q4_find_corners import annotate  # noqa: E402

VIDEO_PATH = "orange_board.mp4"
OUT_VIDEO = "outputs/annotated.mp4"
OUT_JSONL = "outputs/corners.jsonl"


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # mp4v 编码器 + .mp4 后缀；输出分辨率按题目要求 960x540
    writer = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (960, 540))

    n_valid = n_invalid = 0
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            valid, corners = detect_corners(frame)   # 每帧独立检测，禁止沿用上一帧结果
            if valid:
                vis = annotate(frame, corners)
                n_valid += 1
            else:
                vis = frame.copy()
                cv2.putText(vis, "NOT FOUND", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                n_invalid += 1

            writer.write(cv2.resize(vis, (960, 540)))
            f.write(json.dumps({"frame_index": frame_index,
                                "valid": valid, "corners": corners}) + "\n")
            frame_index += 1

    cap.release()
    writer.release()
    print(f"总帧数 {frame_index}, 成功 {n_valid}, 失败 {n_invalid}")


if __name__ == "__main__":
    main()
