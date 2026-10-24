"""第一问：读取视频，逐帧保存并统计信息。

运行（在仓库根目录）：python src/q1_extract_frames.py
输出：outputs/frames/frame_000000.png ... 和 outputs/video_info.json
"""
import json
import os

import cv2

VIDEO_PATH = "orange_board.mp4"
OUT_DIR = "outputs/frames"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {VIDEO_PATH}")

    # 注意：这里的 fps/width/height 只是"属性"，帧数必须靠实际读取统计
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:          # 读到结尾或出错都会返回 False
            break
        cv2.imwrite(os.path.join(OUT_DIR, f"frame_{frame_count:06d}.png"), frame)
        frame_count += 1
    cap.release()

    info = {"width": width, "height": height, "fps": fps, "frame_count": frame_count}
    with open("outputs/video_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
