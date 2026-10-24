"""第三问配套：把各处理方案的掩膜质量量化成表格，用于"下判断"。

用法（在仓库根目录）：
    .venv\\Scripts\\python.exe src\\q3_metrics.py

输出一张表格，列的含义：
    comps    连通域总数（含噪点）       —— 越少越干净
    noise    小噪点个数（面积<100）      —— 越少越干净
    area     最大区域面积（像素）        —— 反映板子是否被腐蚀变小
    holes    最大区域内部孔洞个数        —— 越少越完整
    holeArea 孔洞总面积                  —— 同上
    dCorner  四角点相对基准的平均偏移(px) —— 越小越贴近真实角点
    rough    边缘粗糙度 = 周长²/(4π·面积) —— 越接近 1 越平滑
"""
import os

import cv2
import numpy as np

FRAMES = [0, 109, 217]
FRAME_DIR = "outputs/frames"
HSV_LOWER = np.array([3, 120, 150])
HSV_UPPER = np.array([14, 255, 255])


def seg(bgr):
    return cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), HSV_LOWER, HSV_UPPER)


def four_corners(mask):
    """从掩膜取四角点（与第四问同一套逻辑，供比较用）。"""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    for s in (0.02, 0.015, 0.01, 0.03, 0.04, 0.05):
        cand = cv2.approxPolyDP(hull, s * peri, True)
        if len(cand) == 4:
            return cand.reshape(-1, 2).astype(np.float64)
    return cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float64)


def metrics(mask, base_corners):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    comps = n - 1
    small = sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] < 100)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return dict(comps=comps, noise=small, area=0, holes=0, holeArea=0,
                    dCorner=float("nan"), rough=float("nan"))
    big = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(big)

    # 孔洞：把最大轮廓内部填满，再减去原掩膜，剩下的黑块就是洞
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [big], -1, 255, -1)
    holes_img = cv2.bitwise_and(filled, cv2.bitwise_not(mask))
    hn, _, hstats, _ = cv2.connectedComponentsWithStats(holes_img, 8)
    holes = sum(1 for i in range(1, hn) if hstats[i, cv2.CC_STAT_AREA] > 20)
    hole_area = int(holes_img.sum() / 255)

    corners = four_corners(mask)
    if corners is None or base_corners is None:
        dcorner = float("nan")
    else:
        # 两组四点一一对应：都按同一规则排序后比较
        dcorner = float(np.mean(np.linalg.norm(
            sort_pts(corners) - sort_pts(base_corners), axis=1)))

    peri = cv2.arcLength(big, True)
    rough = peri ** 2 / (4 * np.pi * area) if area else float("nan")
    return dict(comps=comps, noise=small, area=int(area), holes=holes,
                holeArea=hole_area, dCorner=dcorner, rough=rough)


def sort_pts(p):
    p = np.asarray(p, dtype=np.float64).reshape(-1, 2)
    c = p.mean(axis=0)
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    p = p[np.argsort(ang)]
    s = p[:, 0] + p[:, 1]
    k = min(range(len(p)), key=lambda i: (s[i], p[i][1], p[i][0]))
    return np.roll(p, -k, axis=0)


def main():
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    k7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    print(f"{'frame':>5} {'method':<16}{'comps':>6}{'noise':>7}{'area':>9}"
          f"{'holes':>7}{'holeArea':>9}{'dCorner':>9}{'rough':>8}")
    print("-" * 80)

    for idx in FRAMES:
        bgr = cv2.imread(os.path.join(FRAME_DIR, f"frame_{idx:06d}.png"))
        raw = seg(bgr)
        base = four_corners(raw)          # 以原始掩膜的角点作为基准

        cases = {
            "1 raw": raw,
            "2 gauss3-pre": seg(cv2.GaussianBlur(bgr, (3, 3), 0)),
            "3 median3-pre": seg(cv2.medianBlur(bgr, 3)),
            "4 open3": cv2.morphologyEx(raw, cv2.MORPH_OPEN, k3),
            "5 close3": cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k3),
            "6 open7": cv2.morphologyEx(raw, cv2.MORPH_OPEN, k7),
            "7 close7": cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k7),
            "8 open5+close5": cv2.morphologyEx(
                cv2.morphologyEx(raw, cv2.MORPH_OPEN, k5), cv2.MORPH_CLOSE, k5),
        }
        for name, m in cases.items():
            r = metrics(m, base)
            print(f"{idx:>5} {name:<16}{r['comps']:>6}{r['noise']:>7}{r['area']:>9}"
                  f"{r['holes']:>7}{r['holeArea']:>9}{r['dCorner']:>9.2f}{r['rough']:>8.3f}")
        print("-" * 80)


if __name__ == "__main__":
    main()
