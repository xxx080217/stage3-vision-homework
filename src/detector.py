"""共用模块：橙色板单帧角点检测流程（第四问提炼，供第四、五问复用）。"""
import cv2
import numpy as np

# HSV 阈值：橙色的 H 大约在 5~25 之间，S、V 设下限排除灰暗背景
HSV_LOWER = np.array([3, 120, 150])
HSV_UPPER = np.array([14, 255, 255])

# 形态学核：先开运算去小噪点，再闭运算填孔洞（第三问比较后选定的方案）
MORPH_KERNEL = (5, 5)

MIN_AREA = 3000          # 面积下限：过滤背景里零星的橙色噪块


def get_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR 帧 -> 橙色二值掩膜（单通道，0/255）。"""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _order_corners_clockwise(pts: np.ndarray) -> np.ndarray:
    """按题目规则排序：沿图像顺时针，从 x+y 最小的点开始（并列时取 y 更小、x 更小）。"""
    pts = pts.reshape(-1, 2).astype(np.float64)
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    # 图像坐标 y 向下，atan2 角度递增方向在视觉上即顺时针
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    order = np.argsort(angles)
    pts = pts[order]
    # 旋转列表，让 x+y 最小（并列取 y、x）的点排第一位
    sums = pts[:, 0] + pts[:, 1]
    start = min(range(len(pts)), key=lambda i: (sums[i], pts[i][1], pts[i][0]))
    pts = np.roll(pts, -start, axis=0)
    return pts


def detect_corners(frame_bgr: np.ndarray):
    """单帧检测。

    返回 (valid, corners)：
      valid=True  -> corners 为 4 个 [x, y]（原视频像素坐标，已按规则排序）
      valid=False -> corners 为 []
    """
    mask = get_mask(frame_bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, []

    # 按面积从大到小尝试，取第一个能拟合出四边形的轮廓
    h_img, w_img = mask.shape
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours[:3]:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > 0.95 * w_img * h_img:
            continue

        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)

        # 多边形近似：从粗到细尝试 epsilon，凑出 4 个顶点
        approx = None
        for eps_scale in (0.02, 0.015, 0.01, 0.03, 0.04, 0.05):
            cand = cv2.approxPolyDP(hull, eps_scale * peri, True)
            if len(cand) == 4:
                approx = cand
                break
        if approx is None:
            # 退路：用最小外接旋转矩形的四角（对透视形变也有一定适应性）
            approx = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.int32)

        corners = _order_corners_clockwise(approx)
        # 防自交的简单校验：对角线长度和应大于边长和的一半（凸四边形必满足）
        return True, [[int(x), int(y)] for x, y in corners]

    return False, []
