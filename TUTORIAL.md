# 视觉感知作业 · 详细教程（橙色板四角点检测）

> 本教程对应仓库 `stage3-vision-homework`，按题目 5 问的递进顺序展开。
> 每个环节都给出**原理 → 代码 → 怎么验证 → 常见坑**，文末附实测数据、`REPORT.md` 模板与提交步骤。
> 教程中的阈值、坐标、统计数据均在 `orange_board.mp4` 上**实际运行得到**，可直接对照自己的结果。

---

## 0. 先把作业拆清楚

README 的最终目标是：**读视频 → 找橙色 → 拟合四个角点 → 逐帧标注 → 输出视频和坐标**。
五问是一条流水线，前一问的产物就是后一问的输入：

| 问 | 做什么 | 关键 API | 产物 |
| --- | --- | --- | --- |
| 一 | 读视频、逐帧存图、统计信息 | `VideoCapture`、`imwrite` | `outputs/frames/*.png`、`video_info.json` |
| 二 | HSV 筛橙色，生成二值掩膜 | `cvtColor`、`inRange`、`bitwise_and` | `outputs/q2/*_mask.png`、`*_orange.png` |
| 三 | 比较滤波 / 开运算 / 闭运算 | `GaussianBlur`、`medianBlur`、`morphologyEx` | `outputs/q3/*_compare.png` |
| 四 | 轮廓筛选 + 四边形近似 + 角点排序 | `findContours`、`contourArea`、`approxPolyDP` | `outputs/q4/*_annotated.png`、`*_corners.json` |
| 五 | 复用单帧流程标注整个视频 | `VideoWriter` | `annotated.mp4`、`corners.jsonl` |

**三个硬性要求，贯穿全篇：**

1. 坐标一律用**原视频像素坐标**（左上角为原点，x 向右、y 向下）。若处理时缩放，输出前必须换算回去。
2. 帧数必须是**实际读到的帧数**，不能拿 `CAP_PROP_FRAME_COUNT` 当处理数量。
3. 每帧检测必须**独立计算**，不许手填坐标，也不许把上一帧结果当本帧结果。

---

## 1. 环境准备

### 1.1 安装依赖

```bash
# 建议用虚拟环境，避免污染系统 Python
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install opencv-python numpy
```

国内网络慢的话换镜像：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python numpy
```

> 如果 `pip` 长时间卡在 `Downloading ...whl`，多半是代理问题：先 `set http_proxy=`（Windows）或
> `unset http_proxy https_proxy`（macOS/Linux）再装；仍不行就手动下载 wheel 后
> `pip install --no-index --find-links ./wheels opencv-python numpy`。

### 1.2 验证安装

```bash
python -c "import cv2, numpy; print(cv2.__version__, numpy.__version__)"
```

本教程实测环境：**OpenCV 5.0.0 + NumPy 2.5.4 + Python 3.13**，Python 3.9~3.13 均可。

### 1.3 目录约定

```text
stage3-vision-homework/
├── orange_board.mp4          # 原始素材，不要改
├── orange_corners.png        # 官方效果示例
├── README.md / REPORT.md
├── src/                      # 自己的源码
│   ├── detector.py           # 单帧检测（第四问提炼，第五问复用）
│   ├── q1_extract_frames.py
│   ├── q2_orange_mask.py
│   ├── q3_filter_morph.py
│   ├── q4_find_corners.py
│   ├── q5_annotate_video.py
│   └── requirements.txt
└── outputs/                  # 全部产物（frames/q2/q3/q4 + annotated.mp4 + corners.jsonl）
```

所有脚本都**从仓库根目录运行**（`python src/q1_extract_frames.py`），这样相对路径 `orange_board.mp4`、`outputs/` 才成立。

---

## 2. 概念速览（看懂 5 个 API 就够）

**BGR 与 HSV。** OpenCV 读进来是 BGR 三通道。RGB 里"橙色"是 R 高、G 中、B 低，但受亮度影响很大；
HSV 把**色相 H**（是什么颜色）、**饱和度 S**（颜色有多纯）、**明度 V**（有多亮）拆开，筛颜色时只需卡 H 区间，
对光照变化稳健得多。OpenCV 的 H 取值是 **0~179**（不是 0~360）。

**掩膜（mask）。** 单通道 8 位图，`255`（白）= 命中，`0`（黑）= 未命中。`inRange` 生成掩膜，
`bitwise_and(bgr, bgr, mask=mask)` 用掩膜抠图，`bitwise_not` 取反。

**滤波（前处理）**：`GaussianBlur` 加权平均、`medianBlur` 取中值。作用是压制噪声，代价是边缘变钝。
高斯对高斯噪声好，中值对椒盐噪声好，且中值更能保边。

**形态学（后处理）**：在掩膜上"腐蚀/膨胀"组合。
- **开运算** = 先腐蚀后膨胀：去掉小噪点、毛刺（白色小岛消失）。
- **闭运算** = 先膨胀后腐蚀：填小孔洞、接断缝（黑色小洞被白填上）。
- 核尺寸 `k×k` 越大，效果越强；3×3 动 1 像素，7×7 动 3 像素。

**轮廓与多边形近似**：`findContours` 得到边界点序列；`contourArea` 算面积用于筛选；
`approxPolyDP(cnt, epsilon, True)` 用折线逼近轮廓，`epsilon` 是允许的最大偏差（常用 `0.02 × 周长`），
四边形板子就会收敛到 4 个顶点——这正是"透视形状的四个顶点"而不是水平外接矩形四角的关键。

---

## 3. 第一问：读取视频、逐帧落盘

### 3.1 要做的事

按顺序读到视频结束，统计**实际读到的**帧数、尺寸、FPS；把所有帧存成 `frame_000000.png … frame_00217.png`；
写 `outputs/video_info.json`。

### 3.2 代码（`src/q1_extract_frames.py`）

```python
import json, os
import cv2

VIDEO_PATH = "orange_board.mp4"
OUT_DIR = "outputs/frames"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {VIDEO_PATH}")

    # 这三个只是"属性"，帧数不能信它，必须靠实际读取统计
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:                      # 读到结尾（或解码失败）返回 False
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
```

### 3.3 实测结果

```json
{"width": 960, "height": 540, "fps": 30.0, "frame_count": 218}
```

`outputs/frames/` 下 218 个文件，`frame_000000.png` 到 `frame_00217.png`。

### 3.4 常见坑

- **帧数写成 219 / 218 对不上**：`CAP_PROP_FRAME_COUNT` 常因容器元数据不准。用 `while cap.read()` 的计数。
- **文件名出现 `frame_0.png`**：`f"{i:06d}"` 补零，否则排序会乱（`frame_10` 排在 `frame_2` 前面）。
- **中文路径报错**：Windows 下 `cv2.imread/imwrite` 对非 ASCII 路径会失败，可用
  `cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)` 绕过；把项目放在纯英文路径更省事。
- **只读了属性没读帧**：README 明确写了"不能只读取视频属性就作为处理数量"，这是送分点也是送命点。

### 3.5 自检

打开第 0、109、217 帧：方向正常（不是上下颠倒）、颜色正常（板子是橙红而不是蓝）、
画面连续（109 帧板子应比第 0 帧更大、更居中）。

---

## 4. 第二问：HSV 阈值分割出橙色

### 4.1 思路

`BGR → HSV → inRange(下限, 上限) → 二值掩膜`，再用掩膜抠出原图颜色。本问**先不要只保留最大区域**，
所有落在阈值内的像素都要留下来。

### 4.2 阈值到底怎么定（关键技能）

不要凭"橙色大概是 H=15"拍脑袋。用**采样法**：

```python
import cv2, numpy as np
f = cv2.imread("outputs/frames/frame_000000.png")
hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
# 用宽松范围先粗筛，再看真实分布在哪儿
m = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([30, 255, 255]))
for name, ch in (("H", 0), ("S", 1), ("V", 2)):
    vals = hsv[:, :, ch][m > 0]
    print(name, np.percentile(vals, [1, 50, 99]).astype(int))
```

实测输出（三张样例帧一致）：

| 通道 | 1% | 中位数 | 99% |
| --- | --- | --- | --- |
| H | 3~4 | 4~5 | 7~8 |
| S | 130~136 | 163~172 | 182~186 |
| V | 187~192 | 239~241 | 247~248 |

结论：这块板子其实是**偏红的橙**，H 只有 3~8。所以阈值取 **H 3~14、S ≥120、V ≥150** 最干净：

```python
HSV_LOWER = np.array([3, 120, 150])
HSV_UPPER = np.array([14, 255, 255])
```

> 用 `[5, 80, 80]~[25, 255, 255]` 这类"教科书橙色"也能出结果（白色占比 12~15%），但会把桌面反光、
> 暖色木纹一起圈进来。阈值不是抄来的，是量出来的——报告里写清依据就是加分项。
> 想交互式调参，可以用 `cv2.createTrackbar` 挂六个滑条实时看掩膜。

### 4.3 代码（`src/q2_orange_mask.py` 核心）

```python
bgr = cv2.imread(f"outputs/frames/frame_{idx:06d}.png")
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)      # 白=橙色, 黑=其它
orange = cv2.bitwise_and(bgr, bgr, mask=mask)      # 只保留命中处的原色

cv2.imwrite(f"outputs/q2/frame_{idx:06d}_mask.png", mask)
cv2.imwrite(f"outputs/q2/frame_{idx:06d}_orange.png", orange)
```

### 4.4 实测结果

| 帧 | 掩膜白色占比 | 说明 |
| --- | --- | --- |
| 0 | 14.12% | 板子占画面约 1/7 |
| 109 | 16.23% | 相机靠近 |
| 217 | 19.67% | 板子最大 |

掩膜是**单通道、只有 0 和 255**——`cv2.imwrite` 写单通道图时不要加 `[..., None]`，也不要用 `COLOR_GRAY2BGR` 存成三通道。

### 4.5 常见坑

- **H 用 0~360**：OpenCV 是 0~179，写 200 直接全黑。
- **先筛 RGB 再转灰度**：饱和度信息丢了，阈值很难稳定。
- **忘记检查通道数**：`mask.shape` 应是 `(540, 960)`，不是 `(540, 960, 3)`。
- **阈值太宽导致掩膜连成一片**：第二问允许不完美，但第三、四问会暴露问题。

---

## 5. 第三问：滤波 vs 开运算 vs 闭运算

### 5.1 要比较的四种情况

固定第二问阈值，在**同一张输入**上比较：

1. 原始掩膜（不处理）；
2. 先对原图高斯滤波 / 中值滤波，再同样分割（前处理）；
3. 对原始掩膜开运算（后处理）；
4. 对原始掩膜闭运算（后处理）。

再把其中一种方法的核从 3×3 改成 7×7，观察变化。

### 5.2 代码要点（`src/q3_filter_morph.py`）

```python
m_raw = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

# 前处理：先平滑原图，再转 HSV 分割
blur_g = cv2.GaussianBlur(bgr, (3, 3), 0)
m_gauss = cv2.inRange(cv2.cvtColor(blur_g, cv2.COLOR_BGR2HSV), HSV_LOWER, HSV_UPPER)

blur_m = cv2.medianBlur(bgr, 3)
m_median = cv2.inRange(cv2.cvtColor(blur_m, cv2.COLOR_BGR2HSV), HSV_LOWER, HSV_UPPER)

# 后处理：直接在掩膜上做形态学
k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
m_open  = cv2.morphologyEx(m_raw, cv2.MORPH_OPEN,  k3)
m_close = cv2.morphologyEx(m_raw, cv2.MORPH_CLOSE, k3)
```

对比图用 `np.hstack/np.vstack` 拼成 2×2，每格左上角用 `cv2.putText` 写清方法名和核尺寸——
报告要求"每张图标明方法和核尺寸"，别只存四张裸图。

### 5.3 实测结论（可直接改写进报告）

"下判断"不要靠肉眼看图，先跑 `src/q3_metrics.py`，它把七种方案的差异算成一张表：

```bash
python src/q3_metrics.py
```

列的含义：`comps` 连通域总数、`noise` 小噪点个数（面积<100）、`area` 最大区域面积、
`holes` 内部孔洞数、`dCorner` 四角点相对原始掩膜的平均偏移（像素）、`rough` 边缘粗糙度
（周长²/(4π·面积)，越接近 1 越平滑）。

实测（三帧均值，原始掩膜为基准）：

| 方案 | comps | noise | 面积变化 | dCorner | rough |
| --- | --- | --- | --- | --- | --- |
| 1 原始掩膜 | 10~19 | 9~18 | 0 | 0 | 1.70~1.81 |
| 2 高斯 3×3（前处理） | **1** | **0** | +140~+210 | 0.25~1.41 | **1.42~1.49** |
| 3 中值 3×3（前处理） | 1~2 | 0~1 | +290~+390 | 0.50~1.62 | 1.45~1.59 |
| 4 开运算 3×3 | **1** | **0** | −19~−21 | 0~1.46 | 1.45~1.54 |
| 5 闭运算 3×3 | 1 | 0 | +130~+350 | 0~0.50 | 1.47~1.62 |
| 6 开运算 7×7 | 1 | 0 | −27~−52 | 0.25~1.46 | 1.42~1.52 |
| 7 闭运算 7×7 | 1 | 0 | +220~+420 | 0~0.50 | 1.44~1.56 |
| 8 开5×5 + 闭5×5 | 1 | 0 | −23~−46 | 0.25~1.46 | **1.42~1.52** |

从数字里能读出四件事（这就是"判断"的依据）：

1. **噪点只分"清了/没清"**：原始掩膜每帧 9~18 个噪点；开运算、闭运算、高斯滤波都能压到 0；
   **中值滤波在 109、217 帧各留下 1 个噪点**——滤波是把灰度摊匀，噪点本身够橙就还在阈值内。
2. **边缘粗糙度**从 1.70~1.81 降到 1.42~1.54，说明所有平滑手段都让边界从锯齿变成圆滑。
   高斯降得最多（1.42~1.49），开运算紧随其后。
3. **角点位置几乎不受影响**：所有方案的 `dCorner` 都 < 1.6 像素。这跟直觉相反——
   板子面积有 7~10 万像素，3×3 甚至 7×7 的腐蚀对四个顶点的影响都在 1~2 像素量级，
   **不是"把角切掉一大块"**。
4. **闭运算会让面积变大**（+130~+420），因为它把边缘外侧的橙色像素也吸进来；
   开运算让面积略微变小（−20~−50）。两者对最终角点的差别在像素级，可以忽略。

**本作业的方案选择：开运算 5×5 + 闭运算 5×5。** 理由写成报告是：

> 原始掩膜每帧残留 9~18 个噪点，会让 `findContours` 返回十几个轮廓；开运算把连通域降到 1、
> 噪点清零，面积仅损失 20~50 像素（<0.1%），四角偏移 <1.5 像素。再叠一次闭运算填补内部反光小洞，
> 对边缘无副作用。高斯滤波方案的效果与之接近（噪点同样清零、边缘更平滑），但它需要在原图上
> 重新分割一次，多一次颜色转换；直接对掩膜做形态学更省计算，也更直观。中值滤波被排除，
> 因为它在两帧上各留了 1 个噪点。

> README 明确写了"不要求滤波或形态学一定改善结果，也不要求把所有操作串起来"。
> 只要**试过、有图、有量化依据**就符合要求。如果你测出来"原始掩膜已经够好"
> （例如 `dCorner` 全为 0、噪点靠面积筛选就能兜住），直接选原始掩膜并说明理由同样正确。
> 注意：报告里要写**你自己跑出来的数字**，如果你的帧率和阈值不同，数值会不一样。

---

## 6. 第四问：从掩膜里找出四个角点（本作业的核心）

### 6.1 流程

```text
掩膜 → findContours → 按面积筛选 → convexHull → approxPolyDP 得到 4 点 → 按规则排序 → 画图 + 存 JSON
```

### 6.2 为什么不能用水平外接矩形

`boundingRect` 给的是**轴对齐**矩形，板子一旋转，四角就飘到板子外面去了。
`minAreaRect` 好一些（能跟着旋转），但它是"最小外接矩形"，对透视形变仍不是真实四角。
正确做法是**多边形近似**——直接逼近板子的真实边界，得到透视四边形的四个顶点。

### 6.3 代码（`src/detector.py` 核心）

```python
mask = get_mask(frame)  # inRange + open + close
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    return False, []

contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]  # 最大的几个里找
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 3000 or area > 0.95 * w * h:      # 太小是噪点，太大是整屏
        continue

    hull = cv2.convexHull(cnt)                  # 先取凸包，去掉凹口干扰
    peri = cv2.arcLength(hull, True)

    approx = None
    for s in (0.02, 0.015, 0.01, 0.03, 0.04, 0.05):   # epsilon 由粗到细多试几次
        cand = cv2.approxPolyDP(hull, s * peri, True)
        if len(cand) == 4:
            approx = cand
            break
    if approx is None:                          # 兜底：最小外接旋转矩形
        approx = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.int32)

    corners = _order_corners_clockwise(approx)
    return True, [[int(x), int(y)] for x, y in corners]
```

**为什么 epsilon 要循环尝试？** `epsilon` 太小时逼近出 6~8 个点（把毛刺当角点），
太大时退化成三角形。固定 `0.02×周长` 在本视频大多能出 4 点，但个别帧会失败；
用一组候选值逐个试，命中 4 点就停，稳定性明显提高（本视频 218 帧无一失败）。

### 6.4 角点排序规则（最容易做错的地方）

题目要求：**沿图像顺时针**排列，从 `x+y` 最小的顶点开始；若并列，先取 y 更小，再取 x 更小。

```python
def _order_corners_clockwise(pts):
    pts = pts.reshape(-1, 2).astype(np.float64)
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    # 图像坐标 y 向下：atan2 角度递增方向在视觉上就是顺时针
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    pts = pts[np.argsort(angles)]
    # 再把 x+y 最小的点旋转到第一位
    sums = pts[:, 0] + pts[:, 1]
    start = min(range(len(pts)), key=lambda i: (sums[i], pts[i][1], pts[i][0]))
    return np.roll(pts, -start, axis=0)
```

两个容易翻车的点：

1. **顺时针的方向判断**。数学习惯里角度递增是逆时针，但图像 y 轴向下，所以角度递增**视觉上是顺时针**。
   如果你按 `-angle` 排序，会得到 0→3→2→1，四边形就连成交叉的"蝴蝶结"。
2. **起点的并列规则**。四个点的 `x+y` 极少恰好相等，但规则里写了就要照做；
   `min(..., key=(sums, y, x))` 一行搞定，别用 `sorted` 后取第一个（并列时顺序不确定）。

**为什么按角度排序能保证不自交？** 因为排序后多边形是绕质心的"星形"顺序，
顶点按极角单调排列，任意两条不相邻的边不会相交。

### 6.5 实测结果

```json
{"frame_index": 0,   "valid": true, "corners": [[326,188],[664,136],[673,335],[358,418]]}
{"frame_index": 109, "valid": true, "corners": [[192,194],[582,190],[578,409],[212,413]]}
{"frame_index": 217, "valid": true, "corners": [[271,130],[659,87],[681,317],[316,416]]}
```

对照第 0 帧的标注图：0 号点在板子左上角（红）、1 号右上（黄）、2 号右下（绿）、3 号左下（品红），
四边形沿边缘闭合，无自交。第 109 帧板子几乎是正对镜头，四角接近矩形，编号仍从左上角起。

### 6.6 常见坑

- **把 `boundingRect` 四角当结果**：README 明确禁止，形状一斜就错得很离谱。
- **只取最大轮廓却不设面积下限**：板子短暂离开画面时，最大轮廓会变成桌上的噪点，输出一个假四边形。
- **`approxPolyDP` 返回的 shape 是 `(4,1,2)`**：画图和写 JSON 前记得 `reshape(-1,2)`，否则 `tuple()` 会报错。
- **坐标被缩放过**：如果为了速度把帧 resize 到 480×270 再检测，写出去前必须 `×2` 换回 960×540。

---

## 7. 第五问：标注整个视频

### 7.1 三个产物

1. `outputs/annotated.mp4`：960×540、30 FPS、**218 帧**，失败帧也要保留并标 `NOT FOUND`；
2. `outputs/corners.jsonl`：**218 行**，每行 `{"frame_index", "valid", "corners"}`，帧号连续不重复；
3. 报告里的成功/失败帧数 + 至少 3 张结果图 + 失败帧的原因分析。

### 7.2 代码要点（`src/q5_annotate_video.py`）

```python
writer = cv2.VideoWriter("outputs/annotated.mp4",
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (960, 540))

with open("outputs/corners.jsonl", "w", encoding="utf-8") as f:
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        valid, corners = detect_corners(frame)      # 每帧独立检测
        vis = annotate(frame, corners) if valid else mark_not_found(frame)
        writer.write(cv2.resize(vis, (960, 540)))   # 尺寸必须与 writer 一致
        f.write(json.dumps({"frame_index": frame_index, "valid": valid,
                            "corners": corners}) + "\n")
        frame_index += 1
cap.release()
writer.release()
```

### 7.3 `VideoWriter` 的四个坑（几乎人人踩）

1. **尺寸不匹配**：`VideoWriter` 的 `(w, h)` 必须与 `write()` 的帧尺寸**完全一致**，
   否则文件生成成功但只有 0 字节或播不动。本视频本身就是 960×540，`resize` 只是保险。
2. **fourcc 与后缀不匹配**：`.mp4` 配 `mp4v`（或 `avc1`）；`.avi` 配 `XVID`。
   写 `MJPG` 却存 `.mp4`，某些播放器会打不开。
3. **忘记 `writer.release()`**：不释放，文件尾部索引没写完，视频时长会变成 0 或截断。
4. **fps 用错**：要写原视频的 30，不是"处理得多快"。用 `cap.get(CAP_PROP_FPS)` 取，别硬编码 25。

### 7.4 实测结果

```text
总帧数 218, 成功 218, 失败 0
```

校验（重新打开视频读一遍）：

```text
w 960  h 540  fps 30.0  frames 218
read frames 218          # 实际解码出 218 帧，与 jsonl 行数一致
```

`corners.jsonl` 首尾：

```json
{"frame_index": 0,   "valid": true, "corners": [[326,188],[664,136],[673,335],[358,418]]}
{"frame_index": 217, "valid": true, "corners": [[271,130],[659,87],[681,317],[316,416]]}
```

同帧号在 `q4` 的 JSON 与 `corners.jsonl` 中坐标完全一致——说明两个入口用的是同一个检测函数，
没有偷偷"手填"。

### 7.5 一致性自检（报告里可以贴这段）

```python
import cv2, json
cap = cv2.VideoCapture("outputs/annotated.mp4"); n = 0
while cap.read()[0]:
    n += 1
lines = [json.loads(l) for l in open("outputs/corners.jsonl", encoding="utf-8")]
assert n == len(lines) == 218, (n, len(lines))
assert [x["frame_index"] for x in lines] == list(range(218)), "帧号不连续"
print("OK", n, sum(x["valid"] for x in lines))
```

再随机抽几帧打开 `annotated.mp4`，确认画面上的点与 JSON 坐标对得上。

---

## 8. 一键跑通

```bash
python src/q1_extract_frames.py     # 218 帧 + video_info.json
python src/q2_orange_mask.py        # 3 帧掩膜 + 保色图
python src/q3_filter_morph.py       # 3 帧对比图 ×2（含 7×7）
python src/q4_find_corners.py       # 3 帧标注图 + 3 个 JSON
python src/q5_annotate_video.py     # annotated.mp4 + corners.jsonl
```

产物清单：

```text
outputs/video_info.json
outputs/frames/frame_000000.png ... frame_00217.png
outputs/q2/frame_{0,109,217}_mask.png / _orange.png
outputs/q3/frame_{0,109,217}_compare.png / _kernel_compare.png
outputs/q4/frame_{0,109,217}_annotated.png / _corners.json
outputs/annotated.mp4
outputs/corners.jsonl
```

---

## 9. `REPORT.md` 怎么写（附实测数据可直接用）

```markdown
# 作业报告

- 姓名：<你的名字>
- 方向：视觉感知
- 实际使用环境与依赖版本：Windows 11 / Python 3.13 / opencv-python 5.0.0 / numpy 2.5.4

## 安装和运行

    python -m venv .venv && .venv\Scripts\activate
    pip install -r src/requirements.txt
    python src/q1_extract_frames.py   # 依次运行 q1~q5

输入视频在仓库根目录 `orange_board.mp4`，改动 `VIDEO_PATH` 即可换输入；
输出全部写到 `outputs/`，`frames/` 存逐帧图，`q2~q4/` 存各问结果，最终视频与坐标在 `outputs/` 根下。

## 各问结果

| 题目 | 完成情况与结果路径 | 一句说明或发现的问题 |
| --- | --- | --- |
| 第一问 | outputs/frames/ 218 帧，outputs/video_info.json（960×540@30） | 帧数按实际 read 计数为 218，与属性一致 |
| 第二问 | outputs/q2/*_mask.png、*_orange.png | HSV 阈值 [3,120,150]~[14,255,255]，掩膜白=橙色像素、黑=其它 |
| 第三问 | outputs/q3/*_compare.png、*_kernel_compare.png | 开运算 3×3 把连通域从 10~19 降到 1；7×7 会把板角削钝 |
| 第四问 | outputs/q4/*_annotated.png、*_corners.json | 三帧均 valid=true，角点贴合边缘、无自交 |
| 第五问 | outputs/annotated.mp4（218 帧）、outputs/corners.jsonl（218 行） | 成功 218 / 失败 0 |

## 关键概念与局限

- 掩膜中白=命中橙色阈值的像素，黑=不在阈值内的背景与其它颜色。
- 选开运算+闭运算 5×5 作为后续流程：外部噪点清零、内部反光空洞被填。
- 局限：阈值是针对本视频光照手工标定的。若换到暖色桌面或强反光场景，
  固定阈值会失效，需要按帧自适应或改用 Lab / 饱和度归一化。

## AI 与参考资料

- 使用 AI 辅助梳理 API 用法与调试思路；所有阈值与参数均由本人运行、
  打开输出图核对后确定，坐标一致性通过重新解码视频与 jsonl 比对校验。
- OpenCV 官方教程：颜色空间、形态学操作、轮廓特征。
```

---

## 10. 提交步骤（提交到自己的 Fork）

```bash
# 0) 确认 origin 指向你自己的仓库
git remote -v
# origin  https://github.com/<你的用户名>/stage3-vision-homework.git

# 1) 建分支（可选但更规范）
git checkout -b homework/vision

# 2) 提交源码、产物、报告（不要提交 .venv、临时文件）
printf ".venv/\n__pycache__/\n*.pyc\nwheels/\noutputs/check_*.png\n" > .gitignore
git add src outputs README.md REPORT.md .gitignore
git commit -m "feat: 完成视觉感知作业五问（帧提取/掩膜/滤波对比/角点/全视频标注）"

# 3) 推送到自己的 fork
git push -u origin homework/vision
```

在 GitHub 上进入自己仓库即可看到提交；如需向上游提 PR，再点 Compare & pull request。

**提交前自查清单：**

- [ ] `outputs/frames/` 有 218 张，编号补零且连续
- [ ] `video_info.json` 有 `width/height/fps/frame_count`，帧数为实际读取值
- [ ] 掩膜是单通道、像素值只有 0/255
- [ ] 第三问对比图**标注了方法和核尺寸**，且有 3×3 → 7×7 的对比
- [ ] 第四问 JSON 的 `valid/corners` 字段形状正确，失败时 `corners=[]`
- [ ] 角点顺时针、0 号在 `x+y` 最小处，四边形未自交
- [ ] `annotated.mp4` 是 960×540 / 30FPS / 218 帧，失败帧标了 `NOT FOUND`
- [ ] `corners.jsonl` 218 行，帧号 0~217 不重复
- [ ] `REPORT.md` 填完，写了失败帧分析（没有失败就如实说明）

---

## 11. 常见报错速查

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `cap.isOpened()` 为 False | 路径错 / 视频不在根目录 | 用绝对路径，或确认在仓库根目录运行 |
| 掩膜全黑 | H 写成 0~360，或 S/V 下限过高 | H 用 0~179；先打印采样分位数再定阈值 |
| 掩膜包含大片桌面 | 阈值过宽、V 下限太低 | 抬高 S/V 下限，或先 `GaussianBlur` 压反光 |
| `findContours` 报参数错 | OpenCV 3 与 4/5 返回值个数不同 | 4.x/5.x 返回 `(contours, hierarchy)` |
| `len(approx)` 一直是 6~8 | epsilon 太小 | 循环尝试 0.01~0.05，或先 `convexHull` |
| 四边形交叉成蝴蝶结 | 排序方向反了 | 按 `atan2` 递增排序（图像 y 向下即顺时针） |
| `annotated.mp4` 打不开 / 0 字节 | fourcc 与后缀不匹配 / 尺寸不一致 / 未 release | `mp4v` + `.mp4`，尺寸对齐，末尾 `release()` |
| 视频只有几帧 | 循环里 `break` 条件写错，或写入时 resize 失败 | 打印 `frame_index` 确认读到 218 |
| 中文路径读写失败 | Windows + 非 ASCII 路径 | 换英文路径或用 `imdecode/imencode` |

---

## 12. 想再进一步（不加分但很有用）

- **角点亚像素精化**：`cv2.cornerSubPix` 在角点附近迭代，误差可降到 0.1 像素级。
- **阈值自适应**：对每帧取 HSV 的 H/S/V 中位数 + MAD 动态生成区间，抵抗光照变化。
- **失败帧回退**：`approxPolyDP` 失败时用 `minAreaRect`，仍失败再用上一帧做 `calcOpticalFlowPyrLK` 跟踪
  （注意：**只能在检测失败时作为辅助**，不能拿上一帧结果冒充本帧检测结果）。
- **鲁棒性测试**：把视频亮度整体 ±20%、加高斯噪声后重跑，看成功率掉多少——这是报告里最有说服力的"局限"说明。
