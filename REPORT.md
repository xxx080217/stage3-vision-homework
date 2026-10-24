# 作业报告

- 姓名：谢晓旭
- 方向：视觉感知
- 实际使用环境与依赖版本：Windows 11 / Python 3.14.8 / opencv-python 5.0.0 / numpy 2.5.4

## 安装和运行

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r src\requirements.txt

# 依次运行五问（也可直接运行 .\run.cmd 一键跑完）
.\.venv\Scripts\python.exe src\q1_extract_frames.py
.\.venv\Scripts\python.exe src\q2_orange_mask.py
.\.venv\Scripts\python.exe src\q3_filter_morph.py
.\.venv\Scripts\python.exe src\q3_metrics.py
.\.venv\Scripts\python.exe src\q4_find_corners.py
.\.venv\Scripts\python.exe src\q5_annotate_video.py
```

更换输入：修改各脚本顶部的 `VIDEO_PATH`（默认 `orange_board.mp4`，放在仓库根目录）。

输出位置：全部写在 `outputs/` 下——`frames/` 为逐帧图，`q2/q3/q4/` 为各问结果，
`annotated.mp4` 与 `corners.jsonl` 为最终交付。`outputs/frames/` 体积较大（约 70 MB），
已在 `.gitignore` 中排除，可用 `q1` 一条命令重新生成。

## 各问结果

| 题目 | 完成情况与结果路径 | 一句说明或发现的问题 |
| --- | --- | --- |
| 第一问 | `outputs/frames/` 共 218 张；`outputs/video_info.json`（960×540, 30 FPS） | 帧数按 `cap.read()` 实际循环计数得到 218，未使用 `CAP_PROP_FRAME_COUNT` 属性值 |
| 第二问 | `outputs/q2/frame_{0,109,217}_mask.png`、`*_orange.png` | HSV 阈值取 `[3,120,150]~[14,255,255]`：板子实测 H 仅 3~8，属偏红的橙，教科书的 5~25 会带入桌面暖色 |
| 第三问 | `outputs/q3/*_compare.png`、`*_kernel_compare.png`；量化表见 `src/q3_metrics.py` | 原始掩膜每帧 9~18 个噪点，开运算后连通域降为 1、噪点清零、面积仅少 20~50 px；中值滤波在 109/217 帧各残留 1 个噪点 |
| 第四问 | `outputs/q4/frame_{0,109,217}_annotated.png`、`*_corners.json` | 三帧均 `valid=true`，四角贴合板子真实顶点，按顺时针编号、0 号位于 x+y 最小处 |
| 第五问 | `outputs/annotated.mp4`（218 帧）、`outputs/corners.jsonl`（218 行） | 成功 218 帧 / 失败 0 帧；逐帧独立检测，未沿用上一帧结果 |

各问关键数据：

- 掩膜白色像素占比：第 0 帧 14.12%，第 109 帧 16.23%，第 217 帧 19.67%（板子逐渐靠近镜头）。
- 掩膜为单通道 `(540, 960) uint8`，像素值仅 `0` 与 `255`。
- 第三问量化（`dCorner` 为四角相对原始掩膜的平均偏移）：开运算 3×3 噪点清零、`dCorner` 0~1.46 px、面积 −19~−21 px；
  开运算 7×7 面积 −27~−52 px，说明增大核并未显著"切角"；中值滤波 3×3 残留 1 个噪点。
- 第四问角点示例（第 0 帧）：`[[326,188],[664,136],[673,335],[358,418]]`，
  对应 x+y 为 514/800/1008/776，确认 0 号为最小、顺序为顺时针。
- 第五问视频经重新解码校验：960×540、30 FPS、实际解出 218 帧，与 `corners.jsonl` 行数一致；
  同帧号在 `outputs/q4` 与 `corners.jsonl` 中的坐标完全相同。

## 关键概念与局限

**掩膜中黑白的含义**：白色（255）表示该像素的 HSV 落在所选橙色阈值区间内，被判定为橙色板；
黑色（0）表示不在区间内，属于背景或其他颜色。

**最终采用的方案**：HSV 阈值 `[3,120,150]~[14,255,255]` → 开运算 5×5（清外部噪点）→ 闭运算 5×5（填内部反光小洞）
→ `findContours` 取最大轮廓 → `convexHull` + `approxPolyDP` 拟合四点（epsilon 在 0.01~0.05×周长之间逐档尝试，
命中 4 点即停，失败时退化为 `minAreaRect`）→ 按极角顺时针排序并从 x+y 最小点起编。

**选择理由**：开运算把连通域从 10~19 降到 1、噪点清零，面积仅损失 0.03~0.07%，四角偏移 <1.5 px；
闭运算补齐内部空洞且不改变外轮廓。高斯滤波效果相当但需重新分割一次，故未采用；中值滤波因残留噪点被排除。

**局限**：

1. 阈值是针对本视频光照手工标定的。若换到暖色木桌、强反光或白平衡偏黄的场景，固定阈值会把背景误判进来，
   需要改为按帧自适应（如取每帧 H/S 分位数）或改用饱和度归一化。
2. `approxPolyDP` 依赖板子轮廓是"凸四边形"。若板子被部分遮挡、贴边或出现强反光缺口，近似可能得到 3 或 5 个点，
   此时会走到 `minAreaRect` 兜底，角点精度下降。
3. 本视频 218 帧全部检测成功，因此没有失败帧样例可供分析；上述第 1、2 点是基于方法原理的预判，
   未在本视频中实际观察到失败。

## AI 与参考资料

本次作业使用 AI 辅助梳理 OpenCV API 用法、设计量化对比指标（`q3_metrics.py`）以及排查环境问题（PowerShell 执行策略、
虚拟环境路径）。所有阈值、参数与结论均由本人运行脚本、打开输出图片与视频核对后确认：例如第三问的"中值滤波残留噪点"
结论来自 `connectedComponentsWithStats` 的实际计数，而非主观判断；第五问的一致性通过重新解码 `annotated.mp4`
并与 `corners.jsonl` 逐帧比对验证。

参考资料：

- OpenCV 颜色空间与颜色筛选：https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
- OpenCV 形态学操作：https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
- OpenCV 轮廓特征与多边形近似：https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
