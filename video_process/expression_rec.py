# -*- coding: utf-8 -*-
# pip install py-feat opencv-python
from feat import Detector
import numpy as np
import os
detector = Detector()  # 默认包含 AU/VA/情绪

pred = detector.detect_image("./files/frame_0000.jpg")  # 单张图；视频可用 detect_video

# 1) AU 夸张度（简单平均，可换成加权）
au_cols = [c for c in pred.columns if c.endswith("_r")]  # AUxx_r 为强度列
au_score = float(pred[au_cols].clip(lower=0).mean(axis=1).iloc[0]) / 5.0  # 0-1 归一化

# 2) VA 夸张度（幅值）
v, a = float(pred["valence"].iloc[0]), float(pred["arousal"].iloc[0])  # 一般在 [-1,1]
va_score = (np.hypot(v, a) / np.sqrt(2))  # 0-1

# 组合分数（可调权重）
exaggeration = 0.6 * au_score + 0.4 * va_score
print(exaggeration)